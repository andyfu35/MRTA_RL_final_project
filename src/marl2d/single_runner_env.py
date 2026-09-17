from __future__ import annotations

import math
from typing import Any

import numpy as np

from .env import VectorArena2D, _body_frame, _point_to_rect_distance, _wrap_angle


class SingleRunnerArena2D(VectorArena2D):
    """Single-runner navigation environment built on the shared 2D arena geometry.

    Internally the parent arena still stores four state slots so obstacle generation,
    kinematics, collision checks, and seed streams stay identical to the future
    multi-agent experiment. Slots 1-3 are inert ghost states placed outside the map.

    Observation layout (15 values):
      goal relative xy (2), heading sin/cos (2), lidar (8), collision (1),
      previous left/right action (2).
    """

    observation_dim = 15
    action_dim = 2

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        seed: int = 0,
    ) -> None:
        self.single_reward_cfg = dict(reward_cfg)
        self.reward_mode = str(reward_cfg.get("mode", "")).upper()
        if self.reward_mode not in {"R0", "R1", "R2", "R3"}:
            raise ValueError("single-runner reward mode must be one of R0, R1, R2, R3")
        parent_reward = {
            "runner": {
                "goal_bonus": 0.0,
                "team_progress": 0.0,
                "self_progress": 0.0,
                "collision": 0.0,
                "step": 0.0,
            },
            "blocker": {
                "runner_progress": 0.0,
                "timeout_bonus": 0.0,
                "goal_failure": 0.0,
                "collision": 0.0,
                "step": 0.0,
                "proximity": 0.0,
            },
        }
        super().__init__(num_envs, env_cfg, parent_reward, seed=seed)

    def _base_state(self) -> np.ndarray:
        ghost_x = self.width + 100.0
        ghost_y = self.height + 100.0
        return np.array(
            [
                [self.width * 0.10, self.height * 0.50, 0.0],
                [ghost_x, ghost_y, 0.0],
                [ghost_x + 10.0, ghost_y + 10.0, 0.0],
                [ghost_x + 20.0, ghost_y + 20.0, 0.0],
            ],
            dtype=np.float32,
        )

    @property
    def runner_state(self) -> np.ndarray:
        return self.state[:, 0, :]

    def set_runner_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        expected = (self.num_envs, 3)
        if state.shape != expected:
            raise ValueError(f"Expected runner state shape {expected}, got {state.shape}")
        self.state[:, 0, :] = state
        self.collision[:, 0] = False

    def observe(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.observation_dim), dtype=np.float32)
        norm_xy = max(self.width, self.height)
        ray_offsets = np.linspace(-math.pi, math.pi, self.lidar_rays, endpoint=False)

        for env_index in range(self.num_envs):
            x, y, theta = (float(v) for v in self.state[env_index, 0])
            k = 0
            gx, gy = _body_frame(float(self.goal[0] - x), float(self.goal[1] - y), theta)
            obs[env_index, k : k + 2] = (gx / norm_xy, gy / norm_xy)
            k += 2
            obs[env_index, k : k + 2] = (math.sin(theta), math.cos(theta))
            k += 2
            for offset in ray_offsets:
                obs[env_index, k] = (
                    self._ray_distance(env_index, x, y, theta + float(offset)) / self.lidar_range
                )
                k += 1
            obs[env_index, k] = float(self.collision[env_index, 0])
            k += 1
            obs[env_index, k : k + 2] = self.prev_action[env_index, 0]
            k += 2
            if k != self.observation_dim:
                raise RuntimeError(f"Single-runner observation assembly bug: {k}")
        return obs

    def _minimum_clearance(self) -> np.ndarray:
        """Distance from robot footprint to nearest wall/obstacle surface."""
        clearance = np.zeros(self.num_envs, dtype=np.float32)
        for env_index in range(self.num_envs):
            x = float(self.state[env_index, 0, 0])
            y = float(self.state[env_index, 0, 1])
            wall_clearance = min(
                x - self.robot_radius,
                self.width - x - self.robot_radius,
                y - self.robot_radius,
                self.height - y - self.robot_radius,
            )
            best = float(wall_clearance)
            for rect in self.obstacles[env_index]:
                center_to_rect = _point_to_rect_distance(x, y, rect)
                best = min(best, center_to_rect - self.robot_radius)
            clearance[env_index] = float(best)
        return clearance

    def _heading_error(self) -> np.ndarray:
        dx = self.goal[0] - self.state[:, 0, 0]
        dy = self.goal[1] - self.state[:, 0, 1]
        target = np.arctan2(dy, dx)
        return _wrap_angle(target - self.state[:, 0, 2]).astype(np.float32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, 2)
        if actions.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {actions.shape}")
        all_actions = np.zeros((self.num_envs, 4, 2), dtype=np.float32)
        all_actions[:, 0, :] = np.clip(actions, -1.0, 1.0)
        obs, _parent_rewards, parent_done, parent_info = super().step(all_actions)

        collision = parent_info["collision"][:, 0].copy()
        goal_reached = parent_info["goal_reached"].copy()
        timeout = parent_info["timeout"].copy()
        progress = parent_info["team_progress"].copy()
        min_clearance = self._minimum_clearance()
        heading_error = self._heading_error()

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        progress_component = np.zeros_like(rewards)
        time_component = np.zeros_like(rewards)
        safety_component = np.zeros_like(rewards)
        heading_component = np.zeros_like(rewards)

        cfg = self.single_reward_cfg
        if self.reward_mode in {"R1", "R2", "R3"}:
            progress_component = float(cfg["progress_scale"]) * progress
            time_component.fill(float(cfg["step_penalty"]))
            rewards += progress_component + time_component

        if self.reward_mode in {"R2", "R3"}:
            safety_distance = float(cfg["safety_distance"])
            danger = np.maximum(0.0, (safety_distance - min_clearance) / safety_distance)
            safety_component = -float(cfg["safety_scale"]) * danger.astype(np.float32) ** 2
            rewards += safety_component

        if self.reward_mode == "R3":
            heading_component = float(cfg["heading_scale"]) * np.cos(heading_error)
            rewards += heading_component.astype(np.float32)

        # Terminal outcomes override shaping so all ablation modes share the
        # exact same success/failure scale.
        rewards[timeout] = float(cfg["timeout_penalty"])
        rewards[collision] = float(cfg["collision_penalty"])
        rewards[goal_reached] = float(cfg["goal_bonus"])
        done = parent_done | collision

        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "timeout": timeout,
            "progress": progress,
            "min_clearance": min_clearance,
            "heading_error": heading_error,
            "progress_reward": progress_component,
            "safety_reward": safety_component,
            "heading_reward": heading_component,
            "time_reward": time_component,
            "map_seed": parent_info["map_seed"].copy(),
        }
        return obs, rewards, done, info
