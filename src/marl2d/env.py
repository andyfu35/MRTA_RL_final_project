from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import AGENT_IDS

RUNNER_INDICES = (0, 1)
BLOCKER_INDICES = (2, 3)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _body_frame(dx: float, dy: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * dx + s * dy, -s * dx + c * dy


class VectorArena2D:
    """Vectorized deterministic 2D differential-drive arena.

    State layout: [num_envs, 4, (x, y, theta)].
    Action layout: [num_envs, 4, (left, right)] in [-1, 1].
    """

    observation_dim = 21
    action_dim = 2

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        seed: int = 0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.width = float(env_cfg["width"])
        self.height = float(env_cfg["height"])
        self.dt = float(env_cfg["dt"])
        self.wheel_radius = float(env_cfg["wheel_radius"])
        self.wheel_base = float(env_cfg["wheel_base"])
        self.max_wheel_speed = float(env_cfg["max_wheel_speed"])
        self.robot_radius = float(env_cfg["robot_radius"])
        self.goal = np.asarray(env_cfg["goal"], dtype=np.float32)
        self.goal_radius = float(env_cfg["goal_radius"])
        self.max_steps = int(env_cfg["max_steps"])
        self.lidar_rays = int(env_cfg["lidar_rays"])
        self.lidar_range = float(env_cfg["lidar_range"])
        self.reset_jitter = float(env_cfg.get("reset_jitter", 0.0))
        self.obstacles = np.asarray(env_cfg.get("obstacles", []), dtype=np.float32).reshape(-1, 3)
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros((self.num_envs, 4, 3), dtype=np.float32)
        self.prev_action = np.zeros((self.num_envs, 4, 2), dtype=np.float32)
        self.collision = np.zeros((self.num_envs, 4), dtype=bool)
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.reset(seed=seed)

    def _base_state(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 2.0, 0.0],
                [1.0, 4.0, 0.0],
                [6.5, 2.2, math.pi],
                [6.5, 3.8, math.pi],
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        base = self._base_state()
        self.state[:] = base[None, :, :]
        if self.reset_jitter > 0:
            jitter = self.rng.uniform(
                -self.reset_jitter,
                self.reset_jitter,
                size=(self.num_envs, 4, 2),
            ).astype(np.float32)
            self.state[:, :, :2] += jitter
            heading_jitter = self.rng.uniform(-0.08, 0.08, size=(self.num_envs, 4)).astype(np.float32)
            self.state[:, :, 2] = _wrap_angle(self.state[:, :, 2] + heading_jitter)
        self.prev_action.fill(0.0)
        self.collision.fill(False)
        self.steps.fill(0)
        return self.observe()

    def reset_indices(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError("reset mask has wrong shape")
        count = int(mask.sum())
        if count == 0:
            return
        base = self._base_state()
        self.state[mask] = base[None, :, :]
        if self.reset_jitter > 0:
            jitter = self.rng.uniform(-self.reset_jitter, self.reset_jitter, size=(count, 4, 2)).astype(np.float32)
            self.state[mask, :, :2] += jitter
            heading_jitter = self.rng.uniform(-0.08, 0.08, size=(count, 4)).astype(np.float32)
            self.state[mask, :, 2] = _wrap_angle(self.state[mask, :, 2] + heading_jitter)
        self.prev_action[mask] = 0.0
        self.collision[mask] = False
        self.steps[mask] = 0

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.state.shape:
            raise ValueError(f"Expected state shape {self.state.shape}, got {state.shape}")
        self.state[:] = state
        self.collision.fill(False)

    def _goal_distances(self, state: np.ndarray | None = None) -> np.ndarray:
        s = self.state if state is None else state
        diff = s[:, :2, :2] - self.goal[None, None, :]
        return np.linalg.norm(diff, axis=-1)

    def _integrate(self, actions: np.ndarray) -> np.ndarray:
        actions = np.clip(actions, -1.0, 1.0)
        omega_l = actions[:, :, 0] * self.max_wheel_speed
        omega_r = actions[:, :, 1] * self.max_wheel_speed
        v = 0.5 * self.wheel_radius * (omega_r + omega_l)
        yaw_rate = (self.wheel_radius / self.wheel_base) * (omega_r - omega_l)

        old = self.state
        theta = old[:, :, 2]
        dtheta = yaw_rate * self.dt
        new_theta = _wrap_angle(theta + dtheta)

        proposed = old.copy()
        straight = np.abs(yaw_rate) < 1e-8
        proposed[:, :, 0][straight] = old[:, :, 0][straight] + v[straight] * np.cos(theta[straight]) * self.dt
        proposed[:, :, 1][straight] = old[:, :, 1][straight] + v[straight] * np.sin(theta[straight]) * self.dt

        turning = ~straight
        radius = np.zeros_like(v)
        radius[turning] = v[turning] / yaw_rate[turning]
        proposed[:, :, 0][turning] = old[:, :, 0][turning] + radius[turning] * (
            np.sin(new_theta[turning]) - np.sin(theta[turning])
        )
        proposed[:, :, 1][turning] = old[:, :, 1][turning] - radius[turning] * (
            np.cos(new_theta[turning]) - np.cos(theta[turning])
        )
        proposed[:, :, 2] = new_theta
        return proposed

    def _collision_mask(self, proposed: np.ndarray) -> np.ndarray:
        collided = np.zeros((self.num_envs, 4), dtype=bool)
        x = proposed[:, :, 0]
        y = proposed[:, :, 1]
        r = self.robot_radius

        collided |= (x < r) | (x > self.width - r) | (y < r) | (y > self.height - r)

        for ox, oy, oradius in self.obstacles:
            d2 = (x - float(ox)) ** 2 + (y - float(oy)) ** 2
            collided |= d2 < (r + float(oradius)) ** 2

        min_robot_dist2 = (2.0 * r) ** 2
        for a in range(4):
            for b in range(a + 1, 4):
                d2 = np.sum((proposed[:, a, :2] - proposed[:, b, :2]) ** 2, axis=-1)
                hit = d2 < min_robot_dist2
                collided[:, a] |= hit
                collided[:, b] |= hit
        return collided

    def _blocker_proximity(self) -> np.ndarray:
        # [env, blocker] normalized closeness to either runner.
        runners = self.state[:, :2, :2]
        blockers = self.state[:, 2:, :2]
        diff = blockers[:, :, None, :] - runners[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1).min(axis=-1)
        scale = max(self.width, self.height)
        return np.clip(1.0 - dist / scale, 0.0, 1.0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, 4, 2)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)

        old_state = self.state.copy()
        old_goal_dist = self._goal_distances(old_state)
        old_team_dist = old_goal_dist.min(axis=1)

        proposed = self._integrate(actions)
        collision = self._collision_mask(proposed)
        proposed[collision] = old_state[collision]
        self.state[:] = proposed
        self.collision[:] = collision
        self.prev_action[:] = actions
        self.steps += 1

        new_goal_dist = self._goal_distances(self.state)
        new_team_dist = new_goal_dist.min(axis=1)
        team_progress = old_team_dist - new_team_dist
        self_progress = old_goal_dist - new_goal_dist

        goal_reached = new_team_dist <= self.goal_radius
        timeout = self.steps >= self.max_steps
        done = goal_reached | timeout

        rewards = np.zeros((self.num_envs, 4), dtype=np.float32)
        rr = self.reward_cfg["runner"]
        rb = self.reward_cfg["blocker"]
        for idx in RUNNER_INDICES:
            rewards[:, idx] = (
                float(rr["step"])
                + float(rr["team_progress"]) * team_progress
                + float(rr["self_progress"]) * self_progress[:, idx]
                + float(rr["collision"]) * collision[:, idx].astype(np.float32)
                + float(rr["goal_bonus"]) * goal_reached.astype(np.float32)
            )

        proximity = self._blocker_proximity()
        for local_idx, idx in enumerate(BLOCKER_INDICES):
            rewards[:, idx] = (
                float(rb["step"])
                + float(rb["runner_progress"]) * team_progress
                + float(rb["collision"]) * collision[:, idx].astype(np.float32)
                + float(rb.get("proximity", 0.0)) * proximity[:, local_idx]
                + float(rb["goal_failure"]) * goal_reached.astype(np.float32)
                + float(rb["timeout_bonus"]) * (timeout & ~goal_reached).astype(np.float32)
            )

        info = {
            "collision": collision.copy(),
            "goal_reached": goal_reached.copy(),
            "timeout": timeout.copy(),
            "team_progress": team_progress.astype(np.float32),
        }
        return self.observe(), rewards, done, info

    def _ray_distance(self, x: float, y: float, angle: float) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        eps = 1e-9
        best = self.lidar_range

        if abs(dx) > eps:
            for wall_x in (0.0, self.width):
                t = (wall_x - x) / dx
                if t > 0:
                    yy = y + t * dy
                    if 0.0 <= yy <= self.height:
                        best = min(best, t)
        if abs(dy) > eps:
            for wall_y in (0.0, self.height):
                t = (wall_y - y) / dy
                if t > 0:
                    xx = x + t * dx
                    if 0.0 <= xx <= self.width:
                        best = min(best, t)

        for ox, oy, radius in self.obstacles:
            rel_x = float(ox) - x
            rel_y = float(oy) - y
            proj = rel_x * dx + rel_y * dy
            perp2 = rel_x * rel_x + rel_y * rel_y - proj * proj
            radius2 = float(radius) ** 2
            if perp2 > radius2:
                continue
            half = math.sqrt(max(0.0, radius2 - perp2))
            t0 = proj - half
            t1 = proj + half
            candidates = [t for t in (t0, t1) if t > 0.0]
            if candidates:
                best = min(best, min(candidates))
        return float(min(best, self.lidar_range))

    def observe(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, 4, self.observation_dim), dtype=np.float32)
        norm_xy = max(self.width, self.height)
        ray_offsets = np.linspace(-math.pi, math.pi, self.lidar_rays, endpoint=False)

        for e in range(self.num_envs):
            for i in range(4):
                x, y, theta = (float(v) for v in self.state[e, i])
                k = 0

                gx, gy = _body_frame(float(self.goal[0] - x), float(self.goal[1] - y), theta)
                obs[e, i, k : k + 2] = (gx / norm_xy, gy / norm_xy)
                k += 2

                obs[e, i, k : k + 2] = (math.sin(theta), math.cos(theta))
                k += 2

                for j in range(4):
                    if j == i:
                        continue
                    dx = float(self.state[e, j, 0] - x)
                    dy = float(self.state[e, j, 1] - y)
                    bx, by = _body_frame(dx, dy, theta)
                    obs[e, i, k : k + 2] = (bx / norm_xy, by / norm_xy)
                    k += 2

                for offset in ray_offsets:
                    obs[e, i, k] = self._ray_distance(x, y, theta + float(offset)) / self.lidar_range
                    k += 1

                obs[e, i, k] = float(self.collision[e, i])
                k += 1
                obs[e, i, k : k + 2] = self.prev_action[e, i]
                k += 2
                if k != self.observation_dim:
                    raise RuntimeError(f"Observation assembly bug for {AGENT_IDS[i]}: {k}")
        return obs
