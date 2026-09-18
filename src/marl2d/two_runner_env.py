from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .reward_api import RewardContext
from .reward_loader import RewardEngine


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _body_frame(dx: float, dy: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * dx + s * dy, -s * dx + c * dy


def _point_to_rect_distance(point_x: float, point_y: float, rect: np.ndarray) -> float:
    cx, cy, width, height = (float(v) for v in rect)
    dx = max(abs(point_x - cx) - width * 0.5, 0.0)
    dy = max(abs(point_y - cy) - height * 0.5, 0.0)
    return math.hypot(dx, dy)


def _rects_too_close(a: np.ndarray, b: np.ndarray, spacing: float) -> bool:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    return (
        abs(ax - bx) < (aw + bw) * 0.5 + spacing
        and abs(ay - by) < (ah + bh) * 0.5 + spacing
    )


def _ray_aabb_distance(x: float, y: float, dx: float, dy: float, rect: np.ndarray) -> float | None:
    cx, cy, width, height = (float(v) for v in rect)
    xmin = cx - width * 0.5
    xmax = cx + width * 0.5
    ymin = cy - height * 0.5
    ymax = cy + height * 0.5
    t_near = 0.0
    t_far = float("inf")
    eps = 1e-10
    for origin, direction, low, high in ((x, dx, xmin, xmax), (y, dy, ymin, ymax)):
        if abs(direction) < eps:
            if origin < low or origin > high:
                return None
            continue
        t0 = (low - origin) / direction
        t1 = (high - origin) / direction
        if t0 > t1:
            t0, t1 = t1, t0
        t_near = max(t_near, t0)
        t_far = min(t_far, t1)
        if t_near > t_far:
            return None
    if t_far <= 0.0:
        return None
    return t_near if t_near > 0.0 else t_far


class TwoRunnerArena2D:
    """Vectorized two-runner cooperative navigation arena.

    State shape is ``[num_envs, 2, (x, y, theta)]`` and action shape is
    ``[num_envs, 2, (left, right)]``. A collision kills only the colliding
    Runner; the teammate continues until team success, timeout, or both dead.
    """

    observation_dim = 18
    action_dim = 2
    num_agents = 2

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
        self.cfg = copy.deepcopy(env_cfg)
        self.reward_cfg = copy.deepcopy(reward_cfg)
        plugin_path = self.reward_cfg.get("plugin_path")
        self.reward_engine = (
            RewardEngine.from_file(
                plugin_path,
                params=self.reward_cfg.get("params", {}),
            )
            if plugin_path
            else None
        )
        self.width = float(env_cfg["width"])
        self.height = float(env_cfg["height"])
        self.dt = float(env_cfg["dt"])
        self.wheel_radius = float(env_cfg["wheel_radius"])
        self.wheel_base = float(env_cfg["wheel_base"])
        self.max_wheel_linear_speed = float(env_cfg["max_wheel_linear_speed"])
        self.robot_radius = float(env_cfg["robot_radius"])
        self.goal = np.asarray(env_cfg["goal"], dtype=np.float32)
        self.goal_radius = float(env_cfg["goal_radius"])
        self.max_steps = int(env_cfg["max_steps"])
        self.lidar_rays = int(env_cfg["lidar_rays"])
        self.lidar_range = float(env_cfg["lidar_range"])
        self.reset_jitter = float(env_cfg.get("reset_jitter", 0.0))
        if self.lidar_rays != 8:
            raise ValueError("TwoRunnerArena2D fixes lidar_rays=8 so observation size remains 18")

        obstacle_cfg = env_cfg.get("obstacles", {})
        if not isinstance(obstacle_cfg, dict):
            raise ValueError("environment.obstacles must be a mapping")
        self.obstacle_cfg = copy.deepcopy(obstacle_cfg)
        self.obstacle_count = int(obstacle_cfg.get("count", 0))

        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.map_seeds = np.zeros(self.num_envs, dtype=np.int64)
        self.state = np.zeros((self.num_envs, 2, 3), dtype=np.float32)
        self.prev_action = np.zeros((self.num_envs, 2, 2), dtype=np.float32)
        self.collision = np.zeros((self.num_envs, 2), dtype=bool)
        self.alive = np.ones((self.num_envs, 2), dtype=bool)
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.obstacles = np.zeros((self.num_envs, self.obstacle_count, 4), dtype=np.float32)
        self.reset(seed=seed)

    def _base_state(self) -> np.ndarray:
        return np.array(
            [
                [self.width * 0.10, self.height * 0.35, 0.0],
                [self.width * 0.10, self.height * 0.65, 0.0],
            ],
            dtype=np.float32,
        )

    def _map_seed(self, env_index: int) -> int:
        return int(self.base_seed + int(env_index) * 1_000_003 + int(self.episode_counts[env_index]))

    def _generate_obstacles_for_env(self, env_index: int, map_seed: int) -> np.ndarray:
        if self.obstacle_count == 0:
            return np.zeros((0, 4), dtype=np.float32)
        cfg = self.obstacle_cfg
        map_rng = np.random.default_rng(int(map_seed))
        min_width = float(cfg["min_width"])
        max_width = float(cfg["max_width"])
        min_height = float(cfg["min_height"])
        max_height = float(cfg["max_height"])
        min_spacing = float(cfg.get("min_spacing", 0.0))
        spawn_clearance = float(cfg.get("spawn_clearance", 0.0))
        goal_clearance = float(cfg.get("goal_clearance", 0.0))
        border_margin = float(cfg.get("border_margin", 0.0))
        max_attempts = int(cfg.get("max_attempts", max(1000, self.obstacle_count * 100)))
        spawn_points = self.state[env_index, :, :2]
        placed: list[np.ndarray] = []
        for _ in range(max_attempts):
            if len(placed) >= self.obstacle_count:
                break
            width = float(map_rng.uniform(min_width, max_width))
            height = float(map_rng.uniform(min_height, max_height))
            x_low = border_margin + width * 0.5
            x_high = self.width - border_margin - width * 0.5
            y_low = border_margin + height * 0.5
            y_high = self.height - border_margin - height * 0.5
            if x_low >= x_high or y_low >= y_high:
                raise ValueError("Obstacle size and border margin leave no valid placement area")
            rect = np.array(
                [map_rng.uniform(x_low, x_high), map_rng.uniform(y_low, y_high), width, height],
                dtype=np.float32,
            )
            if any(
                _point_to_rect_distance(float(point[0]), float(point[1]), rect) < spawn_clearance
                for point in spawn_points
            ):
                continue
            if _point_to_rect_distance(float(self.goal[0]), float(self.goal[1]), rect) < goal_clearance:
                continue
            if any(_rects_too_close(rect, previous, min_spacing) for previous in placed):
                continue
            placed.append(rect)
        if len(placed) != self.obstacle_count:
            raise ValueError(
                f"Could only place {len(placed)}/{self.obstacle_count} obstacles after {max_attempts} attempts"
            )
        return np.stack(placed).astype(np.float32, copy=False)

    def _regenerate_obstacles(self, indices: np.ndarray) -> None:
        for env_index in np.asarray(indices, dtype=np.int64):
            map_seed = self._map_seed(int(env_index))
            self.map_seeds[env_index] = map_seed
            self.obstacles[env_index] = self._generate_obstacles_for_env(int(env_index), map_seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)
        self.episode_counts.fill(0)
        self.state[:] = self._base_state()[None, :, :]
        if self.reset_jitter > 0:
            self.state[:, :, :2] += self.rng.uniform(
                -self.reset_jitter, self.reset_jitter, size=(self.num_envs, 2, 2)
            ).astype(np.float32)
            heading_jitter = self.rng.uniform(-0.08, 0.08, size=(self.num_envs, 2)).astype(np.float32)
            self.state[:, :, 2] = _wrap_angle(self.state[:, :, 2] + heading_jitter)
        self.prev_action.fill(0.0)
        self.collision.fill(False)
        self.alive.fill(True)
        self.steps.fill(0)
        self._regenerate_obstacles(np.arange(self.num_envs))
        return self.observe()

    def reset_indices(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError("reset mask has wrong shape")
        indices = np.flatnonzero(mask)
        if not len(indices):
            return
        count = len(indices)
        self.state[mask] = self._base_state()[None, :, :]
        if self.reset_jitter > 0:
            self.state[mask, :, :2] += self.rng.uniform(
                -self.reset_jitter, self.reset_jitter, size=(count, 2, 2)
            ).astype(np.float32)
            heading_jitter = self.rng.uniform(-0.08, 0.08, size=(count, 2)).astype(np.float32)
            self.state[mask, :, 2] = _wrap_angle(self.state[mask, :, 2] + heading_jitter)
        self.episode_counts[indices] += 1
        self.prev_action[mask] = 0.0
        self.collision[mask] = False
        self.alive[mask] = True
        self.steps[mask] = 0
        self._regenerate_obstacles(indices)

    def set_state(self, state: np.ndarray, alive: np.ndarray | None = None) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.state.shape:
            raise ValueError(f"Expected state shape {self.state.shape}, got {state.shape}")
        self.state[:] = state
        self.collision.fill(False)
        if alive is not None:
            alive = np.asarray(alive, dtype=bool)
            if alive.shape != self.alive.shape:
                raise ValueError(f"Expected alive shape {self.alive.shape}, got {alive.shape}")
            self.alive[:] = alive

    def _integrate(self, actions: np.ndarray) -> np.ndarray:
        actions = np.clip(actions, -1.0, 1.0)
        v_l = actions[:, :, 0] * self.max_wheel_linear_speed
        v_r = actions[:, :, 1] * self.max_wheel_linear_speed
        v = 0.5 * (v_r + v_l)
        yaw_rate = (v_r - v_l) / self.wheel_base
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
        proposed[~self.alive] = old[~self.alive]
        return proposed

    def _collision_mask(self, proposed: np.ndarray, alive_before: np.ndarray) -> np.ndarray:
        collided = np.zeros((self.num_envs, 2), dtype=bool)
        radius = self.robot_radius
        x = proposed[:, :, 0]
        y = proposed[:, :, 1]
        static_hit = (x < radius) | (x > self.width - radius) | (y < radius) | (y > self.height - radius)
        for env_index in range(self.num_envs):
            for rect in self.obstacles[env_index]:
                cx, cy, width, height = (float(v) for v in rect)
                nearest_x = np.clip(x[env_index], cx - width * 0.5, cx + width * 0.5)
                nearest_y = np.clip(y[env_index], cy - height * 0.5, cy + height * 0.5)
                d2 = (x[env_index] - nearest_x) ** 2 + (y[env_index] - nearest_y) ** 2
                static_hit[env_index] |= d2 < radius**2
        collided |= static_hit & alive_before
        d2_pair = np.sum((proposed[:, 0, :2] - proposed[:, 1, :2]) ** 2, axis=-1)
        pair_hit = d2_pair < (2.0 * radius) ** 2
        collided[:, 0] |= pair_hit & alive_before[:, 0]
        collided[:, 1] |= pair_hit & alive_before[:, 1]
        return collided

    def _goal_distances(self, state: np.ndarray | None = None) -> np.ndarray:
        s = self.state if state is None else state
        return np.linalg.norm(s[:, :, :2] - self.goal[None, None, :], axis=-1)

    def _minimum_clearance(self, state: np.ndarray | None = None) -> np.ndarray:
        s = self.state if state is None else state
        clearance = np.zeros((self.num_envs, 2), dtype=np.float32)
        for env_index in range(self.num_envs):
            for agent_index in range(2):
                x = float(s[env_index, agent_index, 0])
                y = float(s[env_index, agent_index, 1])
                best = min(
                    x - self.robot_radius,
                    self.width - x - self.robot_radius,
                    y - self.robot_radius,
                    self.height - y - self.robot_radius,
                )
                for rect in self.obstacles[env_index]:
                    best = min(best, _point_to_rect_distance(x, y, rect) - self.robot_radius)
                teammate = 1 - agent_index
                teammate_distance = float(
                    np.linalg.norm(s[env_index, agent_index, :2] - s[env_index, teammate, :2])
                ) - 2.0 * self.robot_radius
                best = min(best, teammate_distance)
                clearance[env_index, agent_index] = best
        return clearance

    def _ray_distance(self, env_index: int, x: float, y: float, angle: float) -> float:
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
        for rect in self.obstacles[env_index]:
            distance = _ray_aabb_distance(x, y, dx, dy, rect)
            if distance is not None:
                best = min(best, distance)
        return float(min(best, self.lidar_range))

    def observe(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, 2, self.observation_dim), dtype=np.float32)
        norm_xy = max(self.width, self.height)
        ray_offsets = np.linspace(-math.pi, math.pi, self.lidar_rays, endpoint=False)
        for env_index in range(self.num_envs):
            for agent_index in range(2):
                x, y, theta = (float(v) for v in self.state[env_index, agent_index])
                k = 0
                gx, gy = _body_frame(float(self.goal[0] - x), float(self.goal[1] - y), theta)
                obs[env_index, agent_index, k:k+2] = (gx / norm_xy, gy / norm_xy)
                k += 2
                obs[env_index, agent_index, k:k+2] = (math.sin(theta), math.cos(theta))
                k += 2
                for offset in ray_offsets:
                    obs[env_index, agent_index, k] = (
                        self._ray_distance(env_index, x, y, theta + float(offset)) / self.lidar_range
                    )
                    k += 1
                obs[env_index, agent_index, k] = float(self.collision[env_index, agent_index])
                k += 1
                obs[env_index, agent_index, k:k+2] = self.prev_action[env_index, agent_index]
                k += 2
                teammate = 1 - agent_index
                dx_other = float(self.state[env_index, teammate, 0] - x)
                dy_other = float(self.state[env_index, teammate, 1] - y)
                bx, by = _body_frame(dx_other, dy_other, theta)
                obs[env_index, agent_index, k:k+2] = (bx / norm_xy, by / norm_xy)
                k += 2
                obs[env_index, agent_index, k] = float(self.alive[env_index, teammate])
                k += 1
                if k != self.observation_dim:
                    raise RuntimeError(f"Two-runner observation assembly bug: {k}")
        return obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, 2, 2)
        if actions.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)
        alive_before = self.alive.copy()
        effective_actions = actions.copy()
        effective_actions[~alive_before] = 0.0
        old_state = self.state.copy()
        old_goal_dist = self._goal_distances(old_state)
        proposed = self._integrate(effective_actions)

        raw_collision = self._collision_mask(proposed, alive_before)
        proposed_goal_dist = self._goal_distances(proposed)
        goal_reached = alive_before & (proposed_goal_dist <= self.goal_radius)
        team_success = goal_reached.any(axis=1)
        collision = raw_collision & ~team_success[:, None]
        proposed[collision] = old_state[collision]
        self.state[:] = proposed
        self.collision[:] = collision
        self.prev_action[:] = effective_actions
        self.prev_action[collision] = 0.0
        self.alive[collision] = False
        self.steps += 1

        new_goal_dist = self._goal_distances(self.state)
        self_progress = old_goal_dist - new_goal_dist
        self_progress[~alive_before] = 0.0
        team_progress = np.zeros(self.num_envs, dtype=np.float32)
        for env_index in range(self.num_envs):
            active = alive_before[env_index]
            if np.any(active):
                old_team_dist = float(old_goal_dist[env_index, active].min())
                new_team_dist = float(new_goal_dist[env_index, active].min())
                team_progress[env_index] = old_team_dist - new_team_dist

        min_clearance = self._minimum_clearance(self.state)
        cfg = self.reward_cfg
        reward_params = cfg.get("params", {}) if self.reward_engine is not None else cfg
        safety_distance = float(reward_params.get("safety_distance", 0.5))
        if safety_distance <= 0.0:
            raise ValueError("reward safety_distance must be positive")
        danger = np.maximum(
            0.0,
            (safety_distance - min_clearance) / safety_distance,
        )

        new_death = collision & alive_before
        both_dead = ~self.alive.any(axis=1)
        raw_timeout = self.steps >= self.max_steps
        timeout = raw_timeout & ~team_success & ~both_dead
        done = team_success | both_dead | timeout

        reward_breakdown: dict[str, np.ndarray] | None = None
        if self.reward_engine is None:
            # Backward-compatible reward path for historical experiments and
            # checkpoints. New deployments should set plugin_path and keep the
            # reward logic in external reward.py.
            rewards = np.zeros((self.num_envs, 2), dtype=np.float32)
            for agent_index in range(2):
                active = alive_before[:, agent_index]
                rewards[active, agent_index] = (
                    float(cfg["self_progress_scale"])
                    * self_progress[active, agent_index]
                    + float(cfg["team_progress_scale"])
                    * team_progress[active]
                    + float(cfg["step_penalty"])
                    - float(cfg["safety_scale"])
                    * danger[active, agent_index] ** 2
                )
            rewards[new_death] = float(cfg["collision_penalty"])
            if np.any(timeout):
                timeout_alive = timeout[:, None] & self.alive
                rewards[timeout_alive] = float(cfg["timeout_penalty"])
            if np.any(team_success):
                rewards[
                    team_success[:, None] & alive_before
                ] = float(cfg["team_success_bonus"])
        else:
            reward_eval = self.reward_engine.evaluate(
                RewardContext(
                    old_state=old_state.copy(),
                    new_state=self.state.copy(),
                    actions=effective_actions.copy(),
                    alive_before=alive_before.copy(),
                    alive_after=self.alive.copy(),
                    old_goal_distance=old_goal_dist.astype(
                        np.float32, copy=True
                    ),
                    new_goal_distance=new_goal_dist.astype(
                        np.float32, copy=True
                    ),
                    self_progress=self_progress.astype(
                        np.float32, copy=True
                    ),
                    team_progress=team_progress.astype(
                        np.float32, copy=True
                    ),
                    min_clearance=min_clearance.astype(
                        np.float32, copy=True
                    ),
                    danger=danger.astype(np.float32, copy=True),
                    collision=collision.copy(),
                    new_death=new_death.copy(),
                    goal_reached=goal_reached.copy(),
                    team_success=team_success.copy(),
                    both_dead=both_dead.copy(),
                    timeout=timeout.copy(),
                    steps=self.steps.copy(),
                    params={},
                )
            )
            rewards = reward_eval.total
            reward_breakdown = reward_eval.terms

        info = {
            "collision": collision.copy(),
            "new_death": new_death.copy(),
            "goal_reached": goal_reached.copy(),
            "team_success": team_success.copy(),
            "both_dead": both_dead.copy(),
            "timeout": timeout.copy(),
            "alive_before": alive_before.copy(),
            "alive_after": self.alive.copy(),
            "self_progress": self_progress.astype(np.float32),
            "team_progress": team_progress.astype(np.float32),
            "min_clearance": min_clearance.astype(np.float32),
            "map_seed": self.map_seeds.copy(),
            "reward_breakdown": reward_breakdown,
            "reward_plugin_sha256": (
                None
                if self.reward_engine is None
                else self.reward_engine.source_sha256
            ),
        }
        return self.observe(), rewards, done, info

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "episode_counts": self.episode_counts.copy(),
            "map_seeds": self.map_seeds.copy(),
            "state": self.state.copy(),
            "prev_action": self.prev_action.copy(),
            "collision": self.collision.copy(),
            "alive": self.alive.copy(),
            "steps": self.steps.copy(),
            "obstacles": self.obstacles.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.base_seed = int(state["base_seed"])
        for name in ("episode_counts", "map_seeds", "state", "prev_action", "collision", "alive", "steps", "obstacles"):
            target = getattr(self, name)
            target[...] = np.asarray(state[name], dtype=target.dtype)
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
