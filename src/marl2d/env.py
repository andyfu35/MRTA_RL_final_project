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


class VectorArena2D:
    """Vectorized deterministic 2D differential-drive arena.

    State layout: [num_envs, 4, (x, y, theta)].
    Action layout: [num_envs, 4, (left, right)] in [-1, 1].
    Obstacles: [num_envs, obstacle_count, (center_x, center_y, width, height)].
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
        self.max_wheel_linear_speed = float(env_cfg["max_wheel_linear_speed"])
        self.robot_radius = float(env_cfg["robot_radius"])
        self.body_length = float(env_cfg.get("body_length", self.robot_radius * 2.0))
        self.body_width = float(env_cfg.get("body_width", self.robot_radius * 1.4))
        self.wheel_length = float(env_cfg.get("wheel_length", self.body_length * 0.5))
        self.wheel_width = float(env_cfg.get("wheel_width", self.body_width * 0.22))
        self.goal = np.asarray(env_cfg["goal"], dtype=np.float32)
        self.goal_radius = float(env_cfg["goal_radius"])
        self.max_steps = int(env_cfg["max_steps"])
        self.lidar_rays = int(env_cfg["lidar_rays"])
        self.lidar_range = float(env_cfg["lidar_range"])
        self.reset_jitter = float(env_cfg.get("reset_jitter", 0.0))

        obstacle_cfg = env_cfg.get("obstacles", {})
        if not isinstance(obstacle_cfg, dict):
            raise ValueError("environment.obstacles must be a mapping for random rectangular obstacles")
        self.obstacle_cfg = obstacle_cfg
        self.obstacle_count = int(obstacle_cfg.get("count", 0))
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.map_seeds = np.zeros(self.num_envs, dtype=np.int64)

        self.state = np.zeros((self.num_envs, 4, 3), dtype=np.float32)
        self.prev_action = np.zeros((self.num_envs, 4, 2), dtype=np.float32)
        self.collision = np.zeros((self.num_envs, 4), dtype=bool)
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.obstacles = np.zeros((self.num_envs, self.obstacle_count, 4), dtype=np.float32)
        self.reset(seed=seed)

    def _base_state(self) -> np.ndarray:
        return np.array(
            [
                [self.width * 0.10, self.height * 0.35, 0.0],
                [self.width * 0.10, self.height * 0.65, 0.0],
                [self.width * 0.72, self.height * 0.375, math.pi],
                [self.width * 0.72, self.height * 0.625, math.pi],
            ],
            dtype=np.float32,
        )

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

        placed: list[np.ndarray] = []
        spawn_points = self.state[env_index, :, :2]
        attempts = 0
        while len(placed) < self.obstacle_count and attempts < max_attempts:
            attempts += 1
            width = float(map_rng.uniform(min_width, max_width))
            height = float(map_rng.uniform(min_height, max_height))
            x_low = border_margin + width * 0.5
            x_high = self.width - border_margin - width * 0.5
            y_low = border_margin + height * 0.5
            y_high = self.height - border_margin - height * 0.5
            if x_low >= x_high or y_low >= y_high:
                raise ValueError("Obstacle size and border margin leave no valid placement area")

            rect = np.array(
                [
                    map_rng.uniform(x_low, x_high),
                    map_rng.uniform(y_low, y_high),
                    width,
                    height,
                ],
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
                f"Could only place {len(placed)}/{self.obstacle_count} obstacles after {max_attempts} attempts; "
                "reduce obstacle count/spacing or increase map size"
            )
        return np.stack(placed).astype(np.float32, copy=False)

    def _map_seed(self, env_index: int) -> int:
        return int(self.base_seed + int(env_index) * 1_000_003 + int(self.episode_counts[env_index]))

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
        self._regenerate_obstacles(np.arange(self.num_envs))
        self.prev_action.fill(0.0)
        self.collision.fill(False)
        self.steps.fill(0)
        return self.observe()

    def reset_indices(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError("reset mask has wrong shape")
        indices = np.flatnonzero(mask)
        count = int(indices.size)
        if count == 0:
            return
        base = self._base_state()
        self.state[mask] = base[None, :, :]
        if self.reset_jitter > 0:
            jitter = self.rng.uniform(-self.reset_jitter, self.reset_jitter, size=(count, 4, 2)).astype(np.float32)
            self.state[mask, :, :2] += jitter
            heading_jitter = self.rng.uniform(-0.08, 0.08, size=(count, 4)).astype(np.float32)
            self.state[mask, :, 2] = _wrap_angle(self.state[mask, :, 2] + heading_jitter)
        self.episode_counts[indices] += 1
        self._regenerate_obstacles(indices)
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
        return proposed

    def _collision_mask(self, proposed: np.ndarray) -> np.ndarray:
        collided = np.zeros((self.num_envs, 4), dtype=bool)
        x = proposed[:, :, 0]
        y = proposed[:, :, 1]
        radius = self.robot_radius

        collided |= (x < radius) | (x > self.width - radius) | (y < radius) | (y > self.height - radius)

        for env_index in range(self.num_envs):
            for rect in self.obstacles[env_index]:
                cx, cy, width, height = (float(v) for v in rect)
                nearest_x = np.clip(x[env_index], cx - width * 0.5, cx + width * 0.5)
                nearest_y = np.clip(y[env_index], cy - height * 0.5, cy + height * 0.5)
                d2 = (x[env_index] - nearest_x) ** 2 + (y[env_index] - nearest_y) ** 2
                collided[env_index] |= d2 < radius**2

        min_robot_dist2 = (2.0 * radius) ** 2
        for a in range(4):
            for b in range(a + 1, 4):
                d2 = np.sum((proposed[:, a, :2] - proposed[:, b, :2]) ** 2, axis=-1)
                hit = d2 < min_robot_dist2
                collided[:, a] |= hit
                collided[:, b] |= hit
        return collided

    def _blocker_proximity(self) -> np.ndarray:
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
            "map_seed": self.map_seeds.copy(),
        }
        return self.observe(), rewards, done, info

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
                    dx_other = float(self.state[e, j, 0] - x)
                    dy_other = float(self.state[e, j, 1] - y)
                    bx, by = _body_frame(dx_other, dy_other, theta)
                    obs[e, i, k : k + 2] = (bx / norm_xy, by / norm_xy)
                    k += 2

                for offset in ray_offsets:
                    obs[e, i, k] = self._ray_distance(e, x, y, theta + float(offset)) / self.lidar_range
                    k += 1

                obs[e, i, k] = float(self.collision[e, i])
                k += 1
                obs[e, i, k : k + 2] = self.prev_action[e, i]
                k += 2
                if k != self.observation_dim:
                    raise RuntimeError(f"Observation assembly bug for {AGENT_IDS[i]}: {k}")
        return obs
