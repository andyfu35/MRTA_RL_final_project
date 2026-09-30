from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gene import Gene
from .planner import (
    build_occupancy,
    points_are_connected,
    precompute_path_to_tasks,
)


BASELINE_CODES = {
    "uninformed": 1,
    "nearest": 2,
    "nearest_path": 3,
    "shortest_service": 4,
    "shortest_total_time": 5,
    "shortest_path_time": 6,
    "highest_priority": 7,
    "priority_per_time": 8,
    "priority_per_path_time": 9,
    "earliest_deadline": 10,
    "least_laxity": 11,
    "max_battery_margin": 12,
}


@dataclass(frozen=True)
class EnvConfig:
    world_size: float = 100.0
    num_robots: int = 4
    num_tasks: int = 20
    robot_speed: float = 4.0
    service_time_min: float = 2.0
    service_time_max: float = 35.0
    priority_min: float = 0.1
    priority_max: float = 1.0
    deadline_min: float = 25.0
    deadline_max: float = 50.0
    episode_time: float = 50.0
    obstacle_count: int = 10
    obstacle_size_min: float = 12.0
    obstacle_size_max: float = 20.0
    obstacle_clearance: float = 4.0
    grid_resolution: float = 5.0
    battery_capacity: float = 70.0
    initial_battery_min: float = 35.0
    initial_battery_max: float = 70.0
    energy_per_distance: float = 1.0

    @property
    def diagonal(self) -> float:
        return math.sqrt(2.0) * self.world_size

    @property
    def path_cost_scale(self) -> float:
        return self.diagonal

    @property
    def service_time_range(self) -> float:
        return self.service_time_max - self.service_time_min

    @property
    def priority_range(self) -> float:
        return self.priority_max - self.priority_min


@dataclass(frozen=True)
class World:
    robot_positions: np.ndarray
    robot_initial_batteries: np.ndarray
    task_positions: np.ndarray
    task_service_times: np.ndarray
    task_priorities: np.ndarray
    task_deadlines: np.ndarray
    obstacles: np.ndarray
    path_to_tasks: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    completion: float
    efficiency: float
    priority_satisfaction: float
    deadline_satisfaction: float
    balance: float
    time_optimality: float
    route_efficiency: float
    completed_tasks: float
    completed_priority: float
    total_priority: float
    on_time_tasks: float
    total_travel: float
    total_euclidean_travel: float
    detour_ratio: float
    mean_initial_battery: float
    mean_final_battery: float
    battery_remaining_fraction: float
    energy_consumed: float
    battery_blocked_pair_events: float
    robot_task_counts: tuple[float, ...]
    robot_workloads: tuple[float, ...]
    robot_final_batteries: tuple[float, ...]

    @property
    def min_axis(self) -> float:
        return min(
            self.completion,
            self.efficiency,
            self.priority_satisfaction,
            self.deadline_satisfaction,
            self.balance,
            self.time_optimality,
        )

    def axes(self) -> np.ndarray:
        return np.array(
            [
                self.completion,
                self.efficiency,
                self.priority_satisfaction,
                self.deadline_satisfaction,
                self.balance,
                self.time_optimality,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "completion": self.completion,
            "efficiency": self.efficiency,
            "priority_satisfaction": self.priority_satisfaction,
            "deadline_satisfaction": self.deadline_satisfaction,
            "balance": self.balance,
            "time_optimality": self.time_optimality,
            "route_efficiency": self.route_efficiency,
            "min_axis": self.min_axis,
            "completed_tasks": self.completed_tasks,
            "completed_priority": self.completed_priority,
            "total_priority": self.total_priority,
            "on_time_tasks": self.on_time_tasks,
            "total_travel": self.total_travel,
            "total_euclidean_travel": self.total_euclidean_travel,
            "detour_ratio": self.detour_ratio,
            "mean_initial_battery": self.mean_initial_battery,
            "mean_final_battery": self.mean_final_battery,
            "battery_remaining_fraction": self.battery_remaining_fraction,
            "energy_consumed": self.energy_consumed,
            "battery_blocked_pair_events": self.battery_blocked_pair_events,
            "robot_task_counts": list(self.robot_task_counts),
            "robot_workloads": list(self.robot_workloads),
            "robot_final_batteries": list(self.robot_final_batteries),
        }


def _validate_config(config: EnvConfig) -> None:
    if config.service_time_max < config.service_time_min:
        raise ValueError("service_time_max must be >= service_time_min")
    if config.priority_max < config.priority_min:
        raise ValueError("priority_max must be >= priority_min")
    if config.priority_min < 0.0:
        raise ValueError("priority_min must be non-negative")
    if config.deadline_max < config.deadline_min:
        raise ValueError("deadline_max must be >= deadline_min")
    if config.deadline_min < 0.0:
        raise ValueError("deadline_min must be non-negative")
    if config.deadline_max > config.episode_time + 1e-12:
        raise ValueError("deadline_max must not exceed episode_time")
    if config.obstacle_count < 0:
        raise ValueError("obstacle_count must be non-negative")
    if config.obstacle_size_min <= 0.0:
        raise ValueError("obstacle_size_min must be positive")
    if config.obstacle_size_max < config.obstacle_size_min:
        raise ValueError("obstacle_size_max must be >= obstacle_size_min")
    if config.obstacle_size_max >= config.world_size:
        raise ValueError("obstacle_size_max must be smaller than world_size")
    if config.grid_resolution <= 0.0:
        raise ValueError("grid_resolution must be positive")
    if config.battery_capacity <= 0.0:
        raise ValueError("battery_capacity must be positive")
    if config.initial_battery_min < 0.0:
        raise ValueError("initial_battery_min must be non-negative")
    if config.initial_battery_max < config.initial_battery_min:
        raise ValueError("initial_battery_max must be >= initial_battery_min")
    if config.initial_battery_max > config.battery_capacity + 1e-12:
        raise ValueError("initial_battery_max must not exceed battery_capacity")
    if config.energy_per_distance <= 0.0:
        raise ValueError("energy_per_distance must be positive")


def _rect_hits_points(
    rect: np.ndarray,
    points: np.ndarray,
    margin: float,
) -> bool:
    x0, y0, x1, y1 = rect
    points = np.asarray(points, dtype=np.float64)
    return bool(
        np.any(
            (points[:, 0] >= x0 - margin)
            & (points[:, 0] <= x1 + margin)
            & (points[:, 1] >= y0 - margin)
            & (points[:, 1] <= y1 + margin)
        )
    )


def _rect_overlaps_any(
    rect: np.ndarray,
    rectangles: list[np.ndarray],
    margin: float = 1.0,
) -> bool:
    x0, y0, x1, y1 = rect
    for other in rectangles:
        ox0, oy0, ox1, oy1 = other
        separated = (
            x1 + margin < ox0
            or ox1 + margin < x0
            or y1 + margin < oy0
            or oy1 + margin < y0
        )
        if not separated:
            return True
    return False


def _generate_obstacles(
    config: EnvConfig,
    rng: np.random.Generator,
    protected_points: np.ndarray,
) -> np.ndarray:
    if config.obstacle_count == 0:
        return np.zeros((0, 4), dtype=np.float64)

    protected_margin = max(
        config.obstacle_clearance,
        0.75 * config.grid_resolution,
    )

    for _layout_attempt in range(50):
        rectangles: list[np.ndarray] = []
        max_attempts = max(500, config.obstacle_count * 300)

        for _ in range(max_attempts):
            if len(rectangles) >= config.obstacle_count:
                break

            side = float(
                rng.uniform(
                    config.obstacle_size_min,
                    config.obstacle_size_max,
                )
            )
            x0 = float(rng.uniform(0.0, config.world_size - side))
            y0 = float(rng.uniform(0.0, config.world_size - side))
            rect = np.array(
                [x0, y0, x0 + side, y0 + side],
                dtype=np.float64,
            )

            if _rect_hits_points(rect, protected_points, protected_margin):
                continue
            if _rect_overlaps_any(rect, rectangles):
                continue
            rectangles.append(rect)

        if len(rectangles) != config.obstacle_count:
            continue

        obstacles = np.stack(rectangles, axis=0)
        occupancy = build_occupancy(
            config.world_size,
            config.grid_resolution,
            obstacles,
        )
        if points_are_connected(
            occupancy,
            protected_points,
            config.world_size,
            config.grid_resolution,
        ):
            return obstacles

    raise RuntimeError(
        "Could not generate a connected obstacle layout. "
        "Reduce obstacle density or obstacle size."
    )


def build_world(
    config: EnvConfig,
    robot_positions: np.ndarray,
    robot_initial_batteries: np.ndarray,
    task_positions: np.ndarray,
    task_service_times: np.ndarray,
    task_priorities: np.ndarray,
    task_deadlines: np.ndarray,
    obstacles: np.ndarray,
) -> World:
    _validate_config(config)

    robots = np.asarray(robot_positions, dtype=np.float64)
    batteries = np.asarray(robot_initial_batteries, dtype=np.float64)
    tasks = np.asarray(task_positions, dtype=np.float64)
    services = np.asarray(task_service_times, dtype=np.float64)
    priorities = np.asarray(task_priorities, dtype=np.float64)
    deadlines = np.asarray(task_deadlines, dtype=np.float64)
    obstacle_array = np.asarray(obstacles, dtype=np.float64)

    if robots.shape != (config.num_robots, 2):
        raise ValueError(f"robot_positions has invalid shape {robots.shape}")
    if batteries.shape != (config.num_robots,):
        raise ValueError(f"robot_initial_batteries has invalid shape {batteries.shape}")
    if np.any(batteries < -1e-12) or np.any(
        batteries > config.battery_capacity + 1e-12
    ):
        raise ValueError("robot_initial_batteries must lie within battery capacity")
    if tasks.shape != (config.num_tasks, 2):
        raise ValueError(f"task_positions has invalid shape {tasks.shape}")
    if services.shape != (config.num_tasks,):
        raise ValueError(f"task_service_times has invalid shape {services.shape}")
    if priorities.shape != (config.num_tasks,):
        raise ValueError(f"task_priorities has invalid shape {priorities.shape}")
    if deadlines.shape != (config.num_tasks,):
        raise ValueError(f"task_deadlines has invalid shape {deadlines.shape}")
    if obstacle_array.size == 0:
        obstacle_array = np.zeros((0, 4), dtype=np.float64)
    if obstacle_array.ndim != 2 or obstacle_array.shape[1] != 4:
        raise ValueError(f"obstacles has invalid shape {obstacle_array.shape}")

    path_to_tasks = precompute_path_to_tasks(
        robots,
        tasks,
        obstacle_array,
        config.world_size,
        config.grid_resolution,
    )

    return World(
        robot_positions=robots.copy(),
        robot_initial_batteries=batteries.copy(),
        task_positions=tasks.copy(),
        task_service_times=services.copy(),
        task_priorities=priorities.copy(),
        task_deadlines=deadlines.copy(),
        obstacles=obstacle_array.copy(),
        path_to_tasks=path_to_tasks,
    )


def generate_world(config: EnvConfig, seed: int) -> World:
    _validate_config(config)
    rng = np.random.default_rng(seed)
    robots = rng.uniform(0.0, config.world_size, size=(config.num_robots, 2))
    tasks = rng.uniform(0.0, config.world_size, size=(config.num_tasks, 2))
    services = rng.uniform(
        config.service_time_min,
        config.service_time_max,
        size=config.num_tasks,
    )
    priorities = rng.uniform(
        config.priority_min,
        config.priority_max,
        size=config.num_tasks,
    )
    deadlines = rng.uniform(
        config.deadline_min,
        config.deadline_max,
        size=config.num_tasks,
    )
    protected = np.vstack([robots, tasks])
    obstacles = _generate_obstacles(config, rng, protected)
    batteries = rng.uniform(
        config.initial_battery_min,
        config.initial_battery_max,
        size=config.num_robots,
    )

    return build_world(
        config,
        robots,
        batteries,
        tasks,
        services,
        priorities,
        deadlines,
        obstacles,
    )


def jain_fairness(loads: np.ndarray) -> float:
    loads = np.asarray(loads, dtype=np.float64)
    total = float(loads.sum())
    if total <= 0.0:
        return 0.0
    denom = float(loads.size * np.square(loads).sum())
    return (total * total) / denom if denom > 0.0 else 0.0


def _evaluate_policy_batch(
    weights: np.ndarray,
    policy_codes: np.ndarray,
    worlds: list[World],
    config: EnvConfig,
    *,
    return_time_matrix: bool = False,
) -> list[Evaluation] | tuple[list[Evaluation], np.ndarray]:
    if not worlds:
        raise ValueError("At least one world is required")

    weights = np.asarray(weights, dtype=np.float64)
    policy_codes = np.asarray(policy_codes, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[1] != 8:
        raise ValueError(f"Expected weights shape (N, 8), got {weights.shape}")
    if policy_codes.shape != (weights.shape[0],):
        raise ValueError("policy_codes must have one entry per policy")

    policy_count = weights.shape[0]
    world_count = len(worlds)
    batch = policy_count * world_count
    robot_count = config.num_robots
    task_count = config.num_tasks

    base_robots = np.stack([world.robot_positions for world in worlds], axis=0)
    base_batteries = np.stack(
        [world.robot_initial_batteries for world in worlds],
        axis=0,
    )
    base_tasks = np.stack([world.task_positions for world in worlds], axis=0)
    base_services = np.stack([world.task_service_times for world in worlds], axis=0)
    base_priorities = np.stack([world.task_priorities for world in worlds], axis=0)
    base_deadlines = np.stack([world.task_deadlines for world in worlds], axis=0)
    base_paths = np.stack([world.path_to_tasks for world in worlds], axis=0)

    robot_positions = np.repeat(base_robots[None, ...], policy_count, axis=0).reshape(
        batch, robot_count, 2
    ).copy()
    initial_batteries = np.repeat(
        base_batteries[None, ...],
        policy_count,
        axis=0,
    ).reshape(batch, robot_count)
    battery_remaining = initial_batteries.copy()

    task_positions = np.repeat(base_tasks[None, ...], policy_count, axis=0).reshape(
        batch, task_count, 2
    )
    task_services = np.repeat(base_services[None, ...], policy_count, axis=0).reshape(
        batch, task_count
    )
    task_priorities = np.repeat(
        base_priorities[None, ...],
        policy_count,
        axis=0,
    ).reshape(batch, task_count)
    task_deadlines = np.repeat(
        base_deadlines[None, ...],
        policy_count,
        axis=0,
    ).reshape(batch, task_count)
    path_tables = np.repeat(base_paths[None, ...], policy_count, axis=0).reshape(
        batch,
        robot_count + task_count,
        task_count,
    )

    batch_weights = np.repeat(weights, world_count, axis=0)
    batch_codes = np.repeat(policy_codes, world_count, axis=0)
    current_node_ids = np.tile(
        np.arange(robot_count, dtype=np.int64),
        (batch, 1),
    )

    task_available = np.ones((batch, task_count), dtype=bool)
    busy_until = np.zeros((batch, robot_count), dtype=np.float64)
    task_counts = np.zeros((batch, robot_count), dtype=np.int64)
    robot_workloads = np.zeros((batch, robot_count), dtype=np.float64)
    total_travel = np.zeros(batch, dtype=np.float64)
    total_euclidean_travel = np.zeros(batch, dtype=np.float64)
    route_efficiency_sum = np.zeros(batch, dtype=np.float64)
    completion_time_score_sum = np.zeros(batch, dtype=np.float64)
    completed_priority_sum = np.zeros(batch, dtype=np.float64)
    total_priority_sum = task_priorities.sum(axis=1)
    on_time_count = np.zeros(batch, dtype=np.float64)
    battery_blocked_pair_events = np.zeros(batch, dtype=np.float64)

    max_events = task_count + robot_count + 2
    batch_ids = np.arange(batch)

    service_range = config.service_time_range
    if service_range > 0.0:
        service_norm_base = np.clip(
            (task_services - config.service_time_min) / service_range,
            0.0,
            1.0,
        )
    else:
        service_norm_base = np.zeros_like(task_services)

    priority_range = config.priority_range
    if priority_range > 0.0:
        priority_norm_base = np.clip(
            (task_priorities - config.priority_min) / priority_range,
            0.0,
            1.0,
        )
    else:
        priority_norm_base = np.zeros_like(task_priorities)

    for _ in range(max_events):
        has_tasks = task_available.any(axis=1)
        now = busy_until.min(axis=1)
        active = has_tasks & (now < config.episode_time - 1e-12)
        if not np.any(active):
            break

        free = (
            np.isclose(busy_until, now[:, None], rtol=0.0, atol=1e-12)
            & active[:, None]
        )

        delta = robot_positions[:, :, None, :] - task_positions[:, None, :, :]
        euclidean = np.linalg.norm(delta, axis=-1)
        euclidean_norm = np.clip(
            euclidean / config.diagonal,
            0.0,
            1.0,
        )

        path_lengths = path_tables[
            batch_ids[:, None],
            current_node_ids,
            :,
        ]
        path_norm = np.clip(
            path_lengths / config.path_cost_scale,
            0.0,
            1.0,
        )
        path_norm = np.where(np.isfinite(path_norm), path_norm, 1.0)

        service_norm = np.broadcast_to(
            service_norm_base[:, None, :],
            euclidean.shape,
        )
        priority_norm = np.broadcast_to(
            priority_norm_base[:, None, :],
            euclidean.shape,
        )
        deadline_remaining_base = np.clip(
            (task_deadlines - now[:, None]) / max(config.episode_time, 1e-12),
            0.0,
            1.0,
        )
        deadline_remaining_norm = np.broadcast_to(
            deadline_remaining_base[:, None, :],
            euclidean.shape,
        )

        battery_norm = np.clip(
            battery_remaining / config.battery_capacity,
            0.0,
            1.0,
        )[:, :, None]
        battery_norm = np.broadcast_to(battery_norm, euclidean.shape)

        workload_norm = np.clip(
            robot_workloads / max(config.episode_time, 1e-12),
            0.0,
            1.0,
        )[:, :, None]
        workload_norm = np.broadcast_to(workload_norm, euclidean.shape)

        duration_path = path_lengths / config.robot_speed + task_services[:, None, :]
        duration_euclidean = (
            euclidean / config.robot_speed
            + task_services[:, None, :]
        )
        energy_required = path_lengths * config.energy_per_distance
        finishes = now[:, None, None] + duration_path

        time_eligible = (
            free[:, :, None]
            & task_available[:, None, :]
            & np.isfinite(path_lengths)
            & (finishes <= config.episode_time + 1e-12)
        )
        battery_feasible = (
            energy_required
            <= battery_remaining[:, :, None] + 1e-12
        )
        battery_blocked_pair_events += (
            time_eligible & ~battery_feasible
        ).sum(axis=(1, 2))
        pair_eligible = time_eligible & battery_feasible

        other_is_closer = (
            path_lengths[:, None, :, :]
            < path_lengths[:, :, None, :]
        )
        other_is_eligible = pair_eligible[:, None, :, :]
        closer_eligible = (other_is_closer & other_is_eligible).sum(axis=2)
        eligible_count = pair_eligible.sum(axis=1)
        competition_denom = np.maximum(eligible_count - 1, 1)[:, None, :]
        competition = closer_eligible / competition_denom

        observations = np.stack(
            [
                euclidean_norm,
                path_norm,
                service_norm,
                priority_norm,
                deadline_remaining_norm,
                battery_norm,
                workload_norm,
                competition,
            ],
            axis=-1,
        )
        bids = np.einsum("brtf,bf->brt", observations, batch_weights)

        for code in np.unique(batch_codes):
            rows = batch_codes == code
            if not np.any(rows) or code == 0:
                continue
            if code == BASELINE_CODES["uninformed"]:
                bids[rows] = 0.0
            elif code == BASELINE_CODES["nearest"]:
                bids[rows] = -euclidean[rows]
            elif code == BASELINE_CODES["nearest_path"]:
                bids[rows] = -path_lengths[rows]
            elif code == BASELINE_CODES["shortest_service"]:
                bids[rows] = -task_services[rows, None, :]
            elif code == BASELINE_CODES["shortest_total_time"]:
                bids[rows] = -duration_euclidean[rows]
            elif code == BASELINE_CODES["shortest_path_time"]:
                bids[rows] = -duration_path[rows]
            elif code == BASELINE_CODES["highest_priority"]:
                bids[rows] = task_priorities[rows, None, :]
            elif code == BASELINE_CODES["priority_per_time"]:
                bids[rows] = (
                    task_priorities[rows, None, :]
                    / np.maximum(duration_euclidean[rows], 1e-12)
                )
            elif code == BASELINE_CODES["priority_per_path_time"]:
                bids[rows] = (
                    task_priorities[rows, None, :]
                    / np.maximum(duration_path[rows], 1e-12)
                )
            elif code == BASELINE_CODES["earliest_deadline"]:
                bids[rows] = -task_deadlines[rows, None, :]
            elif code == BASELINE_CODES["least_laxity"]:
                laxity = (
                    task_deadlines[rows, None, :]
                    - now[rows, None, None]
                    - duration_path[rows]
                )
                can_meet = laxity >= -1e-12
                bids[rows] = np.where(
                    can_meet,
                    1_000_000.0 - laxity,
                    -1_000_000.0 - duration_path[rows],
                )
            elif code == BASELINE_CODES["max_battery_margin"]:
                bids[rows] = (
                    battery_remaining[rows, :, None]
                    - energy_required[rows]
                )
            else:
                raise ValueError(f"Unknown policy code: {code}")

        row_open = free.copy()
        col_open = task_available.copy()
        matched_rows = np.zeros_like(free)

        for _match in range(robot_count):
            valid = pair_eligible & row_open[:, :, None] & col_open[:, None, :]
            scores = np.where(valid, bids, -np.inf)
            flat_scores = scores.reshape(batch, -1)
            flat_idx = np.argmax(flat_scores, axis=1)
            best_scores = flat_scores[batch_ids, flat_idx]
            has_match = np.isfinite(best_scores) & active
            if not np.any(has_match):
                break

            ids = batch_ids[has_match]
            rows = flat_idx[has_match] // task_count
            cols = flat_idx[has_match] % task_count

            path_ds = path_lengths[ids, rows, cols]
            euclidean_ds = euclidean[ids, rows, cols]
            services = task_services[ids, cols]
            priorities = task_priorities[ids, cols]
            deadlines = task_deadlines[ids, cols]
            energy = energy_required[ids, rows, cols]
            assignment_duration = path_ds / config.robot_speed + services
            assigned_finishes = now[ids] + assignment_duration

            busy_until[ids, rows] = assigned_finishes
            robot_positions[ids, rows] = task_positions[ids, cols]
            current_node_ids[ids, rows] = robot_count + cols
            battery_remaining[ids, rows] -= energy
            battery_remaining[ids, rows] = np.maximum(
                battery_remaining[ids, rows],
                0.0,
            )
            task_available[ids, cols] = False
            task_counts[ids, rows] += 1
            robot_workloads[ids, rows] += assignment_duration
            total_travel[ids] += path_ds
            total_euclidean_travel[ids] += euclidean_ds
            completed_priority_sum[ids] += priorities
            on_time_count[ids] += (
                assigned_finishes <= deadlines + 1e-12
            ).astype(np.float64)
            route_efficiency_sum[ids] += 1.0 - np.clip(
                path_ds / config.diagonal,
                0.0,
                1.0,
            )
            completion_time_score_sum[ids] += 1.0 - np.clip(
                assigned_finishes / max(config.episode_time, 1e-12),
                0.0,
                1.0,
            )
            matched_rows[ids, rows] = True
            row_open[ids, rows] = False
            col_open[ids, cols] = False

        unmatched_free = free & ~matched_rows
        busy_until[unmatched_free] = config.episode_time

    completed = task_counts.sum(axis=1)
    completion = completed / task_count
    efficiency = route_efficiency_sum / task_count
    route_efficiency = np.divide(
        route_efficiency_sum,
        completed,
        out=np.zeros_like(route_efficiency_sum),
        where=completed > 0,
    )
    priority_satisfaction = np.divide(
        completed_priority_sum,
        total_priority_sum,
        out=np.zeros_like(completed_priority_sum),
        where=total_priority_sum > 0.0,
    )
    deadline_satisfaction = on_time_count / task_count
    time_optimality = completion_time_score_sum / task_count

    workload_sum = robot_workloads.sum(axis=1)
    workload_sq_sum = np.square(robot_workloads).sum(axis=1)
    fairness = np.divide(
        np.square(workload_sum),
        robot_count * workload_sq_sum,
        out=np.zeros(batch, dtype=np.float64),
        where=workload_sq_sum > 0.0,
    )
    balance = completion * fairness
    detour_ratio = np.divide(
        total_travel,
        total_euclidean_travel,
        out=np.ones(batch, dtype=np.float64),
        where=total_euclidean_travel > 0.0,
    )

    initial_battery_sum = initial_batteries.sum(axis=1)
    final_battery_sum = battery_remaining.sum(axis=1)
    battery_remaining_fraction = np.divide(
        final_battery_sum,
        initial_battery_sum,
        out=np.zeros(batch, dtype=np.float64),
        where=initial_battery_sum > 0.0,
    )
    energy_consumed = initial_battery_sum - final_battery_sum

    completion = completion.reshape(policy_count, world_count)
    efficiency = efficiency.reshape(policy_count, world_count)
    priority_satisfaction = priority_satisfaction.reshape(policy_count, world_count)
    deadline_satisfaction = deadline_satisfaction.reshape(policy_count, world_count)
    route_efficiency = route_efficiency.reshape(policy_count, world_count)
    balance = balance.reshape(policy_count, world_count)
    time_optimality = time_optimality.reshape(policy_count, world_count)
    completed = completed.reshape(policy_count, world_count)
    completed_priority_sum = completed_priority_sum.reshape(policy_count, world_count)
    total_priority_sum = total_priority_sum.reshape(policy_count, world_count)
    on_time_count = on_time_count.reshape(policy_count, world_count)
    total_travel = total_travel.reshape(policy_count, world_count)
    total_euclidean_travel = total_euclidean_travel.reshape(
        policy_count,
        world_count,
    )
    detour_ratio = detour_ratio.reshape(policy_count, world_count)
    initial_batteries = initial_batteries.reshape(
        policy_count,
        world_count,
        robot_count,
    )
    battery_remaining = battery_remaining.reshape(
        policy_count,
        world_count,
        robot_count,
    )
    battery_remaining_fraction = battery_remaining_fraction.reshape(
        policy_count,
        world_count,
    )
    energy_consumed = energy_consumed.reshape(policy_count, world_count)
    battery_blocked_pair_events = battery_blocked_pair_events.reshape(
        policy_count,
        world_count,
    )
    task_counts = task_counts.reshape(policy_count, world_count, robot_count)
    robot_workloads = robot_workloads.reshape(
        policy_count,
        world_count,
        robot_count,
    )

    evaluations: list[Evaluation] = []
    for policy_idx in range(policy_count):
        mean_counts = task_counts[policy_idx].mean(axis=0)
        mean_workloads = robot_workloads[policy_idx].mean(axis=0)
        mean_final_batteries = battery_remaining[policy_idx].mean(axis=0)
        evaluations.append(
            Evaluation(
                completion=float(completion[policy_idx].mean()),
                efficiency=float(efficiency[policy_idx].mean()),
                priority_satisfaction=float(
                    priority_satisfaction[policy_idx].mean()
                ),
                deadline_satisfaction=float(
                    deadline_satisfaction[policy_idx].mean()
                ),
                balance=float(balance[policy_idx].mean()),
                time_optimality=float(time_optimality[policy_idx].mean()),
                route_efficiency=float(route_efficiency[policy_idx].mean()),
                completed_tasks=float(completed[policy_idx].mean()),
                completed_priority=float(
                    completed_priority_sum[policy_idx].mean()
                ),
                total_priority=float(total_priority_sum[policy_idx].mean()),
                on_time_tasks=float(on_time_count[policy_idx].mean()),
                total_travel=float(total_travel[policy_idx].mean()),
                total_euclidean_travel=float(
                    total_euclidean_travel[policy_idx].mean()
                ),
                detour_ratio=float(detour_ratio[policy_idx].mean()),
                mean_initial_battery=float(
                    initial_batteries[policy_idx].mean()
                ),
                mean_final_battery=float(
                    battery_remaining[policy_idx].mean()
                ),
                battery_remaining_fraction=float(
                    battery_remaining_fraction[policy_idx].mean()
                ),
                energy_consumed=float(
                    energy_consumed[policy_idx].mean()
                ),
                battery_blocked_pair_events=float(
                    battery_blocked_pair_events[policy_idx].mean()
                ),
                robot_task_counts=tuple(float(x) for x in mean_counts.tolist()),
                robot_workloads=tuple(float(x) for x in mean_workloads.tolist()),
                robot_final_batteries=tuple(
                    float(x) for x in mean_final_batteries.tolist()
                ),
            )
        )
    if return_time_matrix:
        return evaluations, time_optimality.copy()
    return evaluations


def evaluate_genes_time_optimality_matrix(
    genes: list[Gene],
    worlds: list[World],
    config: EnvConfig,
) -> np.ndarray:
    """Return shape (num_genes, num_worlds) without changing rollout semantics."""
    if not genes:
        return np.zeros((0, len(worlds)), dtype=np.float64)
    weights = np.stack([gene.weights for gene in genes], axis=0)
    policy_codes = np.zeros(len(genes), dtype=np.int64)
    result = _evaluate_policy_batch(
        weights,
        policy_codes,
        worlds,
        config,
        return_time_matrix=True,
    )
    _, matrix = result
    return matrix


def evaluate_genes_on_worlds(
    genes: list[Gene],
    worlds: list[World],
    config: EnvConfig,
) -> list[Evaluation]:
    if not genes:
        return []
    weights = np.stack([gene.weights for gene in genes], axis=0)
    policy_codes = np.zeros(len(genes), dtype=np.int64)
    return _evaluate_policy_batch(weights, policy_codes, worlds, config)


def evaluate_gene(gene: Gene, world: World, config: EnvConfig) -> Evaluation:
    return evaluate_genes_on_worlds([gene], [world], config)[0]


def evaluate_gene_on_worlds(
    gene: Gene,
    worlds: list[World],
    config: EnvConfig,
) -> Evaluation:
    return evaluate_genes_on_worlds([gene], worlds, config)[0]


def evaluate_baseline_on_worlds(
    name: str,
    worlds: list[World],
    config: EnvConfig,
) -> Evaluation:
    if name not in BASELINE_CODES:
        raise ValueError(
            f"Unknown baseline '{name}'. Expected one of {sorted(BASELINE_CODES)}"
        )
    weights = np.zeros((1, 8), dtype=np.float64)
    codes = np.array([BASELINE_CODES[name]], dtype=np.int64)
    return _evaluate_policy_batch(weights, codes, worlds, config)[0]
