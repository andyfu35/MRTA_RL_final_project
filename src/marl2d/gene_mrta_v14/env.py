from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gene import Gene


BASELINE_CODES = {
    "uninformed": 1,
    "nearest": 2,
    "shortest_service": 3,
    "shortest_total_time": 4,
    "highest_priority": 5,
    "priority_per_time": 6,
    "earliest_deadline": 7,
    "least_laxity": 8,
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

    @property
    def diagonal(self) -> float:
        return math.sqrt(2.0) * self.world_size

    @property
    def service_time_range(self) -> float:
        return self.service_time_max - self.service_time_min

    @property
    def priority_range(self) -> float:
        return self.priority_max - self.priority_min


@dataclass(frozen=True)
class World:
    robot_positions: np.ndarray
    task_positions: np.ndarray
    task_service_times: np.ndarray
    task_priorities: np.ndarray
    task_deadlines: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    completion: float
    efficiency: float
    priority_satisfaction: float
    deadline_satisfaction: float
    balance: float
    route_efficiency: float
    completed_tasks: float
    completed_priority: float
    total_priority: float
    on_time_tasks: float
    total_travel: float
    robot_task_counts: tuple[float, ...]
    robot_workloads: tuple[float, ...]

    @property
    def min_axis(self) -> float:
        return min(
            self.completion,
            self.efficiency,
            self.priority_satisfaction,
            self.deadline_satisfaction,
            self.balance,
        )

    def axes(self) -> np.ndarray:
        return np.array(
            [
                self.completion,
                self.efficiency,
                self.priority_satisfaction,
                self.deadline_satisfaction,
                self.balance,
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
            "route_efficiency": self.route_efficiency,
            "min_axis": self.min_axis,
            "completed_tasks": self.completed_tasks,
            "completed_priority": self.completed_priority,
            "total_priority": self.total_priority,
            "on_time_tasks": self.on_time_tasks,
            "total_travel": self.total_travel,
            "robot_task_counts": list(self.robot_task_counts),
            "robot_workloads": list(self.robot_workloads),
        }


def generate_world(config: EnvConfig, seed: int) -> World:
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
    return World(
        robot_positions=robots.astype(np.float64),
        task_positions=tasks.astype(np.float64),
        task_service_times=services.astype(np.float64),
        task_priorities=priorities.astype(np.float64),
        task_deadlines=deadlines.astype(np.float64),
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
) -> list[Evaluation]:
    if not worlds:
        raise ValueError("At least one world is required")

    weights = np.asarray(weights, dtype=np.float64)
    policy_codes = np.asarray(policy_codes, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[1] != 6:
        raise ValueError(f"Expected weights shape (N, 6), got {weights.shape}")
    if policy_codes.shape != (weights.shape[0],):
        raise ValueError("policy_codes must have one entry per policy")

    policy_count = weights.shape[0]
    world_count = len(worlds)
    batch = policy_count * world_count
    robot_count = config.num_robots
    task_count = config.num_tasks

    base_robots = np.stack([world.robot_positions for world in worlds], axis=0)
    base_tasks = np.stack([world.task_positions for world in worlds], axis=0)
    base_services = np.stack([world.task_service_times for world in worlds], axis=0)
    base_priorities = np.stack([world.task_priorities for world in worlds], axis=0)
    base_deadlines = np.stack([world.task_deadlines for world in worlds], axis=0)

    robot_positions = np.repeat(base_robots[None, ...], policy_count, axis=0).reshape(
        batch, robot_count, 2
    ).copy()
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
    batch_weights = np.repeat(weights, world_count, axis=0)
    batch_codes = np.repeat(policy_codes, world_count, axis=0)

    task_available = np.ones((batch, task_count), dtype=bool)
    busy_until = np.zeros((batch, robot_count), dtype=np.float64)
    task_counts = np.zeros((batch, robot_count), dtype=np.int64)
    robot_workloads = np.zeros((batch, robot_count), dtype=np.float64)
    total_travel = np.zeros(batch, dtype=np.float64)
    route_efficiency_sum = np.zeros(batch, dtype=np.float64)
    completed_priority_sum = np.zeros(batch, dtype=np.float64)
    total_priority_sum = task_priorities.sum(axis=1)
    on_time_count = np.zeros(batch, dtype=np.float64)

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
        distances = np.linalg.norm(delta, axis=-1)
        distance_norm = np.clip(distances / config.diagonal, 0.0, 1.0)

        service_norm = np.broadcast_to(
            service_norm_base[:, None, :],
            distances.shape,
        )
        priority_norm = np.broadcast_to(
            priority_norm_base[:, None, :],
            distances.shape,
        )
        deadline_remaining_base = np.clip(
            (task_deadlines - now[:, None]) / max(config.episode_time, 1e-12),
            0.0,
            1.0,
        )
        deadline_remaining_norm = np.broadcast_to(
            deadline_remaining_base[:, None, :],
            distances.shape,
        )

        workload_norm = np.clip(
            robot_workloads / max(config.episode_time, 1e-12),
            0.0,
            1.0,
        )[:, :, None]
        workload_norm = np.broadcast_to(workload_norm, distances.shape)

        duration = distances / config.robot_speed + task_services[:, None, :]
        finishes = now[:, None, None] + duration
        pair_eligible = (
            free[:, :, None]
            & task_available[:, None, :]
            & (finishes <= config.episode_time + 1e-12)
        )

        other_is_closer = distances[:, None, :, :] < distances[:, :, None, :]
        other_is_eligible = pair_eligible[:, None, :, :]
        closer_eligible = (other_is_closer & other_is_eligible).sum(axis=2)
        eligible_count = pair_eligible.sum(axis=1)
        competition_denom = np.maximum(eligible_count - 1, 1)[:, None, :]
        competition = closer_eligible / competition_denom

        observations = np.stack(
            [
                distance_norm,
                service_norm,
                priority_norm,
                deadline_remaining_norm,
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
                bids[rows] = -distances[rows]
            elif code == BASELINE_CODES["shortest_service"]:
                bids[rows] = -task_services[rows, None, :]
            elif code == BASELINE_CODES["shortest_total_time"]:
                bids[rows] = -duration[rows]
            elif code == BASELINE_CODES["highest_priority"]:
                bids[rows] = task_priorities[rows, None, :]
            elif code == BASELINE_CODES["priority_per_time"]:
                bids[rows] = (
                    task_priorities[rows, None, :]
                    / np.maximum(duration[rows], 1e-12)
                )
            elif code == BASELINE_CODES["earliest_deadline"]:
                bids[rows] = -task_deadlines[rows, None, :]
            elif code == BASELINE_CODES["least_laxity"]:
                laxity = (
                    task_deadlines[rows, None, :]
                    - now[rows, None, None]
                    - duration[rows]
                )
                can_meet = laxity >= -1e-12
                bids[rows] = np.where(
                    can_meet,
                    1_000_000.0 - laxity,
                    -1_000_000.0 - duration[rows],
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

            ds = distances[ids, rows, cols]
            services = task_services[ids, cols]
            priorities = task_priorities[ids, cols]
            deadlines = task_deadlines[ids, cols]
            assignment_duration = ds / config.robot_speed + services
            assigned_finishes = now[ids] + assignment_duration

            busy_until[ids, rows] = assigned_finishes
            robot_positions[ids, rows] = task_positions[ids, cols]
            task_available[ids, cols] = False
            task_counts[ids, rows] += 1
            robot_workloads[ids, rows] += assignment_duration
            total_travel[ids] += ds
            completed_priority_sum[ids] += priorities
            on_time_count[ids] += (
                assigned_finishes <= deadlines + 1e-12
            ).astype(np.float64)
            route_efficiency_sum[ids] += 1.0 - np.clip(
                ds / config.diagonal,
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

    workload_sum = robot_workloads.sum(axis=1)
    workload_sq_sum = np.square(robot_workloads).sum(axis=1)
    fairness = np.divide(
        np.square(workload_sum),
        robot_count * workload_sq_sum,
        out=np.zeros(batch, dtype=np.float64),
        where=workload_sq_sum > 0.0,
    )
    balance = completion * fairness

    completion = completion.reshape(policy_count, world_count)
    efficiency = efficiency.reshape(policy_count, world_count)
    priority_satisfaction = priority_satisfaction.reshape(policy_count, world_count)
    deadline_satisfaction = deadline_satisfaction.reshape(policy_count, world_count)
    route_efficiency = route_efficiency.reshape(policy_count, world_count)
    balance = balance.reshape(policy_count, world_count)
    completed = completed.reshape(policy_count, world_count)
    completed_priority_sum = completed_priority_sum.reshape(policy_count, world_count)
    total_priority_sum = total_priority_sum.reshape(policy_count, world_count)
    on_time_count = on_time_count.reshape(policy_count, world_count)
    total_travel = total_travel.reshape(policy_count, world_count)
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
                route_efficiency=float(route_efficiency[policy_idx].mean()),
                completed_tasks=float(completed[policy_idx].mean()),
                completed_priority=float(
                    completed_priority_sum[policy_idx].mean()
                ),
                total_priority=float(total_priority_sum[policy_idx].mean()),
                on_time_tasks=float(on_time_count[policy_idx].mean()),
                total_travel=float(total_travel[policy_idx].mean()),
                robot_task_counts=tuple(float(x) for x in mean_counts.tolist()),
                robot_workloads=tuple(float(x) for x in mean_workloads.tolist()),
            )
        )
    return evaluations


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
    weights = np.zeros((1, 6), dtype=np.float64)
    codes = np.array([BASELINE_CODES[name]], dtype=np.int64)
    return _evaluate_policy_batch(weights, codes, worlds, config)[0]
