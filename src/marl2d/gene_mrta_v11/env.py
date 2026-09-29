from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gene import Gene


@dataclass(frozen=True)
class EnvConfig:
    world_size: float = 100.0
    num_robots: int = 4
    num_tasks: int = 20
    robot_speed: float = 2.0
    service_time: float = 3.0
    episode_time: float = 50.0

    @property
    def diagonal(self) -> float:
        return math.sqrt(2.0) * self.world_size

    @property
    def fair_task_share(self) -> float:
        return self.num_tasks / self.num_robots


@dataclass(frozen=True)
class World:
    robot_positions: np.ndarray
    task_positions: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    completion: float
    efficiency: float
    balance: float
    route_efficiency: float
    completed_tasks: float
    total_travel: float
    robot_task_counts: tuple[float, ...]

    @property
    def min_axis(self) -> float:
        return min(self.completion, self.efficiency, self.balance)

    def axes(self) -> np.ndarray:
        return np.array([self.completion, self.efficiency, self.balance], dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "completion": self.completion,
            "efficiency": self.efficiency,
            "balance": self.balance,
            "route_efficiency": self.route_efficiency,
            "min_axis": self.min_axis,
            "completed_tasks": self.completed_tasks,
            "total_travel": self.total_travel,
            "robot_task_counts": list(self.robot_task_counts),
        }


def generate_world(config: EnvConfig, seed: int) -> World:
    rng = np.random.default_rng(seed)
    robots = rng.uniform(0.0, config.world_size, size=(config.num_robots, 2))
    tasks = rng.uniform(0.0, config.world_size, size=(config.num_tasks, 2))
    return World(robots.astype(np.float64), tasks.astype(np.float64))


def jain_fairness(loads: np.ndarray) -> float:
    loads = np.asarray(loads, dtype=np.float64)
    total = float(loads.sum())
    if total <= 0.0:
        return 0.0
    denom = float(loads.size * np.square(loads).sum())
    return (total * total) / denom if denom > 0.0 else 0.0


def evaluate_genes_on_worlds(
    genes: list[Gene],
    worlds: list[World],
    config: EnvConfig,
) -> list[Evaluation]:
    if not genes:
        return []
    if not worlds:
        raise ValueError("At least one world is required")

    gene_count = len(genes)
    world_count = len(worlds)
    batch = gene_count * world_count
    robot_count = config.num_robots
    task_count = config.num_tasks

    base_robots = np.stack([world.robot_positions for world in worlds], axis=0)
    base_tasks = np.stack([world.task_positions for world in worlds], axis=0)
    robot_positions = np.repeat(base_robots[None, ...], gene_count, axis=0).reshape(batch, robot_count, 2).copy()
    task_positions = np.repeat(base_tasks[None, ...], gene_count, axis=0).reshape(batch, task_count, 2)
    weights = np.repeat(np.stack([gene.weights for gene in genes], axis=0), world_count, axis=0)

    task_available = np.ones((batch, task_count), dtype=bool)
    busy_until = np.zeros((batch, robot_count), dtype=np.float64)
    task_counts = np.zeros((batch, robot_count), dtype=np.int64)
    total_travel = np.zeros(batch, dtype=np.float64)
    route_efficiency_sum = np.zeros(batch, dtype=np.float64)

    max_events = task_count + robot_count + 2
    batch_ids = np.arange(batch)

    for _ in range(max_events):
        has_tasks = task_available.any(axis=1)
        now = busy_until.min(axis=1)
        active = has_tasks & (now < config.episode_time - 1e-12)
        if not np.any(active):
            break

        free = np.isclose(busy_until, now[:, None], rtol=0.0, atol=1e-12) & active[:, None]

        delta = robot_positions[:, :, None, :] - task_positions[:, None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        distance_norm = np.clip(distances / config.diagonal, 0.0, 1.0)

        load_norm = np.clip(
            task_counts / max(config.fair_task_share, 1.0),
            0.0,
            1.0,
        )[:, :, None]
        load_norm = np.broadcast_to(load_norm, distances.shape)

        other_free = free[:, None, :, None]
        is_closer = distances[:, :, None, :] < distances[:, None, :, :]
        closer_free = (is_closer & other_free).sum(axis=2)
        free_count = free.sum(axis=1)
        competition_denom = np.maximum(free_count - 1, 1)[:, None, None]
        competition = closer_free / competition_denom

        duration = distances / config.robot_speed + config.service_time
        remaining = config.episode_time - now
        slack_norm = np.clip(
            (remaining[:, None, None] - duration) / config.episode_time,
            -1.0,
            1.0,
        )

        observations = np.stack(
            [distance_norm, load_norm, competition, slack_norm],
            axis=-1,
        )
        bids = np.einsum("brtf,bf->brt", observations, weights)

        pair_feasible = (
            free[:, :, None]
            & task_available[:, None, :]
            & (now[:, None, None] + duration <= config.episode_time + 1e-12)
        )

        row_open = free.copy()
        col_open = task_available.copy()
        matched_rows = np.zeros_like(free)

        for _match in range(robot_count):
            valid = pair_feasible & row_open[:, :, None] & col_open[:, None, :]
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
            finishes = now[ids] + ds / config.robot_speed + config.service_time

            busy_until[ids, rows] = finishes
            robot_positions[ids, rows] = task_positions[ids, cols]
            task_available[ids, cols] = False
            task_counts[ids, rows] += 1
            total_travel[ids] += ds
            route_efficiency_sum[ids] += 1.0 - np.clip(ds / config.diagonal, 0.0, 1.0)
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

    denom = robot_count * np.square(task_counts.astype(np.float64)).sum(axis=1)
    balance = np.divide(
        np.square(completed.astype(np.float64)),
        denom,
        out=np.zeros(batch, dtype=np.float64),
        where=denom > 0.0,
    )

    completion = completion.reshape(gene_count, world_count)
    efficiency = efficiency.reshape(gene_count, world_count)
    route_efficiency = route_efficiency.reshape(gene_count, world_count)
    balance = balance.reshape(gene_count, world_count)
    completed = completed.reshape(gene_count, world_count)
    total_travel = total_travel.reshape(gene_count, world_count)
    task_counts = task_counts.reshape(gene_count, world_count, robot_count)

    evaluations: list[Evaluation] = []
    for gene_idx in range(gene_count):
        mean_counts = task_counts[gene_idx].mean(axis=0)
        evaluations.append(
            Evaluation(
                completion=float(completion[gene_idx].mean()),
                efficiency=float(efficiency[gene_idx].mean()),
                balance=float(balance[gene_idx].mean()),
                route_efficiency=float(route_efficiency[gene_idx].mean()),
                completed_tasks=float(completed[gene_idx].mean()),
                total_travel=float(total_travel[gene_idx].mean()),
                robot_task_counts=tuple(float(x) for x in mean_counts.tolist()),
            )
        )
    return evaluations


def evaluate_gene(gene: Gene, world: World, config: EnvConfig) -> Evaluation:
    return evaluate_genes_on_worlds([gene], [world], config)[0]


def evaluate_gene_on_worlds(gene: Gene, worlds: list[World], config: EnvConfig) -> Evaluation:
    return evaluate_genes_on_worlds([gene], worlds, config)[0]
