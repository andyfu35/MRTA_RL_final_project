from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, World
from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v18.rollout import _build_observations


@dataclass(frozen=True)
class RobustMetrics:
    time_optimality: float
    continuation_preservation: float
    fleet_option_reserve: float


def _future_graph_masses(
    *,
    world: World,
    config: EnvConfig,
    now: float,
    current_node_ids: np.ndarray,
    battery_remaining: np.ndarray,
    busy_until: np.ndarray,
    task_available: np.ndarray,
) -> tuple[float, float]:
    """
    Return (option_mass, reserve_mass), both normalized by total task count.

    option_mass:
      Sum over remaining tasks of the best one-step future T utility available
      from any robot.

    reserve_mass:
      Sum over remaining tasks of min(number_of_feasible_robots, 2) / 2.
      A stranded task contributes 0, a singleton task 0.5, and a task with at
      least two feasible robots contributes 1.

    These are external evaluation signals for Gene Bank axes. They are not
    observations added to the policy and they do not choose actions.
    """
    R = config.num_robots
    N = config.num_tasks

    earliest_start = np.maximum(
        float(now),
        np.asarray(busy_until, dtype=np.float64),
    )
    paths = world.path_to_tasks[current_node_ids, :]
    finishes = (
        earliest_start[:, None]
        + paths / config.robot_speed
        + world.task_service_times[None, :]
    )
    energy = paths * config.energy_per_distance

    feasible = (
        task_available[None, :]
        & np.isfinite(paths)
        & (finishes <= config.episode_time + 1e-12)
        & (energy <= battery_remaining[:, None] + 1e-12)
    )

    utility = np.where(
        feasible,
        1.0
        - np.clip(
            finishes / max(config.episode_time, 1e-12),
            0.0,
            1.0,
        ),
        0.0,
    )
    best_task_utility = np.max(utility, axis=0)
    best_task_utility = np.where(
        task_available,
        best_task_utility,
        0.0,
    )
    option_mass = float(
        np.sum(best_task_utility) / max(N, 1)
    )

    task_degree = np.sum(feasible, axis=0).astype(np.float64)
    reserve_contribution = np.minimum(task_degree, 2.0) / 2.0
    reserve_contribution = np.where(
        task_available,
        reserve_contribution,
        0.0,
    )
    reserve_mass = float(
        np.sum(reserve_contribution) / max(N, 1)
    )

    return option_mass, reserve_mass


def rollout_robust_metrics(
    gene: ConsequenceAwareDirectGene,
    world: World,
    config: EnvConfig,
) -> RobustMetrics:
    """
    Replay the unchanged V1.8 direct policy while measuring two V1.9 external
    capability signals.

    Per event continuation conservation:
        C_t = clip((U_t + O_after) / O_before, 0, 1)

    where U_t is normalized immediate T utility realized by assignments and O
    is normalized fleet option mass.

    Per event reserve conservation:
        R_t = clip((A_t/N + Q_after) / Q_before, 0, 1)

    where A_t is number of completed assignments in the event and Q is
    normalized two-owner reserve mass.

    Episode scores are means over events with a positive corresponding
    pre-decision mass. Empty cases score 1.
    """
    R = config.num_robots
    N = config.num_tasks

    robot_positions = world.robot_positions.copy()
    battery_remaining = world.robot_initial_batteries.copy()
    current_node_ids = np.arange(R, dtype=np.int64)
    task_available = np.ones(N, dtype=bool)
    busy_until = np.zeros(R, dtype=np.float64)
    robot_workloads = np.zeros(R, dtype=np.float64)

    time_score_sum = 0.0
    continuation_scores: list[float] = []
    reserve_scores: list[float] = []

    max_events = N + R + 2
    for _ in range(max_events):
        if not np.any(task_available):
            break

        (
            now,
            free,
            observations,
            eligible,
            path_lengths,
            _euclidean,
        ) = _build_observations(
            world=world,
            config=config,
            robot_positions=robot_positions,
            current_node_ids=current_node_ids,
            battery_remaining=battery_remaining,
            robot_workloads=robot_workloads,
            busy_until=busy_until,
            task_available=task_available,
        )

        if now >= config.episode_time - 1e-12:
            break

        option_before, reserve_before = _future_graph_masses(
            world=world,
            config=config,
            now=now,
            current_node_ids=current_node_ids,
            battery_remaining=battery_remaining,
            busy_until=busy_until,
            task_available=task_available,
        )

        assignments = gene.assign(
            observations,
            eligible,
            free,
            task_available,
        )

        matched = np.zeros(R, dtype=bool)
        event_utility = 0.0

        for robot, task in assignments:
            if not eligible[robot, task]:
                raise RuntimeError("V1.9 metric replay received infeasible pair")
            if not task_available[task] or matched[robot]:
                raise RuntimeError("V1.9 metric replay received duplicate pair")

            path_d = float(path_lengths[robot, task])
            service = float(world.task_service_times[task])
            energy = path_d * config.energy_per_distance
            finish = now + path_d / config.robot_speed + service
            utility = 1.0 - float(
                np.clip(
                    finish / max(config.episode_time, 1e-12),
                    0.0,
                    1.0,
                )
            )

            event_utility += utility / max(N, 1)
            time_score_sum += utility / max(N, 1)

            busy_until[robot] = finish
            robot_positions[robot] = world.task_positions[task]
            current_node_ids[robot] = R + task
            battery_remaining[robot] = max(
                0.0,
                battery_remaining[robot] - energy,
            )
            task_available[task] = False
            robot_workloads[robot] += (
                path_d / config.robot_speed + service
            )
            matched[robot] = True

        unmatched_free = free & ~matched
        if np.any(unmatched_free):
            future_times = busy_until[busy_until > now + 1e-12]
            if future_times.size > 0:
                next_event = float(np.min(future_times))
                busy_until[unmatched_free] = next_event
            else:
                busy_until[unmatched_free] = config.episode_time

        option_after, reserve_after = _future_graph_masses(
            world=world,
            config=config,
            now=now,
            current_node_ids=current_node_ids,
            battery_remaining=battery_remaining,
            busy_until=busy_until,
            task_available=task_available,
        )

        if option_before > 1e-12:
            continuation_scores.append(
                float(
                    np.clip(
                        (event_utility + option_after) / option_before,
                        0.0,
                        1.0,
                    )
                )
            )

        if reserve_before > 1e-12:
            completed_fraction = len(assignments) / max(N, 1)
            reserve_scores.append(
                float(
                    np.clip(
                        (completed_fraction + reserve_after) / reserve_before,
                        0.0,
                        1.0,
                    )
                )
            )

    continuation = (
        float(np.mean(continuation_scores))
        if continuation_scores
        else 1.0
    )
    reserve = (
        float(np.mean(reserve_scores))
        if reserve_scores
        else 1.0
    )

    return RobustMetrics(
        time_optimality=float(time_score_sum),
        continuation_preservation=continuation,
        fleet_option_reserve=reserve,
    )


def evaluate_robust_population(
    genes: list[ConsequenceAwareDirectGene],
    worlds: list[World],
    config: EnvConfig,
) -> np.ndarray:
    """
    Return [gene, world, 3] with columns:
      0 time_optimality
      1 continuation_preservation
      2 fleet_option_reserve
    """
    tensor = np.zeros(
        (len(genes), len(worlds), 3),
        dtype=np.float64,
    )
    for gene_idx, gene in enumerate(genes):
        for world_idx, world in enumerate(worlds):
            metrics = rollout_robust_metrics(gene, world, config)
            tensor[gene_idx, world_idx, 0] = metrics.time_optimality
            tensor[gene_idx, world_idx, 1] = (
                metrics.continuation_preservation
            )
            tensor[gene_idx, world_idx, 2] = metrics.fleet_option_reserve
    return tensor
