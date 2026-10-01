from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    Evaluation,
    World,
    generate_world,
    jain_fairness,
)

from .direct_gene import ConsequenceAwareDirectGene


BASE_FEATURE_NAMES = (
    "euclidean",
    "path",
    "service",
    "priority",
    "deadline_remaining",
    "battery",
    "workload",
    "competition",
)

CONSEQUENCE_FEATURE_NAMES = (
    "self_future_reachability",
    "self_future_best_time_utility",
    "other_robot_opportunity_cost",
    "residual_battery",
)

FEATURE_NAMES = BASE_FEATURE_NAMES + CONSEQUENCE_FEATURE_NAMES

AXES = (
    "completion",
    "efficiency",
    "priority_satisfaction",
    "deadline_satisfaction",
    "balance",
    "time_optimality",
)


@dataclass(frozen=True)
class DirectRollout:
    evaluation: Evaluation
    assignment_events: tuple[tuple[tuple[int, int], ...], ...]


def _other_robot_opportunity_cost(
    *,
    world: World,
    config: EnvConfig,
    now: float,
    current_node_ids: np.ndarray,
    battery_remaining: np.ndarray,
    busy_until: np.ndarray,
    task_available: np.ndarray,
) -> np.ndarray:
    """
    Return [R,T] candidate opportunity cost.

    For each candidate owner i and task j, estimate how costly it is to remove
    j from every other robot's future option set. A task is costly to take
    when it is both valuable in time utility and scarce among another robot's
    remaining feasible tasks.

    No assignment optimization is solved here; this is a deterministic state
    feature derived from one-task feasibility.
    """
    R = config.num_robots
    T = config.num_tasks

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
    option_count = np.sum(feasible, axis=1)
    scarcity_weighted = utility / np.maximum(
        option_count[:, None],
        1,
    )

    result = np.zeros((R, T), dtype=np.float64)
    for owner in range(R):
        other_mask = np.ones(R, dtype=bool)
        other_mask[owner] = False
        if np.any(other_mask):
            result[owner] = np.max(
                scarcity_weighted[other_mask],
                axis=0,
            )

    return np.clip(result, 0.0, 1.0)


def _consequence_features(
    *,
    world: World,
    config: EnvConfig,
    now: float,
    current_node_ids: np.ndarray,
    battery_remaining: np.ndarray,
    busy_until: np.ndarray,
    task_available: np.ndarray,
    eligible: np.ndarray,
    path_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = config.num_robots
    T = config.num_tasks

    service = world.task_service_times
    finish_current = (
        now
        + path_lengths / config.robot_speed
        + service[None, :]
    )
    energy_current = (
        path_lengths * config.energy_per_distance
    )

    residual_battery = np.clip(
        (
            battery_remaining[:, None]
            - energy_current
        )
        / max(config.battery_capacity, 1e-12),
        0.0,
        1.0,
    )
    residual_battery = np.where(
        np.isfinite(residual_battery),
        residual_battery,
        0.0,
    )

    future_reachability = np.zeros((R, T), dtype=np.float64)
    future_best_time = np.zeros((R, T), dtype=np.float64)

    available_count = int(np.sum(task_available))

    for robot in range(R):
        for task in range(T):
            if not eligible[robot, task]:
                continue

            finish = float(finish_current[robot, task])
            battery_after = float(
                battery_remaining[robot]
                - energy_current[robot, task]
            )

            remaining = task_available.copy()
            remaining[task] = False
            denominator = max(available_count - 1, 1)

            paths_after = world.path_to_tasks[R + task, :]
            finish_after = (
                finish
                + paths_after / config.robot_speed
                + service
            )
            energy_after = (
                paths_after * config.energy_per_distance
            )
            feasible_after = (
                remaining
                & np.isfinite(paths_after)
                & (
                    finish_after
                    <= config.episode_time + 1e-12
                )
                & (
                    energy_after
                    <= battery_after + 1e-12
                )
            )

            future_reachability[robot, task] = (
                float(np.sum(feasible_after))
                / float(denominator)
            )

            if np.any(feasible_after):
                utilities = (
                    1.0
                    - np.clip(
                        finish_after[feasible_after]
                        / max(config.episode_time, 1e-12),
                        0.0,
                        1.0,
                    )
                )
                future_best_time[robot, task] = float(
                    np.max(utilities)
                )

    opportunity_cost = _other_robot_opportunity_cost(
        world=world,
        config=config,
        now=now,
        current_node_ids=current_node_ids,
        battery_remaining=battery_remaining,
        busy_until=busy_until,
        task_available=task_available,
    )

    return (
        np.clip(future_reachability, 0.0, 1.0),
        np.clip(future_best_time, 0.0, 1.0),
        opportunity_cost,
        residual_battery,
    )


def _build_observations(
    *,
    world: World,
    config: EnvConfig,
    robot_positions: np.ndarray,
    current_node_ids: np.ndarray,
    battery_remaining: np.ndarray,
    robot_workloads: np.ndarray,
    busy_until: np.ndarray,
    task_available: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    R = config.num_robots
    T = config.num_tasks

    now = float(np.min(busy_until))
    free = np.isclose(
        busy_until,
        now,
        rtol=0.0,
        atol=1e-12,
    )

    delta = (
        robot_positions[:, None, :]
        - world.task_positions[None, :, :]
    )
    euclidean = np.linalg.norm(delta, axis=-1)
    euclidean_norm = np.clip(
        euclidean / config.diagonal,
        0.0,
        1.0,
    )

    path_lengths = world.path_to_tasks[
        current_node_ids,
        :,
    ]
    path_norm = np.clip(
        path_lengths / config.path_cost_scale,
        0.0,
        1.0,
    )
    path_norm = np.where(
        np.isfinite(path_norm),
        path_norm,
        1.0,
    )

    if config.service_time_range > 0.0:
        service_base = np.clip(
            (
                world.task_service_times
                - config.service_time_min
            )
            / config.service_time_range,
            0.0,
            1.0,
        )
    else:
        service_base = np.zeros(T, dtype=np.float64)
    service_norm = np.broadcast_to(
        service_base[None, :],
        (R, T),
    )

    if config.priority_range > 0.0:
        priority_base = np.clip(
            (
                world.task_priorities
                - config.priority_min
            )
            / config.priority_range,
            0.0,
            1.0,
        )
    else:
        priority_base = np.zeros(T, dtype=np.float64)
    priority_norm = np.broadcast_to(
        priority_base[None, :],
        (R, T),
    )

    deadline_remaining = np.clip(
        (
            world.task_deadlines - now
        )
        / max(config.episode_time, 1e-12),
        0.0,
        1.0,
    )
    deadline_norm = np.broadcast_to(
        deadline_remaining[None, :],
        (R, T),
    )

    battery_norm = np.broadcast_to(
        np.clip(
            battery_remaining
            / max(config.battery_capacity, 1e-12),
            0.0,
            1.0,
        )[:, None],
        (R, T),
    )

    workload_norm = np.broadcast_to(
        np.clip(
            robot_workloads
            / max(config.episode_time, 1e-12),
            0.0,
            1.0,
        )[:, None],
        (R, T),
    )

    duration = (
        path_lengths / config.robot_speed
        + world.task_service_times[None, :]
    )
    finishes = now + duration
    energy_required = (
        path_lengths * config.energy_per_distance
    )

    time_eligible = (
        free[:, None]
        & task_available[None, :]
        & np.isfinite(path_lengths)
        & (
            finishes
            <= config.episode_time + 1e-12
        )
    )
    battery_feasible = (
        energy_required
        <= battery_remaining[:, None] + 1e-12
    )
    eligible = time_eligible & battery_feasible

    competition = np.zeros((R, T), dtype=np.float64)
    for task in range(T):
        eligible_robots = np.flatnonzero(
            eligible[:, task]
        )
        denom = max(
            len(eligible_robots) - 1,
            1,
        )
        for robot in eligible_robots:
            competition[robot, task] = (
                np.sum(
                    path_lengths[
                        eligible_robots,
                        task,
                    ]
                    < path_lengths[robot, task]
                )
                / denom
            )

    (
        future_reachability,
        future_best_time,
        opportunity_cost,
        residual_battery,
    ) = _consequence_features(
        world=world,
        config=config,
        now=now,
        current_node_ids=current_node_ids,
        battery_remaining=battery_remaining,
        busy_until=busy_until,
        task_available=task_available,
        eligible=eligible,
        path_lengths=path_lengths,
    )

    observations = np.stack(
        [
            euclidean_norm,
            path_norm,
            service_norm,
            priority_norm,
            deadline_norm,
            battery_norm,
            workload_norm,
            competition,
            future_reachability,
            future_best_time,
            opportunity_cost,
            residual_battery,
        ],
        axis=-1,
    )

    return (
        now,
        free,
        observations,
        eligible,
        path_lengths,
        euclidean,
    )


def rollout_direct_gene(
    gene: ConsequenceAwareDirectGene,
    world: World,
    config: EnvConfig,
) -> DirectRollout:
    R = config.num_robots
    N = config.num_tasks

    robot_positions = world.robot_positions.copy()
    battery_remaining = (
        world.robot_initial_batteries.copy()
    )
    current_node_ids = np.arange(
        R,
        dtype=np.int64,
    )
    task_available = np.ones(N, dtype=bool)
    busy_until = np.zeros(R, dtype=np.float64)
    task_counts = np.zeros(R, dtype=np.int64)
    robot_workloads = np.zeros(
        R,
        dtype=np.float64,
    )

    total_travel = 0.0
    total_euclidean = 0.0
    route_efficiency_sum = 0.0
    time_score_sum = 0.0
    priority_sum = 0.0
    on_time_count = 0.0
    battery_blocked_pair_events = 0.0
    assignment_events: list[
        tuple[tuple[int, int], ...]
    ] = []

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
            euclidean,
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

        duration = (
            path_lengths / config.robot_speed
            + world.task_service_times[None, :]
        )
        time_eligible = (
            free[:, None]
            & task_available[None, :]
            & np.isfinite(path_lengths)
            & (
                now + duration
                <= config.episode_time + 1e-12
            )
        )
        energy_required = (
            path_lengths
            * config.energy_per_distance
        )
        battery_feasible = (
            energy_required
            <= battery_remaining[:, None]
            + 1e-12
        )
        battery_blocked_pair_events += float(
            np.sum(
                time_eligible
                & ~battery_feasible
            )
        )

        assignments = gene.assign(
            observations,
            eligible,
            free,
            task_available,
        )
        assignment_events.append(
            tuple(assignments)
        )

        matched = np.zeros(R, dtype=bool)

        for robot, task in assignments:
            if not eligible[robot, task]:
                raise RuntimeError(
                    "V1.8 policy emitted infeasible assignment"
                )
            if (
                not task_available[task]
                or matched[robot]
            ):
                raise RuntimeError(
                    "V1.8 policy emitted duplicate assignment"
                )

            path_d = float(
                path_lengths[robot, task]
            )
            euclidean_d = float(
                euclidean[robot, task]
            )
            service = float(
                world.task_service_times[task]
            )
            energy = (
                path_d
                * config.energy_per_distance
            )
            finish = (
                now
                + path_d / config.robot_speed
                + service
            )

            busy_until[robot] = finish
            robot_positions[robot] = (
                world.task_positions[task]
            )
            current_node_ids[robot] = R + task
            battery_remaining[robot] = max(
                0.0,
                battery_remaining[robot]
                - energy,
            )
            task_available[task] = False
            task_counts[robot] += 1
            robot_workloads[robot] += (
                path_d / config.robot_speed
                + service
            )

            total_travel += path_d
            total_euclidean += euclidean_d
            priority_sum += float(
                world.task_priorities[task]
            )
            on_time_count += float(
                finish
                <= world.task_deadlines[task]
                + 1e-12
            )
            route_efficiency_sum += (
                1.0
                - float(
                    np.clip(
                        path_d
                        / config.diagonal,
                        0.0,
                        1.0,
                    )
                )
            )
            time_score_sum += (
                1.0
                - float(
                    np.clip(
                        finish
                        / max(
                            config.episode_time,
                            1e-12,
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
            matched[robot] = True

        unmatched_free = free & ~matched
        if np.any(unmatched_free):
            future_times = busy_until[
                busy_until > now + 1e-12
            ]
            if future_times.size > 0:
                next_event = float(
                    np.min(future_times)
                )
                busy_until[
                    unmatched_free
                ] = next_event
            else:
                busy_until[
                    unmatched_free
                ] = config.episode_time

    completed = float(np.sum(task_counts))
    completion = completed / N
    efficiency = route_efficiency_sum / N
    route_efficiency = (
        route_efficiency_sum / completed
        if completed > 0.0
        else 0.0
    )

    total_priority = float(
        np.sum(world.task_priorities)
    )
    priority_satisfaction = (
        priority_sum / total_priority
        if total_priority > 0.0
        else 0.0
    )
    deadline_satisfaction = (
        on_time_count / N
    )
    time_optimality = (
        time_score_sum / N
    )

    fairness = jain_fairness(
        robot_workloads
    )
    balance = completion * fairness

    detour_ratio = (
        total_travel / total_euclidean
        if total_euclidean > 0.0
        else 1.0
    )

    initial_battery_sum = float(
        np.sum(
            world.robot_initial_batteries
        )
    )
    final_battery_sum = float(
        np.sum(battery_remaining)
    )
    battery_remaining_fraction = (
        final_battery_sum
        / initial_battery_sum
        if initial_battery_sum > 0.0
        else 0.0
    )

    evaluation = Evaluation(
        completion=float(completion),
        efficiency=float(efficiency),
        priority_satisfaction=float(
            priority_satisfaction
        ),
        deadline_satisfaction=float(
            deadline_satisfaction
        ),
        balance=float(balance),
        time_optimality=float(
            time_optimality
        ),
        route_efficiency=float(
            route_efficiency
        ),
        completed_tasks=completed,
        completed_priority=float(
            priority_sum
        ),
        total_priority=total_priority,
        on_time_tasks=float(
            on_time_count
        ),
        total_travel=float(
            total_travel
        ),
        total_euclidean_travel=float(
            total_euclidean
        ),
        detour_ratio=float(
            detour_ratio
        ),
        mean_initial_battery=float(
            np.mean(
                world.robot_initial_batteries
            )
        ),
        mean_final_battery=float(
            np.mean(
                battery_remaining
            )
        ),
        battery_remaining_fraction=float(
            battery_remaining_fraction
        ),
        energy_consumed=float(
            initial_battery_sum
            - final_battery_sum
        ),
        battery_blocked_pair_events=float(
            battery_blocked_pair_events
        ),
        robot_task_counts=tuple(
            float(x)
            for x in task_counts.tolist()
        ),
        robot_workloads=tuple(
            float(x)
            for x in robot_workloads.tolist()
        ),
        robot_final_batteries=tuple(
            float(x)
            for x in battery_remaining.tolist()
        ),
    )

    return DirectRollout(
        evaluation=evaluation,
        assignment_events=tuple(
            assignment_events
        ),
    )


def evaluate_direct_population(
    genes: list[ConsequenceAwareDirectGene],
    worlds: list[World],
    config: EnvConfig,
) -> tuple[list[Evaluation], np.ndarray]:
    if not genes:
        return (
            [],
            np.zeros(
                (
                    0,
                    len(worlds),
                    len(AXES),
                ),
                dtype=np.float64,
            ),
        )

    tensor = np.zeros(
        (
            len(genes),
            len(worlds),
            len(AXES),
        ),
        dtype=np.float64,
    )

    all_rollouts: list[
        list[DirectRollout]
    ] = []

    for gene_idx, gene in enumerate(genes):
        gene_rollouts: list[
            DirectRollout
        ] = []
        for world_idx, world in enumerate(
            worlds
        ):
            rollout = rollout_direct_gene(
                gene,
                world,
                config,
            )
            gene_rollouts.append(
                rollout
            )
            for axis_idx, axis in enumerate(
                AXES
            ):
                tensor[
                    gene_idx,
                    world_idx,
                    axis_idx,
                ] = getattr(
                    rollout.evaluation,
                    axis,
                )
        all_rollouts.append(
            gene_rollouts
        )

    means: list[Evaluation] = []

    for gene_rollouts in all_rollouts:
        evs = [
            rollout.evaluation
            for rollout in gene_rollouts
        ]

        def mean(name: str) -> float:
            return float(
                np.mean(
                    [
                        getattr(ev, name)
                        for ev in evs
                    ]
                )
            )

        means.append(
            Evaluation(
                completion=mean(
                    "completion"
                ),
                efficiency=mean(
                    "efficiency"
                ),
                priority_satisfaction=mean(
                    "priority_satisfaction"
                ),
                deadline_satisfaction=mean(
                    "deadline_satisfaction"
                ),
                balance=mean(
                    "balance"
                ),
                time_optimality=mean(
                    "time_optimality"
                ),
                route_efficiency=mean(
                    "route_efficiency"
                ),
                completed_tasks=mean(
                    "completed_tasks"
                ),
                completed_priority=mean(
                    "completed_priority"
                ),
                total_priority=mean(
                    "total_priority"
                ),
                on_time_tasks=mean(
                    "on_time_tasks"
                ),
                total_travel=mean(
                    "total_travel"
                ),
                total_euclidean_travel=mean(
                    "total_euclidean_travel"
                ),
                detour_ratio=mean(
                    "detour_ratio"
                ),
                mean_initial_battery=mean(
                    "mean_initial_battery"
                ),
                mean_final_battery=mean(
                    "mean_final_battery"
                ),
                battery_remaining_fraction=mean(
                    "battery_remaining_fraction"
                ),
                energy_consumed=mean(
                    "energy_consumed"
                ),
                battery_blocked_pair_events=mean(
                    "battery_blocked_pair_events"
                ),
                robot_task_counts=tuple(
                    float(
                        np.mean(
                            [
                                ev.robot_task_counts[r]
                                for ev in evs
                            ]
                        )
                    )
                    for r in range(
                        config.num_robots
                    )
                ),
                robot_workloads=tuple(
                    float(
                        np.mean(
                            [
                                ev.robot_workloads[r]
                                for ev in evs
                            ]
                        )
                    )
                    for r in range(
                        config.num_robots
                    )
                ),
                robot_final_batteries=tuple(
                    float(
                        np.mean(
                            [
                                ev.robot_final_batteries[r]
                                for ev in evs
                            ]
                        )
                    )
                    for r in range(
                        config.num_robots
                    )
                ),
            )
        )

    return means, tensor


def make_worlds(
    config: EnvConfig,
    base_seed: int,
    count: int,
) -> list[World]:
    return [
        generate_world(
            config,
            base_seed + idx,
        )
        for idx in range(count)
    ]
