from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    Evaluation,
    World,
    jain_fairness,
)

from .direct_gene import RouteTailDirectGene


EPS = 1e-12


@dataclass(frozen=True)
class RouteTailStep:
    step: int
    robot: int
    task: int
    start_time: float
    finish_time: float
    path_distance: float
    euclidean_distance: float
    service_time: float
    energy_used: float

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "robot": self.robot,
            "task": self.task,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "path_distance": self.path_distance,
            "euclidean_distance": self.euclidean_distance,
            "service_time": self.service_time,
            "energy_used": self.energy_used,
        }


@dataclass(frozen=True)
class RouteTailPlan:
    routes: tuple[tuple[int, ...], ...]
    selection_sequence: tuple[RouteTailStep, ...]
    stopped_by_policy: bool
    remaining_tasks: tuple[int, ...]
    final_tail_times: tuple[float, ...]
    final_batteries: tuple[float, ...]
    final_workloads: tuple[float, ...]
    battery_blocked_pair_events: float

    def to_dict(self) -> dict[str, object]:
        return {
            "routes": [
                list(route)
                for route in self.routes
            ],
            "selection_sequence": [
                item.to_dict()
                for item
                in self.selection_sequence
            ],
            "stopped_by_policy": (
                self.stopped_by_policy
            ),
            "remaining_tasks": list(
                self.remaining_tasks
            ),
            "final_tail_times": list(
                self.final_tail_times
            ),
            "final_batteries": list(
                self.final_batteries
            ),
            "final_workloads": list(
                self.final_workloads
            ),
            "battery_blocked_pair_events": (
                self.battery_blocked_pair_events
            ),
        }


@dataclass(frozen=True)
class RouteTailRollout:
    evaluation: Evaluation
    plan: RouteTailPlan


def _route_tail_opportunity_cost(
    *,
    world: World,
    config: EnvConfig,
    tail_node_ids: np.ndarray,
    tail_times: np.ndarray,
    battery_remaining: np.ndarray,
    task_available: np.ndarray,
) -> np.ndarray:
    """
    Candidate opportunity cost for route-tail planning.

    Every robot is considered from the end of its currently planned route,
    not from the physical state at time zero.
    """
    R = config.num_robots
    T = config.num_tasks

    paths = world.path_to_tasks[
        tail_node_ids,
        :,
    ]
    finishes = (
        tail_times[:, None]
        + paths / config.robot_speed
        + world.task_service_times[
            None,
            :,
        ]
    )
    energy = (
        paths
        * config.energy_per_distance
    )
    feasible = (
        task_available[None, :]
        & np.isfinite(paths)
        & (
            finishes
            <= config.episode_time
            + EPS
        )
        & (
            energy
            <= battery_remaining[
                :, None
            ]
            + EPS
        )
    )
    utility = np.where(
        feasible,
        1.0
        - np.clip(
            finishes
            / max(
                config.episode_time,
                EPS,
            ),
            0.0,
            1.0,
        ),
        0.0,
    )
    option_count = np.sum(
        feasible,
        axis=1,
    )
    scarcity_weighted = (
        utility
        / np.maximum(
            option_count[:, None],
            1,
        )
    )

    result = np.zeros(
        (R, T),
        dtype=np.float64,
    )
    for owner in range(R):
        other = np.ones(
            R,
            dtype=bool,
        )
        other[owner] = False
        if np.any(other):
            result[owner] = np.max(
                scarcity_weighted[
                    other
                ],
                axis=0,
            )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def _route_tail_consequence_features(
    *,
    world: World,
    config: EnvConfig,
    tail_node_ids: np.ndarray,
    tail_times: np.ndarray,
    battery_remaining: np.ndarray,
    task_available: np.ndarray,
    eligible: np.ndarray,
    path_lengths: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    R = config.num_robots
    T = config.num_tasks
    service = (
        world.task_service_times
    )

    finish_current = (
        tail_times[:, None]
        + path_lengths
        / config.robot_speed
        + service[None, :]
    )
    energy_current = (
        path_lengths
        * config.energy_per_distance
    )
    residual_battery = np.clip(
        (
            battery_remaining[:, None]
            - energy_current
        )
        / max(
            config.battery_capacity,
            EPS,
        ),
        0.0,
        1.0,
    )
    residual_battery = np.where(
        np.isfinite(
            residual_battery
        ),
        residual_battery,
        0.0,
    )

    future_reachability = np.zeros(
        (R, T),
        dtype=np.float64,
    )
    future_best_time = np.zeros(
        (R, T),
        dtype=np.float64,
    )

    available_count = int(
        np.sum(
            task_available
        )
    )

    for robot in range(R):
        for task in range(T):
            if not eligible[
                robot,
                task,
            ]:
                continue

            finish = float(
                finish_current[
                    robot,
                    task,
                ]
            )
            battery_after = float(
                battery_remaining[
                    robot
                ]
                - energy_current[
                    robot,
                    task,
                ]
            )
            remaining = (
                task_available.copy()
            )
            remaining[task] = False
            denominator = max(
                available_count - 1,
                1,
            )

            paths_after = (
                world.path_to_tasks[
                    R + task,
                    :,
                ]
            )
            finishes_after = (
                finish
                + paths_after
                / config.robot_speed
                + service
            )
            energy_after = (
                paths_after
                * config.energy_per_distance
            )
            feasible_after = (
                remaining
                & np.isfinite(
                    paths_after
                )
                & (
                    finishes_after
                    <= config.episode_time
                    + EPS
                )
                & (
                    energy_after
                    <= battery_after
                    + EPS
                )
            )

            future_reachability[
                robot,
                task,
            ] = (
                float(
                    np.sum(
                        feasible_after
                    )
                )
                / float(
                    denominator
                )
            )

            if np.any(
                feasible_after
            ):
                utilities = (
                    1.0
                    - np.clip(
                        finishes_after[
                            feasible_after
                        ]
                        / max(
                            config.episode_time,
                            EPS,
                        ),
                        0.0,
                        1.0,
                    )
                )
                future_best_time[
                    robot,
                    task,
                ] = float(
                    np.max(
                        utilities
                    )
                )

    opportunity_cost = (
        _route_tail_opportunity_cost(
            world=world,
            config=config,
            tail_node_ids=(
                tail_node_ids
            ),
            tail_times=tail_times,
            battery_remaining=(
                battery_remaining
            ),
            task_available=(
                task_available
            ),
        )
    )

    return (
        np.clip(
            future_reachability,
            0.0,
            1.0,
        ),
        np.clip(
            future_best_time,
            0.0,
            1.0,
        ),
        opportunity_cost,
        residual_battery,
    )


def build_route_tail_observations(
    *,
    world: World,
    config: EnvConfig,
    tail_positions: np.ndarray,
    tail_node_ids: np.ndarray,
    tail_times: np.ndarray,
    battery_remaining: np.ndarray,
    robot_workloads: np.ndarray,
    task_available: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build V1.13 pair observations from each robot's virtual route tail.

    Unlike V1.8, every robot may remain selectable after receiving a task.
    A pair (i,j) means: append task j to the current end of robot i's route.
    """
    R = config.num_robots
    T = config.num_tasks

    tail_positions = np.asarray(
        tail_positions,
        dtype=np.float64,
    )
    tail_node_ids = np.asarray(
        tail_node_ids,
        dtype=np.int64,
    )
    tail_times = np.asarray(
        tail_times,
        dtype=np.float64,
    )
    battery_remaining = np.asarray(
        battery_remaining,
        dtype=np.float64,
    )
    robot_workloads = np.asarray(
        robot_workloads,
        dtype=np.float64,
    )
    task_available = np.asarray(
        task_available,
        dtype=bool,
    )

    if tail_positions.shape != (
        R,
        2,
    ):
        raise ValueError(
            "tail_positions shape mismatch"
        )
    for array, name in (
        (tail_node_ids, "tail_node_ids"),
        (tail_times, "tail_times"),
        (
            battery_remaining,
            "battery_remaining",
        ),
        (
            robot_workloads,
            "robot_workloads",
        ),
    ):
        if array.shape != (R,):
            raise ValueError(
                f"{name} shape mismatch"
            )
    if task_available.shape != (
        T,
    ):
        raise ValueError(
            "task_available shape mismatch"
        )

    delta = (
        tail_positions[
            :, None, :
        ]
        - world.task_positions[
            None, :, :
        ]
    )
    euclidean = np.linalg.norm(
        delta,
        axis=-1,
    )
    euclidean_norm = np.clip(
        euclidean
        / config.diagonal,
        0.0,
        1.0,
    )

    path_lengths = (
        world.path_to_tasks[
            tail_node_ids,
            :,
        ]
    )
    path_norm = np.clip(
        path_lengths
        / config.path_cost_scale,
        0.0,
        1.0,
    )
    path_norm = np.where(
        np.isfinite(
            path_norm
        ),
        path_norm,
        1.0,
    )

    if (
        config.service_time_range
        > 0.0
    ):
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
        service_base = np.zeros(
            T,
            dtype=np.float64,
        )
    service_norm = np.broadcast_to(
        service_base[
            None,
            :,
        ],
        (
            R,
            T,
        ),
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
        priority_base = np.zeros(
            T,
            dtype=np.float64,
        )
    priority_norm = np.broadcast_to(
        priority_base[
            None,
            :,
        ],
        (
            R,
            T,
        ),
    )

    deadline_remaining = np.clip(
        (
            world.task_deadlines[
                None,
                :,
            ]
            - tail_times[
                :,
                None,
            ]
        )
        / max(
            config.episode_time,
            EPS,
        ),
        0.0,
        1.0,
    )

    battery_norm = np.broadcast_to(
        np.clip(
            battery_remaining
            / max(
                config.battery_capacity,
                EPS,
            ),
            0.0,
            1.0,
        )[
            :,
            None,
        ],
        (
            R,
            T,
        ),
    )
    workload_norm = np.broadcast_to(
        np.clip(
            robot_workloads
            / max(
                config.episode_time,
                EPS,
            ),
            0.0,
            1.0,
        )[
            :,
            None,
        ],
        (
            R,
            T,
        ),
    )

    duration = (
        path_lengths
        / config.robot_speed
        + world.task_service_times[
            None,
            :,
        ]
    )
    finishes = (
        tail_times[
            :,
            None,
        ]
        + duration
    )
    energy_required = (
        path_lengths
        * config.energy_per_distance
    )

    time_eligible = (
        task_available[
            None,
            :,
        ]
        & np.isfinite(
            path_lengths
        )
        & (
            finishes
            <= config.episode_time
            + EPS
        )
    )
    battery_feasible = (
        energy_required
        <= battery_remaining[
            :,
            None,
        ]
        + EPS
    )
    eligible = (
        time_eligible
        & battery_feasible
    )

    competition = np.zeros(
        (R, T),
        dtype=np.float64,
    )
    for task in range(T):
        owners = np.flatnonzero(
            eligible[
                :,
                task,
            ]
        )
        denom = max(
            len(owners) - 1,
            1,
        )
        for robot in owners:
            competition[
                robot,
                task,
            ] = (
                np.sum(
                    finishes[
                        owners,
                        task,
                    ]
                    < finishes[
                        robot,
                        task,
                    ]
                )
                / denom
            )

    (
        future_reachability,
        future_best_time,
        opportunity_cost,
        residual_battery,
    ) = (
        _route_tail_consequence_features(
            world=world,
            config=config,
            tail_node_ids=(
                tail_node_ids
            ),
            tail_times=tail_times,
            battery_remaining=(
                battery_remaining
            ),
            task_available=(
                task_available
            ),
            eligible=eligible,
            path_lengths=(
                path_lengths
            ),
        )
    )

    observations = np.stack(
        [
            euclidean_norm,
            path_norm,
            service_norm,
            priority_norm,
            deadline_remaining,
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
        observations,
        eligible,
        path_lengths,
        euclidean,
        time_eligible
        & ~battery_feasible,
    )


def plan_route_tails(
    gene: RouteTailDirectGene,
    world: World,
    config: EnvConfig,
) -> RouteTailPlan:
    R = config.num_robots
    N = config.num_tasks

    tail_positions = (
        world.robot_positions.copy()
    )
    tail_node_ids = np.arange(
        R,
        dtype=np.int64,
    )
    tail_times = np.zeros(
        R,
        dtype=np.float64,
    )
    battery_remaining = (
        world.robot_initial_batteries.copy()
    )
    workloads = np.zeros(
        R,
        dtype=np.float64,
    )
    task_available = np.ones(
        N,
        dtype=bool,
    )

    routes: list[list[int]] = [
        []
        for _ in range(R)
    ]
    sequence: list[
        RouteTailStep
    ] = []
    stopped_by_policy = False
    battery_blocked = 0.0

    # Rows intentionally stay open. Only a selected task column closes.
    row_open = np.ones(
        R,
        dtype=bool,
    )

    for step in range(N):
        if not np.any(
            task_available
        ):
            break

        (
            observations,
            eligible,
            path_lengths,
            euclidean,
            battery_blocked_pairs,
        ) = build_route_tail_observations(
            world=world,
            config=config,
            tail_positions=(
                tail_positions
            ),
            tail_node_ids=(
                tail_node_ids
            ),
            tail_times=tail_times,
            battery_remaining=(
                battery_remaining
            ),
            robot_workloads=(
                workloads
            ),
            task_available=(
                task_available
            ),
        )
        battery_blocked += float(
            np.sum(
                battery_blocked_pairs
            )
        )

        if not np.any(eligible):
            break

        logits, stop_logit = (
            gene.action_logits(
                observations,
                eligible,
                row_open,
                task_available,
                step,
            )
        )
        flat = int(
            np.argmax(
                logits
            )
        )
        best = float(
            logits.flat[
                flat
            ]
        )
        if not np.isfinite(best):
            break
        if stop_logit >= best:
            stopped_by_policy = True
            break

        robot = (
            flat // N
        )
        task = (
            flat % N
        )
        if not eligible[
            robot,
            task,
        ]:
            raise RuntimeError(
                "V1.13 decoder selected "
                "an infeasible route-tail pair"
            )
        if not task_available[
            task
        ]:
            raise RuntimeError(
                "V1.13 decoder selected "
                "an unavailable task"
            )

        path_d = float(
            path_lengths[
                robot,
                task,
            ]
        )
        euclidean_d = float(
            euclidean[
                robot,
                task,
            ]
        )
        service = float(
            world.task_service_times[
                task
            ]
        )
        energy = (
            path_d
            * config.energy_per_distance
        )
        start = float(
            tail_times[
                robot
            ]
        )
        finish = (
            start
            + path_d
            / config.robot_speed
            + service
        )

        sequence.append(
            RouteTailStep(
                step=step,
                robot=robot,
                task=task,
                start_time=start,
                finish_time=finish,
                path_distance=path_d,
                euclidean_distance=(
                    euclidean_d
                ),
                service_time=service,
                energy_used=energy,
            )
        )
        routes[
            robot
        ].append(
            task
        )

        # Task closes, robot row stays open with a new virtual route tail.
        task_available[
            task
        ] = False
        tail_positions[
            robot
        ] = (
            world.task_positions[
                task
            ]
        )
        tail_node_ids[
            robot
        ] = (
            R + task
        )
        tail_times[
            robot
        ] = finish
        battery_remaining[
            robot
        ] = max(
            0.0,
            battery_remaining[
                robot
            ]
            - energy,
        )
        workloads[
            robot
        ] += (
            path_d
            / config.robot_speed
            + service
        )

    return RouteTailPlan(
        routes=tuple(
            tuple(route)
            for route in routes
        ),
        selection_sequence=tuple(
            sequence
        ),
        stopped_by_policy=(
            stopped_by_policy
        ),
        remaining_tasks=tuple(
            int(task)
            for task in np.flatnonzero(
                task_available
            )
        ),
        final_tail_times=tuple(
            float(value)
            for value in tail_times
        ),
        final_batteries=tuple(
            float(value)
            for value
            in battery_remaining
        ),
        final_workloads=tuple(
            float(value)
            for value
            in workloads
        ),
        battery_blocked_pair_events=float(
            battery_blocked
        ),
    )


def evaluate_route_tail_plan(
    plan: RouteTailPlan,
    world: World,
    config: EnvConfig,
) -> Evaluation:
    N = config.num_tasks
    R = config.num_robots

    completed = float(
        len(
            plan.selection_sequence
        )
    )
    total_travel = float(
        sum(
            item.path_distance
            for item
            in plan.selection_sequence
        )
    )
    total_euclidean = float(
        sum(
            item.euclidean_distance
            for item
            in plan.selection_sequence
        )
    )
    route_efficiency_sum = float(
        sum(
            1.0
            - float(
                np.clip(
                    item.path_distance
                    / config.diagonal,
                    0.0,
                    1.0,
                )
            )
            for item
            in plan.selection_sequence
        )
    )
    time_score_sum = float(
        sum(
            1.0
            - float(
                np.clip(
                    item.finish_time
                    / max(
                        config.episode_time,
                        EPS,
                    ),
                    0.0,
                    1.0,
                )
            )
            for item
            in plan.selection_sequence
        )
    )
    priority_sum = float(
        sum(
            world.task_priorities[
                item.task
            ]
            for item
            in plan.selection_sequence
        )
    )
    on_time = float(
        sum(
            item.finish_time
            <= world.task_deadlines[
                item.task
            ]
            + EPS
            for item
            in plan.selection_sequence
        )
    )

    task_counts = np.asarray(
        [
            len(route)
            for route
            in plan.routes
        ],
        dtype=np.float64,
    )
    workloads = np.asarray(
        plan.final_workloads,
        dtype=np.float64,
    )
    final_batteries = np.asarray(
        plan.final_batteries,
        dtype=np.float64,
    )

    completion = (
        completed
        / max(
            N,
            1,
        )
    )
    efficiency = (
        route_efficiency_sum
        / max(
            N,
            1,
        )
    )
    route_efficiency = (
        route_efficiency_sum
        / completed
        if completed > 0.0
        else 0.0
    )
    total_priority = float(
        np.sum(
            world.task_priorities
        )
    )
    priority_satisfaction = (
        priority_sum
        / total_priority
        if total_priority > 0.0
        else 0.0
    )
    deadline_satisfaction = (
        on_time
        / max(
            N,
            1,
        )
    )
    time_optimality = (
        time_score_sum
        / max(
            N,
            1,
        )
    )
    fairness = (
        jain_fairness(
            workloads
        )
    )
    balance = (
        completion
        * fairness
    )
    detour_ratio = (
        total_travel
        / total_euclidean
        if total_euclidean
        > 0.0
        else 1.0
    )

    initial_battery_sum = float(
        np.sum(
            world.robot_initial_batteries
        )
    )
    final_battery_sum = float(
        np.sum(
            final_batteries
        )
    )
    battery_remaining_fraction = (
        final_battery_sum
        / initial_battery_sum
        if initial_battery_sum
        > 0.0
        else 0.0
    )

    return Evaluation(
        completion=float(
            completion
        ),
        efficiency=float(
            efficiency
        ),
        priority_satisfaction=float(
            priority_satisfaction
        ),
        deadline_satisfaction=float(
            deadline_satisfaction
        ),
        balance=float(
            balance
        ),
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
        total_priority=(
            total_priority
        ),
        on_time_tasks=float(
            on_time
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
                final_batteries
            )
        ),
        battery_remaining_fraction=float(
            battery_remaining_fraction
        ),
        energy_consumed=float(
            initial_battery_sum
            - final_battery_sum
        ),
        battery_blocked_pair_events=(
            plan.battery_blocked_pair_events
        ),
        robot_task_counts=tuple(
            float(value)
            for value in task_counts
        ),
        robot_workloads=tuple(
            float(value)
            for value in workloads
        ),
        robot_final_batteries=tuple(
            float(value)
            for value
            in final_batteries
        ),
    )


def rollout_route_tail_gene(
    gene: RouteTailDirectGene,
    world: World,
    config: EnvConfig,
) -> RouteTailRollout:
    plan = plan_route_tails(
        gene,
        world,
        config,
    )
    evaluation = (
        evaluate_route_tail_plan(
            plan,
            world,
            config,
        )
    )
    return RouteTailRollout(
        evaluation=evaluation,
        plan=plan,
    )
