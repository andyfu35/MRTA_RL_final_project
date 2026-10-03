from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, World
from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene

from .direct_gene import RouteTailDirectGene
from .route_tail import plan_route_tails


EPS = 1e-12


@dataclass(frozen=True)
class RouteTailRobustMetrics:
    time_optimality: float
    continuation_preservation: float
    fleet_option_reserve: float
    mean_queue_depth: float
    max_queue_depth: float


def _future_graph_masses(
    *,
    world: World,
    config: EnvConfig,
    tail_node_ids: np.ndarray,
    tail_times: np.ndarray,
    battery_remaining: np.ndarray,
    task_available: np.ndarray,
) -> tuple[float, float]:
    N = config.num_tasks

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
    best_task_utility = np.max(
        utility,
        axis=0,
    )
    best_task_utility = np.where(
        task_available,
        best_task_utility,
        0.0,
    )
    option_mass = float(
        np.sum(
            best_task_utility
        )
        / max(
            N,
            1,
        )
    )

    task_degree = np.sum(
        feasible,
        axis=0,
    ).astype(
        np.float64
    )
    reserve_contribution = (
        np.minimum(
            task_degree,
            2.0,
        )
        / 2.0
    )
    reserve_contribution = np.where(
        task_available,
        reserve_contribution,
        0.0,
    )
    reserve_mass = float(
        np.sum(
            reserve_contribution
        )
        / max(
            N,
            1,
        )
    )

    return (
        option_mass,
        reserve_mass,
    )


def rollout_route_tail_robust_metrics(
    gene: ConsequenceAwareDirectGene,
    world: World,
    config: EnvConfig,
) -> RouteTailRobustMetrics:
    route_gene = (
        gene
        if isinstance(
            gene,
            RouteTailDirectGene,
        )
        else RouteTailDirectGene.from_v18(
            gene
        )
    )
    plan = plan_route_tails(
        route_gene,
        world,
        config,
    )

    R = config.num_robots
    N = config.num_tasks

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
    task_available = np.ones(
        N,
        dtype=bool,
    )

    time_score_sum = 0.0
    continuation_scores: list[
        float
    ] = []
    reserve_scores: list[
        float
    ] = []

    for item in plan.selection_sequence:
        option_before, reserve_before = (
            _future_graph_masses(
                world=world,
                config=config,
                tail_node_ids=(
                    tail_node_ids
                ),
                tail_times=(
                    tail_times
                ),
                battery_remaining=(
                    battery_remaining
                ),
                task_available=(
                    task_available
                ),
            )
        )

        utility = (
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
        )
        event_utility = (
            utility
            / max(
                N,
                1,
            )
        )
        time_score_sum += (
            event_utility
        )

        robot = int(
            item.robot
        )
        task = int(
            item.task
        )
        if not task_available[
            task
        ]:
            raise RuntimeError(
                "Duplicate task in V1.13 route-tail plan"
            )

        tail_node_ids[
            robot
        ] = (
            R + task
        )
        tail_times[
            robot
        ] = float(
            item.finish_time
        )
        battery_remaining[
            robot
        ] = max(
            0.0,
            battery_remaining[
                robot
            ]
            - float(
                item.energy_used
            ),
        )
        task_available[
            task
        ] = False

        option_after, reserve_after = (
            _future_graph_masses(
                world=world,
                config=config,
                tail_node_ids=(
                    tail_node_ids
                ),
                tail_times=(
                    tail_times
                ),
                battery_remaining=(
                    battery_remaining
                ),
                task_available=(
                    task_available
                ),
            )
        )

        if option_before > EPS:
            continuation_scores.append(
                float(
                    np.clip(
                        (
                            event_utility
                            + option_after
                        )
                        / option_before,
                        0.0,
                        1.0,
                    )
                )
            )

        if reserve_before > EPS:
            reserve_scores.append(
                float(
                    np.clip(
                        (
                            1.0
                            / max(
                                N,
                                1,
                            )
                            + reserve_after
                        )
                        / reserve_before,
                        0.0,
                        1.0,
                    )
                )
            )

    queue_depths = np.asarray(
        [
            len(route)
            for route
            in plan.routes
        ],
        dtype=np.float64,
    )

    return RouteTailRobustMetrics(
        time_optimality=float(
            time_score_sum
        ),
        continuation_preservation=(
            float(
                np.mean(
                    continuation_scores
                )
            )
            if continuation_scores
            else 1.0
        ),
        fleet_option_reserve=(
            float(
                np.mean(
                    reserve_scores
                )
            )
            if reserve_scores
            else 1.0
        ),
        mean_queue_depth=float(
            np.mean(
                queue_depths
            )
        ),
        max_queue_depth=float(
            np.max(
                queue_depths
            )
        ),
    )


def evaluate_route_tail_population(
    genes: list[ConsequenceAwareDirectGene],
    worlds: list[World],
    config: EnvConfig,
) -> np.ndarray:
    """
    Return [gene, world, 3]:
      0 = route-tail time optimality
      1 = continuation preservation
      2 = fleet option reserve
    """
    tensor = np.zeros(
        (
            len(genes),
            len(worlds),
            3,
        ),
        dtype=np.float64,
    )
    for gene_idx, gene in enumerate(
        genes
    ):
        for world_idx, world in enumerate(
            worlds
        ):
            metrics = (
                rollout_route_tail_robust_metrics(
                    gene,
                    world,
                    config,
                )
            )
            tensor[
                gene_idx,
                world_idx,
                0,
            ] = (
                metrics.time_optimality
            )
            tensor[
                gene_idx,
                world_idx,
                1,
            ] = (
                metrics.continuation_preservation
            )
            tensor[
                gene_idx,
                world_idx,
                2,
            ] = (
                metrics.fleet_option_reserve
            )
    return tensor
