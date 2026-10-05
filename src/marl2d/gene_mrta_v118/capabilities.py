from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    World,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v113.route_tail import (
    RouteTailPlan,
    rollout_route_tail_gene,
)


EPS = 1e-12

BASE_AXES = (
    "global_time_optimality",
    "global_path_efficiency",
    "global_priority_service",
    "global_deadline_optimality",
    "workload_balance",
)


@dataclass(frozen=True)
class GeneAssessment:
    success: bool
    worst_completion: float
    mean_completion: float
    success_worlds: int
    world_count: int
    scores: dict[str, float]


def _retention(
    value: float,
    optimum: float,
) -> float:
    if optimum <= EPS:
        return (
            1.0
            if value <= EPS
            else 0.0
        )
    return float(
        np.clip(
            value / optimum,
            0.0,
            1.0,
        )
    )


def priority_service_score(
    plan: RouteTailPlan,
    world: World,
    config: EnvConfig,
) -> float:
    total_priority = max(
        float(
            np.sum(
                world.task_priorities
            )
        ),
        EPS,
    )
    weighted_earliness = 0.0
    for step in (
        plan.selection_sequence
    ):
        priority = float(
            world.task_priorities[
                step.task
            ]
        )
        earliness = (
            1.0
            - float(
                np.clip(
                    step.finish_time
                    / max(
                        config.episode_time,
                        EPS,
                    ),
                    0.0,
                    1.0,
                )
            )
        )
        weighted_earliness += (
            priority
            * earliness
        )
    return float(
        weighted_earliness
        / total_priority
    )


def evaluate_gene(
    gene: RouteTailDirectGene,
    worlds: Sequence[World],
    config: EnvConfig,
    *,
    time_optima: Sequence[float],
    path_efficiency_optima: Sequence[float],
    priority_service_optima: Sequence[float],
    deadline_optima: Sequence[float],
) -> GeneAssessment:
    lengths = {
        len(worlds),
        len(time_optima),
        len(path_efficiency_optima),
        len(priority_service_optima),
        len(deadline_optima),
    }
    if len(lengths) != 1:
        raise ValueError(
            "worlds and oracle references must have equal lengths"
        )
    if not worlds:
        raise ValueError(
            "At least one world is required"
        )

    rollouts = [
        rollout_route_tail_gene(
            gene,
            world,
            config,
        )
        for world in worlds
    ]

    completions = np.asarray(
        [
            rollout.evaluation.completion
            for rollout in rollouts
        ],
        dtype=np.float64,
    )
    success_mask = (
        completions
        >= 1.0 - EPS
    )
    success_worlds = int(
        np.sum(
            success_mask
        )
    )
    success = (
        success_worlds
        == len(worlds)
    )

    if not success:
        return GeneAssessment(
            success=False,
            worst_completion=float(
                np.min(
                    completions
                )
            ),
            mean_completion=float(
                np.mean(
                    completions
                )
            ),
            success_worlds=(
                success_worlds
            ),
            world_count=len(
                worlds
            ),
            scores={
                axis: 0.0
                for axis in BASE_AXES
            },
        )

    rows: list[
        dict[str, float]
    ] = []
    for (
        rollout,
        world,
        time_star,
        path_star,
        priority_star,
        deadline_star,
    ) in zip(
        rollouts,
        worlds,
        time_optima,
        path_efficiency_optima,
        priority_service_optima,
        deadline_optima,
        strict=True,
    ):
        evaluation = (
            rollout.evaluation
        )
        raw_fairness = float(
            evaluation.balance
            / max(
                evaluation.completion,
                EPS,
            )
        )
        rows.append(
            {
                "global_time_optimality": (
                    _retention(
                        evaluation.time_optimality,
                        float(
                            time_star
                        ),
                    )
                ),
                "global_path_efficiency": (
                    _retention(
                        evaluation.efficiency,
                        float(
                            path_star
                        ),
                    )
                ),
                "global_priority_service": (
                    _retention(
                        priority_service_score(
                            rollout.plan,
                            world,
                            config,
                        ),
                        float(
                            priority_star
                        ),
                    )
                ),
                "global_deadline_optimality": (
                    _retention(
                        evaluation.deadline_satisfaction,
                        float(
                            deadline_star
                        ),
                    )
                ),
                "workload_balance": float(
                    np.clip(
                        raw_fairness,
                        0.0,
                        1.0,
                    )
                ),
            }
        )

    scores = {
        axis: float(
            np.mean(
                [
                    row[axis]
                    for row in rows
                ]
            )
        )
        for axis in BASE_AXES
    }

    return GeneAssessment(
        success=True,
        worst_completion=1.0,
        mean_completion=1.0,
        success_worlds=len(
            worlds
        ),
        world_count=len(
            worlds
        ),
        scores=scores,
    )
