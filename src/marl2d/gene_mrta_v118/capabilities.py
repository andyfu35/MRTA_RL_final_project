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

# V1.18 capabilities are raw [0, 1] task-quality measurements.
# Completion is intentionally not an axis: it is a hard success gate.
# Deadline remains environment metadata only and is not a Pareto capability.
BASE_AXES = (
    "time_earliness",
    "path_efficiency",
    "priority_service",
)


@dataclass(frozen=True)
class GeneAssessment:
    success: bool
    worst_completion: float
    mean_completion: float
    success_worlds: int
    world_count: int
    scores: dict[str, float]


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
        np.clip(
            weighted_earliness
            / total_priority,
            0.0,
            1.0,
        )
    )


def evaluate_gene(
    gene: RouteTailDirectGene,
    worlds: Sequence[World],
    config: EnvConfig,
) -> GeneAssessment:
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

    for rollout, world in zip(
        rollouts,
        worlds,
        strict=True,
    ):
        evaluation = (
            rollout.evaluation
        )
        rows.append(
            {
                "time_earliness": float(
                    np.clip(
                        evaluation.time_optimality,
                        0.0,
                        1.0,
                    )
                ),
                "path_efficiency": float(
                    np.clip(
                        evaluation.efficiency,
                        0.0,
                        1.0,
                    )
                ),
                "priority_service": (
                    priority_service_score(
                        rollout.plan,
                        world,
                        config,
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
