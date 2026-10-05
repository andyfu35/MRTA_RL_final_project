from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    Evaluation,
    World,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v113.route_tail import (
    rollout_route_tail_gene,
)


EPS = 1e-12

# Stage A contains only objectives/requirements that are known before any
# hard-world analysis. Stage B is intentionally not defined here.
BASE_AXES = (
    "completion",
    "time_retention",
    "path_efficiency",
    "priority_satisfaction",
    "deadline_satisfaction",
    "workload_balance",
)


def base_scores_for_evaluation(
    evaluation: Evaluation,
    *,
    time_optimum: float,
) -> dict[str, float]:
    if time_optimum <= EPS:
        time_retention = (
            1.0
            if evaluation.time_optimality <= EPS
            else 0.0
        )
    else:
        time_retention = float(
            np.clip(
                evaluation.time_optimality
                / time_optimum,
                0.0,
                1.0,
            )
        )

    return {
        "completion": float(
            np.clip(
                evaluation.completion,
                0.0,
                1.0,
            )
        ),
        "time_retention": (
            time_retention
        ),
        "path_efficiency": float(
            np.clip(
                evaluation.efficiency,
                0.0,
                1.0,
            )
        ),
        "priority_satisfaction": float(
            np.clip(
                evaluation.priority_satisfaction,
                0.0,
                1.0,
            )
        ),
        "deadline_satisfaction": float(
            np.clip(
                evaluation.deadline_satisfaction,
                0.0,
                1.0,
            )
        ),
        "workload_balance": float(
            np.clip(
                evaluation.balance,
                0.0,
                1.0,
            )
        ),
    }


def aggregate_base_scores(
    evaluations: Sequence[Evaluation],
    *,
    time_optima: Sequence[float],
) -> dict[str, float]:
    if len(evaluations) != len(time_optima):
        raise ValueError(
            "evaluations and time_optima length mismatch"
        )
    if not evaluations:
        raise ValueError(
            "At least one evaluation is required"
        )

    rows = [
        base_scores_for_evaluation(
            evaluation,
            time_optimum=float(star),
        )
        for evaluation, star in zip(
            evaluations,
            time_optima,
            strict=True,
        )
    ]
    return {
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


def evaluate_gene_base_axes(
    gene: RouteTailDirectGene,
    worlds: Sequence[World],
    config: EnvConfig,
    *,
    time_optima: Sequence[float],
) -> dict[str, float]:
    evaluations = [
        rollout_route_tail_gene(
            gene,
            world,
            config,
        ).evaluation
        for world in worlds
    ]
    return aggregate_base_scores(
        evaluations,
        time_optima=time_optima,
    )
