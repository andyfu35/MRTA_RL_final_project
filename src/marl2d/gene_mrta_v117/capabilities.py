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
    "global_completion_optimality",
    "global_time_optimality",
    "path_efficiency",
    "global_priority_optimality",
    "deadline_satisfaction",
    "workload_balance",
)


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


def base_scores_for_evaluation(
    evaluation: Evaluation,
    *,
    completion_optimum: float,
    time_optimum: float,
    priority_optimum: float,
) -> dict[str, float]:
    completion_retention = _retention(
        evaluation.completion,
        completion_optimum,
    )
    time_retention = _retention(
        evaluation.time_optimality,
        time_optimum,
    )
    priority_retention = _retention(
        evaluation.priority_satisfaction,
        priority_optimum,
    )

    raw_fairness = (
        evaluation.balance
        / evaluation.completion
        if evaluation.completion > EPS
        else 0.0
    )
    normalized_balance = float(
        np.clip(
            completion_retention
            * raw_fairness,
            0.0,
            1.0,
        )
    )

    return {
        "global_completion_optimality": (
            completion_retention
        ),
        "global_time_optimality": (
            time_retention
        ),
        "path_efficiency": float(
            np.clip(
                evaluation.efficiency,
                0.0,
                1.0,
            )
        ),
        "global_priority_optimality": (
            priority_retention
        ),
        "deadline_satisfaction": float(
            np.clip(
                evaluation.deadline_satisfaction,
                0.0,
                1.0,
            )
        ),
        "workload_balance": (
            normalized_balance
        ),
    }


def aggregate_base_scores(
    evaluations: Sequence[Evaluation],
    *,
    completion_optima: Sequence[float],
    time_optima: Sequence[float],
    priority_optima: Sequence[float],
) -> dict[str, float]:
    if (
        len(evaluations) != len(completion_optima)
        or len(evaluations) != len(time_optima)
        or len(evaluations) != len(priority_optima)
    ):
        raise ValueError(
            "evaluations and oracle-reference lengths mismatch"
        )
    if not evaluations:
        raise ValueError(
            "At least one evaluation is required"
        )

    rows = [
        base_scores_for_evaluation(
            evaluation,
            completion_optimum=float(
                completion_star
            ),
            time_optimum=float(time_star),
            priority_optimum=float(priority_star),
        )
        for (
            evaluation,
            completion_star,
            time_star,
            priority_star,
        ) in zip(
            evaluations,
            completion_optima,
            time_optima,
            priority_optima,
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
    completion_optima: Sequence[float],
    time_optima: Sequence[float],
    priority_optima: Sequence[float],
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
        completion_optima=completion_optima,
        time_optima=time_optima,
        priority_optima=priority_optima,
    )
