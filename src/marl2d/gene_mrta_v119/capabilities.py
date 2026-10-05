from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np

from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from .benchmark import (
    MTRPDInstance,
)
from .rollout import (
    rollout_gene,
)


EPS = 1e-12


@dataclass(frozen=True)
class InstanceAssessment:
    instance_id: str
    vertex_count: int
    robot_count: int
    success: bool
    completion: float
    total_latency: float
    optimum_total_latency: float
    optimum_retention: float
    optimality_gap: float
    reference_violation: bool

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "instance_id": (
                self.instance_id
            ),
            "vertex_count": (
                self.vertex_count
            ),
            "robot_count": (
                self.robot_count
            ),
            "success": self.success,
            "completion": (
                self.completion
            ),
            "total_latency": (
                self.total_latency
            ),
            "optimum_total_latency": (
                self.optimum_total_latency
            ),
            "optimum_retention": (
                self.optimum_retention
            ),
            "optimality_gap": (
                self.optimality_gap
            ),
            "reference_violation": (
                self.reference_violation
            ),
        }


@dataclass(frozen=True)
class GeneAssessment:
    success: bool
    worst_completion: float
    mean_completion: float
    success_instances: int
    instance_count: int
    scores: dict[
        str,
        float,
    ]
    overall_optimum_retention: float
    worst_optimum_retention: float
    exact_optimum_matches: int
    reference_violations: int
    instances: tuple[
        InstanceAssessment,
        ...,
    ]

    def to_summary_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "success": self.success,
            "worst_completion": (
                self.worst_completion
            ),
            "mean_completion": (
                self.mean_completion
            ),
            "success_instances": (
                self.success_instances
            ),
            "instance_count": (
                self.instance_count
            ),
            "scores": dict(
                self.scores
            ),
            "overall_optimum_retention": (
                self.overall_optimum_retention
            ),
            "worst_optimum_retention": (
                self.worst_optimum_retention
            ),
            "exact_optimum_matches": (
                self.exact_optimum_matches
            ),
            "reference_violations": (
                self.reference_violations
            ),
        }


def capability_axes(
    instances: Sequence[
        MTRPDInstance
    ],
) -> tuple[
    str,
    ...,
]:
    sizes = sorted(
        {
            int(
                item.vertex_count
            )
            for item in instances
        }
    )
    return tuple(
        f"opt_retention_v{size}"
        for size in sizes
    )


def evaluate_gene(
    gene: RouteTailDirectGene,
    instances: Sequence[
        MTRPDInstance
    ],
    *,
    strict_reference: bool = True,
) -> GeneAssessment:
    if not instances:
        raise ValueError(
            "At least one MTRPD instance is required"
        )

    axes = capability_axes(
        instances
    )
    sizes = sorted(
        {
            item.vertex_count
            for item in instances
        }
    )

    rows: list[
        InstanceAssessment
    ] = []
    completions: list[
        float
    ] = []

    for instance in instances:
        if (
            not instance.proven_optimal
            or instance.optimum_total_latency
            is None
        ):
            raise ValueError(
                "Training/evaluation requires proven optimum: "
                + instance.instance_id
            )

        rollout = rollout_gene(
            gene,
            instance,
        )
        completions.append(
            rollout.completion
        )

        optimum = float(
            instance.optimum_total_latency
        )
        if rollout.success:
            if (
                rollout.total_latency
                <= EPS
            ):
                raise RuntimeError(
                    "Successful MTRPD rollout has non-positive latency"
                )
            retention = float(
                optimum
                / rollout.total_latency
            )
            gap = float(
                (
                    rollout.total_latency
                    - optimum
                )
                / optimum
            )
        else:
            retention = 0.0
            gap = float(
                "inf"
            )

        violation = bool(
            rollout.success
            and retention
            > 1.0 + 1e-7
        )
        if (
            strict_reference
            and violation
        ):
            raise RuntimeError(
                "SEGB result is better than published proven optimum; "
                "distance convention or imported reference is inconsistent: "
                f"{instance.instance_id} retention={retention:.12f}"
            )

        rows.append(
            InstanceAssessment(
                instance_id=(
                    instance.instance_id
                ),
                vertex_count=(
                    instance.vertex_count
                ),
                robot_count=(
                    instance.robot_count
                ),
                success=(
                    rollout.success
                ),
                completion=float(
                    rollout.completion
                ),
                total_latency=float(
                    rollout.total_latency
                ),
                optimum_total_latency=(
                    optimum
                ),
                optimum_retention=float(
                    retention
                ),
                optimality_gap=float(
                    gap
                ),
                reference_violation=(
                    violation
                ),
            )
        )

    completion_array = np.asarray(
        completions,
        dtype=np.float64,
    )
    success_instances = sum(
        int(
            row.success
        )
        for row in rows
    )
    all_success = bool(
        success_instances
        == len(
            rows
        )
    )

    scores: dict[
        str,
        float,
    ] = {}
    for size, axis in zip(
        sizes,
        axes,
        strict=True,
    ):
        values = [
            row.optimum_retention
            for row in rows
            if row.vertex_count
            == size
        ]
        scores[
            axis
        ] = float(
            np.mean(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    retention_values = np.asarray(
        [
            row.optimum_retention
            for row in rows
        ],
        dtype=np.float64,
    )
    exact_matches = sum(
        int(
            row.success
            and abs(
                row.total_latency
                - row.optimum_total_latency
            )
            <= 1e-9
        )
        for row in rows
    )
    violations = sum(
        int(
            row.reference_violation
        )
        for row in rows
    )

    return GeneAssessment(
        success=all_success,
        worst_completion=float(
            np.min(
                completion_array
            )
        ),
        mean_completion=float(
            np.mean(
                completion_array
            )
        ),
        success_instances=int(
            success_instances
        ),
        instance_count=len(
            rows
        ),
        scores=scores,
        overall_optimum_retention=float(
            np.mean(
                retention_values
            )
        ),
        worst_optimum_retention=float(
            np.min(
                retention_values
            )
        ),
        exact_optimum_matches=int(
            exact_matches
        ),
        reference_violations=int(
            violations
        ),
        instances=tuple(
            rows
        ),
    )


def paired_delta(
    parent: GeneAssessment,
    child: GeneAssessment,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    parent_by_id = {
        row.instance_id: row
        for row in parent.instances
    }
    child_by_id = {
        row.instance_id: row
        for row in child.instances
    }
    if (
        set(
            parent_by_id
        )
        != set(
            child_by_id
        )
    ):
        raise ValueError(
            "Parent and child must be evaluated on identical instances"
        )

    wins = 0
    ties = 0
    losses = 0
    success_gains = 0
    success_losses = 0
    deltas: list[
        float
    ] = []

    for instance_id in sorted(
        parent_by_id
    ):
        p = parent_by_id[
            instance_id
        ]
        c = child_by_id[
            instance_id
        ]
        if (
            not p.success
            and c.success
        ):
            success_gains += 1
        if (
            p.success
            and not c.success
        ):
            success_losses += 1

        delta = (
            c.optimum_retention
            - p.optimum_retention
        )
        deltas.append(
            delta
        )
        if delta > tie_tolerance:
            wins += 1
        elif delta < -tie_tolerance:
            losses += 1
        else:
            ties += 1

    axis_delta = {
        axis: float(
            child.scores[
                axis
            ]
            - parent.scores[
                axis
            ]
        )
        for axis in (
            parent.scores
        )
    }

    return {
        "overall_delta": float(
            child.overall_optimum_retention
            - parent.overall_optimum_retention
        ),
        "worst_delta": float(
            child.worst_optimum_retention
            - parent.worst_optimum_retention
        ),
        "axis_delta": (
            axis_delta
        ),
        "wins": int(
            wins
        ),
        "ties": int(
            ties
        ),
        "losses": int(
            losses
        ),
        "success_gains": int(
            success_gains
        ),
        "success_losses": int(
            success_losses
        ),
        "mean_instance_delta": float(
            np.mean(
                np.asarray(
                    deltas,
                    dtype=np.float64,
                )
            )
        ),
    }
