from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np

from .benchmark import (
    MinMaxMTSPInstance,
)
from .gene import (
    ScalableRouteTailGene,
)
from .rollout import (
    rollout_gene,
)


@dataclass(frozen=True)
class InstanceAssessment:
    instance_id: str
    benchmark_set: str
    size_band: str
    vertex_count: int
    robot_count: int
    reference_kind: str
    reference_value: float
    success: bool
    completion: float
    objective: float | None
    reference_retention: float
    gap_to_reference: float | None
    beat_bks: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "benchmark_set": self.benchmark_set,
            "size_band": self.size_band,
            "vertex_count": self.vertex_count,
            "robot_count": self.robot_count,
            "reference_kind": self.reference_kind,
            "reference_value": self.reference_value,
            "success": self.success,
            "completion": self.completion,
            "objective": self.objective,
            "reference_retention": self.reference_retention,
            "gap_to_reference": self.gap_to_reference,
            "beat_bks": self.beat_bks,
        }


@dataclass(frozen=True)
class GeneAssessment:
    success: bool
    worst_completion: float
    mean_completion: float
    success_instances: int
    instance_count: int
    scores: dict[str, float]
    overall_reference_retention: float
    worst_reference_retention: float
    exact_matches: int
    bks_improvements: int
    instances: tuple[InstanceAssessment, ...]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "worst_completion": self.worst_completion,
            "mean_completion": self.mean_completion,
            "success_instances": self.success_instances,
            "instance_count": self.instance_count,
            "scores": dict(self.scores),
            "overall_reference_retention": self.overall_reference_retention,
            "worst_reference_retention": self.worst_reference_retention,
            "exact_matches": self.exact_matches,
            "bks_improvements": self.bks_improvements,
        }


def capability_axes(
    instances: Sequence[MinMaxMTSPInstance],
) -> tuple[str, ...]:
    bands = [
        band
        for band in ("small", "medium", "large")
        if any(item.size_band == band for item in instances)
    ]
    return tuple(f"retention_{band}" for band in bands)


def evaluate_gene(
    gene: ScalableRouteTailGene,
    instances: Sequence[MinMaxMTSPInstance],
    *,
    candidate_k: int = 32,
    exact_tolerance: float = 1e-6,
) -> GeneAssessment:
    if not instances:
        raise ValueError("At least one benchmark instance is required")

    rows: list[InstanceAssessment] = []
    for instance in instances:
        rollout = rollout_gene(
            gene,
            instance,
            candidate_k=candidate_k,
        )
        if rollout.success:
            objective = float(rollout.objective)
            retention = float(instance.reference_value / objective)
            gap = float(
                (objective - instance.reference_value)
                / instance.reference_value
            )
        else:
            objective = None
            retention = 0.0
            gap = None

        if (
            instance.is_exact_optimum
            and rollout.success
            and retention > 1.0 + exact_tolerance
        ):
            raise RuntimeError(
                f"{instance.instance_id}: Gene objective {objective} is "
                f"better than published exact optimum {instance.reference_value}; "
                "distance convention or benchmark parsing is inconsistent"
            )

        beat_bks = bool(
            instance.reference_kind == "best_known"
            and rollout.success
            and retention > 1.0 + exact_tolerance
        )

        rows.append(
            InstanceAssessment(
                instance_id=instance.instance_id,
                benchmark_set=instance.benchmark_set,
                size_band=instance.size_band,
                vertex_count=instance.vertex_count,
                robot_count=instance.robot_count,
                reference_kind=instance.reference_kind,
                reference_value=float(instance.reference_value),
                success=rollout.success,
                completion=float(rollout.completion),
                objective=objective,
                reference_retention=retention,
                gap_to_reference=gap,
                beat_bks=beat_bks,
            )
        )

    completions = np.asarray(
        [row.completion for row in rows],
        dtype=np.float64,
    )
    retentions = np.asarray(
        [row.reference_retention for row in rows],
        dtype=np.float64,
    )

    scores: dict[str, float] = {}
    for band in ("small", "medium", "large"):
        values = [
            row.reference_retention
            for row in rows
            if row.size_band == band
        ]
        if values:
            scores[f"retention_{band}"] = float(np.mean(values))

    success_instances = sum(int(row.success) for row in rows)
    exact_matches = sum(
        int(
            row.success
            and row.reference_kind == "exact_optimum"
            and row.objective is not None
            and abs(row.objective - row.reference_value)
            <= exact_tolerance * max(1.0, row.reference_value)
        )
        for row in rows
    )
    bks_improvements = sum(int(row.beat_bks) for row in rows)

    return GeneAssessment(
        success=bool(success_instances == len(rows)),
        worst_completion=float(np.min(completions)),
        mean_completion=float(np.mean(completions)),
        success_instances=int(success_instances),
        instance_count=len(rows),
        scores=scores,
        overall_reference_retention=float(np.mean(retentions)),
        worst_reference_retention=float(np.min(retentions)),
        exact_matches=int(exact_matches),
        bks_improvements=int(bks_improvements),
        instances=tuple(rows),
    )


def paired_delta(
    parent: GeneAssessment,
    child: GeneAssessment,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    p = {row.instance_id: row for row in parent.instances}
    c = {row.instance_id: row for row in child.instances}
    if set(p) != set(c):
        raise ValueError(
            "Parent and child must be evaluated on identical benchmark instances"
        )

    wins = ties = losses = 0
    deltas: list[float] = []
    for instance_id in sorted(p):
        delta = (
            c[instance_id].reference_retention
            - p[instance_id].reference_retention
        )
        deltas.append(delta)
        if delta > tolerance:
            wins += 1
        elif delta < -tolerance:
            losses += 1
        else:
            ties += 1

    axes = tuple(parent.scores)
    if set(axes) != set(child.scores):
        raise ValueError("Parent/child capability axes differ")

    return {
        "overall_delta": float(
            child.overall_reference_retention
            - parent.overall_reference_retention
        ),
        "worst_delta": float(
            child.worst_reference_retention
            - parent.worst_reference_retention
        ),
        "axis_delta": {
            axis: float(child.scores[axis] - parent.scores[axis])
            for axis in axes
        },
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "mean_instance_delta": float(np.mean(deltas)),
        "bks_improvement_delta": int(
            child.bks_improvements - parent.bks_improvements
        ),
        "exact_match_delta": int(
            child.exact_matches - parent.exact_matches
        ),
    }
