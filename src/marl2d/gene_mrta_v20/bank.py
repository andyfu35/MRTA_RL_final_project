from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping

import numpy as np

from .gene import SetAssignmentGene
from .rollout import AXES, AXIS_DIRECTIONS, GeneEvaluation


DEFAULT_RESOLUTIONS = {
    "total_time": 0.5,
    "priority": 0.10,
    "on_time_completed_tasks": 0.5,
}


@dataclass(frozen=True)
class BankRecord:
    record_id: str
    gene: SetAssignmentGene
    scores: dict[str, float]
    round_index: int
    parent_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "gene": self.gene.to_dict(),
            "scores": dict(self.scores),
            "round_index": self.round_index,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "BankRecord":
        return cls(
            record_id=str(data["record_id"]),
            gene=SetAssignmentGene.from_dict(data["gene"]),
            scores={
                str(k): float(v)
                for k, v in dict(data["scores"]).items()
            },
            round_index=int(data["round_index"]),
            parent_id=(
                None
                if data.get("parent_id") is None
                else str(data["parent_id"])
            ),
        )


def make_record(
    gene: SetAssignmentGene,
    evaluation: GeneEvaluation,
    round_index: int,
    parent_id: str | None,
) -> BankRecord:
    digest = hashlib.sha1(
        np.asarray(
            gene.vector_data,
            dtype=np.float64,
        ).tobytes()
    ).hexdigest()[:20]
    return BankRecord(
        record_id=digest,
        gene=gene,
        scores=evaluation.axes(),
        round_index=round_index,
        parent_id=parent_id,
    )


def _no_worse(
    a: float,
    b: float,
    axis: str,
) -> bool:
    return (
        a <= b
        if AXIS_DIRECTIONS[axis] == "min"
        else a >= b
    )


def _strictly_better(
    a: float,
    b: float,
    axis: str,
) -> bool:
    return (
        a < b
        if AXIS_DIRECTIONS[axis] == "min"
        else a > b
    )


def dominates(
    a: BankRecord,
    b: BankRecord,
) -> bool:
    return all(
        _no_worse(
            a.scores[x],
            b.scores[x],
            x,
        )
        for x in AXES
    ) and any(
        _strictly_better(
            a.scores[x],
            b.scores[x],
            x,
        )
        for x in AXES
    )


def _validated_resolutions(
    resolutions: Mapping[str, float] | None,
) -> dict[str, float]:
    values = dict(DEFAULT_RESOLUTIONS)
    if resolutions is not None:
        values.update(
            {
                str(k): float(v)
                for k, v in resolutions.items()
            }
        )
    for axis in AXES:
        if axis not in values or values[axis] <= 0.0:
            raise ValueError(
                f"Resolution for {axis} must be positive"
            )
    return values


def _cell(
    record: BankRecord,
    resolutions: Mapping[str, float],
) -> tuple[int, ...]:
    return tuple(
        int(
            math.floor(
                record.scores[axis]
                / resolutions[axis]
                + 1e-12
            )
        )
        for axis in AXES
    )


def _cell_dominates(
    a: tuple[int, ...],
    b: tuple[int, ...],
) -> bool:
    no_worse = True
    strict = False
    for index, axis in enumerate(AXES):
        if AXIS_DIRECTIONS[axis] == "min":
            no_worse = (
                no_worse
                and a[index] <= b[index]
            )
            strict = (
                strict
                or a[index] < b[index]
            )
        else:
            no_worse = (
                no_worse
                and a[index] >= b[index]
            )
            strict = (
                strict
                or a[index] > b[index]
            )
    return no_worse and strict


def _within_cell_quality(
    record: BankRecord,
    resolutions: Mapping[str, float],
) -> float:
    quality = 0.0
    for axis in AXES:
        resolution = resolutions[axis]
        value = record.scores[axis]
        base = (
            math.floor(value / resolution)
            * resolution
        )
        fraction = (
            value - base
        ) / resolution
        quality += (
            1.0 - fraction
            if AXIS_DIRECTIONS[axis] == "min"
            else fraction
        )
    return quality / len(AXES)


def rebuild_bank(
    records: Iterable[BankRecord],
    resolutions: Mapping[str, float] | None = None,
) -> dict[str, BankRecord]:
    """
    Unbounded epsilon-grid Pareto archive.

    There is no hard Bank size cap. Scores are quantized only for archive
    resolution, so near-identical trade-offs occupy one cell. Exact per-axis
    champions are always retained even when they share a cell.
    """
    resolutions = _validated_resolutions(
        resolutions
    )

    deduped: dict[
        tuple[float, ...],
        BankRecord,
    ] = {}
    for record in records:
        key = record.gene.key()
        existing = deduped.get(key)
        if (
            existing is None
            or record.round_index
            >= existing.round_index
        ):
            deduped[key] = record

    rows = list(deduped.values())
    by_cell: dict[
        tuple[int, ...],
        BankRecord,
    ] = {}
    for record in rows:
        cell = _cell(
            record,
            resolutions,
        )
        existing = by_cell.get(cell)
        if existing is None:
            by_cell[cell] = record
            continue
        q_new = _within_cell_quality(
            record,
            resolutions,
        )
        q_old = _within_cell_quality(
            existing,
            resolutions,
        )
        if (
            q_new > q_old + 1e-12
            or (
                abs(q_new - q_old)
                <= 1e-12
                and record.round_index
                > existing.round_index
            )
        ):
            by_cell[cell] = record

    cell_rows = list(by_cell.values())
    cells = [
        _cell(
            record,
            resolutions,
        )
        for record in cell_rows
    ]
    keep: dict[
        str,
        BankRecord,
    ] = {}
    for i, candidate in enumerate(
        cell_rows
    ):
        if any(
            j != i
            and _cell_dominates(
                cells[j],
                cells[i],
            )
            for j in range(
                len(cell_rows)
            )
        ):
            continue
        keep[
            candidate.record_id
        ] = candidate

    for axis in AXES:
        champion = (
            min(
                rows,
                key=lambda row: (
                    row.scores[axis]
                ),
            )
            if AXIS_DIRECTIONS[
                axis
            ] == "min"
            else max(
                rows,
                key=lambda row: (
                    row.scores[axis]
                ),
            )
        )
        keep[
            champion.record_id
        ] = champion

    return keep


def rank_selection_scores(
    records: Mapping[str, BankRecord],
) -> tuple[list[str], np.ndarray]:
    ids = list(records)
    if not ids:
        raise ValueError(
            "Cannot rank an empty bank"
        )
    n = len(ids)
    total = np.zeros(
        n,
        dtype=np.float64,
    )

    for axis in AXES:
        values = np.asarray(
            [
                records[rid].scores[
                    axis
                ]
                for rid in ids
            ],
            dtype=np.float64,
        )
        order = np.argsort(values)
        if (
            AXIS_DIRECTIONS[
                axis
            ] == "max"
        ):
            order = order[::-1]
        axis_rank = np.empty(
            n,
            dtype=np.float64,
        )
        axis_rank[order] = (
            n
            - np.arange(
                n,
                dtype=np.float64,
            )
        ) / n
        total += axis_rank

    total /= len(AXES)
    weights = np.square(
        np.maximum(
            total,
            1e-12,
        )
    )
    weights /= np.sum(weights)
    return ids, weights


def representative_id(
    records: Mapping[
        str,
        BankRecord,
    ],
) -> str:
    ids, probs = rank_selection_scores(
        records
    )
    return ids[
        int(np.argmax(probs))
    ]
