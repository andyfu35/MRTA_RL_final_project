from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np

from .gene import SetAssignmentGene
from .rollout import AXES, AXIS_DIRECTIONS, GeneEvaluation


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


def rebuild_bank(
    records: Iterable[BankRecord],
) -> dict[str, BankRecord]:
    deduped: dict[
        tuple[float, ...],
        BankRecord,
    ] = {}
    for record in records:
        key = record.gene.key()
        existing = deduped.get(key)
        if (
            existing is None
            or record.round_index >= existing.round_index
        ):
            deduped[key] = record

    rows = list(deduped.values())
    keep: dict[str, BankRecord] = {}
    for i, candidate in enumerate(rows):
        if any(
            j != i
            and dominates(other, candidate)
            for j, other in enumerate(rows)
        ):
            continue
        keep[candidate.record_id] = candidate
    return keep


def rank_selection_scores(
    records: Mapping[str, BankRecord],
) -> tuple[list[str], np.ndarray]:
    ids = list(records)
    if not ids:
        raise ValueError("Cannot rank an empty bank")
    n = len(ids)
    total = np.zeros(n, dtype=np.float64)

    for axis in AXES:
        values = np.asarray(
            [
                records[rid].scores[axis]
                for rid in ids
            ],
            dtype=np.float64,
        )
        order = np.argsort(values)
        if AXIS_DIRECTIONS[axis] == "max":
            order = order[::-1]
        axis_rank = np.empty(
            n,
            dtype=np.float64,
        )
        axis_rank[order] = (
            n - np.arange(
                n,
                dtype=np.float64,
            )
        ) / n
        total += axis_rank

    total /= len(AXES)
    weights = np.square(
        np.maximum(total, 1e-12)
    )
    weights /= np.sum(weights)
    return ids, weights


def representative_id(
    records: Mapping[str, BankRecord],
) -> str:
    ids, probs = rank_selection_scores(records)
    return ids[int(np.argmax(probs))]
