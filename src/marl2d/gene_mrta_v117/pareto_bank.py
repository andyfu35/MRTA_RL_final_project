from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.stage_a_train import Record


@dataclass(frozen=True)
class ParetoRebuildResult:
    records: dict[str, Record]
    nondominated_ids: tuple[str, ...]
    dominated_ids: tuple[str, ...]
    epsilon_duplicate_ids: tuple[str, ...]
    crowding_removed_ids: tuple[str, ...]


def score_vector(
    scores: Mapping[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            float(scores[axis])
            for axis in BASE_AXES
        ],
        dtype=np.float64,
    )


def dominates(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> bool:
    av = score_vector(a)
    bv = score_vector(b)
    no_worse = bool(
        np.all(
            av >= bv - tolerance
        )
    )
    strictly_better = bool(
        np.any(
            av > bv + tolerance
        )
    )
    return (
        no_worse
        and strictly_better
    )


def pareto_front_ids(
    records: Mapping[str, Record],
    *,
    tolerance: float = 1e-12,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    ids = list(records)
    dominated: set[str] = set()

    for i, rid in enumerate(ids):
        if rid in dominated:
            continue
        for j, other_id in enumerate(ids):
            if i == j:
                continue
            if dominates(
                records[other_id].scores,
                records[rid].scores,
                tolerance=tolerance,
            ):
                dominated.add(rid)
                break

    front = tuple(
        rid
        for rid in ids
        if rid not in dominated
    )
    return (
        front,
        tuple(
            sorted(dominated)
        ),
    )


def _epsilon_key(
    scores: Mapping[str, float],
    *,
    epsilon: float,
) -> tuple[int, ...]:
    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be positive"
        )
    vector = score_vector(
        scores
    )
    return tuple(
        int(
            np.floor(
                value / epsilon
                + 1e-12
            )
        )
        for value in vector
    )


def epsilon_deduplicate_ids(
    records: Mapping[str, Record],
    ids: Sequence[str],
    *,
    epsilon: float,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    if epsilon <= 0.0:
        return (
            tuple(ids),
            (),
        )

    cell_owner: dict[
        tuple[int, ...],
        str,
    ] = {}
    removed: list[str] = []

    for rid in ids:
        key = _epsilon_key(
            records[rid].scores,
            epsilon=epsilon,
        )
        if key not in cell_owner:
            cell_owner[key] = rid
            continue

        # Same epsilon-cell means the capability vectors are treated as
        # practically equivalent for storage. Keep the existing owner to
        # avoid introducing any extra scalar preference between axes.
        removed.append(rid)

    return (
        tuple(
            cell_owner.values()
        ),
        tuple(
            sorted(removed)
        ),
    )


def crowding_distance(
    records: Mapping[str, Record],
    ids: Sequence[str],
) -> dict[str, float]:
    result = {
        rid: 0.0
        for rid in ids
    }
    if len(ids) <= 2:
        return {
            rid: float("inf")
            for rid in ids
        }

    for axis in BASE_AXES:
        ordered = sorted(
            ids,
            key=lambda rid: (
                records[rid].scores[
                    axis
                ]
            ),
        )
        low = float(
            records[
                ordered[0]
            ].scores[axis]
        )
        high = float(
            records[
                ordered[-1]
            ].scores[axis]
        )

        result[
            ordered[0]
        ] = float("inf")
        result[
            ordered[-1]
        ] = float("inf")

        span = high - low
        if span <= 1e-12:
            continue

        for index in range(
            1,
            len(ordered) - 1,
        ):
            rid = ordered[index]
            if np.isinf(
                result[rid]
            ):
                continue
            prev_value = float(
                records[
                    ordered[
                        index - 1
                    ]
                ].scores[axis]
            )
            next_value = float(
                records[
                    ordered[
                        index + 1
                    ]
                ].scores[axis]
            )
            result[rid] += (
                next_value
                - prev_value
            ) / span

    return result


def crowding_trim_ids(
    records: Mapping[str, Record],
    ids: Sequence[str],
    *,
    max_size: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    if max_size <= 0:
        raise ValueError(
            "max_size must be positive"
        )
    if len(ids) <= max_size:
        return (
            tuple(ids),
            (),
        )

    distance = crowding_distance(
        records,
        ids,
    )
    ordered = sorted(
        ids,
        key=lambda rid: (
            distance[rid],
            rid,
        ),
        reverse=True,
    )
    keep = tuple(
        ordered[:max_size]
    )
    removed = tuple(
        sorted(
            ordered[max_size:]
        )
    )
    return (
        keep,
        removed,
    )


def rebuild_pareto_bank(
    records: Mapping[str, Record],
    *,
    max_size: int = 256,
    epsilon: float = 0.005,
    tolerance: float = 1e-12,
) -> ParetoRebuildResult:
    front_ids, dominated = (
        pareto_front_ids(
            records,
            tolerance=tolerance,
        )
    )
    deduped, epsilon_removed = (
        epsilon_deduplicate_ids(
            records,
            front_ids,
            epsilon=epsilon,
        )
    )
    kept, crowding_removed = (
        crowding_trim_ids(
            records,
            deduped,
            max_size=max_size,
        )
    )

    kept_records = {
        rid: records[rid]
        for rid in kept
    }
    return ParetoRebuildResult(
        records=kept_records,
        nondominated_ids=(
            front_ids
        ),
        dominated_ids=(
            dominated
        ),
        epsilon_duplicate_ids=(
            epsilon_removed
        ),
        crowding_removed_ids=(
            crowding_removed
        ),
    )


def analysis_best_by_axis(
    records: Mapping[str, Record],
) -> dict[str, str | None]:
    if not records:
        return {
            axis: None
            for axis in BASE_AXES
        }
    return {
        axis: max(
            records,
            key=lambda rid: (
                records[rid].scores[
                    axis
                ]
            ),
        )
        for axis in BASE_AXES
    }


def maximin_gene_id(
    records: Mapping[str, Record],
) -> str | None:
    if not records:
        return None
    return max(
        records,
        key=lambda rid: (
            min(
                float(
                    records[rid].scores[
                        axis
                    ]
                )
                for axis in BASE_AXES
            )
        ),
    )


def capability_distance(
    a: Mapping[str, float],
    b: Mapping[str, float],
) -> float:
    av = score_vector(a)
    bv = score_vector(b)
    return float(
        np.linalg.norm(
            av - bv
        )
    )
