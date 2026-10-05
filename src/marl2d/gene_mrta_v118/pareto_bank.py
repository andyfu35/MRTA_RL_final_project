from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from marl2d.gene_mrta_v117.stage_a_train import Record
from marl2d.gene_mrta_v118.capabilities import BASE_AXES


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
            float(
                scores[axis]
            )
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
    return bool(
        np.all(
            av >= bv - tolerance
        )
        and np.any(
            av > bv + tolerance
        )
    )


def pareto_front_ids(
    records: Mapping[str, Record],
    *,
    tolerance: float = 1e-12,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    ids = list(
        records
    )
    dominated_ids: set[
        str
    ] = set()

    for rid in ids:
        for other in ids:
            if other == rid:
                continue
            if dominates(
                records[
                    other
                ].scores,
                records[
                    rid
                ].scores,
                tolerance=(
                    tolerance
                ),
            ):
                dominated_ids.add(
                    rid
                )
                break

    front = tuple(
        rid
        for rid in ids
        if rid
        not in dominated_ids
    )
    return (
        front,
        tuple(
            sorted(
                dominated_ids
            )
        ),
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

    kept: list[str] = []
    removed: list[str] = []

    for rid in ids:
        vector = score_vector(
            records[
                rid
            ].scores
        )
        duplicate = False
        for owner in kept:
            owner_vector = (
                score_vector(
                    records[
                        owner
                    ].scores
                )
            )
            if bool(
                np.all(
                    np.abs(
                        vector
                        - owner_vector
                    )
                    <= epsilon
                    + 1e-12
                )
            ):
                duplicate = True
                break

        if duplicate:
            removed.append(
                rid
            )
        else:
            kept.append(
                rid
            )

    return (
        tuple(
            kept
        ),
        tuple(
            sorted(
                removed
            )
        ),
    )


def crowding_distance(
    records: Mapping[str, Record],
    ids: Sequence[str],
) -> dict[str, float]:
    distance = {
        rid: 0.0
        for rid in ids
    }
    if len(ids) <= 2:
        return {
            rid: float(
                "inf"
            )
            for rid in ids
        }

    for axis in BASE_AXES:
        ordered = sorted(
            ids,
            key=lambda rid: (
                records[
                    rid
                ].scores[
                    axis
                ]
            ),
        )
        low = float(
            records[
                ordered[0]
            ].scores[
                axis
            ]
        )
        high = float(
            records[
                ordered[-1]
            ].scores[
                axis
            ]
        )
        distance[
            ordered[0]
        ] = float(
            "inf"
        )
        distance[
            ordered[-1]
        ] = float(
            "inf"
        )
        span = high - low
        if span <= 1e-12:
            continue

        for index in range(
            1,
            len(ordered) - 1,
        ):
            rid = ordered[
                index
            ]
            if np.isinf(
                distance[
                    rid
                ]
            ):
                continue
            previous = float(
                records[
                    ordered[
                        index - 1
                    ]
                ].scores[
                    axis
                ]
            )
            following = float(
                records[
                    ordered[
                        index + 1
                    ]
                ].scores[
                    axis
                ]
            )
            distance[rid] += (
                following
                - previous
            ) / span

    return distance


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
            distance[
                rid
            ],
            rid,
        ),
        reverse=True,
    )
    return (
        tuple(
            ordered[
                :max_size
            ]
        ),
        tuple(
            sorted(
                ordered[
                    max_size:
                ]
            )
        ),
    )


def rebuild_pareto_bank(
    records: Mapping[str, Record],
    *,
    max_size: int = 256,
    epsilon: float = 0.005,
) -> ParetoRebuildResult:
    front, dominated = (
        pareto_front_ids(
            records
        )
    )
    deduped, epsilon_removed = (
        epsilon_deduplicate_ids(
            records,
            front,
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
    return ParetoRebuildResult(
        records={
            rid: records[rid]
            for rid in kept
        },
        nondominated_ids=(
            front
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
) -> dict[
    str,
    str | None,
]:
    if not records:
        return {
            axis: None
            for axis in BASE_AXES
        }
    return {
        axis: max(
            records,
            key=lambda rid: (
                records[
                    rid
                ].scores[
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
        key=lambda rid: min(
            records[
                rid
            ].scores[
                axis
            ]
            for axis in BASE_AXES
        ),
    )
