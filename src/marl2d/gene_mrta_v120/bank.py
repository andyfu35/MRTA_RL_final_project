from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BankRecord:
    record_id: str
    vector_data: tuple[
        float,
        ...,
    ]
    hidden_dim: int
    scores: dict[
        str,
        float,
    ]
    overall_retention: float
    worst_retention: float
    generation: int
    origin: str
    parents: tuple[
        str,
        ...,
    ] = ()

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "vector_data": list(
                self.vector_data
            ),
            "hidden_dim": (
                self.hidden_dim
            ),
            "scores": dict(
                self.scores
            ),
            "overall_retention": (
                self.overall_retention
            ),
            "worst_retention": (
                self.worst_retention
            ),
            "generation": (
                self.generation
            ),
            "origin": (
                self.origin
            ),
            "parents": list(
                self.parents
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[
            str,
            object,
        ],
    ) -> "BankRecord":
        return cls(
            record_id=str(
                data[
                    "record_id"
                ]
            ),
            vector_data=tuple(
                float(
                    value
                )
                for value in data[
                    "vector_data"
                ]
            ),
            hidden_dim=int(
                data[
                    "hidden_dim"
                ]
            ),
            scores={
                str(
                    key
                ): float(
                    value
                )
                for key, value
                in dict(
                    data[
                        "scores"
                    ]
                ).items()
            },
            overall_retention=float(
                data[
                    "overall_retention"
                ]
            ),
            worst_retention=float(
                data[
                    "worst_retention"
                ]
            ),
            generation=int(
                data[
                    "generation"
                ]
            ),
            origin=str(
                data[
                    "origin"
                ]
            ),
            parents=tuple(
                str(
                    value
                )
                for value in data.get(
                    "parents",
                    [],
                )
            ),
        )


@dataclass(frozen=True)
class ParetoResult:
    records: dict[
        str,
        BankRecord,
    ]
    dominated_ids: tuple[
        str,
        ...,
    ]
    epsilon_removed_ids: tuple[
        str,
        ...,
    ]
    crowding_removed_ids: tuple[
        str,
        ...,
    ]


def score_vector(
    record: BankRecord,
    axes: Sequence[
        str
    ],
) -> np.ndarray:
    return np.asarray(
        [
            float(
                record.scores[
                    axis
                ]
            )
            for axis in axes
        ],
        dtype=np.float64,
    )


def dominates(
    a: BankRecord,
    b: BankRecord,
    axes: Sequence[
        str
    ],
    *,
    tolerance: float = 1e-12,
) -> bool:
    av = score_vector(
        a,
        axes,
    )
    bv = score_vector(
        b,
        axes,
    )
    return bool(
        np.all(
            av
            >= bv
            - tolerance
        )
        and np.any(
            av
            > bv
            + tolerance
        )
    )


def _pareto_front(
    records: Mapping[
        str,
        BankRecord,
    ],
    axes: Sequence[
        str
    ],
) -> tuple[
    list[
        str
    ],
    list[
        str
    ],
]:
    ids = list(
        records
    )
    dominated: list[
        str
    ] = []

    for rid in ids:
        for other in ids:
            if other == rid:
                continue
            if dominates(
                records[
                    other
                ],
                records[
                    rid
                ],
                axes,
            ):
                dominated.append(
                    rid
                )
                break

    dominated_set = set(
        dominated
    )
    front = [
        rid
        for rid in ids
        if rid
        not in dominated_set
    ]
    return (
        front,
        sorted(
            dominated_set
        ),
    )


def _epsilon_dedup(
    records: Mapping[
        str,
        BankRecord,
    ],
    ids: Sequence[
        str
    ],
    axes: Sequence[
        str
    ],
    *,
    epsilon: float,
) -> tuple[
    list[
        str
    ],
    list[
        str
    ],
]:
    if epsilon <= 0.0:
        return (
            list(
                ids
            ),
            [],
        )

    ordered = sorted(
        ids,
        key=lambda rid: (
            records[
                rid
            ].overall_retention,
            records[
                rid
            ].worst_retention,
            rid,
        ),
        reverse=True,
    )
    kept: list[
        str
    ] = []
    removed: list[
        str
    ] = []

    for rid in ordered:
        vector = score_vector(
            records[
                rid
            ],
            axes,
        )
        duplicate = False
        for owner in kept:
            owner_vector = (
                score_vector(
                    records[
                        owner
                    ],
                    axes,
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
        kept,
        sorted(
            removed
        ),
    )


def _crowding_distance(
    records: Mapping[
        str,
        BankRecord,
    ],
    ids: Sequence[
        str
    ],
    axes: Sequence[
        str
    ],
) -> dict[
    str,
    float,
]:
    distance = {
        rid: 0.0
        for rid in ids
    }
    if len(
        ids
    ) <= 2:
        return {
            rid: float(
                "inf"
            )
            for rid in ids
        }

    for axis in axes:
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
                ordered[
                    0
                ]
            ].scores[
                axis
            ]
        )
        high = float(
            records[
                ordered[
                    -1
                ]
            ].scores[
                axis
            ]
        )
        distance[
            ordered[
                0
            ]
        ] = float(
            "inf"
        )
        distance[
            ordered[
                -1
            ]
        ] = float(
            "inf"
        )
        span = (
            high - low
        )
        if span <= 1e-12:
            continue

        for index in range(
            1,
            len(
                ordered
            )
            - 1,
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
            prev_value = float(
                records[
                    ordered[
                        index - 1
                    ]
                ].scores[
                    axis
                ]
            )
            next_value = float(
                records[
                    ordered[
                        index + 1
                    ]
                ].scores[
                    axis
                ]
            )
            distance[
                rid
            ] += (
                next_value
                - prev_value
            ) / span

    return distance


def rebuild_bank(
    records: Mapping[
        str,
        BankRecord,
    ],
    axes: Sequence[
        str
    ],
    *,
    epsilon: float,
    max_size: int,
) -> ParetoResult:
    if not axes:
        raise ValueError(
            "At least one capability axis is required"
        )
    if max_size <= 0:
        raise ValueError(
            "max_size must be positive"
        )

    front, dominated = (
        _pareto_front(
            records,
            axes,
        )
    )
    deduped, eps_removed = (
        _epsilon_dedup(
            records,
            front,
            axes,
            epsilon=epsilon,
        )
    )

    crowd_removed: list[
        str
    ] = []
    kept = list(
        deduped
    )
    if len(
        kept
    ) > max_size:
        distance = (
            _crowding_distance(
                records,
                kept,
                axes,
            )
        )
        ordered = sorted(
            kept,
            key=lambda rid: (
                distance[
                    rid
                ],
                records[
                    rid
                ].overall_retention,
                records[
                    rid
                ].worst_retention,
                rid,
            ),
            reverse=True,
        )
        kept = ordered[
            :max_size
        ]
        crowd_removed = sorted(
            ordered[
                max_size:
            ]
        )

    return ParetoResult(
        records={
            rid: records[
                rid
            ]
            for rid in kept
        },
        dominated_ids=tuple(
            dominated
        ),
        epsilon_removed_ids=tuple(
            eps_removed
        ),
        crowding_removed_ids=tuple(
            crowd_removed
        ),
    )


def best_by_axis(
    records: Mapping[
        str,
        BankRecord,
    ],
    axes: Sequence[
        str
    ],
) -> dict[
    str,
    str | None,
]:
    if not records:
        return {
            axis: None
            for axis in axes
        }
    return {
        axis: max(
            records,
            key=lambda rid: (
                records[
                    rid
                ].scores[
                    axis
                ],
                records[
                    rid
                ].overall_retention,
            ),
        )
        for axis in axes
    }


def global_best_id(
    records: Mapping[
        str,
        BankRecord,
    ],
) -> str | None:
    if not records:
        return None
    return max(
        records,
        key=lambda rid: (
            records[
                rid
            ].overall_retention,
            records[
                rid
            ].worst_retention,
            rid,
        ),
    )
