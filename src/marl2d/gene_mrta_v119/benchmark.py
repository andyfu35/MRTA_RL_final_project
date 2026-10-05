from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MANIFEST_VERSION = "mtrpd_public_canonical_v1"
PUBLIC_SOURCE_URL = (
    "https://www.computational-logistics.org/orlib/mtrpd/"
)
PUBLIC_FAMILIES = (
    "brd14051",
    "d15112",
    "d18512",
    "fnl4461",
    "nrw1379",
    "pr1002",
)
PUBLIC_VERTEX_COUNTS = (
    30,
    40,
    50,
)


def tsplib_euc_2d(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    delta = np.asarray(
        a,
        dtype=np.float64,
    ) - np.asarray(
        b,
        dtype=np.float64,
    )
    raw = float(
        np.sqrt(
            np.dot(
                delta,
                delta,
            )
        )
    )
    return float(
        math.floor(
            raw + 0.5
        )
    )


def pairwise_tsplib_euc_2d(
    coordinates: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(
        coordinates,
        dtype=np.float64,
    )
    if (
        coords.ndim != 2
        or coords.shape[1] != 2
    ):
        raise ValueError(
            "coordinates must have shape (N,2)"
        )
    delta = (
        coords[:, None, :]
        - coords[None, :, :]
    )
    raw = np.linalg.norm(
        delta,
        axis=-1,
    )
    return np.floor(
        raw + 0.5
    ).astype(
        np.float64
    )


@dataclass(frozen=True)
class MTRPDInstance:
    instance_id: str
    family: str
    replicate_index: int
    robot_count: int
    route_limit: float
    optimum_total_latency: float | None
    proven_optimal: bool
    split: str
    coordinates: np.ndarray
    distance_matrix: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(
            self.coordinates.shape[0]
        )

    @property
    def task_count(self) -> int:
        return (
            self.vertex_count - 1
        )

    @property
    def depot(self) -> np.ndarray:
        return self.coordinates[
            0
        ]

    @property
    def task_positions(
        self,
    ) -> np.ndarray:
        return self.coordinates[
            1:
        ]

    def validate(
        self,
    ) -> None:
        if not self.instance_id:
            raise ValueError(
                "instance_id is required"
            )
        if self.robot_count <= 0:
            raise ValueError(
                "robot_count must be positive"
            )
        if self.route_limit <= 0.0:
            raise ValueError(
                "route_limit must be positive"
            )
        if (
            self.coordinates.ndim != 2
            or self.coordinates.shape[
                1
            ]
            != 2
            or self.coordinates.shape[
                0
            ]
            < 2
        ):
            raise ValueError(
                "coordinates must contain depot + at least one task"
            )
        if self.distance_matrix.shape != (
            self.vertex_count,
            self.vertex_count,
        ):
            raise ValueError(
                "distance_matrix shape mismatch"
            )
        if np.any(
            self.distance_matrix
            < -1e-12
        ):
            raise ValueError(
                "distance_matrix must be nonnegative"
            )
        if not np.allclose(
            np.diag(
                self.distance_matrix
            ),
            0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "distance_matrix diagonal must be zero"
            )
        if self.split not in {
            "evolution",
            "validation",
            "protected_test",
        }:
            raise ValueError(
                f"unsupported split: {self.split}"
            )
        if self.proven_optimal:
            if (
                self.optimum_total_latency
                is None
                or self.optimum_total_latency
                <= 0.0
            ):
                raise ValueError(
                    "proven_optimal instances require a positive optimum"
                )


def default_split_for_replicate(
    replicate_index: int,
) -> str:
    """
    Balanced 4/3/3 split within every family x vertex-count group.

    Ten public replicates per group:
      0..3 -> evolution
      4..6 -> validation
      7..9 -> protected test
    """
    idx = int(
        replicate_index
    )
    if idx < 0 or idx > 9:
        raise ValueError(
            "replicate_index must be in [0,9]"
        )
    if idx <= 3:
        return "evolution"
    if idx <= 6:
        return "validation"
    return "protected_test"


def _parse_instance(
    row: dict[str, Any],
) -> MTRPDInstance:
    coords = np.asarray(
        row[
            "coordinates"
        ],
        dtype=np.float64,
    )
    if (
        "distance_matrix"
        in row
        and row[
            "distance_matrix"
        ]
        is not None
    ):
        distance = np.asarray(
            row[
                "distance_matrix"
            ],
            dtype=np.float64,
        )
    else:
        rule = str(
            row.get(
                "distance_rule",
                "TSPLIB_EUC_2D",
            )
        ).upper()
        if rule != "TSPLIB_EUC_2D":
            raise ValueError(
                "Only TSPLIB_EUC_2D is supported when distance_matrix is absent"
            )
        distance = (
            pairwise_tsplib_euc_2d(
                coords
            )
        )

    optimum_raw = row.get(
        "optimum_total_latency"
    )
    optimum = (
        None
        if optimum_raw is None
        else float(
            optimum_raw
        )
    )
    split = str(
        row.get(
            "split"
        )
        or default_split_for_replicate(
            int(
                row[
                    "replicate_index"
                ]
            )
        )
    )

    instance = MTRPDInstance(
        instance_id=str(
            row[
                "instance_id"
            ]
        ),
        family=str(
            row[
                "family"
            ]
        ),
        replicate_index=int(
            row[
                "replicate_index"
            ]
        ),
        robot_count=int(
            row[
                "robot_count"
            ]
        ),
        route_limit=float(
            row[
                "route_limit"
            ]
        ),
        optimum_total_latency=(
            optimum
        ),
        proven_optimal=bool(
            row.get(
                "proven_optimal",
                optimum is not None,
            )
        ),
        split=split,
        coordinates=coords,
        distance_matrix=distance,
    )
    instance.validate()
    return instance


def load_manifest(
    path: Path,
) -> list[
    MTRPDInstance
]:
    data = json.loads(
        Path(
            path
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        data.get(
            "version"
        )
        != MANIFEST_VERSION
    ):
        raise ValueError(
            "Unsupported MTRPD manifest version"
        )
    rows = [
        _parse_instance(
            row
        )
        for row in data.get(
            "instances",
            []
        )
    ]
    if not rows:
        raise ValueError(
            "MTRPD manifest contains no instances"
        )
    ids = [
        row.instance_id
        for row in rows
    ]
    if (
        len(
            ids
        )
        != len(
            set(
                ids
            )
        )
    ):
        raise ValueError(
            "Duplicate MTRPD instance_id"
        )
    return rows


def save_manifest(
    path: Path,
    instances: Iterable[
        MTRPDInstance
    ],
    *,
    source_url: str = (
        PUBLIC_SOURCE_URL
    ),
) -> None:
    rows = []
    for item in instances:
        item.validate()
        rows.append(
            {
                "instance_id": (
                    item.instance_id
                ),
                "family": (
                    item.family
                ),
                "replicate_index": int(
                    item.replicate_index
                ),
                "robot_count": int(
                    item.robot_count
                ),
                "route_limit": float(
                    item.route_limit
                ),
                "optimum_total_latency": (
                    None
                    if item.optimum_total_latency
                    is None
                    else float(
                        item.optimum_total_latency
                    )
                ),
                "proven_optimal": bool(
                    item.proven_optimal
                ),
                "split": item.split,
                "coordinates": (
                    item.coordinates.tolist()
                ),
                "distance_matrix": (
                    item.distance_matrix.tolist()
                ),
            }
        )
    payload = {
        "version": (
            MANIFEST_VERSION
        ),
        "source_url": (
            source_url
        ),
        "instances": rows,
    }
    target = Path(
        path
    )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def select_split(
    instances: Iterable[
        MTRPDInstance
    ],
    split: str,
    *,
    require_proven_optimum: bool = False,
) -> list[
    MTRPDInstance
]:
    selected = [
        item
        for item in instances
        if item.split == split
    ]
    if not selected:
        raise ValueError(
            f"No instances in split={split}"
        )
    if require_proven_optimum:
        missing = [
            item.instance_id
            for item in selected
            if (
                not item.proven_optimal
                or item.optimum_total_latency
                is None
            )
        ]
        if missing:
            raise ValueError(
                "Evolution split contains instances without proven optimum: "
                + ", ".join(
                    missing
                )
            )
    return selected


def describe_manifest(
    instances: Iterable[
        MTRPDInstance
    ],
) -> dict[str, Any]:
    items = list(
        instances
    )
    by_split: dict[
        str,
        int,
    ] = {}
    by_vertex_count: dict[
        str,
        int,
    ] = {}
    robot_counts: set[
        int
    ] = set()

    for item in items:
        by_split[
            item.split
        ] = (
            by_split.get(
                item.split,
                0,
            )
            + 1
        )
        key = str(
            item.vertex_count
        )
        by_vertex_count[
            key
        ] = (
            by_vertex_count.get(
                key,
                0,
            )
            + 1
        )
        robot_counts.add(
            item.robot_count
        )

    return {
        "instance_count": len(
            items
        ),
        "by_split": by_split,
        "by_vertex_count": (
            by_vertex_count
        ),
        "robot_counts": sorted(
            robot_counts
        ),
        "proven_optimum_count": sum(
            int(
                item.proven_optimal
                and item.optimum_total_latency
                is not None
            )
            for item in items
        ),
    }
