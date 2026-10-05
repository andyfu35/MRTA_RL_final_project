from __future__ import annotations

from pathlib import Path

import numpy as np

from .benchmark import (
    MTRPDInstance,
    pairwise_tsplib_euc_2d,
    save_manifest,
)


def _fixture_instance(
    *,
    vertex_count: int,
    robot_count: int,
    replicate_index: int,
    split: str,
) -> MTRPDInstance:
    task_count = (
        vertex_count - 1
    )

    # Every task lies on its own ray with increasing radius. For smoke only,
    # we provision one robot per task so the published/reference optimum is
    # analytically the sum of direct depot-to-task distances.
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        task_count,
        endpoint=False,
    )
    radii = np.linspace(
        2.0,
        2.0
        + task_count
        - 1,
        task_count,
    )
    tasks = np.stack(
        [
            radii
            * np.cos(
                angles
            ),
            radii
            * np.sin(
                angles
            ),
        ],
        axis=1,
    )
    coords = np.vstack(
        [
            np.zeros(
                (
                    1,
                    2,
                ),
                dtype=np.float64,
            ),
            tasks,
        ]
    )
    distance = (
        pairwise_tsplib_euc_2d(
            coords
        )
    )
    direct = distance[
        0,
        1:
    ]
    optimum = float(
        np.sum(
            direct
        )
    )
    route_limit = float(
        2.0
        * np.max(
            direct
        )
        + 2.0
    )

    return MTRPDInstance(
        instance_id=(
            f"smoke-v{vertex_count}-r{replicate_index}"
        ),
        family="smoke",
        replicate_index=(
            replicate_index
        ),
        robot_count=(
            robot_count
        ),
        route_limit=(
            route_limit
        ),
        optimum_total_latency=(
            optimum
        ),
        proven_optimal=True,
        split=split,
        coordinates=coords,
        distance_matrix=distance,
    )


def write_smoke_manifest(
    path: Path,
) -> None:
    rows = []
    specs = (
        (6, 5),
        (8, 7),
        (10, 9),
    )
    for vertex_count, robot_count in specs:
        rows.append(
            _fixture_instance(
                vertex_count=(
                    vertex_count
                ),
                robot_count=(
                    robot_count
                ),
                replicate_index=0,
                split="evolution",
            )
        )
        rows.append(
            _fixture_instance(
                vertex_count=(
                    vertex_count
                ),
                robot_count=(
                    robot_count
                ),
                replicate_index=4,
                split="validation",
            )
        )
        rows.append(
            _fixture_instance(
                vertex_count=(
                    vertex_count
                ),
                robot_count=(
                    robot_count
                ),
                replicate_index=7,
                split="protected_test",
            )
        )

    save_manifest(
        Path(
            path
        ),
        rows,
        source_url=(
            "synthetic_v119_structure_smoke_only"
        ),
    )
