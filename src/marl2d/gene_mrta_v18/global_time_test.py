from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    generate_world,
)
from marl2d.gene_mrta_v16t.gene import (
    Gene as LinearGene,
)
from marl2d.gene_mrta_v16t.global_optimal_core import (
    solve_global_time_optimum,
)
from marl2d.gene_mrta_v16t.hungarian_benchmark import (
    rollout_world,
)
from marl2d.gene_mrta_v17.direct_gene import (
    DirectAssignmentGene as V17DirectAssignmentGene,
)
from marl2d.gene_mrta_v17.rollout import (
    rollout_direct_gene as rollout_v17,
)

from .direct_gene import (
    ConsequenceAwareDirectGene,
)
from .rollout import (
    rollout_direct_gene as rollout_v18,
)


def _load_v18(
    run_dir: Path,
) -> ConsequenceAwareDirectGene:
    data = json.loads(
        (run_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    gene_data = data[
        "final"
    ][
        "time_specialist"
    ][
        "gene"
    ]
    return (
        ConsequenceAwareDirectGene.from_dict(
            gene_data
        )
    )


def _load_v17(
    run_dir: Path,
) -> V17DirectAssignmentGene:
    data = json.loads(
        (run_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    final = data["final"]
    if "time_specialist" in final:
        gene_data = final[
            "time_specialist"
        ]["gene"]
    else:
        gene_data = final[
            "axis_specialists"
        ][
            "time_optimality"
        ][
            "gene"
        ]
    return (
        V17DirectAssignmentGene.from_dict(
            gene_data
        )
    )


def _load_v16to(
    run_dir: Path,
) -> LinearGene:
    data = json.loads(
        (run_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    weights = data[
        "final"
    ][
        "global_optimality_specialist"
    ][
        "gene"
    ][
        "weights"
    ]
    return LinearGene(
        np.asarray(
            weights,
            dtype=np.float64,
        )
    )


def _aggregate(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    proven = [
        row
        for row in rows
        if row["optimal"]
    ]

    out: dict[str, object] = {
        "worlds_requested": len(rows),
        "worlds_proven_optimal": len(
            proven
        ),
    }

    if not proven:
        return out

    names = (
        "v18",
        "v17",
        "v16to",
        "hungarian",
    )
    arrays = {
        name: np.asarray(
            [
                float(
                    row[
                        f"{name}_retention"
                    ]
                )
                for row in proven
            ],
            dtype=np.float64,
        )
        for name in names
    }

    for name, values in arrays.items():
        out[
            f"{name}_retention_mean"
        ] = float(
            values.mean()
        )
        out[
            f"{name}_retention_std"
        ] = (
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0
        )
        out[
            f"{name}_retention_min"
        ] = float(
            values.min()
        )
        out[
            f"{name}_retention_max"
        ] = float(
            values.max()
        )

    for baseline in (
        "v17",
        "v16to",
        "hungarian",
    ):
        delta = (
            arrays["v18"]
            - arrays[baseline]
        )
        out[
            f"v18_minus_{baseline}_mean"
        ] = float(
            delta.mean()
        )
        out[
            f"v18_wins_vs_{baseline}"
        ] = int(
            np.sum(delta > 1e-12)
        )
        out[
            f"v18_ties_vs_{baseline}"
        ] = int(
            np.sum(
                np.isclose(
                    delta,
                    0.0,
                    atol=1e-12,
                )
            )
        )
        out[
            f"v18_losses_vs_{baseline}"
        ] = int(
            np.sum(delta < -1e-12)
        )

    out[
        "solve_seconds_mean"
    ] = float(
        np.mean(
            [
                float(
                    row[
                        "solve_seconds"
                    ]
                )
                for row in proven
            ]
        )
    )

    return out


def run(
    args: argparse.Namespace,
) -> Path:
    config = EnvConfig()

    v18_gene = _load_v18(
        Path(args.v18_run)
    )
    v17_gene = _load_v17(
        Path(args.v17_run)
    )
    v16to_gene = _load_v16to(
        Path(args.v16to_run)
    )

    rows: list[
        dict[str, object]
    ] = []

    for index in range(
        args.worlds
    ):
        seed = (
            args.world_seed
            + index
        )
        world = generate_world(
            config,
            seed,
        )

        v18_eval = rollout_v18(
            v18_gene,
            world,
            config,
        ).evaluation
        v17_eval = rollout_v17(
            v17_gene,
            world,
            config,
        ).evaluation
        v16to_eval = rollout_world(
            world,
            config,
            score_mode="gene",
            matcher="greedy",
            gene=v16to_gene,
        ).evaluation
        hungarian_eval = rollout_world(
            world,
            config,
            score_mode="path_time",
            matcher="hungarian",
        ).evaluation

        oracle = solve_global_time_optimum(
            world,
            config,
            time_limit=(
                args.time_limit
            ),
        )

        row: dict[str, object] = {
            "world_seed": seed,
            "v18_time": float(
                v18_eval.time_optimality
            ),
            "v17_time": float(
                v17_eval.time_optimality
            ),
            "v16to_time": float(
                v16to_eval.time_optimality
            ),
            "hungarian_time": float(
                hungarian_eval.time_optimality
            ),
            "global_time": (
                oracle.time_optimality
            ),
            "optimal": (
                oracle.optimal
            ),
            "mip_gap": (
                oracle.mip_gap
            ),
            "solve_seconds": (
                oracle.solve_seconds
            ),
            "v18_completed_tasks": (
                v18_eval.completed_tasks
            ),
            "v17_completed_tasks": (
                v17_eval.completed_tasks
            ),
        }

        if (
            oracle.optimal
            and oracle.time_optimality
        ):
            star = float(
                oracle.time_optimality
            )
            row[
                "v18_retention"
            ] = (
                float(
                    v18_eval.time_optimality
                )
                / star
            )
            row[
                "v17_retention"
            ] = (
                float(
                    v17_eval.time_optimality
                )
                / star
            )
            row[
                "v16to_retention"
            ] = (
                float(
                    v16to_eval.time_optimality
                )
                / star
            )
            row[
                "hungarian_retention"
            ] = (
                float(
                    hungarian_eval.time_optimality
                )
                / star
            )

        rows.append(row)

        print(
            f"WORLD {index:03d} "
            f"seed={seed} | "
            f"V18={row['v18_time']:.6f} "
            f"V17={row['v17_time']:.6f} "
            f"V16TO={row['v16to_time']:.6f} "
            f"Hungarian={row['hungarian_time']:.6f} "
            f"Global={oracle.time_optimality} "
            f"optimal={oracle.optimal}"
        )

        if "v18_retention" in row:
            print(
                "  retention "
                f"V18={100*float(row['v18_retention']):.3f}% "
                f"V17={100*float(row['v17_retention']):.3f}% "
                f"V16TO={100*float(row['v16to_retention']):.3f}% "
                f"Hungarian="
                f"{100*float(row['hungarian_retention']):.3f}%"
            )

    aggregate = _aggregate(
        rows
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "v18_global_time_test.json"
    )
    csv_path = (
        output_dir
        / "v18_global_time_test.csv"
    )

    json_path.write_text(
        json.dumps(
            {
                "v18_run": args.v18_run,
                "v17_run": args.v17_run,
                "v16to_run": args.v16to_run,
                "test_world_seed_base": (
                    args.world_seed
                ),
                "held_out_from_train_probe": (
                    True
                ),
                "rows": rows,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fields = [
        "world_seed",
        "v18_time",
        "v17_time",
        "v16to_time",
        "hungarian_time",
        "global_time",
        "optimal",
        "mip_gap",
        "solve_seconds",
        "v18_completed_tasks",
        "v17_completed_tasks",
        "v18_retention",
        "v17_retention",
        "v16to_retention",
        "hungarian_retention",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(
        "\nV1.8 CONSEQUENCE-AWARE "
        "HELD-OUT GLOBAL TIME TEST"
    )
    print(
        json.dumps(
            aggregate,
            indent=2,
        )
    )
    print(
        f"RESULT_JSON={json_path}"
    )
    print(
        f"RESULT_CSV={csv_path}"
    )

    return json_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--v18-run",
        required=True,
    )
    p.add_argument(
        "--v17-run",
        required=True,
    )
    p.add_argument(
        "--v16to-run",
        required=True,
    )
    p.add_argument(
        "--worlds",
        type=int,
        default=20,
    )
    p.add_argument(
        "--world-seed",
        type=int,
        default=97_000_000,
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "gene_mrta_v18_global_time_test"
        ),
    )

    return p


def main() -> None:
    run(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
