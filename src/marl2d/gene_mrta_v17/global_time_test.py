from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v16t.gene import Gene as LinearGene
from marl2d.gene_mrta_v16t.global_optimal_core import (
    solve_global_time_optimum,
)
from marl2d.gene_mrta_v16t.hungarian_benchmark import rollout_world

from .direct_gene import DirectAssignmentGene
from .rollout import rollout_direct_gene


def _load_direct_time_gene(run_dir: Path) -> DirectAssignmentGene:
    data = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    return DirectAssignmentGene.from_dict(
        data["final"]["axis_specialists"]["time_optimality"]["gene"]
    )


def _load_v16to_gene(run_dir: Path) -> LinearGene:
    data = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    weights = data["final"]["global_optimality_specialist"]["gene"][
        "weights"
    ]
    return LinearGene(np.asarray(weights, dtype=np.float64))


def run(args: argparse.Namespace) -> Path:
    direct_run = Path(args.direct_run)
    direct_gene = _load_direct_time_gene(direct_run)

    v16to_gene = None
    if args.v16to_run:
        v16to_gene = _load_v16to_gene(Path(args.v16to_run))

    config = EnvConfig()
    rows: list[dict[str, object]] = []

    for k in range(args.worlds):
        seed = args.world_seed + k
        world = generate_world(config, seed)

        direct_eval = rollout_direct_gene(
            direct_gene,
            world,
            config,
        ).evaluation
        direct_time = float(direct_eval.time_optimality)

        hungarian_time = float(
            rollout_world(
                world,
                config,
                score_mode="path_time",
                matcher="hungarian",
            ).evaluation.time_optimality
        )

        v16to_time = None
        if v16to_gene is not None:
            v16to_time = float(
                rollout_world(
                    world,
                    config,
                    score_mode="gene",
                    matcher="greedy",
                    gene=v16to_gene,
                ).evaluation.time_optimality
            )

        oracle = solve_global_time_optimum(
            world,
            config,
            time_limit=args.time_limit,
        )

        row: dict[str, object] = {
            "world_seed": seed,
            "direct_time": direct_time,
            "hungarian_time": hungarian_time,
            "v16to_time": v16to_time,
            "global_time": oracle.time_optimality,
            "optimal": oracle.optimal,
            "mip_gap": oracle.mip_gap,
            "solve_seconds": oracle.solve_seconds,
            "direct_completed_tasks": direct_eval.completed_tasks,
        }

        if oracle.optimal and oracle.time_optimality:
            star = float(oracle.time_optimality)
            row["direct_retention"] = direct_time / star
            row["hungarian_retention"] = hungarian_time / star
            row["v16to_retention"] = (
                v16to_time / star
                if v16to_time is not None
                else None
            )

        rows.append(row)

        print(
            f"WORLD {k:03d} seed={seed} | "
            f"Direct={direct_time:.6f} "
            f"V16TO={v16to_time if v16to_time is not None else 'NA'} "
            f"Hungarian={hungarian_time:.6f} "
            f"Global={oracle.time_optimality} "
            f"optimal={oracle.optimal} gap={oracle.mip_gap}"
        )
        if "direct_retention" in row:
            msg = (
                f"  retention Direct={100*float(row['direct_retention']):.3f}% "
                f"Hungarian={100*float(row['hungarian_retention']):.3f}%"
            )
            if row["v16to_retention"] is not None:
                msg += (
                    f" V16TO={100*float(row['v16to_retention']):.3f}%"
                )
            print(msg)

    proven = [row for row in rows if row["optimal"]]
    aggregate: dict[str, object] = {
        "worlds_requested": len(rows),
        "worlds_proven_optimal": len(proven),
    }

    if proven:
        direct = np.asarray(
            [float(row["direct_retention"]) for row in proven],
            dtype=np.float64,
        )
        hungarian = np.asarray(
            [float(row["hungarian_retention"]) for row in proven],
            dtype=np.float64,
        )
        aggregate.update(
            {
                "direct_retention_mean": float(direct.mean()),
                "direct_retention_std": float(direct.std(ddof=1))
                if len(direct) > 1 else 0.0,
                "direct_retention_min": float(direct.min()),
                "direct_retention_max": float(direct.max()),
                "hungarian_retention_mean": float(hungarian.mean()),
                "direct_minus_hungarian_mean": float(
                    (direct - hungarian).mean()
                ),
            }
        )

        if v16to_gene is not None:
            old = np.asarray(
                [float(row["v16to_retention"]) for row in proven],
                dtype=np.float64,
            )
            aggregate.update(
                {
                    "v16to_retention_mean": float(old.mean()),
                    "direct_minus_v16to_mean": float(
                        (direct - old).mean()
                    ),
                    "direct_wins_vs_v16to": int(np.sum(direct > old)),
                    "ties_vs_v16to": int(
                        np.sum(np.isclose(direct, old, atol=1e-12))
                    ),
                    "losses_vs_v16to": int(np.sum(direct < old)),
                }
            )

        aggregate["solve_seconds_mean"] = float(
            np.mean(
                [float(row["solve_seconds"]) for row in proven]
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v17_direct_global_time_test.json"
    csv_path = output_dir / "v17_direct_global_time_test.csv"

    json_path.write_text(
        json.dumps(
            {
                "direct_run": str(direct_run),
                "v16to_run": args.v16to_run,
                "test_world_seed_base": args.world_seed,
                "held_out_from_v17_train_probe": True,
                "rows": rows,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fields = [
        "world_seed",
        "direct_time",
        "v16to_time",
        "hungarian_time",
        "global_time",
        "optimal",
        "mip_gap",
        "solve_seconds",
        "direct_completed_tasks",
        "direct_retention",
        "v16to_retention",
        "hungarian_retention",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nV1.7 DIRECT HELD-OUT GLOBAL TIME TEST")
    print(json.dumps(aggregate, indent=2))
    print(f"RESULT_JSON={json_path}")
    print(f"RESULT_CSV={csv_path}")
    return json_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--direct-run", required=True)
    p.add_argument("--v16to-run", default="")
    p.add_argument("--worlds", type=int, default=20)
    p.add_argument("--world-seed", type=int, default=97_000_000)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v17_global_time_test",
    )
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
