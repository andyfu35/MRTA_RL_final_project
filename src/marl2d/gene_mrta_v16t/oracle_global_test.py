from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .env import EnvConfig, generate_world
from .gene import Gene
from .global_optimal_core import solve_global_time_optimum
from .hungarian_benchmark import rollout_world


def _load_oracle_gene(run_dir: Path) -> Gene:
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    weights = data["final"]["global_optimality_specialist"]["gene"]["weights"]
    return Gene(np.asarray(weights, dtype=np.float64))


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    gene = _load_oracle_gene(run_dir)
    config = EnvConfig()
    rows = []

    for k in range(args.worlds):
        seed = args.world_seed + k
        world = generate_world(config, seed)

        gene_score = rollout_world(
            world,
            config,
            score_mode="gene",
            matcher="greedy",
            gene=gene,
        ).evaluation.time_optimality
        hungarian_score = rollout_world(
            world,
            config,
            score_mode="path_time",
            matcher="hungarian",
        ).evaluation.time_optimality
        oracle = solve_global_time_optimum(
            world,
            config,
            time_limit=args.time_limit,
        )

        row = {
            "world_seed": seed,
            "gene_time": float(gene_score),
            "hungarian_time": float(hungarian_score),
            "global_time": oracle.time_optimality,
            "optimal": oracle.optimal,
            "mip_gap": oracle.mip_gap,
            "solve_seconds": oracle.solve_seconds,
        }
        if oracle.optimal and oracle.time_optimality:
            row["gene_retention"] = gene_score / oracle.time_optimality
            row["hungarian_retention"] = (
                hungarian_score / oracle.time_optimality
            )
        rows.append(row)

        print(
            f"WORLD {k:03d} seed={seed} | "
            f"Gene={gene_score:.6f} "
            f"Hungarian={hungarian_score:.6f} "
            f"Global={oracle.time_optimality} "
            f"optimal={oracle.optimal} gap={oracle.mip_gap}"
        )
        if "gene_retention" in row:
            print(
                f"  retention Gene={100*row['gene_retention']:.3f}% "
                f"Hungarian={100*row['hungarian_retention']:.3f}%"
            )

    proven = [row for row in rows if row["optimal"]]
    aggregate = {
        "worlds_requested": len(rows),
        "worlds_proven_optimal": len(proven),
    }
    if proven:
        gene_ret = np.asarray(
            [row["gene_retention"] for row in proven],
            dtype=np.float64,
        )
        hung_ret = np.asarray(
            [row["hungarian_retention"] for row in proven],
            dtype=np.float64,
        )
        aggregate.update(
            {
                "gene_retention_mean": float(gene_ret.mean()),
                "gene_retention_std": float(gene_ret.std(ddof=1))
                if len(gene_ret) > 1
                else 0.0,
                "gene_retention_min": float(gene_ret.min()),
                "gene_retention_max": float(gene_ret.max()),
                "hungarian_retention_mean": float(hung_ret.mean()),
                "gene_minus_hungarian_retention_mean": float(
                    (gene_ret - hung_ret).mean()
                ),
                "solve_seconds_mean": float(
                    np.mean([row["solve_seconds"] for row in proven])
                ),
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "oracle_guided_global_test.json"
    cpath = out / "oracle_guided_global_test.csv"

    jpath.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "test_world_seed_base": args.world_seed,
                "test_worlds_are_not_oracle_train_or_probe": True,
                "rows": rows,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with cpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "world_seed",
                "gene_time",
                "hungarian_time",
                "global_time",
                "optimal",
                "mip_gap",
                "solve_seconds",
                "gene_retention",
                "hungarian_retention",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nORACLE-GUIDED GLOBAL TEST")
    print(json.dumps(aggregate, indent=2))
    print(f"RESULT_JSON={jpath}")
    print(f"RESULT_CSV={cpath}")
    return jpath


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worlds", type=int, default=20)
    p.add_argument("--world-seed", type=int, default=97_000_000)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v16to_global_test",
    )
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
