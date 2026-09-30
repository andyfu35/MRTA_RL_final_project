from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .env import EnvConfig, generate_world
from .global_optimal_core import solve_global_time_optimum
from .hungarian_benchmark import _load_gene, _summary_paths, rollout_world


def run(args: argparse.Namespace) -> Path:
    suite = Path(args.suite_dir)
    genes = [_load_gene(p, args.gene_source) for p in _summary_paths(suite)]
    config = EnvConfig()
    rows = []

    for k in range(args.worlds):
        seed = args.world_seed + k
        world = generate_world(config, seed)

        gene_scores = [
            rollout_world(
                world,
                config,
                score_mode="gene",
                matcher="greedy",
                gene=gene,
            ).evaluation.time_optimality
            for gene in genes
        ]
        hungarian = rollout_world(
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
            "world_index": k,
            "world_seed": seed,
            "gene_mean": float(np.mean(gene_scores)),
            "gene_best": float(np.max(gene_scores)),
            "gene_per_seed": [float(x) for x in gene_scores],
            "hungarian_path_time": float(hungarian),
            "global_time_optimality": oracle.time_optimality,
            "optimal": oracle.optimal,
            "status": oracle.status,
            "message": oracle.message,
            "mip_gap": oracle.mip_gap,
            "solve_seconds": oracle.solve_seconds,
            "completed_tasks": oracle.completed_tasks,
            "routes": [list(x) for x in oracle.routes],
        }

        if oracle.optimal and oracle.time_optimality:
            opt = oracle.time_optimality
            row["gene_mean_retention"] = row["gene_mean"] / opt
            row["gene_best_retention"] = row["gene_best"] / opt
            row["hungarian_retention"] = row["hungarian_path_time"] / opt

        rows.append(row)

        print(
            f"WORLD {k:03d} seed={seed} | "
            f"GeneMean={row['gene_mean']:.6f} | "
            f"Hungarian={hungarian:.6f} | "
            f"Global={oracle.time_optimality} | "
            f"optimal={oracle.optimal} | gap={oracle.mip_gap} | "
            f"solve={oracle.solve_seconds:.3f}s"
        )
        if oracle.optimal and oracle.time_optimality:
            print(
                f"  retention GeneMean={100*row['gene_mean_retention']:.3f}% "
                f"GeneBest={100*row['gene_best_retention']:.3f}% "
                f"Hungarian={100*row['hungarian_retention']:.3f}%"
            )
            print(f"  routes={oracle.routes}")

    proven = [r for r in rows if r["optimal"]]
    aggregate = {
        "worlds_requested": len(rows),
        "worlds_proven_optimal": len(proven),
    }
    if proven:
        aggregate.update(
            {
                "global_time_optimality_mean": float(
                    np.mean([r["global_time_optimality"] for r in proven])
                ),
                "gene_mean_retention_mean": float(
                    np.mean([r["gene_mean_retention"] for r in proven])
                ),
                "gene_best_retention_mean": float(
                    np.mean([r["gene_best_retention"] for r in proven])
                ),
                "hungarian_retention_mean": float(
                    np.mean([r["hungarian_retention"] for r in proven])
                ),
                "solve_seconds_mean": float(
                    np.mean([r["solve_seconds"] for r in proven])
                ),
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "global_optimal_benchmark.json"
    cpath = out / "global_optimal_benchmark.csv"

    jpath.write_text(
        json.dumps(
            {
                "objective": "maximize sum(1 - finish_time/H) over completed tasks / N",
                "global_optimum_rule": (
                    "Only status=optimal from HiGHS with requested mip_rel_gap=0 "
                    "is treated as proven global optimum."
                ),
                "suite_dir": str(suite),
                "gene_source": args.gene_source,
                "rows": rows,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "world_seed",
                "gene_mean",
                "gene_best",
                "hungarian_path_time",
                "global_time_optimality",
                "optimal",
                "mip_gap",
                "solve_seconds",
                "gene_mean_retention",
                "gene_best_retention",
                "hungarian_retention",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["world_seed"],
                    r["gene_mean"],
                    r["gene_best"],
                    r["hungarian_path_time"],
                    r["global_time_optimality"],
                    r["optimal"],
                    r["mip_gap"],
                    r["solve_seconds"],
                    r.get("gene_mean_retention"),
                    r.get("gene_best_retention"),
                    r.get("hungarian_retention"),
                ]
            )

    print("\nGLOBAL OPTIMAL SUMMARY")
    print(json.dumps(aggregate, indent=2))
    print(f"RESULT_JSON={jpath}")
    print(f"RESULT_CSV={cpath}")
    return jpath


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-dir", required=True)
    p.add_argument("--gene-source", default="time_optimality")
    p.add_argument("--worlds", type=int, default=1)
    p.add_argument("--world-seed", type=int, default=97_000_000)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v16t_global_optimal",
    )
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
