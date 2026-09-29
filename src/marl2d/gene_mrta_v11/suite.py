from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .train import build_parser as build_train_parser
from .train import train


AXES = ("completion", "efficiency", "balance")


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def run_suite(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.output_dir) / f"multiseed_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, object]] = []
    train_parser = build_train_parser()

    for seed in args.seeds:
        seed_args = train_parser.parse_args([
            "--generations", str(args.generations),
            "--population", str(args.population),
            "--worlds-per-generation", str(args.worlds_per_generation),
            "--probe-worlds", str(args.probe_worlds),
            "--validation-worlds", str(args.validation_worlds),
            "--archive-per-axis", str(args.archive_per_axis),
            "--seed", str(seed),
            "--log-every", str(args.log_every),
            "--output-dir", str(suite_dir / "runs"),
        ])
        run_dir = train(seed_args)
        with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        ref = summary["final"]["reference_gene"]["validation_metrics"]
        nearest = summary["final"]["validation_baselines"]["nearest"]
        row: dict[str, object] = {"seed": seed, "run_dir": str(run_dir)}
        for axis in AXES:
            row[f"gene_{axis}"] = float(ref[axis])
            row[f"nearest_{axis}"] = float(nearest[axis])
            row[f"delta_{axis}"] = float(ref[axis]) - float(nearest[axis])
        per_seed.append(row)

    aggregate: dict[str, object] = {
        "experiment": "gene_homogeneous_mrta_v1_1_multiseed",
        "seeds": list(args.seeds),
        "per_seed": per_seed,
        "aggregate": {},
    }
    aggregate_metrics: dict[str, object] = {}
    for axis in AXES:
        aggregate_metrics[axis] = {
            "gene": _mean_std([float(row[f"gene_{axis}"]) for row in per_seed]),
            "nearest": _mean_std([float(row[f"nearest_{axis}"]) for row in per_seed]),
            "delta_gene_minus_nearest": _mean_std([float(row[f"delta_{axis}"]) for row in per_seed]),
        }
    aggregate["aggregate"] = aggregate_metrics

    with (suite_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    csv_fields = list(per_seed[0].keys())
    with (suite_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_seed)

    print("\nMULTISEED AGGREGATE")
    print(json.dumps(aggregate_metrics, indent=2, ensure_ascii=False))
    print(f"\nSUITE_DIR={suite_dir}")
    return suite_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Five-seed Gene MRTA v1.1 suite")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--worlds-per-generation", type=int, default=8)
    parser.add_argument("--probe-worlds", type=int, default=64)
    parser.add_argument("--validation-worlds", type=int, default=128)
    parser.add_argument("--archive-per-axis", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/gene_mrta_v11_suite")
    return parser


def main() -> None:
    run_suite(build_parser().parse_args())


if __name__ == "__main__":
    main()
