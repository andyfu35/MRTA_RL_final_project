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
BASELINES = ("nearest", "shortest_service", "shortest_total_time")


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
        seed_args = train_parser.parse_args(
            [
                "--generations",
                str(args.generations),
                "--population",
                str(args.population),
                "--worlds-per-generation",
                str(args.worlds_per_generation),
                "--probe-worlds",
                str(args.probe_worlds),
                "--validation-worlds",
                str(args.validation_worlds),
                "--archive-per-axis",
                str(args.archive_per_axis),
                "--seed",
                str(seed),
                "--log-every",
                str(args.log_every),
                "--output-dir",
                str(suite_dir / "runs"),
            ]
        )

        run_dir = train(seed_args)
        with (run_dir / "summary.json").open(
            "r",
            encoding="utf-8",
        ) as f:
            summary = json.load(f)

        ref = summary["final"]["reference_gene"]["validation_metrics"]
        baselines = summary["final"]["validation_baselines"]

        row: dict[str, object] = {
            "seed": seed,
            "run_dir": str(run_dir),
        }

        for axis in AXES:
            row[f"gene_{axis}"] = float(ref[axis])

        for baseline_name in BASELINES:
            baseline = baselines[baseline_name]
            for axis in AXES:
                row[f"{baseline_name}_{axis}"] = float(
                    baseline[axis]
                )
                row[
                    f"delta_gene_minus_{baseline_name}_{axis}"
                ] = (
                    float(ref[axis])
                    - float(baseline[axis])
                )

        per_seed.append(row)

    aggregate_metrics: dict[str, object] = {}
    for axis in AXES:
        axis_result: dict[str, object] = {
            "gene": _mean_std(
                [float(row[f"gene_{axis}"]) for row in per_seed]
            )
        }
        for baseline_name in BASELINES:
            axis_result[baseline_name] = _mean_std(
                [
                    float(row[f"{baseline_name}_{axis}"])
                    for row in per_seed
                ]
            )
            axis_result[
                f"delta_gene_minus_{baseline_name}"
            ] = _mean_std(
                [
                    float(
                        row[
                            f"delta_gene_minus_"
                            f"{baseline_name}_{axis}"
                        ]
                    )
                    for row in per_seed
                ]
            )
        aggregate_metrics[axis] = axis_result

    aggregate = {
        "experiment": "gene_homogeneous_mrta_v1_2_multiseed",
        "seeds": list(args.seeds),
        "per_seed": per_seed,
        "aggregate": aggregate_metrics,
    }

    with (suite_dir / "aggregate_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            aggregate,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (suite_dir / "per_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(per_seed[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_seed)

    print("\nMULTISEED AGGREGATE")
    print(
        json.dumps(
            aggregate_metrics,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nSUITE_DIR={suite_dir}")
    return suite_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-seed Gene MRTA v1.2 suite"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27, 37, 47],
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument(
        "--worlds-per-generation",
        type=int,
        default=8,
    )
    parser.add_argument("--probe-worlds", type=int, default=64)
    parser.add_argument(
        "--validation-worlds",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--archive-per-axis",
        type=int,
        default=8,
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v12_suite",
    )
    return parser


def main() -> None:
    run_suite(build_parser().parse_args())


if __name__ == "__main__":
    main()
