from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .train import AXES, BASELINES
from .train import build_parser as build_train_parser
from .train import train


REPORT_BASELINES = tuple(name for name in BASELINES if name != "uninformed")


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
                "--generations", str(args.generations),
                "--population", str(args.population),
                "--worlds-per-generation", str(args.worlds_per_generation),
                "--probe-worlds", str(args.probe_worlds),
                "--validation-worlds", str(args.validation_worlds),
                "--archive-per-axis", str(args.archive_per_axis),
                "--seed", str(seed),
                "--log-every", str(args.log_every),
                "--robot-speed", str(args.robot_speed),
                "--service-time-min", str(args.service_time_min),
                "--service-time-max", str(args.service_time_max),
                "--priority-min", str(args.priority_min),
                "--priority-max", str(args.priority_max),
                "--deadline-min", str(args.deadline_min),
                "--deadline-max", str(args.deadline_max),
                "--episode-time", str(args.episode_time),
                "--obstacle-count", str(args.obstacle_count),
                "--obstacle-size-min", str(args.obstacle_size_min),
                "--obstacle-size-max", str(args.obstacle_size_max),
                "--obstacle-clearance", str(args.obstacle_clearance),
                "--grid-resolution", str(args.grid_resolution),
                "--battery-capacity", str(args.battery_capacity),
                "--initial-battery-min", str(args.initial_battery_min),
                "--initial-battery-max", str(args.initial_battery_max),
                "--energy-per-distance", str(args.energy_per_distance),
                "--output-dir", str(suite_dir / "runs"),
            ]
        )

        run_dir = train(seed_args)
        with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        final = summary["final"]
        ref = final["reference_gene"]["validation_metrics"]
        specialists = final["axis_best_selected_on_probe"]
        baselines = final["validation_baselines"]

        row: dict[str, object] = {
            "seed": seed,
            "run_dir": str(run_dir),
        }

        for metric in AXES:
            row[f"reference_{metric}"] = float(ref[metric])
        row["reference_battery_remaining_fraction"] = float(
            ref["battery_remaining_fraction"]
        )
        row["reference_battery_blocked_pair_events"] = float(
            ref["battery_blocked_pair_events"]
        )

        for specialist_axis in AXES:
            metrics = specialists[specialist_axis]["validation_metrics"]
            for metric in AXES:
                row[
                    f"specialist_{specialist_axis}_{metric}"
                ] = float(metrics[metric])

        for baseline_name in REPORT_BASELINES:
            baseline = baselines[baseline_name]
            for metric in AXES:
                row[f"{baseline_name}_{metric}"] = float(baseline[metric])
                row[
                    f"delta_reference_minus_{baseline_name}_{metric}"
                ] = float(ref[metric]) - float(baseline[metric])

        per_seed.append(row)

    aggregate: dict[str, object] = {
        "reference": {},
        "specialists": {},
        "baselines": {},
        "delta_reference_minus_baseline": {},
        "specialist_own_axis": {},
        "battery_diagnostics": {},
    }

    for metric in AXES:
        aggregate["reference"][metric] = _mean_std(
            [float(row[f"reference_{metric}"]) for row in per_seed]
        )

    aggregate["battery_diagnostics"]["reference_battery_remaining_fraction"] = (
        _mean_std(
            [
                float(row["reference_battery_remaining_fraction"])
                for row in per_seed
            ]
        )
    )
    aggregate["battery_diagnostics"]["reference_battery_blocked_pair_events"] = (
        _mean_std(
            [
                float(row["reference_battery_blocked_pair_events"])
                for row in per_seed
            ]
        )
    )

    for specialist_axis in AXES:
        aggregate["specialists"][specialist_axis] = {}
        for metric in AXES:
            aggregate["specialists"][specialist_axis][metric] = _mean_std(
                [
                    float(
                        row[
                            f"specialist_{specialist_axis}_{metric}"
                        ]
                    )
                    for row in per_seed
                ]
            )

    for baseline_name in REPORT_BASELINES:
        aggregate["baselines"][baseline_name] = {}
        aggregate["delta_reference_minus_baseline"][baseline_name] = {}
        for metric in AXES:
            aggregate["baselines"][baseline_name][metric] = _mean_std(
                [
                    float(row[f"{baseline_name}_{metric}"])
                    for row in per_seed
                ]
            )
            aggregate["delta_reference_minus_baseline"][baseline_name][metric] = (
                _mean_std(
                    [
                        float(
                            row[
                                f"delta_reference_minus_"
                                f"{baseline_name}_{metric}"
                            ]
                        )
                        for row in per_seed
                    ]
                )
            )

    for specialist_axis in AXES:
        own_values = [
            float(
                row[
                    f"specialist_{specialist_axis}_{specialist_axis}"
                ]
            )
            for row in per_seed
        ]
        own_summary: dict[str, object] = {
            "specialist": _mean_std(own_values),
            "vs_baselines": {},
        }
        for baseline_name in REPORT_BASELINES:
            baseline_values = [
                float(row[f"{baseline_name}_{specialist_axis}"])
                for row in per_seed
            ]
            own_summary["vs_baselines"][baseline_name] = _mean_std(
                [
                    gene_value - baseline_value
                    for gene_value, baseline_value
                    in zip(own_values, baseline_values)
                ]
            )
        aggregate["specialist_own_axis"][specialist_axis] = own_summary

    payload = {
        "experiment": "gene_homogeneous_mrta_v1_6_multiseed",
        "seeds": list(args.seeds),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }

    with (suite_dir / "aggregate_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

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
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"\nSUITE_DIR={suite_dir}")
    return suite_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-seed Gene MRTA v1.6 battery-aware suite"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27, 37, 47],
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--worlds-per-generation", type=int, default=8)
    parser.add_argument("--probe-worlds", type=int, default=64)
    parser.add_argument("--validation-worlds", type=int, default=128)
    parser.add_argument("--archive-per-axis", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--robot-speed", type=float, default=4.0)
    parser.add_argument("--service-time-min", type=float, default=2.0)
    parser.add_argument("--service-time-max", type=float, default=35.0)
    parser.add_argument("--priority-min", type=float, default=0.1)
    parser.add_argument("--priority-max", type=float, default=1.0)
    parser.add_argument("--deadline-min", type=float, default=25.0)
    parser.add_argument("--deadline-max", type=float, default=50.0)
    parser.add_argument("--episode-time", type=float, default=50.0)
    parser.add_argument("--obstacle-count", type=int, default=10)
    parser.add_argument("--obstacle-size-min", type=float, default=12.0)
    parser.add_argument("--obstacle-size-max", type=float, default=20.0)
    parser.add_argument("--obstacle-clearance", type=float, default=4.0)
    parser.add_argument("--grid-resolution", type=float, default=5.0)
    parser.add_argument("--battery-capacity", type=float, default=70.0)
    parser.add_argument("--initial-battery-min", type=float, default=35.0)
    parser.add_argument("--initial-battery-max", type=float, default=70.0)
    parser.add_argument("--energy-per-distance", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v16_suite",
    )
    return parser


def main() -> None:
    run_suite(build_parser().parse_args())


if __name__ == "__main__":
    main()
