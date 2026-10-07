from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from .bank import BankRecord
from .rollout import GeneEvaluation, evaluate_population
from .suite import FIXED_WORLD_SEEDS, WorldConfig, generate_world


PROTOCOL = "gene_global_set_mrta_v20_zero_shot_size_ood_v1"
TRAIN_ROBOT_RANGE = (5, 20)
TRAIN_TASK_RANGE = (10, 100)
DEFAULT_CELLS = (
    (10, 50),
    (20, 100),
    (25, 100),
    (40, 100),
    (60, 100),
    (20, 125),
    (20, 150),
    (20, 200),
    (20, 300),
    (25, 125),
    (40, 200),
    (60, 300),
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _parse_cells(text: str | None) -> list[tuple[int, int]]:
    if text is None or not text.strip():
        return list(DEFAULT_CELLS)
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid cell {raw!r}; expected ROBOTSxTASKS, e.g. 20x100"
            )
        robots, tasks = (int(parts[0]), int(parts[1]))
        if robots <= 0 or tasks <= 0:
            raise ValueError("Robot/task counts must be positive")
        cell = (robots, tasks)
        if cell not in seen:
            cells.append(cell)
            seen.add(cell)
    if not cells:
        raise ValueError("At least one OOD cell is required")
    return cells


def _regime(robots: int, tasks: int) -> str:
    r_in = TRAIN_ROBOT_RANGE[0] <= robots <= TRAIN_ROBOT_RANGE[1]
    t_in = TRAIN_TASK_RANGE[0] <= tasks <= TRAIN_TASK_RANGE[1]
    if robots == TRAIN_ROBOT_RANGE[1] and tasks == TRAIN_TASK_RANGE[1]:
        return "train_boundary"
    if r_in and t_in:
        return "in_distribution"
    if not r_in and not t_in:
        return "both_ood"
    if not r_in:
        return "robot_ood"
    return "task_ood"


def _unseen_seeds(base_seed: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("seed count must be positive")
    blocked = set(FIXED_WORLD_SEEDS)
    rng = np.random.default_rng(int(base_seed))
    out: list[int] = []
    used: set[int] = set()
    while len(out) < count:
        value = int(rng.integers(1, 2_000_000_000))
        if value in blocked or value in used:
            continue
        used.add(value)
        out.append(value)
    return out


def _actual_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_checkpoint_records(
    checkpoint: Path,
) -> tuple[dict[str, object], dict[str, BankRecord]]:
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    records = {
        str(row["record_id"]): BankRecord.from_dict(row)
        for row in data["bank_records"]
    }
    if not records:
        raise ValueError("Checkpoint Gene Bank is empty")
    return data, records


def _select_record(
    records: dict[str, BankRecord],
    axis: str,
    requested_id: str | None,
) -> BankRecord:
    if requested_id is not None:
        try:
            return records[requested_id]
        except KeyError as exc:
            raise ValueError(
                f"Requested Gene {requested_id!r} is not present in checkpoint"
            ) from exc
    if axis == "total_time":
        return min(records.values(), key=lambda row: row.scores[axis])
    if axis == "on_time_completed_tasks":
        return max(records.values(), key=lambda row: row.scores[axis])
    raise ValueError(f"Unsupported champion axis: {axis}")


def _worlds_for_cell(
    robots: int,
    tasks: int,
    seeds: Iterable[int],
) -> tuple[WorldConfig, list[object]]:
    config = WorldConfig(
        robot_min=robots,
        robot_max=robots,
        task_min=tasks,
        task_max=tasks,
    )
    worlds = [generate_world(seed, config) for seed in seeds]
    if any(w.robot_count != robots or w.task_count != tasks for w in worlds):
        raise RuntimeError("Exact-size OOD world generation failed")
    return config, worlds


def _summary_row(
    model_name: str,
    record: BankRecord,
    evaluation: GeneEvaluation,
    worlds: list[object],
    robots: int,
    tasks: int,
    seed_count: int,
    cell_runtime_s: float,
) -> dict[str, object]:
    ratios = np.asarray(
        [
            metric.total_time / max(float(world.baseline_makespan), 1e-12)
            for metric, world in zip(evaluation.per_seed, worlds)
        ],
        dtype=np.float64,
    )
    baselines = np.asarray(
        [float(world.baseline_makespan) for world in worlds],
        dtype=np.float64,
    )
    return {
        "model": model_name,
        "record_id": record.record_id,
        "robots": robots,
        "tasks": tasks,
        "regime": _regime(robots, tasks),
        "seed_count": seed_count,
        "total_time_s": evaluation.total_time,
        "baseline_time_s": float(np.mean(baselines)),
        "time_over_baseline": float(np.mean(ratios)),
        "time_over_baseline_median": float(np.median(ratios)),
        "priority_rank": evaluation.priority,
        "on_time_percent": 100.0 * evaluation.deadline_completion_rate,
        "total_distance": evaluation.total_distance,
        "cell_runtime_s": cell_runtime_s,
    }


def _per_seed_rows(
    model_name: str,
    record: BankRecord,
    evaluation: GeneEvaluation,
    worlds: list[object],
    robots: int,
    tasks: int,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for metric, world in zip(evaluation.per_seed, worlds):
        baseline = float(world.baseline_makespan)
        out.append(
            {
                "model": model_name,
                "record_id": record.record_id,
                "robots": robots,
                "tasks": tasks,
                "regime": _regime(robots, tasks),
                "seed": metric.seed,
                "total_time_s": metric.total_time,
                "baseline_time_s": baseline,
                "time_over_baseline": metric.total_time / max(baseline, 1e-12),
                "priority_rank": metric.priority,
                "on_time_percent": 100.0 * metric.deadline_completion_rate,
                "total_distance": metric.total_distance,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint)
    data, records = _load_checkpoint_records(checkpoint)

    total_record = _select_record(
        records,
        "total_time",
        args.total_time_gene_id,
    )
    ontime_record = _select_record(
        records,
        "on_time_completed_tasks",
        args.on_time_gene_id,
    )
    models = [
        ("total_time", total_record),
        ("on_time", ontime_record),
    ]

    cells = _parse_cells(args.cells)
    seeds = _unseen_seeds(args.seed, args.seeds)
    device = _actual_device(args.device)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    protocol = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint),
        "checkpoint_completed_round": data.get("completed_round"),
        "device": device,
        "training_robot_range": list(TRAIN_ROBOT_RANGE),
        "training_task_range": list(TRAIN_TASK_RANGE),
        "cells": [{"robots": r, "tasks": t, "regime": _regime(r, t)} for r, t in cells],
        "unseen_seed_base": args.seed,
        "unseen_seeds": seeds,
        "seed_count_per_cell": args.seeds,
        "frozen_genes": {
            "total_time": {
                "record_id": total_record.record_id,
                "training_scores": dict(total_record.scores),
            },
            "on_time": {
                "record_id": ontime_record.record_id,
                "training_scores": dict(ontime_record.scores),
            },
        },
        "selection_feedback": False,
        "retraining": False,
        "notes": (
            "Zero-shot size generalization only. OOD results never enter "
            "the Gene Bank, mutation, parent selection, or training."
        ),
    }
    _atomic_json(run_dir / "protocol.json", protocol)
    print("V20_OOD_PROTOCOL " + json.dumps(protocol, ensure_ascii=False), flush=True)

    summary_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []

    for robots, tasks in cells:
        config, worlds = _worlds_for_cell(robots, tasks, seeds)

        started = time.perf_counter()
        evaluations = evaluate_population(
            [record.gene for _, record in models],
            worlds,
            config=config,
            device=device,
            gene_batch_size=2,
        )
        cell_runtime_s = time.perf_counter() - started

        for (model_name, record), evaluation in zip(models, evaluations):
            row = _summary_row(
                model_name,
                record,
                evaluation,
                worlds,
                robots,
                tasks,
                args.seeds,
                cell_runtime_s,
            )
            summary_rows.append(row)
            per_seed_rows.extend(
                _per_seed_rows(
                    model_name,
                    record,
                    evaluation,
                    worlds,
                    robots,
                    tasks,
                )
            )
            print(
                "V20_OOD_CELL " + json.dumps(row, ensure_ascii=False),
                flush=True,
            )

    _write_csv(run_dir / "summary.csv", summary_rows)
    _atomic_json(
        run_dir / "summary.json",
        {
            "protocol": PROTOCOL,
            "rows": summary_rows,
        },
    )
    with (run_dir / "per_seed.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_seed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"V20_OOD_RUN_DIR={run_dir}", flush=True)
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed", type=int, default=20070001)
    p.add_argument(
        "--cells",
        default=",".join(f"{r}x{t}" for r, t in DEFAULT_CELLS),
    )
    p.add_argument("--total-time-gene-id")
    p.add_argument("--on-time-gene-id")
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
