from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bank import (
    BankRecord,
    make_record,
    rank_selection_scores,
    rebuild_bank,
    representative_id,
)
from .gene import SetAssignmentGene
from .rollout import (
    AXES,
    AXIS_DIRECTIONS,
    GeneEvaluation,
    evaluate_population,
)
from .suite import (
    FIXED_WORLD_SEEDS,
    WorldConfig,
    fixed_worlds,
)


CHECKPOINT_VERSION = "gene_global_set_mrta_v20_checkpoint_v1"
PROTOCOL = "gene_global_set_mrta_v20_fixed100"


def _atomic_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def _axis_stats(
    values: list[float],
) -> dict[str, float]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
    }


def _best_index(
    evaluations: list[GeneEvaluation],
    axis: str,
) -> int:
    values = [
        getattr(x, axis)
        for x in evaluations
    ]
    return int(
        np.argmin(values)
        if AXIS_DIRECTIONS[axis] == "min"
        else np.argmax(values)
    )


def _seed_progress(
    current: GeneEvaluation,
    previous: GeneEvaluation | None,
) -> dict[str, int] | None:
    if previous is None:
        return None
    old = {
        row.seed: row
        for row in previous.per_seed
    }
    result = {
        "total_time_improved": 0,
        "priority_improved": 0,
        "completed_tasks_improved": 0,
        "deadline_rate_improved": 0,
        "all_three_improved": 0,
    }
    for row in current.per_seed:
        before = old[row.seed]
        ti = (
            row.total_time
            < before.total_time - 1e-9
        )
        pi = (
            row.priority
            < before.priority - 1e-9
        )
        ci = (
            row.completed_tasks
            > before.completed_tasks + 1e-9
        )
        di = (
            row.deadline_completion_rate
            > before.deadline_completion_rate + 1e-9
        )
        result["total_time_improved"] += int(ti)
        result["priority_improved"] += int(pi)
        result["completed_tasks_improved"] += int(ci)
        result["deadline_rate_improved"] += int(di)
        result["all_three_improved"] += int(
            ti and pi and ci
        )
    return result


def _evaluation_from_dict(
    data: dict[str, object],
) -> GeneEvaluation:
    from .rollout import WorldMetrics

    rows = tuple(
        WorldMetrics(**row)
        for row in data["per_seed"]
    )
    return GeneEvaluation(
        total_time=float(data["total_time"]),
        priority=float(data["priority"]),
        completed_tasks=float(
            data["completed_tasks"]
        ),
        deadline_completion_rate=float(
            data["deadline_completion_rate"]
        ),
        all_tasks_completed=float(
            data["all_tasks_completed"]
        ),
        total_distance=float(
            data["total_distance"]
        ),
        per_seed=rows,
    )


def _evaluation_dict(
    evaluation: GeneEvaluation,
) -> dict[str, object]:
    return {
        "total_time": evaluation.total_time,
        "priority": evaluation.priority,
        "completed_tasks": (
            evaluation.completed_tasks
        ),
        "deadline_completion_rate": (
            evaluation.deadline_completion_rate
        ),
        "all_tasks_completed": (
            evaluation.all_tasks_completed
        ),
        "total_distance": (
            evaluation.total_distance
        ),
        "per_seed": [
            row.to_dict()
            for row in evaluation.per_seed
        ],
    }


def run(
    args: argparse.Namespace,
) -> Path:
    config = WorldConfig()
    worlds = fixed_worlds(
        config,
        count=args.world_count,
    )
    if (
        args.world_count == 100
        and tuple(
            w.seed for w in worlds
        ) != FIXED_WORLD_SEEDS
    ):
        raise RuntimeError(
            "Fixed 100-seed suite changed unexpectedly"
        )

    rng = np.random.default_rng(
        args.seed
    )
    run_dir = Path(args.run_dir)
    checkpoint_path = (
        run_dir / "checkpoint.json"
    )
    representatives_dir = (
        run_dir / "representatives"
    )

    if checkpoint_path.exists():
        data = json.loads(
            checkpoint_path.read_text(
                encoding="utf-8"
            )
        )
        if (
            data.get("version")
            != CHECKPOINT_VERSION
        ):
            raise ValueError(
                "Incompatible V2.0 checkpoint"
            )
        immutable = {
            "world_count": args.world_count,
            "genes_per_round": (
                args.genes_per_round
            ),
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "mutation_sigma": (
                args.mutation_sigma
            ),
            "mutation_rate": (
                args.mutation_rate
            ),
        }
        for key, expected in immutable.items():
            if data.get(key) != expected:
                raise ValueError(
                    "Cannot resume because "
                    f"{key} changed"
                )
        rng.bit_generator.state = (
            data["rng_state"]
        )
        bank = {
            row["record_id"]:
                BankRecord.from_dict(row)
            for row in data["bank_records"]
        }
        history = list(
            data.get("history", [])
        )
        start_round = (
            int(data["completed_round"]) + 1
        )
    else:
        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )
        representatives_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        bank: dict[
            str,
            BankRecord,
        ] = {}
        history: list[
            dict[str, Any]
        ] = []
        start_round = 0
        _atomic_json(
            run_dir / "fixed_worlds.json",
            {
                "protocol": PROTOCOL,
                "world_config": asdict(config),
                "seeds": list(
                    FIXED_WORLD_SEEDS[
                        : args.world_count
                    ]
                ),
                "worlds": [
                    {
                        "seed": w.seed,
                        "robots": (
                            w.robot_count
                        ),
                        "tasks": (
                            w.task_count
                        ),
                        "baseline_makespan": (
                            w.baseline_makespan
                        ),
                        "min_deadline_margin": float(
                            np.min(
                                w.task_deadlines
                                - w.baseline_completion_times
                            )
                        ),
                    }
                    for w in worlds
                ],
            },
        )

    actual_device = args.device
    if actual_device == "auto":
        actual_device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        "V20_PROTOCOL "
        + json.dumps(
            {
                "protocol": PROTOCOL,
                "device": actual_device,
                "rounds": args.rounds,
                "genes_per_round": (
                    args.genes_per_round
                ),
                "fixed_worlds": (
                    args.world_count
                ),
                "robot_range": [
                    config.robot_min,
                    config.robot_max,
                ],
                "task_range": [
                    config.task_min,
                    config.task_max,
                ],
                "task_input": [
                    "x",
                    "y",
                    "priority",
                    "deadline",
                    "service_time",
                ],
                "robot_input": [
                    "x",
                    "y",
                    "accumulated_distance",
                    "estimated_finish_time",
                ],
                "axes": {
                    "total_time": (
                        "min raw makespan"
                    ),
                    "priority": (
                        "min priority-weighted "
                        "completion rank"
                    ),
                    "completed_tasks": (
                        "max tasks completed "
                        "before own deadline"
                    ),
                },
                "axis_normalization": (
                    "none"
                ),
                "parent_selection": (
                    "equal-axis percentile "
                    "ranks squared"
                ),
                "bank": (
                    "unbounded pure Pareto"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    previous_rep: (
        GeneEvaluation | None
    ) = None
    if start_round > 0:
        path = (
            representatives_dir
            / f"round_{start_round - 1:03d}.json"
        )
        if path.exists():
            previous_rep = (
                _evaluation_from_dict(
                    json.loads(
                        path.read_text(
                            encoding="utf-8"
                        )
                    )["evaluation"]
                )
            )

    for round_index in range(
        start_round,
        args.rounds,
    ):
        genes: list[
            SetAssignmentGene
        ] = []
        parent_ids: list[
            str | None
        ] = []

        if round_index == 0:
            for _ in range(
                args.genes_per_round
            ):
                genes.append(
                    SetAssignmentGene.random(
                        rng,
                        hidden_dim=(
                            args.hidden_dim
                        ),
                        scale=(
                            args.initial_scale
                        ),
                    )
                )
                parent_ids.append(None)
        else:
            ids, probs = (
                rank_selection_scores(bank)
            )
            for _ in range(
                args.genes_per_round
            ):
                parent_id = str(
                    rng.choice(
                        ids,
                        p=probs,
                    )
                )
                parent = bank[parent_id]
                genes.append(
                    parent.gene.mutated(
                        rng,
                        sigma=(
                            args.mutation_sigma
                        ),
                        mutation_rate=(
                            args.mutation_rate
                        ),
                    )
                )
                parent_ids.append(
                    parent_id
                )

        evaluations = (
            evaluate_population(
                genes,
                worlds,
                config=config,
                device=actual_device,
                gene_batch_size=(
                    args.gene_batch_size
                ),
            )
        )
        candidates = [
            make_record(
                g,
                ev,
                round_index,
                parent_id,
            )
            for (
                g,
                ev,
                parent_id,
            ) in zip(
                genes,
                evaluations,
                parent_ids,
            )
        ]
        bank = rebuild_bank(
            list(bank.values())
            + candidates
        )

        rep_id = representative_id(
            bank
        )
        rep_record = bank[rep_id]
        rep_eval = evaluate_population(
            [rep_record.gene],
            worlds,
            config=config,
            device=actual_device,
            gene_batch_size=1,
        )[0]

        summary: dict[
            str,
            Any,
        ] = {
            "round": round_index,
            "bank_size": len(bank),
            "population_axis_stats": {
                axis: _axis_stats(
                    [
                        getattr(ev, axis)
                        for ev in evaluations
                    ]
                )
                for axis in AXES
            },
            "population_best": {
                axis: getattr(
                    evaluations[
                        _best_index(
                            evaluations,
                            axis,
                        )
                    ],
                    axis,
                )
                for axis in AXES
            },
            "representative_id": (
                rep_id
            ),
            "representative": {
                "total_time": (
                    rep_eval.total_time
                ),
                "priority": (
                    rep_eval.priority
                ),
                "completed_tasks": (
                    rep_eval.completed_tasks
                ),
                "deadline_completion_rate": (
                    rep_eval.deadline_completion_rate
                ),
                "all_tasks_completed": (
                    rep_eval.all_tasks_completed
                ),
                "total_distance": (
                    rep_eval.total_distance
                ),
            },
            (
                "seed_progress_vs_"
                "previous_representative"
            ): _seed_progress(
                rep_eval,
                previous_rep,
            ),
        }
        history.append(summary)

        _atomic_json(
            representatives_dir
            / f"round_{round_index:03d}.json",
            {
                "round": (
                    round_index
                ),
                "record_id": rep_id,
                "evaluation": (
                    _evaluation_dict(
                        rep_eval
                    )
                ),
            },
        )
        _atomic_json(
            checkpoint_path,
            {
                "version": (
                    CHECKPOINT_VERSION
                ),
                "protocol": PROTOCOL,
                "completed_round": (
                    round_index
                ),
                "rounds": args.rounds,
                "world_count": (
                    args.world_count
                ),
                "genes_per_round": (
                    args.genes_per_round
                ),
                "hidden_dim": (
                    args.hidden_dim
                ),
                "seed": args.seed,
                "initial_scale": (
                    args.initial_scale
                ),
                "mutation_sigma": (
                    args.mutation_sigma
                ),
                "mutation_rate": (
                    args.mutation_rate
                ),
                "rng_state": (
                    rng.bit_generator.state
                ),
                "bank_records": [
                    row.to_dict()
                    for row in (
                        bank.values()
                    )
                ],
                "history": history,
            },
        )

        print(
            "V20_ROUND "
            + json.dumps(
                summary,
                ensure_ascii=False,
            ),
            flush=True,
        )
        previous_rep = rep_eval

    print(
        f"V20_RUN_DIR={run_dir}",
        flush=True,
    )
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        required=True,
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=50,
    )
    p.add_argument(
        "--genes-per-round",
        type=int,
        default=64,
    )
    p.add_argument(
        "--world-count",
        type=int,
        default=100,
    )
    p.add_argument(
        "--gene-batch-size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=8,
    )
    p.add_argument(
        "--initial-scale",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.12,
    )
    p.add_argument(
        "--mutation-rate",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=200,
    )
    p.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "mps",
            "cuda",
        ),
        default="auto",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if (
        args.rounds <= 0
        or args.genes_per_round <= 0
        or args.gene_batch_size <= 0
    ):
        raise ValueError(
            "rounds, genes-per-round "
            "and gene-batch-size "
            "must be positive"
        )
    run(args)


if __name__ == "__main__":
    main()
