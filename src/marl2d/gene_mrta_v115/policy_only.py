from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, World
from marl2d.gene_mrta_v113.route_tail import (
    evaluate_route_tail_plan,
    plan_route_tails,
)
from marl2d.gene_mrta_v115.scaling import (
    BenchmarkTimeout,
    _case_label,
    _load_frozen_gene,
    _pair_slot_count,
    _parse_cases,
    _rss_mb,
    _safe_mean,
    _time_limit,
    scale_config,
)


def build_euclidean_world(
    config: EnvConfig,
    seed: int,
) -> tuple[World, float]:
    rng = np.random.default_rng(seed)

    robots = rng.uniform(
        0.0,
        config.world_size,
        size=(config.num_robots, 2),
    )
    tasks = rng.uniform(
        0.0,
        config.world_size,
        size=(config.num_tasks, 2),
    )
    services = rng.uniform(
        config.service_time_min,
        config.service_time_max,
        size=config.num_tasks,
    )
    priorities = rng.uniform(
        config.priority_min,
        config.priority_max,
        size=config.num_tasks,
    )
    deadlines = rng.uniform(
        config.deadline_min,
        config.deadline_max,
        size=config.num_tasks,
    )
    batteries = rng.uniform(
        config.initial_battery_min,
        config.initial_battery_max,
        size=config.num_robots,
    )

    nodes = np.vstack(
        [
            robots,
            tasks,
        ]
    )

    start = time.perf_counter()
    node_sq = np.sum(
        nodes * nodes,
        axis=1,
    )[:, None]
    task_sq = np.sum(
        tasks * tasks,
        axis=1,
    )[None, :]
    path_to_tasks = (
        node_sq
        + task_sq
        - 2.0
        * (
            nodes
            @ tasks.T
        )
    )
    np.maximum(
        path_to_tasks,
        0.0,
        out=path_to_tasks,
    )
    np.sqrt(
        path_to_tasks,
        out=path_to_tasks,
    )
    build_seconds = (
        time.perf_counter()
        - start
    )

    world = World(
        robot_positions=robots,
        robot_initial_batteries=batteries,
        task_positions=tasks,
        task_service_times=services,
        task_priorities=priorities,
        task_deadlines=deadlines,
        obstacles=np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
        path_to_tasks=path_to_tasks,
    )
    return world, build_seconds


def _world_row(
    *,
    case_index: int,
    world_index: int,
    robots: int,
    tasks: int,
    seed: int,
    config: EnvConfig,
    gene,
    world_timeout: float,
    policy_timeout: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_index": case_index,
        "case": _case_label(
            robots,
            tasks,
        ),
        "world_index": world_index,
        "seed": seed,
        "robots": robots,
        "tasks": tasks,
        "tasks_per_robot": (
            float(tasks)
            / float(robots)
        ),
        "world_size": (
            config.world_size
        ),
        "initial_pair_count": (
            robots
            * tasks
        ),
        "status": "started",
        "failure_stage": "",
        "error": "",
    }

    rss_before = _rss_mb()

    try:
        with _time_limit(
            world_timeout,
            "euclidean path-table build",
        ):
            world, build_seconds = (
                build_euclidean_world(
                    config,
                    seed,
                )
            )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_stage": (
                    "euclidean_table"
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "rss_before_mb": (
                    rss_before
                ),
                "rss_peak_mb": (
                    _rss_mb()
                ),
            }
        )
        return row

    row[
        "euclidean_table_seconds"
    ] = build_seconds
    row[
        "path_table_entries"
    ] = int(
        world.path_to_tasks.size
    )
    row[
        "path_table_mb"
    ] = float(
        world.path_to_tasks.nbytes
        / (
            1024.0
            * 1024.0
        )
    )

    start = time.perf_counter()
    try:
        with _time_limit(
            policy_timeout,
            "policy planning",
        ):
            plan = plan_route_tails(
                gene,
                world,
                config,
            )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_stage": (
                    "policy"
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "policy_seconds": (
                    time.perf_counter()
                    - start
                ),
                "rss_before_mb": (
                    rss_before
                ),
                "rss_peak_mb": (
                    _rss_mb()
                ),
            }
        )
        return row

    policy_seconds = (
        time.perf_counter()
        - start
    )

    evaluation = (
        evaluate_route_tail_plan(
            plan,
            world,
            config,
        )
    )

    decoder_steps = len(
        plan.selection_sequence
    )
    pair_slots = _pair_slot_count(
        robots,
        tasks,
        decoder_steps,
    )
    queue_depths = np.asarray(
        [
            len(route)
            for route in plan.routes
        ],
        dtype=np.float64,
    )

    row.update(
        {
            "status": "ok",
            "policy_seconds": (
                policy_seconds
            ),
            "decoder_steps": (
                decoder_steps
            ),
            "pair_slots_scored": (
                pair_slots
            ),
            "policy_seconds_per_decoder_step": (
                policy_seconds
                / max(
                    decoder_steps,
                    1,
                )
            ),
            "policy_seconds_per_pair_slot": (
                policy_seconds
                / max(
                    pair_slots,
                    1,
                )
            ),
            "assigned_tasks": (
                decoder_steps
            ),
            "unassigned_tasks": (
                tasks
                - decoder_steps
            ),
            "assignment_fraction": (
                decoder_steps
                / max(
                    tasks,
                    1,
                )
            ),
            "completion": (
                evaluation.completion
            ),
            "raw_time_utility": (
                evaluation.time_optimality
            ),
            "balance": (
                evaluation.balance
            ),
            "route_efficiency": (
                evaluation.route_efficiency
            ),
            "mean_queue_depth": (
                float(
                    np.mean(
                        queue_depths
                    )
                )
            ),
            "max_queue_depth": (
                float(
                    np.max(
                        queue_depths
                    )
                )
            ),
            "battery_blocked_pair_events": (
                plan.battery_blocked_pair_events
            ),
            "rss_before_mb": (
                rss_before
            ),
            "rss_peak_mb": (
                _rss_mb()
            ),
        }
    )
    return row


def _case_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successes = [
        row
        for row in rows
        if row[
            "status"
        ] == "ok"
    ]
    first = rows[0]
    keys = (
        "euclidean_table_seconds",
        "path_table_mb",
        "policy_seconds",
        "decoder_steps",
        "pair_slots_scored",
        "assignment_fraction",
        "completion",
        "raw_time_utility",
        "balance",
        "mean_queue_depth",
        "max_queue_depth",
        "rss_peak_mb",
    )
    return {
        "case": first[
            "case"
        ],
        "robots": first[
            "robots"
        ],
        "tasks": first[
            "tasks"
        ],
        "world_size": first[
            "world_size"
        ],
        "requested_worlds": len(
            rows
        ),
        "successful_worlds": len(
            successes
        ),
        "failed_worlds": (
            len(rows)
            - len(successes)
        ),
        **{
            f"mean_{key}": (
                _safe_mean(
                    successes,
                    key,
                )
            )
            for key in keys
        },
        "failures": [
            {
                "seed": row[
                    "seed"
                ],
                "stage": row.get(
                    "failure_stage",
                    "",
                ),
                "error": row.get(
                    "error",
                    "",
                ),
            }
            for row in rows
            if row[
                "status"
            ] != "ok"
        ],
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def _fit_exponent(
    summaries: list[dict[str, Any]],
    key: str,
) -> float | None:
    points = [
        (
            float(item["tasks"]),
            float(item[key]),
        )
        for item in summaries
        if (
            item.get(key)
            is not None
            and float(
                item[key]
            ) > 0.0
        )
    ]
    if len(points) < 2:
        return None
    tasks = np.asarray(
        [
            item[0]
            for item in points
        ],
        dtype=np.float64,
    )
    values = np.asarray(
        [
            item[1]
            for item in points
        ],
        dtype=np.float64,
    )
    return float(
        np.polyfit(
            np.log(tasks),
            np.log(values),
            1,
        )[0]
    )


def run(
    args: argparse.Namespace,
) -> Path:
    cases = _parse_cases(
        args.cases
    )
    (
        gene_id,
        gene,
        gene_scores,
        gene_retention,
    ) = _load_frozen_gene(
        Path(
            args.v113_checkpoint
        ),
        archive_size=(
            args.archive_size
        ),
        hybrid_limit=(
            args.hybrid_limit
        ),
        threshold=(
            args.certification_threshold
        ),
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        Path(
            args.output_dir
        )
        / (
            "gene_mrta_v115b_policy_only_"
            f"{stamp}_seed{args.seed_base}"
        )
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[
        dict[str, Any]
    ] = []
    summaries: list[
        dict[str, Any]
    ] = []

    for case_index, (
        robots,
        tasks,
    ) in enumerate(cases):
        config = scale_config(
            robots,
            tasks,
            preserve_spatial_density=True,
        )
        case_rows = []

        for world_index in range(
            args.worlds_per_case
        ):
            seed = (
                args.seed_base
                + case_index
                * 10_000
                + world_index
            )
            print(
                f"POLICY_ONLY {robots}R/{tasks}T "
                f"WORLD {world_index + 1}/"
                f"{args.worlds_per_case} "
                f"seed={seed}",
                flush=True,
            )
            row = _world_row(
                case_index=case_index,
                world_index=world_index,
                robots=robots,
                tasks=tasks,
                seed=seed,
                config=config,
                gene=gene,
                world_timeout=(
                    args.world_timeout
                ),
                policy_timeout=(
                    args.policy_timeout
                ),
            )
            rows.append(
                row
            )
            case_rows.append(
                row
            )
            print(
                "RESULT "
                + json.dumps(
                    row,
                    ensure_ascii=False,
                ),
                flush=True,
            )

        summary = _case_summary(
            case_rows
        )
        summaries.append(
            summary
        )
        print(
            "CASE_SUMMARY "
            + json.dumps(
                summary,
                ensure_ascii=False,
            ),
            flush=True,
        )

        if (
            args.stop_after_zero_success
            and summary[
                "successful_worlds"
            ] == 0
        ):
            print(
                "STOP_LADDER="
                "zero successful worlds",
                flush=True,
            )
            break

    _write_csv(
        run_dir
        / "policy_only_results.csv",
        rows,
    )
    _write_csv(
        run_dir
        / "policy_only_case_summary.csv",
        summaries,
    )

    policy_exponent = (
        _fit_exponent(
            summaries,
            "mean_policy_seconds",
        )
    )
    table_exponent = (
        _fit_exponent(
            summaries,
            "mean_euclidean_table_seconds",
        )
    )

    payload = {
        "experiment": (
            "gene_mrta_v115b_policy_only_scaling"
        ),
        "status": "completed",
        "seed_base": (
            args.seed_base
        ),
        "cases_requested": [
            {
                "robots": r,
                "tasks": t,
            }
            for r, t in cases
        ],
        "worlds_per_case": (
            args.worlds_per_case
        ),
        "frozen_policy": {
            "record_id": (
                gene_id
            ),
            "parameter_count": int(
                gene.vector_data.size
            ),
            "baseline_scores": (
                gene_scores
            ),
            "baseline_retention": (
                gene_retention
            ),
        },
        "path_model": (
            "dense vectorized Euclidean "
            "robot/task and task/task distance table; no A*"
        ),
        "policy_timeout_seconds": (
            args.policy_timeout
        ),
        "world_timeout_seconds": (
            args.world_timeout
        ),
        "case_summaries": (
            summaries
        ),
        "empirical_exponents_vs_task_count": {
            "policy_seconds": (
                policy_exponent
            ),
            "euclidean_table_seconds": (
                table_exponent
            ),
        },
        "guardrails": [
            (
                "Policy-only benchmark: do not compare its "
                "behavioral quality directly with obstacle-A* V1.15A."
            ),
            (
                "Same 148-parameter route-tail decoder and "
                "original feature normalization are retained."
            ),
            "No scale-specific retraining.",
            "99M untouched.",
        ],
    }

    (
        run_dir
        / "policy_only_summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "V115B_POLICY_ONLY_FINISHED"
    )
    print(
        f"FROZEN_GENE={gene_id}"
    )
    print(
        f"PARAMETERS="
        f"{gene.vector_data.size}"
    )
    print(
        f"POLICY_EXPONENT="
        f"{policy_exponent}"
    )
    print(
        f"RUN_DIR={run_dir}"
    )
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v113-checkpoint",
        required=True,
    )
    p.add_argument(
        "--cases",
        default=(
            "64x320,128x640"
        ),
    )
    p.add_argument(
        "--worlds-per-case",
        type=int,
        default=1,
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=115_100_000,
    )
    p.add_argument(
        "--archive-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--hybrid-limit",
        type=int,
        default=128,
    )
    p.add_argument(
        "--certification-threshold",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--world-timeout",
        type=float,
        default=120.0,
    )
    p.add_argument(
        "--policy-timeout",
        type=float,
        default=300.0,
    )
    p.add_argument(
        "--stop-after-zero-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "gene_mrta_v115b_policy_only"
        ),
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.worlds_per_case <= 0:
        raise ValueError(
            "worlds-per-case must be positive"
        )
    run(args)


if __name__ == "__main__":
    main()
