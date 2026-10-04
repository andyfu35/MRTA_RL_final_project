from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.global_optimal_core import (
    solve_global_time_optimum,
)
from marl2d.gene_mrta_v113.route_tail import (
    evaluate_route_tail_plan,
    plan_route_tails,
)
from marl2d.gene_mrta_v115.scaling import (
    BenchmarkStageError,
    _load_frozen_gene,
    _parse_cases,
    _rss_mb,
    _time_limit,
    generate_world_timed,
    scale_config,
)


EPS = 1e-12


def milp_problem_size(
    robots: int,
    tasks: int,
) -> dict[str, int]:
    r = int(robots)
    n = int(tasks)
    start_binaries = r * n
    arc_binaries = (
        r * n * (n - 1)
    )
    selected_binaries = r * n
    binary_variables = (
        start_binaries
        + arc_binaries
        + selected_binaries
    )
    continuous_variables = r * n
    total_variables = (
        binary_variables
        + continuous_variables
    )
    approximate_constraints = (
        r * n * (n - 1)
        + 4 * r * n
        + 2 * r
        + n
    )
    return {
        "milp_start_binary_variables": (
            start_binaries
        ),
        "milp_arc_binary_variables": (
            arc_binaries
        ),
        "milp_selected_binary_variables": (
            selected_binaries
        ),
        "milp_binary_variables": (
            binary_variables
        ),
        "milp_continuous_variables": (
            continuous_variables
        ),
        "milp_total_variables": (
            total_variables
        ),
        "milp_approx_constraints": (
            approximate_constraints
        ),
    }


def gap_metrics(
    *,
    policy_score: float,
    milp_incumbent_score: float | None,
    milp_upper_bound_score: float | None,
    optimal: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exact_absolute_gap": None,
        "exact_relative_gap": None,
        "exact_retention": None,
        "policy_retention_lower_bound": None,
        "policy_gap_upper_bound": None,
        "bound_consistent": None,
        "policy_minus_milp_incumbent": None,
        "policy_to_milp_incumbent_ratio": None,
    }

    if milp_incumbent_score is not None:
        result[
            "policy_minus_milp_incumbent"
        ] = float(
            policy_score
            - milp_incumbent_score
        )
        if (
            milp_incumbent_score
            > EPS
        ):
            result[
                "policy_to_milp_incumbent_ratio"
            ] = float(
                policy_score
                / milp_incumbent_score
            )

    if (
        milp_upper_bound_score
        is not None
        and milp_upper_bound_score
        > EPS
    ):
        consistent = bool(
            policy_score
            <= milp_upper_bound_score
            + 1e-7
        )
        result[
            "bound_consistent"
        ] = consistent
        if consistent:
            result[
                "policy_retention_lower_bound"
            ] = float(
                policy_score
                / milp_upper_bound_score
            )
            result[
                "policy_gap_upper_bound"
            ] = float(
                max(
                    0.0,
                    milp_upper_bound_score
                    - policy_score,
                )
            )

    if (
        optimal
        and milp_incumbent_score
        is not None
    ):
        optimum = float(
            milp_incumbent_score
        )
        absolute_gap = float(
            optimum
            - policy_score
        )
        result[
            "exact_absolute_gap"
        ] = absolute_gap
        if optimum > EPS:
            result[
                "exact_relative_gap"
            ] = float(
                absolute_gap
                / optimum
            )
            result[
                "exact_retention"
            ] = float(
                policy_score
                / optimum
            )

    return result


def _case_label(
    robots: int,
    tasks: int,
) -> str:
    return f"{robots}R_{tasks}T"


def _safe_mean(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]
    if not values:
        return None
    return float(
        np.mean(
            values
        )
    )


def _safe_median(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]
    if not values:
        return None
    return float(
        np.median(
            values
        )
    )


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


def _case_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = rows[0]
    valid = [
        row
        for row in rows
        if row.get(
            "status"
        ) == "ok"
    ]
    proven = [
        row
        for row in valid
        if row.get(
            "milp_optimal"
        )
    ]

    result: dict[str, Any] = {
        "case": first["case"],
        "robots": first["robots"],
        "tasks": first["tasks"],
        "requested_worlds": len(
            rows
        ),
        "valid_worlds": len(
            valid
        ),
        "milp_optimal_worlds": len(
            proven
        ),
        "milp_optimal_proof_rate": (
            len(proven)
            / max(
                len(valid),
                1,
            )
        ),
        "mean_common_preprocess_seconds": (
            _safe_mean(
                valid,
                "common_preprocess_seconds",
            )
        ),
        "mean_policy_seconds": (
            _safe_mean(
                valid,
                "policy_seconds",
            )
        ),
        "median_policy_seconds": (
            _safe_median(
                valid,
                "policy_seconds",
            )
        ),
        "mean_milp_total_seconds": (
            _safe_mean(
                valid,
                "milp_total_seconds",
            )
        ),
        "median_milp_total_seconds": (
            _safe_median(
                valid,
                "milp_total_seconds",
            )
        ),
        "mean_milp_solver_seconds": (
            _safe_mean(
                valid,
                "milp_solver_seconds",
            )
        ),
        "mean_milp_over_policy_time_ratio": (
            _safe_mean(
                valid,
                "milp_over_policy_time_ratio",
            )
        ),
        "mean_policy_score": (
            _safe_mean(
                valid,
                "policy_time_optimality",
            )
        ),
        "mean_exact_retention": (
            _safe_mean(
                proven,
                "exact_retention",
            )
        ),
        "mean_exact_relative_gap": (
            _safe_mean(
                proven,
                "exact_relative_gap",
            )
        ),
        "mean_exact_absolute_gap": (
            _safe_mean(
                proven,
                "exact_absolute_gap",
            )
        ),
        "min_exact_retention": (
            min(
                [
                    float(
                        row[
                            "exact_retention"
                        ]
                    )
                    for row in proven
                    if row.get(
                        "exact_retention"
                    )
                    is not None
                ],
                default=None,
            )
        ),
        "mean_policy_retention_lower_bound": (
            _safe_mean(
                valid,
                "policy_retention_lower_bound",
            )
        ),
        "mean_policy_to_milp_incumbent_ratio": (
            _safe_mean(
                valid,
                "policy_to_milp_incumbent_ratio",
            )
        ),
        "milp_binary_variables": (
            first[
                "milp_binary_variables"
            ]
        ),
        "milp_continuous_variables": (
            first[
                "milp_continuous_variables"
            ]
        ),
        "milp_total_variables": (
            first[
                "milp_total_variables"
            ]
        ),
        "milp_approx_constraints": (
            first[
                "milp_approx_constraints"
            ]
        ),
    }
    return result


def _run_world(
    *,
    case_index: int,
    world_index: int,
    robots: int,
    tasks: int,
    seed: int,
    gene,
    geometry_timeout: float,
    path_timeout: float,
    policy_timeout: float,
    milp_time_limit: float,
) -> dict[str, Any]:
    config = scale_config(
        robots,
        tasks,
        preserve_spatial_density=True,
    )
    row: dict[str, Any] = {
        "case_index": case_index,
        "case": _case_label(
            robots,
            tasks,
        ),
        "world_index": (
            world_index
        ),
        "seed": seed,
        "robots": robots,
        "tasks": tasks,
        "tasks_per_robot": (
            tasks
            / robots
        ),
        "world_size": (
            config.world_size
        ),
        "obstacle_count": (
            config.obstacle_count
        ),
        "status": "started",
        "failure_stage": "",
        "error": "",
        **milp_problem_size(
            robots,
            tasks,
        ),
    }

    rss_before = _rss_mb()

    try:
        (
            world,
            geometry_seconds,
            path_seconds,
        ) = generate_world_timed(
            config,
            seed,
            geometry_timeout=(
                geometry_timeout
            ),
            path_timeout=(
                path_timeout
            ),
        )
    except BenchmarkStageError as exc:
        row.update(
            {
                "status": "failed",
                "failure_stage": (
                    exc.stage
                ),
                "error": str(
                    exc
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

    common_preprocess = (
        geometry_seconds
        + path_seconds
    )
    row.update(
        {
            "geometry_seconds": (
                geometry_seconds
            ),
            "path_precompute_seconds": (
                path_seconds
            ),
            "common_preprocess_seconds": (
                common_preprocess
            ),
            "path_table_entries": int(
                world.path_to_tasks.size
            ),
            "path_table_mb": float(
                world.path_to_tasks.nbytes
                / (
                    1024.0
                    * 1024.0
                )
            ),
        }
    )

    policy_start = (
        time.perf_counter()
    )
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
                    - policy_start
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
        - policy_start
    )
    evaluation = (
        evaluate_route_tail_plan(
            plan,
            world,
            config,
        )
    )
    policy_score = float(
        evaluation.time_optimality
    )

    milp_start = (
        time.perf_counter()
    )
    try:
        oracle = (
            solve_global_time_optimum(
                world,
                config,
                time_limit=(
                    milp_time_limit
                ),
            )
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_stage": (
                    "milp"
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "policy_seconds": (
                    policy_seconds
                ),
                "policy_time_optimality": (
                    policy_score
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

    milp_total_seconds = (
        time.perf_counter()
        - milp_start
    )
    milp_formulation_seconds = (
        max(
            0.0,
            milp_total_seconds
            - oracle.solve_seconds,
        )
    )

    row.update(
        {
            "status": "ok",
            "policy_seconds": (
                policy_seconds
            ),
            "policy_time_optimality": (
                policy_score
            ),
            "policy_completed_tasks": (
                evaluation.completed_tasks
            ),
            "policy_end_to_end_seconds": (
                common_preprocess
                + policy_seconds
            ),
            "milp_optimal": (
                oracle.optimal
            ),
            "milp_status": (
                oracle.status
            ),
            "milp_message": (
                oracle.message
            ),
            "milp_incumbent_score": (
                oracle.time_optimality
            ),
            "milp_upper_bound_score": (
                oracle.time_optimality_upper_bound
            ),
            "milp_completed_tasks": (
                oracle.completed_tasks
            ),
            "milp_gap": (
                oracle.mip_gap
            ),
            "milp_node_count": (
                oracle.mip_node_count
            ),
            "milp_solver_seconds": (
                oracle.solve_seconds
            ),
            "milp_formulation_seconds": (
                milp_formulation_seconds
            ),
            "milp_total_seconds": (
                milp_total_seconds
            ),
            "milp_end_to_end_seconds": (
                common_preprocess
                + milp_total_seconds
            ),
            "milp_over_policy_time_ratio": (
                milp_total_seconds
                / max(
                    policy_seconds,
                    EPS,
                )
            ),
            "rss_before_mb": (
                rss_before
            ),
            "rss_peak_mb": (
                _rss_mb()
            ),
            **gap_metrics(
                policy_score=(
                    policy_score
                ),
                milp_incumbent_score=(
                    oracle.time_optimality
                ),
                milp_upper_bound_score=(
                    oracle.time_optimality_upper_bound
                ),
                optimal=oracle.optimal,
            ),
        }
    )
    return row


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
            "gene_mrta_v116_milp_policy_"
            f"{stamp}_seed{args.seed_base}"
        )
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    jsonl_path = (
        run_dir
        / "per_world.jsonl"
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
                f"MILP_POLICY "
                f"{robots}R/{tasks}T "
                f"WORLD {world_index + 1}/"
                f"{args.worlds_per_case} "
                f"seed={seed}",
                flush=True,
            )

            row = _run_world(
                case_index=(
                    case_index
                ),
                world_index=(
                    world_index
                ),
                robots=robots,
                tasks=tasks,
                seed=seed,
                gene=gene,
                geometry_timeout=(
                    args.geometry_timeout
                ),
                path_timeout=(
                    args.path_timeout
                ),
                policy_timeout=(
                    args.policy_timeout
                ),
                milp_time_limit=(
                    args.milp_time_limit
                ),
            )

            rows.append(
                row
            )
            case_rows.append(
                row
            )
            with jsonl_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
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
            args.stop_after_zero_optimal
            and summary[
                "milp_optimal_worlds"
            ] == 0
        ):
            print(
                "STOP_LADDER="
                "zero MILP proven-optimal worlds",
                flush=True,
            )
            break

    _write_csv(
        run_dir
        / "milp_policy_results.csv",
        rows,
    )
    _write_csv(
        run_dir
        / "milp_policy_case_summary.csv",
        summaries,
    )

    payload = {
        "experiment": (
            "gene_mrta_v116_milp_vs_policy_scaling"
        ),
        "status": "completed",
        "seed_base": (
            args.seed_base
        ),
        "milp_time_limit_seconds": (
            args.milp_time_limit
        ),
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
        "objective": (
            "T=(1/N)*sum_completed"
            "(1-finish_time/H)"
        ),
        "exact_gap_rule": (
            "Exact Policy-vs-optimum gap is "
            "reported only when MILP optimal=True."
        ),
        "case_summaries": (
            summaries
        ),
        "guardrails": [
            (
                "same World and A* path table "
                "for Policy and MILP"
            ),
            (
                "shared preprocessing timed "
                "separately"
            ),
            (
                "non-optimal MILP incumbents "
                "are not labelled optimum"
            ),
            "no Policy retraining",
            "116M dedicated seeds",
            "99M untouched",
        ],
    }
    (
        run_dir
        / "milp_policy_summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "V116_MILP_POLICY_FINISHED"
    )
    print(
        f"FROZEN_GENE={gene_id}"
    )
    print(
        f"PARAMETERS="
        f"{gene.vector_data.size}"
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
            "2x10,3x15,4x20"
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
        default=116_000_000,
    )
    p.add_argument(
        "--milp-time-limit",
        type=float,
        default=60.0,
    )
    p.add_argument(
        "--geometry-timeout",
        type=float,
        default=60.0,
    )
    p.add_argument(
        "--path-timeout",
        type=float,
        default=120.0,
    )
    p.add_argument(
        "--policy-timeout",
        type=float,
        default=60.0,
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
        "--stop-after-zero-optimal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "gene_mrta_v116_milp_policy"
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
