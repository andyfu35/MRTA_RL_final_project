from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import resource
import signal
import time
from typing import Any, Iterator

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    World,
    _generate_obstacles,
    build_world,
)
from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v113.evolve import (
    AXES,
    GeneRecord,
    _active_parent_ids,
    _best_by_axis,
)
from marl2d.gene_mrta_v113.robust_metrics import _future_graph_masses
from marl2d.gene_mrta_v113.route_tail import (
    RouteTailPlan,
    evaluate_route_tail_plan,
    plan_route_tails,
)


EPS = 1e-12
BASE_ROBOTS = 4
BASE_TASKS = 20
BASE_WORLD_SIZE = 100.0
BASE_OBSTACLES = 10


class BenchmarkTimeout(RuntimeError):
    pass


class BenchmarkStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage


@contextmanager
def _time_limit(
    seconds: float,
    label: str,
) -> Iterator[None]:
    if (
        seconds <= 0.0
        or not hasattr(
            signal,
            "SIGALRM",
        )
        or not hasattr(
            signal,
            "setitimer",
        )
    ):
        yield
        return

    previous = signal.getsignal(
        signal.SIGALRM
    )

    def handler(
        _signum,
        _frame,
    ):
        raise BenchmarkTimeout(
            f"{label} exceeded "
            f"{seconds:.1f}s"
        )

    signal.signal(
        signal.SIGALRM,
        handler,
    )
    signal.setitimer(
        signal.ITIMER_REAL,
        seconds,
    )
    try:
        yield
    finally:
        signal.setitimer(
            signal.ITIMER_REAL,
            0.0,
        )
        signal.signal(
            signal.SIGALRM,
            previous,
        )


def _rss_mb() -> float:
    value = float(
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
    )
    # macOS reports bytes, Linux reports KiB.
    if platform.system() == "Darwin":
        return value / (
            1024.0 * 1024.0
        )
    return value / 1024.0


def scale_config(
    robots: int,
    tasks: int,
    *,
    base: EnvConfig | None = None,
    preserve_spatial_density: bool = True,
) -> EnvConfig:
    if robots <= 0 or tasks <= 0:
        raise ValueError(
            "robots/tasks must be positive"
        )
    base = (
        EnvConfig()
        if base is None
        else base
    )

    if preserve_spatial_density:
        area_scale = (
            float(robots)
            / float(BASE_ROBOTS)
        )
        world_size = (
            BASE_WORLD_SIZE
            * math.sqrt(
                area_scale
            )
        )
        obstacle_count = max(
            0,
            int(
                round(
                    BASE_OBSTACLES
                    * area_scale
                )
            ),
        )
    else:
        world_size = (
            BASE_WORLD_SIZE
        )
        obstacle_count = (
            BASE_OBSTACLES
        )

    return replace(
        base,
        world_size=float(
            world_size
        ),
        num_robots=int(
            robots
        ),
        num_tasks=int(
            tasks
        ),
        obstacle_count=int(
            obstacle_count
        ),
    )


def _load_frozen_gene(
    checkpoint: Path,
    *,
    archive_size: int,
    hybrid_limit: int,
    threshold: float,
) -> tuple[
    str,
    RouteTailDirectGene,
    dict[str, float],
    dict[str, float],
]:
    data = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )
    records = {
        str(item["record_id"]): (
            GeneRecord.from_dict(
                item
            )
        )
        for item in data.get(
            "records",
            []
        )
    }
    if not records:
        raise RuntimeError(
            "No V1.13 records found"
        )

    active_ids, _ = (
        _active_parent_ids(
            records,
            archive_size,
            hybrid_limit,
        )
    )
    best = _best_by_axis(
        records,
        active_ids,
    )

    candidates = []
    for record_id in active_ids:
        record = records[
            record_id
        ]
        retention = {
            axis: (
                float(
                    record.scores[
                        axis
                    ]
                )
                / max(
                    float(
                        best[
                            axis
                        ]
                    ),
                    EPS,
                )
            )
            for axis in AXES
        }
        if all(
            value
            >= threshold
            for value
            in retention.values()
        ):
            candidates.append(
                (
                    min(
                        retention.values()
                    ),
                    float(
                        np.mean(
                            list(
                                retention.values()
                            )
                        )
                    ),
                    record_id,
                    record,
                    retention,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No active four-capability "
            "V1.13 Gene meets the frozen "
            f"{threshold:.3f} threshold"
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
        reverse=True,
    )
    (
        _min_ret,
        _mean_ret,
        record_id,
        record,
        retention,
    ) = candidates[0]

    return (
        record_id,
        RouteTailDirectGene.from_v18(
            record.gene
        ),
        {
            axis: float(
                record.scores[
                    axis
                ]
            )
            for axis in AXES
        },
        retention,
    )


def _sample_geometry(
    config: EnvConfig,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(
        seed
    )
    robots = rng.uniform(
        0.0,
        config.world_size,
        size=(
            config.num_robots,
            2,
        ),
    )
    tasks = rng.uniform(
        0.0,
        config.world_size,
        size=(
            config.num_tasks,
            2,
        ),
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
    protected = np.vstack(
        [
            robots,
            tasks,
        ]
    )
    obstacles = _generate_obstacles(
        config,
        rng,
        protected,
    )
    batteries = rng.uniform(
        config.initial_battery_min,
        config.initial_battery_max,
        size=config.num_robots,
    )
    return (
        robots,
        batteries,
        tasks,
        services,
        priorities,
        deadlines,
        obstacles,
    )


def generate_world_timed(
    config: EnvConfig,
    seed: int,
    *,
    geometry_timeout: float,
    path_timeout: float,
) -> tuple[
    World,
    float,
    float,
]:
    start = time.perf_counter()
    try:
        with _time_limit(
            geometry_timeout,
            "geometry generation",
        ):
            (
                robots,
                batteries,
                tasks,
                services,
                priorities,
                deadlines,
                obstacles,
            ) = _sample_geometry(
                config,
                seed,
            )
    except Exception as exc:
        raise BenchmarkStageError(
            "geometry",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    geometry_seconds = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()
    try:
        with _time_limit(
            path_timeout,
            "path precomputation",
        ):
            world = build_world(
                config,
                robots,
                batteries,
                tasks,
                services,
                priorities,
                deadlines,
                obstacles,
            )
    except Exception as exc:
        raise BenchmarkStageError(
            "path",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    path_seconds = (
        time.perf_counter()
        - start
    )

    return (
        world,
        geometry_seconds,
        path_seconds,
    )


def _robust_from_plan(
    plan: RouteTailPlan,
    world: World,
    config: EnvConfig,
) -> tuple[
    float,
    float,
]:
    R = config.num_robots
    N = config.num_tasks

    tail_node_ids = np.arange(
        R,
        dtype=np.int64,
    )
    tail_times = np.zeros(
        R,
        dtype=np.float64,
    )
    battery_remaining = (
        world.robot_initial_batteries.copy()
    )
    task_available = np.ones(
        N,
        dtype=bool,
    )

    continuation_scores: list[
        float
    ] = []
    reserve_scores: list[
        float
    ] = []

    for item in (
        plan.selection_sequence
    ):
        (
            option_before,
            reserve_before,
        ) = _future_graph_masses(
            world=world,
            config=config,
            tail_node_ids=(
                tail_node_ids
            ),
            tail_times=tail_times,
            battery_remaining=(
                battery_remaining
            ),
            task_available=(
                task_available
            ),
        )

        utility = (
            1.0
            - float(
                np.clip(
                    item.finish_time
                    / max(
                        config.episode_time,
                        EPS,
                    ),
                    0.0,
                    1.0,
                )
            )
        )
        event_utility = (
            utility
            / max(
                N,
                1,
            )
        )

        robot = int(
            item.robot
        )
        task = int(
            item.task
        )
        tail_node_ids[
            robot
        ] = (
            R + task
        )
        tail_times[
            robot
        ] = float(
            item.finish_time
        )
        battery_remaining[
            robot
        ] = max(
            0.0,
            battery_remaining[
                robot
            ]
            - float(
                item.energy_used
            ),
        )
        task_available[
            task
        ] = False

        (
            option_after,
            reserve_after,
        ) = _future_graph_masses(
            world=world,
            config=config,
            tail_node_ids=(
                tail_node_ids
            ),
            tail_times=tail_times,
            battery_remaining=(
                battery_remaining
            ),
            task_available=(
                task_available
            ),
        )

        if option_before > EPS:
            continuation_scores.append(
                float(
                    np.clip(
                        (
                            event_utility
                            + option_after
                        )
                        / option_before,
                        0.0,
                        1.0,
                    )
                )
            )

        if reserve_before > EPS:
            reserve_scores.append(
                float(
                    np.clip(
                        (
                            1.0
                            / max(
                                N,
                                1,
                            )
                            + reserve_after
                        )
                        / reserve_before,
                        0.0,
                        1.0,
                    )
                )
            )

    continuation = (
        float(
            np.mean(
                continuation_scores
            )
        )
        if continuation_scores
        else 1.0
    )
    reserve = (
        float(
            np.mean(
                reserve_scores
            )
        )
        if reserve_scores
        else 1.0
    )
    return (
        continuation,
        reserve,
    )


def _initial_distance_stats(
    world: World,
    config: EnvConfig,
) -> dict[str, float]:
    delta = (
        world.robot_positions[
            :, None, :
        ]
        - world.task_positions[
            None, :, :
        ]
    )
    euclidean = np.linalg.norm(
        delta,
        axis=-1,
    )
    nearest = np.min(
        euclidean,
        axis=0,
    )
    return {
        "mean_nearest_robot_distance": (
            float(
                np.mean(
                    nearest
                )
            )
        ),
        "median_nearest_robot_distance": (
            float(
                np.median(
                    nearest
                )
            )
        ),
        "mean_nearest_robot_distance_normalized": (
            float(
                np.mean(
                    nearest
                    / max(
                        config.diagonal,
                        EPS,
                    )
                )
            )
        ),
    }


def _pair_slot_count(
    robots: int,
    tasks: int,
    decoder_steps: int,
) -> int:
    steps = min(
        max(
            int(
                decoder_steps
            ),
            0,
        ),
        tasks,
    )
    return int(
        robots
        * (
            steps
            * tasks
            - (
                steps
                * (
                    steps - 1
                )
                // 2
            )
        )
    )


def _case_label(
    robots: int,
    tasks: int,
) -> str:
    return (
        f"{robots}R_{tasks}T"
    )


def _parse_cases(
    value: str,
) -> list[
    tuple[int, int]
]:
    result = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(
                f"Invalid case: {item}"
            )
        r_text, t_text = (
            item.split(
                "x",
                1,
            )
        )
        robots = int(
            r_text
        )
        tasks = int(
            t_text
        )
        if robots <= 0 or tasks <= 0:
            raise ValueError(
                f"Invalid case: {item}"
            )
        result.append(
            (
                robots,
                tasks,
            )
        )
    if not result:
        raise ValueError(
            "No scaling cases"
        )
    return result


def _world_row(
    *,
    case_index: int,
    world_index: int,
    robots: int,
    tasks: int,
    seed: int,
    config: EnvConfig,
    gene: RouteTailDirectGene,
    geometry_timeout: float,
    path_timeout: float,
    policy_timeout: float,
) -> dict[str, Any]:
    label = _case_label(
        robots,
        tasks,
    )
    row: dict[str, Any] = {
        "case_index": case_index,
        "case": label,
        "world_index": (
            world_index
        ),
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
        "obstacle_count": (
            config.obstacle_count
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
                "rss_peak_mb": (
                    _rss_mb()
                ),
                "rss_before_mb": (
                    rss_before
                ),
            }
        )
        return row
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_stage": (
                    "world_unknown"
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "rss_peak_mb": (
                    _rss_mb()
                ),
                "rss_before_mb": (
                    rss_before
                ),
            }
        )
        return row

    row.update(
        _initial_distance_stats(
            world,
            config,
        )
    )
    row[
        "geometry_seconds"
    ] = geometry_seconds
    row[
        "path_precompute_seconds"
    ] = path_seconds
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
                "rss_peak_mb": (
                    _rss_mb()
                ),
                "rss_before_mb": (
                    rss_before
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

    robust_start = (
        time.perf_counter()
    )
    (
        continuation,
        reserve,
    ) = _robust_from_plan(
        plan,
        world,
        config,
    )
    robust_seconds = (
        time.perf_counter()
        - robust_start
    )

    queue_depths = np.asarray(
        [
            len(route)
            for route in plan.routes
        ],
        dtype=np.float64,
    )
    decoder_steps = len(
        plan.selection_sequence
    )
    pair_slots = _pair_slot_count(
        robots,
        tasks,
        decoder_steps,
    )

    row.update(
        {
            "status": "ok",
            "policy_seconds": (
                policy_seconds
            ),
            "robust_metric_seconds": (
                robust_seconds
            ),
            "total_core_seconds": (
                geometry_seconds
                + path_seconds
                + policy_seconds
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
            "continuation_preservation": (
                continuation
            ),
            "fleet_option_reserve": (
                reserve
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
            "stopped_by_policy": (
                plan.stopped_by_policy
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


def _safe_mean(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]
    return (
        float(
            np.mean(
                values
            )
        )
        if values
        else None
    )


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
        "geometry_seconds",
        "path_precompute_seconds",
        "policy_seconds",
        "total_core_seconds",
        "path_table_mb",
        "rss_peak_mb",
        "decoder_steps",
        "assignment_fraction",
        "completion",
        "raw_time_utility",
        "continuation_preservation",
        "fleet_option_reserve",
        "balance",
        "mean_queue_depth",
        "max_queue_depth",
        "mean_nearest_robot_distance",
        "mean_nearest_robot_distance_normalized",
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
        "obstacle_count": first[
            "obstacle_count"
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
    rows: list[
        dict[str, Any]
    ],
) -> None:
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
        for row in rows:
            writer.writerow(
                row
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
            "gene_mrta_v115_scaling_"
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
            preserve_spatial_density=(
                args.preserve_spatial_density
            ),
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
                f"CASE {robots}R/{tasks}T "
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
                geometry_timeout=(
                    args.geometry_timeout
                ),
                path_timeout=(
                    args.path_timeout
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
        / "scaling_results.csv",
        rows,
    )
    _write_csv(
        run_dir
        / "scaling_case_summary.csv",
        summaries,
    )
    _write_csv(
        run_dir
        / "timing_breakdown.csv",
        [
            {
                key: row.get(
                    key
                )
                for key in (
                    "case",
                    "world_index",
                    "seed",
                    "status",
                    "failure_stage",
                    "geometry_seconds",
                    "path_precompute_seconds",
                    "policy_seconds",
                    "robust_metric_seconds",
                    "total_core_seconds",
                    "decoder_steps",
                    "pair_slots_scored",
                    "policy_seconds_per_decoder_step",
                    "policy_seconds_per_pair_slot",
                )
            }
            for row in rows
        ],
    )
    _write_csv(
        run_dir
        / "memory_breakdown.csv",
        [
            {
                key: row.get(
                    key
                )
                for key in (
                    "case",
                    "world_index",
                    "seed",
                    "status",
                    "path_table_entries",
                    "path_table_mb",
                    "rss_before_mb",
                    "rss_peak_mb",
                )
            }
            for row in rows
        ],
    )

    payload = {
        "experiment": (
            "gene_mrta_v115_zero_shot_scaling"
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
        "preserve_spatial_density": (
            args.preserve_spatial_density
        ),
        "map_scaling_formula": (
            "L=100*sqrt(R/4)"
            if args.preserve_spatial_density
            else "L=100 fixed"
        ),
        "obstacle_scaling_formula": (
            "count=round(10*R/4)"
            if args.preserve_spatial_density
            else "count=10 fixed"
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
            "selection": (
                "max min-retention among active "
                "four-capability V1.13 genes"
            ),
        },
        "timeouts_seconds": {
            "geometry": (
                args.geometry_timeout
            ),
            "path": (
                args.path_timeout
            ),
            "policy": (
                args.policy_timeout
            ),
        },
        "case_summaries": (
            summaries
        ),
        "guardrails": [
            "zero-shot: no scale-specific retraining",
            "99M untouched",
            "no exact MILP claim at large scale",
            (
                "distance normalization remains the "
                "original world-diagonal normalization"
            ),
            (
                "path/A* failure is reported separately "
                "from Policy failure"
            ),
        ],
    }

    (
        run_dir
        / "scaling_summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "V115_SCALING_FINISHED"
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
            "4x20,8x40,16x80"
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
        default=115_000_000,
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
        default=120.0,
    )
    p.add_argument(
        "--preserve-spatial-density",
        action=argparse.BooleanOptionalAction,
        default=True,
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
            "gene_mrta_v115_scaling"
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
