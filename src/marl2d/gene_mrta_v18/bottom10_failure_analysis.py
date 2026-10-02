from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v17.failure_trace import (
    _direct_trace as _v17_trace,
    _matcher_trace,
)
from marl2d.gene_mrta_v18.global_time_test import (
    _load_v16to,
    _load_v17,
    _load_v18,
)

from .rollout import (
    FEATURE_NAMES,
    _build_observations,
    rollout_direct_gene,
)


DEFAULT_PUBLICATION_RESULT = (
    "runs/gene_mrta_v18_publication_final_100/"
    "publication_final_100.json"
)
DEFAULT_V18_RUN = (
    "runs/gene_mrta_v18t_scale/"
    "gene_mrta_v18t_scale_20261001_223157_seed7"
)
DEFAULT_V17_RUN = (
    "runs/gene_mrta_v17t_scale/"
    "gene_mrta_v17t_scale_20261001_105828_seed7"
)
DEFAULT_V16TO_RUN = (
    "runs/gene_mrta_v16to/"
    "gene_mrta_v16to_20260930_141419_seed7"
)


def _top_pairs(
    values: np.ndarray,
    eligible: np.ndarray,
    path_lengths: np.ndarray,
    service_times: np.ndarray,
    now: float,
    speed: float,
    *,
    limit: int = 8,
) -> list[dict[str, float | int]]:
    ids = np.argwhere(
        eligible & np.isfinite(values)
    )
    rows: list[dict[str, float | int]] = []

    for robot, task in ids.tolist():
        path_d = float(
            path_lengths[robot, task]
        )
        finish = (
            now
            + path_d / speed
            + float(service_times[task])
        )
        rows.append(
            {
                "robot": int(robot),
                "task": int(task),
                "score": float(
                    values[robot, task]
                ),
                "path": path_d,
                "finish": float(finish),
            }
        )

    rows.sort(
        key=lambda item: float(
            item["score"]
        ),
        reverse=True,
    )
    return rows[:limit]


def _future_graph_snapshot(
    *,
    world,
    config: EnvConfig,
    now: float,
    current_node_ids: np.ndarray,
    battery_remaining: np.ndarray,
    busy_until: np.ndarray,
    task_available: np.ndarray,
) -> dict[str, object]:
    R = config.num_robots
    N = config.num_tasks

    earliest_start = np.maximum(
        float(now),
        np.asarray(
            busy_until,
            dtype=np.float64,
        ),
    )
    paths = world.path_to_tasks[
        current_node_ids,
        :,
    ]
    finishes = (
        earliest_start[:, None]
        + paths / config.robot_speed
        + world.task_service_times[
            None, :
        ]
    )
    energy = (
        paths
        * config.energy_per_distance
    )

    feasible = (
        task_available[None, :]
        & np.isfinite(paths)
        & (
            finishes
            <= config.episode_time + 1e-12
        )
        & (
            energy
            <= battery_remaining[:, None]
            + 1e-12
        )
    )

    utility = np.where(
        feasible,
        1.0
        - np.clip(
            finishes
            / max(
                config.episode_time,
                1e-12,
            ),
            0.0,
            1.0,
        ),
        0.0,
    )

    task_degree = np.sum(
        feasible,
        axis=0,
    ).astype(int)
    robot_degree = np.sum(
        feasible,
        axis=1,
    ).astype(int)

    available_ids = np.flatnonzero(
        task_available
    )
    available_count = int(
        available_ids.size
    )

    if available_count:
        available_degree = task_degree[
            available_ids
        ]
        reachable = int(
            np.sum(
                available_degree > 0
            )
        )
        singleton = int(
            np.sum(
                available_degree == 1
            )
        )
        stranded = int(
            np.sum(
                available_degree == 0
            )
        )
        mean_task_degree = float(
            np.mean(
                available_degree
            )
        )
    else:
        reachable = 0
        singleton = 0
        stranded = 0
        mean_task_degree = 0.0

    best_task_utility = np.max(
        utility,
        axis=0,
    )
    best_task_utility = np.where(
        task_available,
        best_task_utility,
        0.0,
    )

    option_mass = float(
        np.sum(
            best_task_utility
        )
        / max(N, 1)
    )

    edge_count = int(
        np.sum(feasible)
    )
    edge_density = (
        float(
            edge_count
            / max(
                R * available_count,
                1,
            )
        )
        if available_count
        else 0.0
    )

    return {
        "available_tasks": (
            available_count
        ),
        "edge_count": edge_count,
        "edge_density": edge_density,
        "reachable_tasks": reachable,
        "reachable_task_ratio": (
            float(
                reachable
                / max(
                    available_count,
                    1,
                )
            )
            if available_count
            else 1.0
        ),
        "singleton_tasks": singleton,
        "stranded_tasks": stranded,
        "mean_task_degree": (
            mean_task_degree
        ),
        "robot_option_counts": (
            robot_degree.tolist()
        ),
        "task_owner_counts": (
            task_degree.tolist()
        ),
        "option_mass": option_mass,
        "best_task_time_utility": (
            best_task_utility.tolist()
        ),
    }


def _rank_fraction(
    values: np.ndarray,
    active: np.ndarray,
    robot: int,
    task: int,
) -> float:
    pool = np.asarray(
        values[active],
        dtype=np.float64,
    )
    if pool.size <= 1:
        return 1.0

    selected = float(
        values[robot, task]
    )
    strictly_better = int(
        np.sum(
            pool > selected + 1e-12
        )
    )
    return float(
        1.0
        - strictly_better
        / max(
            pool.size - 1,
            1,
        )
    )


def _v18_trace(
    gene,
    world,
    config: EnvConfig,
) -> dict[str, object]:
    R = config.num_robots
    N = config.num_tasks

    robot_positions = (
        world.robot_positions.copy()
    )
    battery_remaining = (
        world.robot_initial_batteries.copy()
    )
    current_node_ids = np.arange(
        R,
        dtype=np.int64,
    )
    task_available = np.ones(
        N,
        dtype=bool,
    )
    busy_until = np.zeros(
        R,
        dtype=np.float64,
    )
    robot_workloads = np.zeros(
        R,
        dtype=np.float64,
    )

    events: list[
        dict[str, object]
    ] = []

    for event_idx in range(
        N + R + 2
    ):
        if not np.any(
            task_available
        ):
            break

        (
            now,
            free,
            observations,
            eligible,
            path_lengths,
            _euclidean,
        ) = _build_observations(
            world=world,
            config=config,
            robot_positions=(
                robot_positions
            ),
            current_node_ids=(
                current_node_ids
            ),
            battery_remaining=(
                battery_remaining
            ),
            robot_workloads=(
                robot_workloads
            ),
            busy_until=busy_until,
            task_available=(
                task_available
            ),
        )

        if (
            now
            >= config.episode_time
            - 1e-12
        ):
            break

        graph_before = (
            _future_graph_snapshot(
                world=world,
                config=config,
                now=now,
                current_node_ids=(
                    current_node_ids
                ),
                battery_remaining=(
                    battery_remaining
                ),
                busy_until=(
                    busy_until
                ),
                task_available=(
                    task_available
                ),
            )
        )

        duration = (
            path_lengths
            / config.robot_speed
            + world.task_service_times[
                None, :
            ]
        )
        finishes = (
            now + duration
        )
        immediate_utility = np.where(
            eligible,
            1.0
            - np.clip(
                finishes
                / max(
                    config.episode_time,
                    1e-12,
                ),
                0.0,
                1.0,
            ),
            -np.inf,
        )

        row_open = free.copy()
        col_open = (
            task_available.copy()
        )
        selected: list[
            tuple[int, int]
        ] = []
        decode_steps: list[
            dict[str, object]
        ] = []

        for step in range(
            int(
                np.sum(
                    row_open
                )
            )
        ):
            logits, stop_logit = (
                gene.action_logits(
                    observations,
                    eligible,
                    row_open,
                    col_open,
                    step,
                )
            )

            active = (
                eligible
                & row_open[:, None]
                & col_open[None, :]
            )
            flat = int(
                np.argmax(
                    logits
                )
            )
            best = float(
                logits.flat[flat]
            )
            stop = (
                not np.isfinite(best)
                or stop_logit >= best
            )

            if np.isfinite(best):
                robot = (
                    flat // N
                )
                task = (
                    flat % N
                )
                obs = {
                    name: float(
                        observations[
                            robot,
                            task,
                            idx,
                        ]
                    )
                    for idx, name
                    in enumerate(
                        FEATURE_NAMES
                    )
                }
                immediate_rank = (
                    _rank_fraction(
                        immediate_utility,
                        active,
                        robot,
                        task,
                    )
                )
            else:
                robot = -1
                task = -1
                obs = None
                immediate_rank = None

            decode_steps.append(
                {
                    "step": step,
                    "stop_logit": float(
                        stop_logit
                    ),
                    "best_pair_logit": (
                        best
                    ),
                    "pair_minus_stop": (
                        float(
                            best
                            - stop_logit
                        )
                        if np.isfinite(
                            best
                        )
                        else None
                    ),
                    "best_pair": (
                        [
                            int(robot),
                            int(task),
                        ]
                        if np.isfinite(
                            best
                        )
                        else None
                    ),
                    "best_pair_observation": (
                        obs
                    ),
                    "selected_immediate_time_rank": (
                        immediate_rank
                    ),
                    "top_policy_pairs": (
                        _top_pairs(
                            logits,
                            active,
                            path_lengths,
                            world.task_service_times,
                            now,
                            config.robot_speed,
                        )
                    ),
                    "top_immediate_time_pairs": (
                        _top_pairs(
                            immediate_utility,
                            active,
                            path_lengths,
                            world.task_service_times,
                            now,
                            config.robot_speed,
                        )
                    ),
                    "decision": (
                        "STOP"
                        if stop
                        else (
                            f"R{robot}->T{task}"
                        )
                    ),
                }
            )

            if stop:
                break

            selected.append(
                (
                    robot,
                    task,
                )
            )
            row_open[robot] = False
            col_open[task] = False

        before_battery = (
            battery_remaining.copy()
        )
        before_workload = (
            robot_workloads.copy()
        )

        realized_utility = 0.0
        details: list[
            dict[str, object]
        ] = []
        matched = np.zeros(
            R,
            dtype=bool,
        )

        for robot, task in selected:
            path_d = float(
                path_lengths[
                    robot,
                    task,
                ]
            )
            service = float(
                world.task_service_times[
                    task
                ]
            )
            energy = (
                path_d
                * config.energy_per_distance
            )
            finish = (
                now
                + path_d
                / config.robot_speed
                + service
            )
            utility = float(
                1.0
                - np.clip(
                    finish
                    / max(
                        config.episode_time,
                        1e-12,
                    ),
                    0.0,
                    1.0,
                )
            )
            realized_utility += (
                utility
                / max(N, 1)
            )

            selected_obs = {
                name: float(
                    observations[
                        robot,
                        task,
                        idx,
                    ]
                )
                for idx, name
                in enumerate(
                    FEATURE_NAMES
                )
            }

            details.append(
                {
                    "robot": int(
                        robot
                    ),
                    "task": int(
                        task
                    ),
                    "path": path_d,
                    "service": service,
                    "finish": float(
                        finish
                    ),
                    "time_utility": (
                        utility
                    ),
                    "battery_before": float(
                        before_battery[
                            robot
                        ]
                    ),
                    "battery_after": float(
                        max(
                            0.0,
                            before_battery[
                                robot
                            ]
                            - energy,
                        )
                    ),
                    "observation": (
                        selected_obs
                    ),
                    "task_owner_count_before": int(
                        graph_before[
                            "task_owner_counts"
                        ][task]
                    ),
                }
            )

            busy_until[robot] = (
                finish
            )
            robot_positions[robot] = (
                world.task_positions[
                    task
                ]
            )
            current_node_ids[robot] = (
                R + task
            )
            battery_remaining[robot] = max(
                0.0,
                battery_remaining[
                    robot
                ]
                - energy,
            )
            task_available[task] = (
                False
            )
            robot_workloads[robot] += (
                path_d
                / config.robot_speed
                + service
            )
            matched[robot] = True

        unmatched_free = (
            free & ~matched
        )
        wait_until = None
        if np.any(
            unmatched_free
        ):
            future_times = (
                busy_until[
                    busy_until
                    > now + 1e-12
                ]
            )
            if (
                future_times.size
                > 0
            ):
                wait_until = float(
                    np.min(
                        future_times
                    )
                )
                busy_until[
                    unmatched_free
                ] = wait_until
            else:
                wait_until = float(
                    config.episode_time
                )
                busy_until[
                    unmatched_free
                ] = (
                    config.episode_time
                )

        graph_after = (
            _future_graph_snapshot(
                world=world,
                config=config,
                now=now,
                current_node_ids=(
                    current_node_ids
                ),
                battery_remaining=(
                    battery_remaining
                ),
                busy_until=(
                    busy_until
                ),
                task_available=(
                    task_available
                ),
            )
        )

        before_mass = float(
            graph_before[
                "option_mass"
            ]
        )
        after_mass = float(
            graph_after[
                "option_mass"
            ]
        )
        conserved = (
            realized_utility
            + after_mass
        )
        conservation_ratio = (
            conserved
            / before_mass
            if before_mass > 1e-12
            else 1.0
        )

        events.append(
            {
                "event": (
                    event_idx
                ),
                "time": float(
                    now
                ),
                "free_robots": (
                    np.flatnonzero(
                        free
                    ).astype(
                        int
                    ).tolist()
                ),
                "eligible_pairs": int(
                    np.sum(
                        eligible
                    )
                ),
                "graph_before": (
                    graph_before
                ),
                "decode_steps": (
                    decode_steps
                ),
                "assignments": [
                    [
                        int(r),
                        int(t),
                    ]
                    for r, t
                    in selected
                ],
                "assignment_details": (
                    details
                ),
                "realized_time_utility_normalized": (
                    float(
                        realized_utility
                    )
                ),
                "unmatched_free_robots": (
                    np.flatnonzero(
                        unmatched_free
                    ).astype(
                        int
                    ).tolist()
                ),
                "wait_until": (
                    wait_until
                ),
                "graph_after": (
                    graph_after
                ),
                "option_conservation_ratio": (
                    float(
                        conservation_ratio
                    )
                ),
                "reachable_ratio_delta": float(
                    graph_after[
                        "reachable_task_ratio"
                    ]
                    - graph_before[
                        "reachable_task_ratio"
                    ]
                ),
                "stranded_task_delta": int(
                    graph_after[
                        "stranded_tasks"
                    ]
                    - graph_before[
                        "stranded_tasks"
                    ]
                ),
            }
        )

    evaluation = (
        rollout_direct_gene(
            gene,
            world,
            config,
        ).evaluation.to_dict()
    )

    return {
        "method": "v18_direct",
        "evaluation": evaluation,
        "events": events,
    }


def _select_bottom_rows(
    publication: dict[str, object],
    count: int,
) -> list[dict[str, object]]:
    rows = [
        row
        for row in publication[
            "rows"
        ]
        if (
            bool(
                row.get(
                    "optimal"
                )
            )
            and row.get(
                "v18_retention"
            )
            is not None
        )
    ]
    rows.sort(
        key=lambda row: float(
            row[
                "v18_retention"
            ]
        )
    )
    return rows[:count]


def _world_mechanism_flags(
    *,
    row: dict[str, object],
    trace: dict[str, object],
) -> dict[str, object]:
    v18 = float(
        row["v18_retention"]
    )
    v17 = float(
        row["v17_retention"]
    )
    v16 = float(
        row["v16to_retention"]
    )
    hungarian = float(
        row["hungarian_retention"]
    )

    events = trace["events"]
    first_half = [
        event
        for event in events
        if float(
            event["time"]
        ) <= 25.0
    ]

    conservation = [
        float(
            event[
                "option_conservation_ratio"
            ]
        )
        for event in first_half
        if event[
            "graph_before"
        ][
            "option_mass"
        ]
        > 1e-12
    ]
    min_conservation = (
        min(conservation)
        if conservation
        else 1.0
    )

    reach_deltas = [
        float(
            event[
                "reachable_ratio_delta"
            ]
        )
        for event in first_half
    ]
    min_reach_delta = (
        min(reach_deltas)
        if reach_deltas
        else 0.0
    )

    selected_future_zero = 0
    selected_high_opportunity = 0
    low_immediate_rank = 0
    selected_total = 0

    for event in events:
        for detail in event[
            "assignment_details"
        ]:
            selected_total += 1
            obs = detail[
                "observation"
            ]
            if (
                float(
                    obs[
                        "self_future_reachability"
                    ]
                )
                <= 0.05
                and float(
                    detail["finish"]
                )
                < 35.0
            ):
                selected_future_zero += 1

            if float(
                obs[
                    "other_robot_opportunity_cost"
                ]
            ) >= 0.08:
                selected_high_opportunity += 1

        for step in event[
            "decode_steps"
        ]:
            if (
                step[
                    "decision"
                ]
                != "STOP"
                and step[
                    "selected_immediate_time_rank"
                ]
                is not None
                and float(
                    step[
                        "selected_immediate_time_rank"
                    ]
                )
                <= 0.35
            ):
                low_immediate_rank += 1

    hard_for_all = (
        max(
            v17,
            v16,
            hungarian,
        )
        < 0.92
    )
    consequence_regression = (
        v17 - v18 >= 0.03
    )
    comparator_advantage = (
        max(
            v16,
            hungarian,
        )
        - v18
        >= 0.03
    )
    continuation_collapse = (
        min_conservation < 0.80
        or min_reach_delta <= -0.20
        or selected_future_zero >= 1
    )
    fleet_reserve_risk = (
        selected_high_opportunity
        >= 1
    )
    immediate_future_imbalance = (
        low_immediate_rank
        >= 1
        and consequence_regression
    )

    tags: list[str] = []
    if hard_for_all:
        tags.append(
            "hard_for_all"
        )
    if consequence_regression:
        tags.append(
            "v18_regression_vs_v17"
        )
    if comparator_advantage:
        tags.append(
            "matcher_or_v16_advantage"
        )
    if continuation_collapse:
        tags.append(
            "continuation_collapse"
        )
    if fleet_reserve_risk:
        tags.append(
            "fleet_reserve_risk"
        )
    if immediate_future_imbalance:
        tags.append(
            "immediate_future_imbalance"
        )

    return {
        "tags": tags,
        "metrics": {
            "v18_retention": v18,
            "v17_retention": v17,
            "v16to_retention": v16,
            "hungarian_retention": (
                hungarian
            ),
            "v17_minus_v18": (
                v17 - v18
            ),
            "best_legacy_minus_v18": (
                max(
                    v16,
                    hungarian,
                )
                - v18
            ),
            "min_early_option_conservation": (
                min_conservation
            ),
            "min_early_reachable_ratio_delta": (
                min_reach_delta
            ),
            "selected_future_zero_count": (
                selected_future_zero
            ),
            "selected_high_opportunity_count": (
                selected_high_opportunity
            ),
            "low_immediate_rank_count": (
                low_immediate_rank
            ),
            "selected_assignment_count": (
                selected_total
            ),
        },
    }


def _candidate_axes(
    analyses: list[
        dict[str, object]
    ],
) -> list[
    dict[str, object]
]:
    counts: dict[str, int] = {}
    for item in analyses:
        for tag in item[
            "mechanism_flags"
        ][
            "tags"
        ]:
            counts[tag] = (
                counts.get(
                    tag,
                    0,
                )
                + 1
            )

    recommendations: list[
        dict[str, object]
    ] = []

    if counts.get(
        "continuation_collapse",
        0,
    ):
        recommendations.append(
            {
                "axis": (
                    "continuation_preservation"
                ),
                "triggered_worlds": (
                    counts[
                        "continuation_collapse"
                    ]
                ),
                "episode_signal": (
                    "mean clipped option-conservation ratio "
                    "(realized immediate T utility + post-decision "
                    "fleet option mass) / pre-decision fleet option mass"
                ),
                "purpose": (
                    "preserve task-chain value instead of maximizing "
                    "only immediate assignment quality"
                ),
            }
        )

    if counts.get(
        "fleet_reserve_risk",
        0,
    ):
        recommendations.append(
            {
                "axis": (
                    "fleet_option_reserve"
                ),
                "triggered_worlds": (
                    counts[
                        "fleet_reserve_risk"
                    ]
                ),
                "episode_signal": (
                    "reachable-task ratio, edge density, singleton-task "
                    "survival, and normalized fleet option mass after decisions"
                ),
                "purpose": (
                    "preserve scarce tasks and cross-robot flexibility"
                ),
            }
        )

    if counts.get(
        "immediate_future_imbalance",
        0,
    ):
        recommendations.append(
            {
                "axis": (
                    "immediate_time_capture"
                ),
                "triggered_worlds": (
                    counts[
                        "immediate_future_imbalance"
                    ]
                ),
                "episode_signal": (
                    "selected pair immediate-time utility rank among "
                    "currently feasible pairs"
                ),
                "purpose": (
                    "prevent consequence features from overvaluing future "
                    "options at the cost of strong immediate T utility"
                ),
            }
        )

    if counts.get(
        "hard_for_all",
        0,
    ):
        recommendations.append(
            {
                "axis": (
                    "hard_world_time_specialist"
                ),
                "triggered_worlds": (
                    counts[
                        "hard_for_all"
                    ]
                ),
                "episode_signal": (
                    "mean T/T* on the fixed 98M Bottom-10 development worlds"
                ),
                "purpose": (
                    "preserve genes specialized for intrinsically hard "
                    "world structures without scalarizing them into mean T"
                ),
            }
        )

    return recommendations


def _write_markdown(
    path: Path,
    *,
    analyses: list[
        dict[str, object]
    ],
    recommendations: list[
        dict[str, object]
    ],
) -> None:
    lines = [
        "# V1.8 Bottom-10 failure analysis",
        "",
        "The 98M publication set is now development/analysis data for V1.9.",
        "Any V1.9 final generalization claim must use a new untouched seed range.",
        "",
        "## Bottom-10",
        "",
        "| Rank | Seed | V1.8 | V1.7 | V1.6-T-O | Hungarian | Mechanism flags |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]

    for rank, item in enumerate(
        analyses,
        start=1,
    ):
        row = item[
            "publication_row"
        ]
        flags = ", ".join(
            item[
                "mechanism_flags"
            ][
                "tags"
            ]
        ) or "none"
        lines.append(
            f"| {rank} | {row['world_seed']} | "
            f"{100*float(row['v18_retention']):.3f}% | "
            f"{100*float(row['v17_retention']):.3f}% | "
            f"{100*float(row['v16to_retention']):.3f}% | "
            f"{100*float(row['hungarian_retention']):.3f}% | "
            f"{flags} |"
        )

    lines.extend(
        [
            "",
            "## Candidate V1.9 capability axes",
            "",
        ]
    )
    for rec in recommendations:
        lines.extend(
            [
                f"### {rec['axis']}",
                "",
                f"Triggered worlds: {rec['triggered_worlds']}",
                "",
                f"Signal: {rec['episode_signal']}",
                "",
                f"Purpose: {rec['purpose']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Per-world diagnostics",
            "",
        ]
    )

    for item in analyses:
        seed = item[
            "publication_row"
        ][
            "world_seed"
        ]
        metrics = item[
            "mechanism_flags"
        ][
            "metrics"
        ]
        lines.extend(
            [
                f"### Seed {seed}",
                "",
                (
                    "Flags: "
                    + (
                        ", ".join(
                            item[
                                "mechanism_flags"
                            ][
                                "tags"
                            ]
                        )
                        or "none"
                    )
                ),
                "",
                (
                    "min early option conservation = "
                    f"{metrics['min_early_option_conservation']:.4f}"
                ),
                (
                    "min early reachable-ratio delta = "
                    f"{metrics['min_early_reachable_ratio_delta']:+.4f}"
                ),
                (
                    "selected future-zero = "
                    f"{metrics['selected_future_zero_count']}, "
                    "high-opportunity = "
                    f"{metrics['selected_high_opportunity_count']}, "
                    "low-immediate-rank = "
                    f"{metrics['low_immediate_rank_count']}"
                ),
                "",
                "V1.8 timeline:",
                "",
            ]
        )

        for event in item[
            "v18_trace"
        ][
            "events"
        ]:
            assignments = (
                ", ".join(
                    f"R{r}->T{t}"
                    for r, t
                    in event[
                        "assignments"
                    ]
                )
                or "NONE"
            )
            lines.append(
                "- "
                f"t={float(event['time']):.3f} "
                f"assign=[{assignments}] "
                f"opt={float(event['option_conservation_ratio']):.3f} "
                f"reachΔ={float(event['reachable_ratio_delta']):+.3f} "
                f"strandedΔ={int(event['stranded_task_delta']):+d}"
            )

        lines.append("")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(
    args: argparse.Namespace,
) -> Path:
    publication_path = Path(
        args.publication_result
    )
    publication = json.loads(
        publication_path.read_text(
            encoding="utf-8"
        )
    )

    bottom = _select_bottom_rows(
        publication,
        args.bottom,
    )

    config = EnvConfig()
    v18_gene = _load_v18(
        Path(
            args.v18_run
        )
    )
    v17_gene = _load_v17(
        Path(
            args.v17_run
        )
    )
    v16to_gene = _load_v16to(
        Path(
            args.v16to_run
        )
    )

    analyses: list[
        dict[str, object]
    ] = []

    for rank, row in enumerate(
        bottom,
        start=1,
    ):
        seed = int(
            row[
                "world_seed"
            ]
        )
        world = generate_world(
            config,
            seed,
        )

        v18_trace = _v18_trace(
            v18_gene,
            world,
            config,
        )
        v17_trace = _v17_trace(
            v17_gene,
            world,
            config,
        )
        v16_trace = _matcher_trace(
            method=(
                "v16to_gene_greedy"
            ),
            world=world,
            config=config,
            score_mode="gene",
            matcher="greedy",
            gene=v16to_gene,
        )
        hungarian_trace = (
            _matcher_trace(
                method=(
                    "hungarian_path_time"
                ),
                world=world,
                config=config,
                score_mode=(
                    "path_time"
                ),
                matcher=(
                    "hungarian"
                ),
                gene=None,
            )
        )

        flags = (
            _world_mechanism_flags(
                row=row,
                trace=v18_trace,
            )
        )

        item = {
            "rank": rank,
            "publication_row": row,
            "mechanism_flags": (
                flags
            ),
            "v18_trace": (
                v18_trace
            ),
            "v17_trace": (
                v17_trace
            ),
            "v16to_trace": (
                v16_trace
            ),
            "hungarian_trace": (
                hungarian_trace
            ),
        }
        analyses.append(
            item
        )

        print(
            f"[{rank:02d}/{len(bottom):02d}] "
            f"seed={seed} "
            f"V18={100*float(row['v18_retention']):.3f}% "
            f"flags="
            + (
                ",".join(
                    flags[
                        "tags"
                    ]
                )
                or "none"
            )
        )

    recommendations = (
        _candidate_axes(
            analyses
        )
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "analysis_status": (
            "98M is development data "
            "for V1.9 from this point forward"
        ),
        "publication_result": str(
            publication_path
        ),
        "bottom_count": len(
            bottom
        ),
        "bottom_seeds": [
            int(
                item[
                    "publication_row"
                ][
                    "world_seed"
                ]
            )
            for item in analyses
        ],
        "analyses": analyses,
        "candidate_v19_axes": (
            recommendations
        ),
        "next_final_seed_policy": (
            "V1.9 final benchmark must use a new untouched "
            "range, planned as 99,000,000-99,000,099"
        ),
    }

    json_path = (
        output_dir
        / "bottom10_failure_analysis.json"
    )
    md_path = (
        output_dir
        / "bottom10_failure_analysis.md"
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_markdown(
        md_path,
        analyses=analyses,
        recommendations=(
            recommendations
        ),
    )

    print(
        "\nBOTTOM10_SEEDS="
        + ",".join(
            str(seed)
            for seed in payload[
                "bottom_seeds"
            ]
        )
    )
    print(
        "CANDIDATE_V19_AXES="
        + ",".join(
            str(
                rec["axis"]
            )
            for rec
            in recommendations
        )
    )
    print(
        f"ANALYSIS_JSON={json_path}"
    )
    print(
        f"ANALYSIS_MD={md_path}"
    )

    return json_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--publication-result",
        default=(
            DEFAULT_PUBLICATION_RESULT
        ),
    )
    p.add_argument(
        "--v18-run",
        default=DEFAULT_V18_RUN,
    )
    p.add_argument(
        "--v17-run",
        default=DEFAULT_V17_RUN,
    )
    p.add_argument(
        "--v16to-run",
        default=DEFAULT_V16TO_RUN,
    )
    p.add_argument(
        "--bottom",
        type=int,
        default=10,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "gene_mrta_v19_failure_analysis"
        ),
    )
    return p


def main() -> None:
    run(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
