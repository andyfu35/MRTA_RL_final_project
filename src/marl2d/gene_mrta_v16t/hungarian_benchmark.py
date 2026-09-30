from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .env import EnvConfig, Evaluation, World, generate_world, jain_fairness
from .gene import Gene


SCALAR_METRICS = (
    "completion",
    "efficiency",
    "priority_satisfaction",
    "deadline_satisfaction",
    "balance",
    "time_optimality",
    "route_efficiency",
    "completed_tasks",
    "completed_priority",
    "on_time_tasks",
    "total_travel",
    "total_euclidean_travel",
    "detour_ratio",
    "mean_initial_battery",
    "mean_final_battery",
    "battery_remaining_fraction",
    "energy_consumed",
    "battery_blocked_pair_events",
)


@dataclass
class RolloutTiming:
    decision_total_ns: int = 0
    matcher_ns: int = 0
    decision_count: int = 0

    def add_decision(self, total_ns: int, matcher_ns: int) -> None:
        self.decision_total_ns += int(total_ns)
        self.matcher_ns += int(matcher_ns)
        self.decision_count += 1


@dataclass
class RolloutResult:
    evaluation: Evaluation
    timing: RolloutTiming
    wall_ns: int
    local_hungarian_time_regret_sum: float = 0.0
    local_hungarian_time_regret_events: int = 0


def _greedy_match(
    scores: np.ndarray,
    eligible: np.ndarray,
    free: np.ndarray,
    task_available: np.ndarray,
) -> list[tuple[int, int]]:
    robot_count, task_count = scores.shape
    row_open = free.copy()
    col_open = task_available.copy()
    assignments: list[tuple[int, int]] = []

    for _ in range(robot_count):
        valid = eligible & row_open[:, None] & col_open[None, :]
        masked = np.where(valid, scores, -np.inf)
        flat_idx = int(np.argmax(masked))
        best = float(masked.flat[flat_idx])
        if not np.isfinite(best):
            break
        robot_idx = flat_idx // task_count
        task_idx = flat_idx % task_count
        assignments.append((robot_idx, task_idx))
        row_open[robot_idx] = False
        col_open[task_idx] = False

    return assignments


def _hungarian_match(
    scores: np.ndarray,
    eligible: np.ndarray,
    free: np.ndarray,
    task_available: np.ndarray,
) -> list[tuple[int, int]]:
    free_ids = np.flatnonzero(free)
    task_ids = np.flatnonzero(task_available)
    if free_ids.size == 0 or task_ids.size == 0:
        return []

    sub_scores = scores[np.ix_(free_ids, task_ids)]
    sub_eligible = eligible[np.ix_(free_ids, task_ids)]
    if not np.any(sub_eligible):
        return []

    feasible_scores = sub_scores[sub_eligible]
    max_abs = float(np.max(np.abs(feasible_scores))) if feasible_scores.size else 1.0
    # First maximize assignment cardinality, then maximize the requested score.
    # One extra feasible assignment must dominate any possible score tradeoff.
    cardinality_bonus = (2.0 * max_abs + 1.0) * (free_ids.size + 1)

    real_utility = np.where(
        sub_eligible,
        cardinality_bonus + sub_scores,
        -cardinality_bonus * (free_ids.size + 1),
    )

    # One private dummy column per free robot allows truly infeasible robots
    # to remain unmatched without forcing an invalid real assignment.
    dummy_utility = np.zeros((free_ids.size, free_ids.size), dtype=np.float64)
    utility = np.concatenate([real_utility, dummy_utility], axis=1)

    row_ind, col_ind = linear_sum_assignment(utility, maximize=True)
    assignments: list[tuple[int, int]] = []
    real_task_count = task_ids.size
    for local_row, local_col in zip(row_ind.tolist(), col_ind.tolist()):
        if local_col >= real_task_count:
            continue
        if not sub_eligible[local_row, local_col]:
            continue
        assignments.append(
            (int(free_ids[local_row]), int(task_ids[local_col]))
        )
    return assignments


def _build_scores(
    score_mode: str,
    gene: Gene | None,
    config: EnvConfig,
    now: float,
    robot_positions: np.ndarray,
    task_positions: np.ndarray,
    task_services: np.ndarray,
    task_priorities: np.ndarray,
    task_deadlines: np.ndarray,
    battery_remaining: np.ndarray,
    robot_workloads: np.ndarray,
    path_lengths: np.ndarray,
    pair_eligible: np.ndarray,
) -> np.ndarray:
    delta = robot_positions[:, None, :] - task_positions[None, :, :]
    euclidean = np.linalg.norm(delta, axis=-1)
    duration_path = path_lengths / config.robot_speed + task_services[None, :]

    if score_mode == "path_time":
        return -duration_path
    if score_mode == "priority_path_time":
        return task_priorities[None, :] / np.maximum(duration_path, 1e-12)
    if score_mode != "gene":
        raise ValueError(f"Unknown score_mode: {score_mode}")
    if gene is None:
        raise ValueError("gene score_mode requires a Gene")

    euclidean_norm = np.clip(euclidean / config.diagonal, 0.0, 1.0)
    path_norm = np.clip(path_lengths / config.path_cost_scale, 0.0, 1.0)
    path_norm = np.where(np.isfinite(path_norm), path_norm, 1.0)

    if config.service_time_range > 0.0:
        service_norm_base = np.clip(
            (task_services - config.service_time_min)
            / config.service_time_range,
            0.0,
            1.0,
        )
    else:
        service_norm_base = np.zeros_like(task_services)

    if config.priority_range > 0.0:
        priority_norm_base = np.clip(
            (task_priorities - config.priority_min) / config.priority_range,
            0.0,
            1.0,
        )
    else:
        priority_norm_base = np.zeros_like(task_priorities)

    deadline_remaining_base = np.clip(
        (task_deadlines - now) / max(config.episode_time, 1e-12),
        0.0,
        1.0,
    )
    battery_norm = np.clip(
        battery_remaining / config.battery_capacity,
        0.0,
        1.0,
    )[:, None]
    workload_norm = np.clip(
        robot_workloads / max(config.episode_time, 1e-12),
        0.0,
        1.0,
    )[:, None]

    other_is_closer = path_lengths[None, :, :] < path_lengths[:, None, :]
    other_is_eligible = pair_eligible[None, :, :]
    closer_eligible = (other_is_closer & other_is_eligible).sum(axis=1)
    eligible_count = pair_eligible.sum(axis=0)
    competition_denom = np.maximum(eligible_count - 1, 1)[None, :]
    competition = closer_eligible / competition_denom

    robot_count, task_count = path_lengths.shape
    observations = np.stack(
        [
            euclidean_norm,
            path_norm,
            np.broadcast_to(service_norm_base[None, :], (robot_count, task_count)),
            np.broadcast_to(priority_norm_base[None, :], (robot_count, task_count)),
            np.broadcast_to(
                deadline_remaining_base[None, :],
                (robot_count, task_count),
            ),
            np.broadcast_to(battery_norm, (robot_count, task_count)),
            np.broadcast_to(workload_norm, (robot_count, task_count)),
            competition,
        ],
        axis=-1,
    )
    return gene.bid(observations)


def rollout_world(
    world: World,
    config: EnvConfig,
    *,
    score_mode: str,
    matcher: str,
    gene: Gene | None = None,
) -> RolloutResult:
    wall_start = perf_counter_ns()

    robot_count = config.num_robots
    task_count = config.num_tasks

    robot_positions = world.robot_positions.copy()
    initial_batteries = world.robot_initial_batteries.copy()
    battery_remaining = initial_batteries.copy()
    task_positions = world.task_positions
    task_services = world.task_service_times
    task_priorities = world.task_priorities
    task_deadlines = world.task_deadlines
    path_table = world.path_to_tasks

    current_node_ids = np.arange(robot_count, dtype=np.int64)
    task_available = np.ones(task_count, dtype=bool)
    busy_until = np.zeros(robot_count, dtype=np.float64)
    task_counts = np.zeros(robot_count, dtype=np.int64)
    robot_workloads = np.zeros(robot_count, dtype=np.float64)

    total_travel = 0.0
    total_euclidean_travel = 0.0
    route_efficiency_sum = 0.0
    completion_time_score_sum = 0.0
    completed_priority_sum = 0.0
    total_priority_sum = float(task_priorities.sum())
    on_time_count = 0.0
    battery_blocked_pair_events = 0.0
    timing = RolloutTiming()
    local_hungarian_time_regret_sum = 0.0
    local_hungarian_time_regret_events = 0
    oracle_diagnostic_ns = 0

    max_events = task_count + robot_count + 2
    for _ in range(max_events):
        if not np.any(task_available):
            break
        now = float(np.min(busy_until))
        if now >= config.episode_time - 1e-12:
            break

        active_free = np.isclose(
            busy_until,
            now,
            rtol=0.0,
            atol=1e-12,
        )
        if not np.any(active_free):
            break

        decision_start = perf_counter_ns()

        delta = robot_positions[:, None, :] - task_positions[None, :, :]
        euclidean = np.linalg.norm(delta, axis=-1)
        path_lengths = path_table[current_node_ids, :]
        duration_path = (
            path_lengths / config.robot_speed
            + task_services[None, :]
        )
        energy_required = path_lengths * config.energy_per_distance
        finishes = now + duration_path

        time_eligible = (
            active_free[:, None]
            & task_available[None, :]
            & np.isfinite(path_lengths)
            & (finishes <= config.episode_time + 1e-12)
        )
        battery_feasible = (
            energy_required
            <= battery_remaining[:, None] + 1e-12
        )
        battery_blocked_pair_events += float(
            np.sum(time_eligible & ~battery_feasible)
        )
        pair_eligible = time_eligible & battery_feasible

        scores = _build_scores(
            score_mode,
            gene,
            config,
            now,
            robot_positions,
            task_positions,
            task_services,
            task_priorities,
            task_deadlines,
            battery_remaining,
            robot_workloads,
            path_lengths,
            pair_eligible,
        )

        matcher_start = perf_counter_ns()
        if matcher == "greedy":
            assignments = _greedy_match(
                scores,
                pair_eligible,
                active_free,
                task_available,
            )
        elif matcher == "hungarian":
            assignments = _hungarian_match(
                scores,
                pair_eligible,
                active_free,
                task_available,
            )
        else:
            raise ValueError(f"Unknown matcher: {matcher}")
        matcher_end = perf_counter_ns()
        decision_end = matcher_end
        timing.add_decision(
            decision_end - decision_start,
            matcher_end - matcher_start,
        )

        oracle_diag_start = perf_counter_ns()
        time_utility = np.where(
            pair_eligible,
            1.0 - np.clip(
                finishes / max(config.episode_time, 1e-12),
                0.0,
                1.0,
            ),
            -np.inf,
        )
        oracle_assignments = _hungarian_match(
            time_utility,
            pair_eligible,
            active_free,
            task_available,
        )
        oracle_utility = sum(
            float(time_utility[r, t]) for r, t in oracle_assignments
        )
        policy_utility = sum(
            float(time_utility[r, t]) for r, t in assignments
        )
        if oracle_utility > 1e-12:
            local_hungarian_time_regret_sum += max(
                0.0,
                (oracle_utility - policy_utility) / oracle_utility,
            )
            local_hungarian_time_regret_events += 1
        oracle_diagnostic_ns += perf_counter_ns() - oracle_diag_start

        matched_rows = np.zeros(robot_count, dtype=bool)
        for robot_idx, task_idx in assignments:
            path_d = float(path_lengths[robot_idx, task_idx])
            euclidean_d = float(euclidean[robot_idx, task_idx])
            service = float(task_services[task_idx])
            priority = float(task_priorities[task_idx])
            deadline = float(task_deadlines[task_idx])
            energy = float(energy_required[robot_idx, task_idx])
            duration = path_d / config.robot_speed + service
            finish = now + duration

            busy_until[robot_idx] = finish
            robot_positions[robot_idx] = task_positions[task_idx]
            current_node_ids[robot_idx] = robot_count + task_idx
            battery_remaining[robot_idx] = max(
                0.0,
                battery_remaining[robot_idx] - energy,
            )
            task_available[task_idx] = False
            task_counts[robot_idx] += 1
            robot_workloads[robot_idx] += duration
            total_travel += path_d
            total_euclidean_travel += euclidean_d
            completed_priority_sum += priority
            on_time_count += float(finish <= deadline + 1e-12)
            route_efficiency_sum += 1.0 - float(
                np.clip(path_d / config.diagonal, 0.0, 1.0)
            )
            completion_time_score_sum += 1.0 - float(
                np.clip(
                    finish / max(config.episode_time, 1e-12),
                    0.0,
                    1.0,
                )
            )
            matched_rows[robot_idx] = True

        unmatched_free = active_free & ~matched_rows
        busy_until[unmatched_free] = config.episode_time

    completed = int(task_counts.sum())
    completion = completed / task_count
    efficiency = route_efficiency_sum / task_count
    route_efficiency = (
        route_efficiency_sum / completed if completed > 0 else 0.0
    )
    priority_satisfaction = (
        completed_priority_sum / total_priority_sum
        if total_priority_sum > 0.0
        else 0.0
    )
    deadline_satisfaction = on_time_count / task_count
    time_optimality = completion_time_score_sum / task_count
    balance = completion * jain_fairness(robot_workloads)

    detour_ratio = (
        total_travel / total_euclidean_travel
        if total_euclidean_travel > 0.0
        else 1.0
    )
    mean_initial_battery = float(initial_batteries.mean())
    mean_final_battery = float(battery_remaining.mean())
    initial_battery_sum = float(initial_batteries.sum())
    final_battery_sum = float(battery_remaining.sum())
    battery_remaining_fraction = (
        final_battery_sum / initial_battery_sum
        if initial_battery_sum > 0.0
        else 0.0
    )
    energy_consumed = initial_battery_sum - final_battery_sum

    evaluation = Evaluation(
        completion=float(completion),
        efficiency=float(efficiency),
        priority_satisfaction=float(priority_satisfaction),
        deadline_satisfaction=float(deadline_satisfaction),
        balance=float(balance),
        time_optimality=float(time_optimality),
        route_efficiency=float(route_efficiency),
        completed_tasks=float(completed),
        completed_priority=float(completed_priority_sum),
        total_priority=float(total_priority_sum),
        on_time_tasks=float(on_time_count),
        total_travel=float(total_travel),
        total_euclidean_travel=float(total_euclidean_travel),
        detour_ratio=float(detour_ratio),
        mean_initial_battery=mean_initial_battery,
        mean_final_battery=mean_final_battery,
        battery_remaining_fraction=float(battery_remaining_fraction),
        energy_consumed=float(energy_consumed),
        battery_blocked_pair_events=float(battery_blocked_pair_events),
        robot_task_counts=tuple(float(x) for x in task_counts.tolist()),
        robot_workloads=tuple(float(x) for x in robot_workloads.tolist()),
        robot_final_batteries=tuple(
            float(x) for x in battery_remaining.tolist()
        ),
    )
    wall_end = perf_counter_ns()
    return RolloutResult(
        evaluation=evaluation,
        timing=timing,
        wall_ns=max(0, wall_end - wall_start - oracle_diagnostic_ns),
        local_hungarian_time_regret_sum=local_hungarian_time_regret_sum,
        local_hungarian_time_regret_events=local_hungarian_time_regret_events,
    )


def _load_gene(summary_path: Path, source: str) -> Gene:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    final = data["final"]
    if source == "reference":
        weights = final["reference_gene"]["gene"]["weights"]
    else:
        weights = final["axis_best_selected_on_probe"][source]["gene"]["weights"]
    return Gene(np.asarray(weights, dtype=np.float64))


def _summary_paths(suite_dir: Path) -> list[Path]:
    paths = sorted((suite_dir / "runs").glob("*/summary.json"))
    if not paths:
        raise FileNotFoundError(
            f"No summary.json files found under {suite_dir / 'runs'}"
        )
    return paths


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _aggregate_rollouts(results: list[RolloutResult]) -> dict[str, object]:
    if not results:
        raise ValueError("No rollout results")

    out: dict[str, object] = {}
    for metric in SCALAR_METRICS:
        out[metric] = _mean_std(
            getattr(result.evaluation, metric)
            for result in results
        )

    decisions = sum(r.timing.decision_count for r in results)
    total_decision_ns = sum(r.timing.decision_total_ns for r in results)
    total_matcher_ns = sum(r.timing.matcher_ns for r in results)
    total_wall_ns = sum(r.wall_ns for r in results)

    regret_events = sum(
        r.local_hungarian_time_regret_events for r in results
    )
    regret_sum = sum(
        r.local_hungarian_time_regret_sum for r in results
    )
    mean_local_regret = (
        regret_sum / regret_events if regret_events else 0.0
    )
    out["local_hungarian_time_regret"] = mean_local_regret
    out["local_hungarian_time_retention"] = 1.0 - mean_local_regret

    out["timing"] = {
        "rollouts": len(results),
        "decisions": decisions,
        "decision_ms_per_world": (
            total_decision_ns / 1e6 / len(results)
        ),
        "decision_us_per_event": (
            total_decision_ns / 1e3 / decisions if decisions else 0.0
        ),
        "matcher_us_per_event": (
            total_matcher_ns / 1e3 / decisions if decisions else 0.0
        ),
        "wall_ms_per_world": total_wall_ns / 1e6 / len(results),
    }
    return out


def _run_method(
    worlds: list[World],
    config: EnvConfig,
    *,
    score_mode: str,
    matcher: str,
    genes: list[Gene | None],
    timing_repeats: int,
) -> list[RolloutResult]:
    results: list[RolloutResult] = []
    for gene in genes:
        for world in worlds:
            first: RolloutResult | None = None
            repeated_decision_ns = 0
            repeated_matcher_ns = 0
            repeated_decisions = 0
            repeated_wall_ns = 0
            for _ in range(timing_repeats):
                result = rollout_world(
                    world,
                    config,
                    score_mode=score_mode,
                    matcher=matcher,
                    gene=gene,
                )
                if first is None:
                    first = result
                repeated_decision_ns += result.timing.decision_total_ns
                repeated_matcher_ns += result.timing.matcher_ns
                repeated_decisions += result.timing.decision_count
                repeated_wall_ns += result.wall_ns

            assert first is not None
            first.timing = RolloutTiming(
                decision_total_ns=int(
                    repeated_decision_ns / timing_repeats
                ),
                matcher_ns=int(repeated_matcher_ns / timing_repeats),
                decision_count=int(
                    round(repeated_decisions / timing_repeats)
                ),
            )
            first.wall_ns = int(repeated_wall_ns / timing_repeats)
            results.append(first)
    return results


def _scaling_benchmark(
    sizes: list[int],
    task_ratio: int,
    repeats: int,
    seed: int,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for robot_count in sizes:
        task_count = robot_count * task_ratio
        scores = rng.normal(
            0.0,
            1.0,
            size=(robot_count, task_count),
        )
        eligible = rng.random((robot_count, task_count)) < 0.85
        free = np.ones(robot_count, dtype=bool)
        available = np.ones(task_count, dtype=bool)

        # Warm up both code paths.
        _greedy_match(scores, eligible, free, available)
        _hungarian_match(scores, eligible, free, available)

        greedy_times: list[int] = []
        hungarian_times: list[int] = []
        for _ in range(repeats):
            t0 = perf_counter_ns()
            _greedy_match(scores, eligible, free, available)
            t1 = perf_counter_ns()
            _hungarian_match(scores, eligible, free, available)
            t2 = perf_counter_ns()
            greedy_times.append(t1 - t0)
            hungarian_times.append(t2 - t1)

        g = np.asarray(greedy_times, dtype=np.float64) / 1e3
        h = np.asarray(hungarian_times, dtype=np.float64) / 1e3
        rows.append(
            {
                "robots": robot_count,
                "tasks": task_count,
                "greedy_median_us": float(np.median(g)),
                "hungarian_median_us": float(np.median(h)),
                "greedy_p95_us": float(np.percentile(g, 95)),
                "hungarian_p95_us": float(np.percentile(h, 95)),
                "hungarian_over_greedy_median": float(
                    np.median(h) / max(np.median(g), 1e-12)
                ),
            }
        )
    return rows


def _metric_gap(
    candidate: dict[str, object],
    baseline: dict[str, object],
    metric: str,
) -> dict[str, float]:
    c = float(candidate[metric]["mean"])
    b = float(baseline[metric]["mean"])
    return {
        "absolute": c - b,
        "relative_percent": (
            100.0 * (c - b) / abs(b) if abs(b) > 1e-12 else 0.0
        ),
        "retention_percent": (
            100.0 * c / b if b > 1e-12 else 0.0
        ),
    }


def benchmark(args: argparse.Namespace) -> Path:
    suite_dir = Path(args.suite_dir)
    summary_paths = _summary_paths(suite_dir)
    genes = [_load_gene(path, args.gene_source) for path in summary_paths]

    config = EnvConfig(
        world_size=args.world_size,
        num_robots=args.robots,
        num_tasks=args.tasks,
        robot_speed=args.robot_speed,
        service_time_min=args.service_time_min,
        service_time_max=args.service_time_max,
        priority_min=args.priority_min,
        priority_max=args.priority_max,
        deadline_min=args.deadline_min,
        deadline_max=args.deadline_max,
        episode_time=args.episode_time,
        obstacle_count=args.obstacle_count,
        obstacle_size_min=args.obstacle_size_min,
        obstacle_size_max=args.obstacle_size_max,
        obstacle_clearance=args.obstacle_clearance,
        grid_resolution=args.grid_resolution,
        battery_capacity=args.battery_capacity,
        initial_battery_min=args.initial_battery_min,
        initial_battery_max=args.initial_battery_max,
        energy_per_distance=args.energy_per_distance,
    )

    # World/A* generation is intentionally outside the timed region.
    worlds = [
        generate_world(config, args.world_seed + idx)
        for idx in range(args.worlds)
    ]

    methods = {
        "gene_greedy": _aggregate_rollouts(
            _run_method(
                worlds,
                config,
                score_mode="gene",
                matcher="greedy",
                genes=genes,
                timing_repeats=args.timing_repeats,
            )
        ),
        "gene_hungarian": _aggregate_rollouts(
            _run_method(
                worlds,
                config,
                score_mode="gene",
                matcher="hungarian",
                genes=genes,
                timing_repeats=args.timing_repeats,
            )
        ),
        "greedy_path_time": _aggregate_rollouts(
            _run_method(
                worlds,
                config,
                score_mode="path_time",
                matcher="greedy",
                genes=[None],
                timing_repeats=args.timing_repeats,
            )
        ),
        "hungarian_path_time": _aggregate_rollouts(
            _run_method(
                worlds,
                config,
                score_mode="path_time",
                matcher="hungarian",
                genes=[None],
                timing_repeats=args.timing_repeats,
            )
        ),
        "hungarian_priority_per_path_time": _aggregate_rollouts(
            _run_method(
                worlds,
                config,
                score_mode="priority_path_time",
                matcher="hungarian",
                genes=[None],
                timing_repeats=args.timing_repeats,
            )
        ),
    }

    comparison = {
        "gene_greedy_vs_gene_hungarian": {
            metric: _metric_gap(
                methods["gene_greedy"],
                methods["gene_hungarian"],
                metric,
            )
            for metric in (
                "completion",
                "efficiency",
                "priority_satisfaction",
                "deadline_satisfaction",
                "balance",
                "time_optimality",
            )
        },
        "gene_greedy_vs_hungarian_path_time": {
            metric: _metric_gap(
                methods["gene_greedy"],
                methods["hungarian_path_time"],
                metric,
            )
            for metric in (
                "completion",
                "efficiency",
                "priority_satisfaction",
                "deadline_satisfaction",
                "balance",
                "time_optimality",
            )
        },
        "speed": {
            "gene_greedy_vs_gene_hungarian_decision_speedup": (
                float(
                    methods["gene_hungarian"]["timing"][
                        "decision_ms_per_world"
                    ]
                )
                / max(
                    float(
                        methods["gene_greedy"]["timing"][
                            "decision_ms_per_world"
                        ]
                    ),
                    1e-12,
                )
            ),
            "gene_greedy_vs_gene_hungarian_matcher_speedup": (
                float(
                    methods["gene_hungarian"]["timing"][
                        "matcher_us_per_event"
                    ]
                )
                / max(
                    float(
                        methods["gene_greedy"]["timing"][
                            "matcher_us_per_event"
                        ]
                    ),
                    1e-12,
                )
            ),
            "gene_greedy_vs_hungarian_path_time_decision_speedup": (
                float(
                    methods["hungarian_path_time"]["timing"][
                        "decision_ms_per_world"
                    ]
                )
                / max(
                    float(
                        methods["gene_greedy"]["timing"][
                            "decision_ms_per_world"
                        ]
                    ),
                    1e-12,
                )
            ),
            "gene_greedy_vs_hungarian_path_time_matcher_speedup": (
                float(
                    methods["hungarian_path_time"]["timing"][
                        "matcher_us_per_event"
                    ]
                )
                / max(
                    float(
                        methods["gene_greedy"]["timing"][
                            "matcher_us_per_event"
                        ]
                    ),
                    1e-12,
                )
            ),
        },
    }

    scaling = _scaling_benchmark(
        args.scaling_sizes,
        args.task_ratio,
        args.scaling_repeats,
        args.world_seed + 999,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "hungarian_benchmark.json"
    result = {
        "benchmark": "gene_mrta_v16t_hungarian_quality_time",
        "interpretation": (
            "Hungarian is exact only for the current event assignment matrix "
            "under the selected score. It is not a globally optimal solver for "
            "the entire sequential MRTA episode."
        ),
        "suite_dir": str(suite_dir),
        "gene_source": args.gene_source,
        "gene_count": len(genes),
        "world_seed": args.world_seed,
        "world_count": args.worlds,
        "timing_excludes_world_generation_and_astar_precompute": True,
        "timing_repeats": args.timing_repeats,
        "config": {
            "robots": config.num_robots,
            "tasks": config.num_tasks,
            "world_size": config.world_size,
        },
        "methods": methods,
        "comparison": comparison,
        "matching_scaling": scaling,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output_dir / "hungarian_method_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "completion",
                "efficiency",
                "priority_satisfaction",
                "deadline_satisfaction",
                "balance",
                "time_optimality",
                "local_hungarian_time_regret",
                "local_hungarian_time_retention",
                "decision_ms_per_world",
                "decision_us_per_event",
                "matcher_us_per_event",
                "wall_ms_per_world",
            ]
        )
        for name, metrics in methods.items():
            timing = metrics["timing"]
            writer.writerow(
                [
                    name,
                    metrics["completion"]["mean"],
                    metrics["efficiency"]["mean"],
                    metrics["priority_satisfaction"]["mean"],
                    metrics["deadline_satisfaction"]["mean"],
                    metrics["balance"]["mean"],
                    metrics["time_optimality"]["mean"],
                    metrics["local_hungarian_time_regret"],
                    metrics["local_hungarian_time_retention"],
                    timing["decision_ms_per_world"],
                    timing["decision_us_per_event"],
                    timing["matcher_us_per_event"],
                    timing["wall_ms_per_world"],
                ]
            )

    scaling_path = output_dir / "hungarian_scaling.csv"
    with scaling_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scaling[0].keys()))
        writer.writeheader()
        writer.writerows(scaling)

    print("\nHUNGARIAN QUALITY/TIME BENCHMARK")
    print(
        "method                         C        E        P        D        B        T"
        "   regret(H-time)   decision-ms/world   matcher-us/event"
    )
    for name, metrics in methods.items():
        timing = metrics["timing"]
        print(
            f"{name:30s} "
            f"{metrics['completion']['mean']:.5f} "
            f"{metrics['efficiency']['mean']:.5f} "
            f"{metrics['priority_satisfaction']['mean']:.5f} "
            f"{metrics['deadline_satisfaction']['mean']:.5f} "
            f"{metrics['balance']['mean']:.5f} "
            f"{metrics['time_optimality']['mean']:.5f} "
            f"{metrics['local_hungarian_time_regret']:14.5f} "
            f"{timing['decision_ms_per_world']:18.6f} "
            f"{timing['matcher_us_per_event']:18.3f}"
        )

    speed = comparison["speed"]
    print("\nGENE GREEDY vs SAME-GENE HUNGARIAN")
    for metric, gap in comparison["gene_greedy_vs_gene_hungarian"].items():
        print(
            f"{metric:24s} retention={gap['retention_percent']:.3f}% "
            f"gap={gap['absolute']:+.6f}"
        )
    print(
        "decision speedup (Hungarian / Greedy) = "
        f"{speed['gene_greedy_vs_gene_hungarian_decision_speedup']:.3f}x"
    )
    print(
        "matcher speedup (Hungarian / Greedy) = "
        f"{speed['gene_greedy_vs_gene_hungarian_matcher_speedup']:.3f}x"
    )
    time_gap = comparison["gene_greedy_vs_hungarian_path_time"]["time_optimality"]
    print("\nGENE GREEDY vs HUNGARIAN PATH-TIME")
    print(
        "time-optimality retention = "
        f"{time_gap['retention_percent']:.3f}% "
        f"gap={time_gap['absolute']:+.6f}"
    )
    print(
        "local Hungarian time retention = "
        f"{100.0 * methods['gene_greedy']['local_hungarian_time_retention']:.3f}%"
    )
    print(
        "decision speedup (H-PathTime / Gene Greedy) = "
        f"{speed['gene_greedy_vs_hungarian_path_time_decision_speedup']:.3f}x"
    )
    print(
        "matcher speedup (H-PathTime / Gene Greedy) = "
        f"{speed['gene_greedy_vs_hungarian_path_time_matcher_speedup']:.3f}x"
    )

    print("\nMATCHER SCALING")
    for row in scaling:
        print(
            f"R={row['robots']:3d} T={row['tasks']:4d} | "
            f"greedy={row['greedy_median_us']:.2f} us | "
            f"hungarian={row['hungarian_median_us']:.2f} us | "
            f"H/G={row['hungarian_over_greedy_median']:.2f}x"
        )

    print(f"\nRESULT_JSON={result_path}")
    print(f"RESULT_CSV={csv_path}")
    print(f"SCALING_CSV={scaling_path}")
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gene greedy assignment with event-based Hungarian "
            "assignment on quality and decision time."
        )
    )
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument(
        "--gene-source",
        choices=[
            "reference",
            "completion",
            "efficiency",
            "priority_satisfaction",
            "deadline_satisfaction",
            "balance",
            "time_optimality",
        ],
        default="reference",
    )
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--world-seed", type=int, default=97_000_000)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--scaling-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 100],
    )
    parser.add_argument("--task-ratio", type=int, default=5)
    parser.add_argument("--scaling-repeats", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v16t_hungarian_benchmark",
    )

    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20)
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
    return parser


def main() -> None:
    benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
