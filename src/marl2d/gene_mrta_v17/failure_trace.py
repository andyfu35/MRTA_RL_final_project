from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v16t.gene import Gene as LinearGene
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum
from marl2d.gene_mrta_v16t.hungarian_benchmark import (
    _build_scores,
    _greedy_match,
    _hungarian_match,
    rollout_world,
)

from .direct_gene import DirectAssignmentGene
from .rollout import _build_observations, rollout_direct_gene


OBS_NAMES = (
    "euclidean",
    "path",
    "service",
    "priority",
    "deadline_remaining",
    "battery",
    "workload",
    "competition",
)


def _load_direct(run_dir: Path) -> DirectAssignmentGene:
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    final = data["final"]
    if "time_specialist" in final:
        item = final["time_specialist"]
    else:
        item = final["axis_specialists"]["time_optimality"]
    return DirectAssignmentGene.from_dict(item["gene"])


def _load_v16to(run_dir: Path) -> LinearGene:
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    weights = data["final"]["global_optimality_specialist"]["gene"]["weights"]
    return LinearGene(np.asarray(weights, dtype=np.float64))


def _top_pairs(
    values: np.ndarray,
    eligible: np.ndarray,
    path_lengths: np.ndarray,
    service_times: np.ndarray,
    now: float,
    speed: float,
    limit: int = 6,
) -> list[dict[str, float | int]]:
    ids = np.argwhere(eligible & np.isfinite(values))
    rows: list[dict[str, float | int]] = []
    for robot, task in ids.tolist():
        path_d = float(path_lengths[robot, task])
        finish = now + path_d / speed + float(service_times[task])
        rows.append(
            {
                "robot": int(robot),
                "task": int(task),
                "score": float(values[robot, task]),
                "path": path_d,
                "finish": float(finish),
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows[:limit]


def _direct_trace(gene: DirectAssignmentGene, world, config: EnvConfig) -> dict[str, object]:
    R = config.num_robots
    N = config.num_tasks

    robot_positions = world.robot_positions.copy()
    battery_remaining = world.robot_initial_batteries.copy()
    current_node_ids = np.arange(R, dtype=np.int64)
    task_available = np.ones(N, dtype=bool)
    busy_until = np.zeros(R, dtype=np.float64)
    robot_workloads = np.zeros(R, dtype=np.float64)

    events: list[dict[str, object]] = []
    max_events = N + R + 2

    for event_idx in range(max_events):
        if not np.any(task_available):
            break

        now, free, observations, eligible, path_lengths, _ = _build_observations(
            world=world,
            config=config,
            robot_positions=robot_positions,
            current_node_ids=current_node_ids,
            battery_remaining=battery_remaining,
            robot_workloads=robot_workloads,
            busy_until=busy_until,
            task_available=task_available,
        )

        if now >= config.episode_time - 1e-12:
            break

        row_open = free.copy()
        col_open = task_available.copy()
        selected: list[tuple[int, int]] = []
        decode_steps: list[dict[str, object]] = []

        for step in range(int(np.sum(row_open))):
            logits, stop_logit = gene.action_logits(
                observations,
                eligible,
                row_open,
                col_open,
                step,
            )
            active = eligible & row_open[:, None] & col_open[None, :]
            top = _top_pairs(
                logits,
                active,
                path_lengths,
                world.task_service_times,
                now,
                config.robot_speed,
            )

            flat = int(np.argmax(logits))
            best = float(logits.flat[flat])
            best_robot = flat // N
            best_task = flat % N
            stop = not np.isfinite(best) or stop_logit >= best

            best_obs = None
            if np.isfinite(best):
                best_obs = {
                    name: float(observations[best_robot, best_task, idx])
                    for idx, name in enumerate(OBS_NAMES)
                }

            decode_steps.append(
                {
                    "step": step,
                    "stop_logit": float(stop_logit),
                    "best_pair_logit": best,
                    "pair_minus_stop": (
                        float(best - stop_logit) if np.isfinite(best) else None
                    ),
                    "best_pair": (
                        [int(best_robot), int(best_task)]
                        if np.isfinite(best)
                        else None
                    ),
                    "best_pair_observation": best_obs,
                    "top_pairs": top,
                    "decision": (
                        "STOP" if stop else f"R{best_robot}->T{best_task}"
                    ),
                }
            )

            if stop:
                break

            selected.append((best_robot, best_task))
            row_open[best_robot] = False
            col_open[best_task] = False

        before_battery = battery_remaining.copy()
        before_workload = robot_workloads.copy()
        assignment_details: list[dict[str, float | int]] = []
        matched = np.zeros(R, dtype=bool)

        for robot, task in selected:
            path_d = float(path_lengths[robot, task])
            service = float(world.task_service_times[task])
            energy = path_d * config.energy_per_distance
            finish = now + path_d / config.robot_speed + service

            busy_until[robot] = finish
            robot_positions[robot] = world.task_positions[task]
            current_node_ids[robot] = R + task
            battery_remaining[robot] = max(
                0.0,
                battery_remaining[robot] - energy,
            )
            task_available[task] = False
            robot_workloads[robot] += path_d / config.robot_speed + service
            matched[robot] = True

            assignment_details.append(
                {
                    "robot": int(robot),
                    "task": int(task),
                    "path": path_d,
                    "service": service,
                    "finish": float(finish),
                    "time_utility": float(
                        1.0 - np.clip(finish / config.episode_time, 0.0, 1.0)
                    ),
                    "battery_before": float(before_battery[robot]),
                    "battery_after": float(battery_remaining[robot]),
                }
            )

        unmatched_free = free & ~matched
        wait_until = None
        if np.any(unmatched_free):
            future_times = busy_until[busy_until > now + 1e-12]
            if future_times.size > 0:
                wait_until = float(np.min(future_times))
                busy_until[unmatched_free] = wait_until
            else:
                wait_until = float(config.episode_time)
                busy_until[unmatched_free] = config.episode_time

        events.append(
            {
                "event": event_idx,
                "time": float(now),
                "free_robots": np.flatnonzero(free).astype(int).tolist(),
                "available_tasks_before": int(np.sum(task_available)) + len(selected),
                "eligible_pairs": int(np.sum(eligible)),
                "battery_before": before_battery.tolist(),
                "workload_before": before_workload.tolist(),
                "decode_steps": decode_steps,
                "assignments": [[int(r), int(t)] for r, t in selected],
                "assignment_details": assignment_details,
                "unmatched_free_robots": (
                    np.flatnonzero(unmatched_free).astype(int).tolist()
                ),
                "wait_until": wait_until,
            }
        )

    evaluation = rollout_direct_gene(gene, world, config).evaluation.to_dict()
    return {"method": "v17_direct", "evaluation": evaluation, "events": events}


def _matcher_trace(
    *,
    method: str,
    world,
    config: EnvConfig,
    score_mode: str,
    matcher: str,
    gene: LinearGene | None,
) -> dict[str, object]:
    R = config.num_robots
    N = config.num_tasks

    robot_positions = world.robot_positions.copy()
    battery_remaining = world.robot_initial_batteries.copy()
    current_node_ids = np.arange(R, dtype=np.int64)
    task_available = np.ones(N, dtype=bool)
    busy_until = np.zeros(R, dtype=np.float64)
    robot_workloads = np.zeros(R, dtype=np.float64)

    events: list[dict[str, object]] = []
    max_events = N + R + 2

    for event_idx in range(max_events):
        if not np.any(task_available):
            break
        now = float(np.min(busy_until))
        if now >= config.episode_time - 1e-12:
            break

        free = np.isclose(busy_until, now, rtol=0.0, atol=1e-12)
        if not np.any(free):
            break

        delta = robot_positions[:, None, :] - world.task_positions[None, :, :]
        _euclidean = np.linalg.norm(delta, axis=-1)
        path_lengths = world.path_to_tasks[current_node_ids, :]
        duration = path_lengths / config.robot_speed + world.task_service_times[None, :]
        finishes = now + duration
        energy_required = path_lengths * config.energy_per_distance

        time_eligible = (
            free[:, None]
            & task_available[None, :]
            & np.isfinite(path_lengths)
            & (finishes <= config.episode_time + 1e-12)
        )
        battery_feasible = (
            energy_required <= battery_remaining[:, None] + 1e-12
        )
        eligible = time_eligible & battery_feasible

        scores = _build_scores(
            score_mode,
            gene,
            config,
            now,
            robot_positions,
            world.task_positions,
            world.task_service_times,
            world.task_priorities,
            world.task_deadlines,
            battery_remaining,
            robot_workloads,
            path_lengths,
            eligible,
        )

        if matcher == "greedy":
            assignments = _greedy_match(scores, eligible, free, task_available)
        elif matcher == "hungarian":
            assignments = _hungarian_match(scores, eligible, free, task_available)
        else:
            raise ValueError(matcher)

        time_utility = np.where(
            eligible,
            1.0 - np.clip(finishes / config.episode_time, 0.0, 1.0),
            -np.inf,
        )

        before_battery = battery_remaining.copy()
        before_workload = robot_workloads.copy()
        details: list[dict[str, float | int]] = []
        matched = np.zeros(R, dtype=bool)

        for robot, task in assignments:
            path_d = float(path_lengths[robot, task])
            service = float(world.task_service_times[task])
            finish = now + path_d / config.robot_speed + service
            energy = path_d * config.energy_per_distance

            busy_until[robot] = finish
            robot_positions[robot] = world.task_positions[task]
            current_node_ids[robot] = R + task
            battery_remaining[robot] = max(
                0.0,
                battery_remaining[robot] - energy,
            )
            task_available[task] = False
            robot_workloads[robot] += path_d / config.robot_speed + service
            matched[robot] = True

            details.append(
                {
                    "robot": int(robot),
                    "task": int(task),
                    "policy_score": float(scores[robot, task]),
                    "path": path_d,
                    "service": service,
                    "finish": float(finish),
                    "time_utility": float(time_utility[robot, task]),
                    "battery_before": float(before_battery[robot]),
                    "battery_after": float(battery_remaining[robot]),
                }
            )

        unmatched_free = free & ~matched
        busy_until[unmatched_free] = config.episode_time

        events.append(
            {
                "event": event_idx,
                "time": float(now),
                "free_robots": np.flatnonzero(free).astype(int).tolist(),
                "available_tasks_before": int(np.sum(task_available)) + len(assignments),
                "eligible_pairs": int(np.sum(eligible)),
                "battery_before": before_battery.tolist(),
                "workload_before": before_workload.tolist(),
                "top_policy_pairs": _top_pairs(
                    scores,
                    eligible,
                    path_lengths,
                    world.task_service_times,
                    now,
                    config.robot_speed,
                ),
                "top_local_time_pairs": _top_pairs(
                    time_utility,
                    eligible,
                    path_lengths,
                    world.task_service_times,
                    now,
                    config.robot_speed,
                ),
                "assignments": [[int(r), int(t)] for r, t in assignments],
                "assignment_details": details,
                "retired_unmatched_free": (
                    np.flatnonzero(unmatched_free).astype(int).tolist()
                ),
            }
        )

    evaluation = rollout_world(
        world,
        config,
        score_mode=score_mode,
        matcher=matcher,
        gene=gene,
    ).evaluation.to_dict()
    return {"method": method, "evaluation": evaluation, "events": events}


def _event_brief(trace: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for event in trace["events"]:
        assignments = ", ".join(
            f"R{pair[0]}->T{pair[1]}" for pair in event["assignments"]
        ) or "NONE"
        extra = ""
        if trace["method"] == "v17_direct":
            steps = event["decode_steps"]
            if steps:
                last = steps[-1]
                if last["decision"] == "STOP":
                    margin = last["pair_minus_stop"]
                    margin_text = "NA" if margin is None else f"{float(margin):+.4f}"
                    extra = f" STOP(pair-stop={margin_text})"
        lines.append(
            f"t={event['time']:.3f} free={event['free_robots']} "
            f"assign=[{assignments}]{extra}"
        )
    return lines


def _write_markdown(
    path: Path,
    *,
    seed: int,
    oracle,
    traces: list[dict[str, object]],
) -> None:
    lines = [
        f"# V1.7 failure trace: seed {seed}",
        "",
        f"Global T*: {oracle.time_optimality}",
        f"MILP optimal: {oracle.optimal}, gap={oracle.mip_gap}",
        "",
        "## Final metrics",
        "",
        "| Method | T | Retention | Completed | Travel |",
        "|---|---:|---:|---:|---:|",
    ]

    for trace in traces:
        ev = trace["evaluation"]
        retention = (
            float(ev["time_optimality"]) / float(oracle.time_optimality)
            if oracle.optimal and oracle.time_optimality
            else float("nan")
        )
        lines.append(
            f"| {trace['method']} | {ev['time_optimality']:.6f} | "
            f"{100.0*retention:.3f}% | {ev['completed_tasks']:.0f} | "
            f"{ev['total_travel']:.3f} |"
        )

    for trace in traces:
        lines.extend(
            [
                "",
                f"## {trace['method']} event timeline",
                "",
            ]
        )
        lines.extend(_event_brief(trace))

        if trace["method"] == "v17_direct":
            lines.extend(["", "### Direct decoder details", ""])
            for event in trace["events"]:
                lines.append(
                    f"Event {event['event']} at t={event['time']:.3f}:"
                )
                for step in event["decode_steps"]:
                    lines.append(
                        "- step "
                        f"{step['step']}: decision={step['decision']}, "
                        f"best_pair={step['best_pair']}, "
                        f"pair_logit={step['best_pair_logit']}, "
                        f"stop_logit={step['stop_logit']:.6f}, "
                        f"pair-stop={step['pair_minus_stop']}"
                    )
                lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    config = EnvConfig()
    world = generate_world(config, args.world_seed)

    direct_gene = _load_direct(Path(args.direct_run))
    v16to_gene = _load_v16to(Path(args.v16to_run))

    direct = _direct_trace(direct_gene, world, config)
    v16to = _matcher_trace(
        method="v16to_gene_greedy",
        world=world,
        config=config,
        score_mode="gene",
        matcher="greedy",
        gene=v16to_gene,
    )
    hungarian = _matcher_trace(
        method="hungarian_path_time",
        world=world,
        config=config,
        score_mode="path_time",
        matcher="hungarian",
        gene=None,
    )

    oracle = solve_global_time_optimum(
        world,
        config,
        time_limit=args.time_limit,
    )

    traces = [direct, v16to, hungarian]
    payload = {
        "world_seed": args.world_seed,
        "direct_run": args.direct_run,
        "v16to_run": args.v16to_run,
        "oracle": {
            "time_optimality": oracle.time_optimality,
            "optimal": oracle.optimal,
            "mip_gap": oracle.mip_gap,
            "solve_seconds": oracle.solve_seconds,
        },
        "traces": traces,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"failure_trace_{args.world_seed}.json"
    md_path = output_dir / f"failure_trace_{args.world_seed}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(
        md_path,
        seed=args.world_seed,
        oracle=oracle,
        traces=traces,
    )

    print(f"FAILURE_TRACE_SEED={args.world_seed}")
    print(
        f"GLOBAL_T_STAR={oracle.time_optimality} "
        f"OPTIMAL={oracle.optimal} GAP={oracle.mip_gap}"
    )
    for trace in traces:
        ev = trace["evaluation"]
        retention = (
            float(ev["time_optimality"]) / float(oracle.time_optimality)
            if oracle.optimal and oracle.time_optimality
            else float("nan")
        )
        print(
            f"{trace['method']}: "
            f"T={ev['time_optimality']:.6f} "
            f"retention={100.0*retention:.3f}% "
            f"completed={ev['completed_tasks']:.0f} "
            f"events={len(trace['events'])}"
        )
        for line in _event_brief(trace):
            print(f"  {line}")

    print(f"TRACE_JSON={json_path}")
    print(f"TRACE_MD={md_path}")
    return json_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--direct-run", required=True)
    p.add_argument("--v16to-run", required=True)
    p.add_argument("--world-seed", type=int, default=97_000_012)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v17_failure_trace",
    )
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
