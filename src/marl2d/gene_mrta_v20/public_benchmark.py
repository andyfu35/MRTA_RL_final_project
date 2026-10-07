from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.request import urlopen

import numpy as np
import torch

from .gene import SetAssignmentGene
from .ood_generalization import _actual_device, _load_checkpoint_records, _select_record
from .rollout import _unpack_gene_batch


PROTOCOL = "gene_global_set_mrta_v20_public_benchmark_v1"

TSPLIB_BASE = "https://raw.githubusercontent.com/mastqe/tsplib/master"
TWPC_BASE = "https://raw.githubusercontent.com/zhanglixuan0720/TWPC-MRTA/main"

# Published mTSPLib values reproduced in ScheduleNet's benchmark table.
# Only CPLEX values marked with * in that table are treated as proven optimum.
MTSPLIB_REFERENCE = {
    ("eil51", 2): {
        "cplex": 222.73, "cplex_proven_optimal": True,
        "ortools": 243.02, "schedulenet": 259.67,
        "som": 278.44, "aco": 248.76, "ea": 276.62,
    },
    ("eil51", 5): {
        "cplex": 110.43, "cplex_proven_optimal": False,
        "ortools": 127.50, "schedulenet": 118.94,
        "som": 157.68, "aco": 135.09, "ea": 151.21,
    },
    ("berlin52", 2): {
        "cplex": 4079.63, "cplex_proven_optimal": False,
        "ortools": 4665.47, "schedulenet": 4816.30,
        "som": 5350.83, "aco": 4388.99, "ea": 5038.33,
    },
    ("berlin52", 5): {
        "cplex": 2056.54, "cplex_proven_optimal": False,
        "ortools": 2482.57, "schedulenet": 2615.57,
        "som": 3461.93, "aco": 2733.56, "ea": 2853.63,
    },
    ("eil76", 2): {
        "cplex": 280.85, "cplex_proven_optimal": True,
        "ortools": 318.00, "schedulenet": 334.10,
        "som": 364.02, "aco": 308.53, "ea": 365.72,
    },
    ("eil76", 5): {
        "cplex": 133.95, "cplex_proven_optimal": False,
        "ortools": 143.38, "schedulenet": 168.03,
        "som": 210.69, "aco": 163.93, "ea": 211.91,
    },
    ("rat99", 2): {
        "cplex": 674.85, "cplex_proven_optimal": False,
        "ortools": 762.19, "schedulenet": 789.98,
        "som": 927.36, "aco": 767.15, "ea": 896.72,
    },
    ("rat99", 5): {
        "cplex": 402.71, "cplex_proven_optimal": False,
        "ortools": 473.66, "schedulenet": 502.49,
        "som": 624.38, "aco": 525.54, "ea": 596.87,
    },
}


def _download_text(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _parse_tsplib(text: str) -> tuple[str, np.ndarray]:
    name = ""
    coords: list[tuple[float, float]] = []
    in_coords = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("NAME"):
            name = line.split(":", 1)[1].strip()
            continue
        if line.upper() == "NODE_COORD_SECTION":
            in_coords = True
            continue
        if line.upper() == "EOF":
            break
        if in_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    if not name or len(coords) < 2:
        raise ValueError("Invalid TSPLIB coordinate instance")
    return name, np.asarray(coords, dtype=np.float64)


def _greedy_open_baseline(coords: np.ndarray, robot_count: int) -> float:
    depot = coords[0]
    tasks = coords[1:]
    tails = np.repeat(depot[None, :], robot_count, axis=0)
    finish = np.zeros(robot_count, dtype=np.float64)
    remaining = np.ones(len(tasks), dtype=bool)
    for _ in range(len(tasks)):
        distances = np.linalg.norm(
            tails[:, None, :] - tasks[None, :, :], axis=-1
        )
        projected = finish[:, None] + distances
        projected[:, ~remaining] = np.inf
        flat = int(np.argmin(projected))
        robot = flat // len(tasks)
        task = flat % len(tasks)
        finish[robot] = projected[robot, task]
        tails[robot] = tasks[task]
        remaining[task] = False
    return max(float(np.max(finish)), 1.0)


def _gene_parts(gene: SetAssignmentGene, device: torch.device):
    parts = _unpack_gene_batch([gene], device)
    return tuple(x[0] for x in parts)


def _score_matrix(
    task_h: torch.Tensor,
    robot_h: torch.Tensor,
    available: torch.Tensor,
    distance_norm: torch.Tensor,
    progress: float,
    decoder_w: torch.Tensor,
    decoder_b: torch.Tensor,
) -> torch.Tensor:
    hidden = task_h.shape[-1]
    mask_f = available.to(torch.float32)
    task_mean = torch.sum(task_h * mask_f[:, None], dim=0) / torch.clamp(
        torch.sum(mask_f), min=1.0
    )
    neg_inf = torch.full_like(task_h, float("-inf"))
    task_max = torch.where(available[:, None], task_h, neg_inf).max(dim=0).values
    robot_mean = robot_h.mean(dim=0)
    robot_max = robot_h.max(dim=0).values

    o = 0
    wr = decoder_w[o:o + hidden]; o += hidden
    wt = decoder_w[o:o + hidden]; o += hidden
    wtm = decoder_w[o:o + hidden]; o += hidden
    wtx = decoder_w[o:o + hidden]; o += hidden
    wrm = decoder_w[o:o + hidden]; o += hidden
    wrx = decoder_w[o:o + hidden]; o += hidden
    wd = decoder_w[o]; o += 1
    wp = decoder_w[o]

    return (
        (robot_h @ wr)[:, None]
        + (task_h @ wt)[None, :]
        + torch.dot(task_mean, wtm)
        + torch.dot(task_max, wtx)
        + torch.dot(robot_mean, wrm)
        + torch.dot(robot_max, wrx)
        + distance_norm * wd
        + float(progress) * wp
        + decoder_b
    )


def _normalize_xy(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lo = np.min(coords, axis=0)
    span = np.max(coords, axis=0) - lo
    scale = max(float(np.max(span)), 1.0)
    return (coords - lo) / scale, lo, scale


def evaluate_mtsplib_gene(
    gene: SetAssignmentGene,
    coords: np.ndarray,
    robot_count: int,
    device: str,
) -> dict[str, object]:
    device_obj = torch.device(device)
    normalized, _, coord_scale = _normalize_xy(coords)
    depot = coords[0]
    tasks = coords[1:]
    task_norm_xy = normalized[1:]
    task_count = len(tasks)
    baseline = _greedy_open_baseline(coords, robot_count)
    diagonal = max(math.sqrt(2.0) * coord_scale, 1.0)
    distance_scale = max(baseline, diagonal, 1.0)

    task_input = np.column_stack([
        task_norm_xy,
        np.ones(task_count),
        np.ones(task_count),
        np.zeros(task_count),
    ]).astype(np.float32)

    task_w, task_b, robot_w, robot_b, decoder_w, decoder_b = _gene_parts(
        gene, device_obj
    )
    task_input_t = torch.as_tensor(task_input, dtype=torch.float32, device=device_obj)
    task_h = torch.tanh(task_input_t @ task_w + task_b)

    robot_pos = np.repeat(depot[None, :], robot_count, axis=0).astype(np.float64)
    current_nodes = np.zeros(robot_count, dtype=np.int64)
    accumulated = np.zeros(robot_count, dtype=np.float64)
    finish = np.zeros(robot_count, dtype=np.float64)
    available_np = np.ones(task_count, dtype=bool)
    routes: list[list[int]] = [[] for _ in range(robot_count)]

    started = time.perf_counter()
    for step in range(task_count):
        robot_norm = (robot_pos - np.min(coords, axis=0)) / coord_scale
        robot_input = np.column_stack([
            robot_norm,
            accumulated / distance_scale,
            finish / baseline,
        ]).astype(np.float32)
        robot_input_t = torch.as_tensor(
            robot_input, dtype=torch.float32, device=device_obj
        )
        robot_h = torch.tanh(robot_input_t @ robot_w + robot_b)

        pair_dist = np.linalg.norm(
            robot_pos[:, None, :] - tasks[None, :, :], axis=-1
        )
        distance_norm_t = torch.as_tensor(
            pair_dist / diagonal, dtype=torch.float32, device=device_obj
        )
        available_t = torch.as_tensor(
            available_np, dtype=torch.bool, device=device_obj
        )
        score = _score_matrix(
            task_h,
            robot_h,
            available_t,
            distance_norm_t,
            float(step) / max(task_count - 1, 1),
            decoder_w,
            decoder_b,
        )
        score[:, ~available_t] = float("-inf")
        flat = int(torch.argmax(score.reshape(-1)).item())
        robot = flat // task_count
        task = flat % task_count

        dist = float(pair_dist[robot, task])
        accumulated[robot] += dist
        finish[robot] += dist
        robot_pos[robot] = tasks[task]
        current_nodes[robot] = task + 1
        available_np[task] = False
        routes[robot].append(task + 1)

    return_distances = np.linalg.norm(robot_pos - depot[None, :], axis=-1)
    closed_costs = accumulated + return_distances
    runtime_s = time.perf_counter() - started

    return {
        "closed_makespan": float(np.max(closed_costs)),
        "total_closed_distance": float(np.sum(closed_costs)),
        "route_sizes": [len(route) for route in routes],
        "routes": routes,
        "runtime_s": runtime_s,
    }


def _parse_twpc_csv(text: str, robot_count: int, task_count: int):
    rows = list(csv.reader(io.StringIO(text)))
    n = robot_count + task_count
    if len(rows) < 1 + 3 * n:
        raise ValueError("TWPC file is shorter than expected")
    info = np.asarray(
        [[float(x) for x in rows[i]] for i in range(1, n + 1)],
        dtype=np.float64,
    )
    precedence = np.asarray(
        [[int(float(x)) for x in rows[i]] for i in range(n + 1, 2 * n + 1)],
        dtype=np.int64,
    )
    distances = np.asarray(
        [[float(x) for x in rows[i]] for i in range(2 * n + 1, 3 * n + 1)],
        dtype=np.float64,
    )
    return info, precedence, distances


def _parse_twpc_solution(text: str) -> dict[str, float | int]:
    finished = re.search(r"Finished Task:\s*(\d+)\s*/\s*(\d+)", text)
    metrics = re.search(
        r"Makespan:\s*([0-9.]+),\s*Total Distance:\s*([0-9.]+),"
        r"\s*Total Time:\s*([0-9.]+)",
        text,
    )
    if not finished or not metrics:
        raise ValueError("Invalid TWPC solution")
    return {
        "finished": int(finished.group(1)),
        "tasks": int(finished.group(2)),
        "makespan": float(metrics.group(1)),
        "distance": float(metrics.group(2)),
        "runtime_s": float(metrics.group(3)),
    }


def _twpc_precedence_state(
    completed: np.ndarray,
    task_finish: np.ndarray,
    predecessors: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    available = np.asarray([
        (not bool(completed[t]))
        and all(bool(completed[p]) for p in predecessors[t])
        for t in range(len(predecessors))
    ], dtype=bool)
    release = np.zeros(len(predecessors), dtype=np.float64)
    for t, pred in enumerate(predecessors):
        if pred and all(bool(completed[p]) for p in pred):
            release[t] = max(float(task_finish[p]) for p in pred)
    return available, release


def evaluate_twpc_gene(
    gene: SetAssignmentGene,
    info: np.ndarray,
    precedence: np.ndarray,
    distances: np.ndarray,
    robot_count: int,
    task_count: int,
    device: str,
) -> dict[str, object]:
    device_obj = torch.device(device)
    n = robot_count + task_count
    task_nodes = np.arange(robot_count, n, dtype=np.int64)

    coords = info[:, 0:2]
    est = info[:, 2]
    twl = info[:, 3]
    duration = info[:, 4]
    deadlines = est + twl

    normalized, _, coord_scale = _normalize_xy(coords)
    task_norm_xy = normalized[task_nodes]
    task_deadlines = deadlines[task_nodes]
    task_duration = duration[task_nodes]

    time_scale = max(float(np.max(task_deadlines)), 1.0)
    distance_scale = max(float(np.max(distances)), 1.0)
    task_input = np.column_stack([
        task_norm_xy,
        np.ones(task_count),
        task_deadlines / time_scale,
        task_duration / time_scale,
    ]).astype(np.float32)

    task_w, task_b, robot_w, robot_b, decoder_w, decoder_b = _gene_parts(
        gene, device_obj
    )
    task_h = torch.tanh(
        torch.as_tensor(task_input, dtype=torch.float32, device=device_obj)
        @ task_w + task_b
    )

    robot_nodes = np.arange(robot_count, dtype=np.int64)
    robot_finish = np.zeros(robot_count, dtype=np.float64)
    robot_distance = np.zeros(robot_count, dtype=np.float64)
    completed = np.zeros(task_count, dtype=bool)
    task_finish = np.full(task_count, np.nan, dtype=np.float64)
    sequence: list[tuple[int, int]] = []

    predecessors: list[list[int]] = []
    for task_node in task_nodes:
        pred_nodes = np.where(precedence[:, task_node] > 0)[0]
        predecessors.append([
            int(x - robot_count) for x in pred_nodes if x >= robot_count
        ])

    started = time.perf_counter()
    while True:
        available, precedence_release = _twpc_precedence_state(
            completed,
            task_finish,
            predecessors,
        )
        if not np.any(available):
            break

        robot_xy = normalized[robot_nodes]
        robot_input = np.column_stack([
            robot_xy,
            robot_distance / distance_scale,
            robot_finish / time_scale,
        ]).astype(np.float32)
        robot_h = torch.tanh(
            torch.as_tensor(robot_input, dtype=torch.float32, device=device_obj)
            @ robot_w + robot_b
        )

        pair_dist = distances[
            robot_nodes[:, None],
            task_nodes[None, :],
        ]
        arrival = robot_finish[:, None] + pair_dist
        start_time = np.maximum(arrival, est[task_nodes][None, :])
        start_time = np.maximum(
            start_time,
            precedence_release[None, :],
        )
        finish_time = start_time + duration[task_nodes][None, :]
        feasible = (
            available[None, :]
            & (finish_time <= deadlines[task_nodes][None, :] + 1e-9)
        )
        if not np.any(feasible):
            break

        distance_norm_t = torch.as_tensor(
            pair_dist / distance_scale,
            dtype=torch.float32,
            device=device_obj,
        )
        available_t = torch.as_tensor(
            available, dtype=torch.bool, device=device_obj
        )
        score = _score_matrix(
            task_h,
            robot_h,
            available_t,
            distance_norm_t,
            len(sequence) / max(task_count - 1, 1),
            decoder_w,
            decoder_b,
        )
        feasible_t = torch.as_tensor(
            feasible, dtype=torch.bool, device=device_obj
        )
        score = torch.where(feasible_t, score, float("-inf"))
        flat = int(torch.argmax(score.reshape(-1)).item())
        robot = flat // task_count
        task = flat % task_count

        robot_finish[robot] = finish_time[robot, task]
        robot_distance[robot] += pair_dist[robot, task]
        robot_nodes[robot] = task_nodes[task]
        completed[task] = True
        task_finish[task] = finish_time[robot, task]
        sequence.append((robot, task))

    runtime_s = time.perf_counter() - started
    completed_count = int(np.sum(completed))
    return {
        "completed": completed_count,
        "tasks": task_count,
        "completion_rate": completed_count / float(task_count),
        "makespan": float(np.max(robot_finish)),
        "distance": float(np.sum(robot_distance)),
        "runtime_s": runtime_s,
        "sequence": sequence,
    }


def _best_feasible_reference(
    reference: dict[str, object],
) -> tuple[str, float]:
    values = {
        k: float(v)
        for k, v in reference.items()
        if k in {"ortools", "schedulenet", "som", "aco", "ea"}
    }
    if bool(reference["cplex_proven_optimal"]):
        values["cplex_opt"] = float(reference["cplex"])
    method = min(values, key=values.get)
    return method, values[method]


def run_mtsplib(args, genes, run_dir: Path, cache_dir: Path, device: str):
    rows = []
    cases = [
        ("eil51", 2), ("eil51", 5),
        ("berlin52", 2), ("berlin52", 5),
        ("eil76", 2), ("eil76", 5),
        ("rat99", 2), ("rat99", 5),
    ]
    for name, robots in cases:
        text = _download_text(
            f"{TSPLIB_BASE}/{name}.tsp",
            cache_dir / "tsplib" / f"{name}.tsp",
        )
        parsed_name, coords = _parse_tsplib(text)
        reference = MTSPLIB_REFERENCE[(name, robots)]
        best_method, best_value = _best_feasible_reference(reference)

        for gene_name, gene in genes:
            result = evaluate_mtsplib_gene(gene, coords, robots, device)
            value = float(result["closed_makespan"])
            row = {
                "benchmark": "mTSPLib",
                "instance": parsed_name,
                "robots": robots,
                "tasks": len(coords) - 1,
                "gene": gene_name,
                "gene_value": value,
                "published_best_feasible_method": best_method,
                "published_best_feasible_value": best_value,
                "gap_to_published_best_feasible_pct": (
                    100.0 * (value - best_value) / best_value
                ),
                "cplex_reference_midpoint_or_optimum": reference["cplex"],
                "cplex_proven_optimal": reference["cplex_proven_optimal"],
                "gap_to_proven_optimum_pct": (
                    100.0
                    * (value - float(reference["cplex"]))
                    / float(reference["cplex"])
                    if bool(reference["cplex_proven_optimal"])
                    else None
                ),
                "ortools": reference["ortools"],
                "schedulenet": reference["schedulenet"],
                "som": reference["som"],
                "aco": reference["aco"],
                "ea": reference["ea"],
                "runtime_s": result["runtime_s"],
                "route_sizes": result["route_sizes"],
            }
            rows.append(row)
            print("V20_PUBLIC_TT " + json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def run_twpc(args, genes, run_dir: Path, cache_dir: Path, device: str):
    rows = []
    robot_count = 5
    task_count = 18
    for sample in range(args.twpc_samples):
        stem = f"r5_t18_m0_{sample}"
        data_text = _download_text(
            f"{TWPC_BASE}/Data/RL5/{stem}.csv",
            cache_dir / "twpc" / "Data" / f"{stem}.csv",
        )
        sol_text = _download_text(
            f"{TWPC_BASE}/Sol/MIP/RL5/{stem}.sol",
            cache_dir / "twpc" / "MIP" / f"{stem}.sol",
        )
        info, precedence, distances = _parse_twpc_csv(
            data_text, robot_count, task_count
        )
        mip = _parse_twpc_solution(sol_text)

        for gene_name, gene in genes:
            result = evaluate_twpc_gene(
                gene,
                info,
                precedence,
                distances,
                robot_count,
                task_count,
                device,
            )
            completed = int(result["completed"])
            mip_finished = int(mip["finished"])
            row = {
                "benchmark": "TWPC-MRTA",
                "instance": stem,
                "robots": robot_count,
                "tasks": task_count,
                "gene": gene_name,
                "gene_completed": completed,
                "gene_completion_rate": float(result["completion_rate"]),
                "mip_finished": mip_finished,
                "mip_completion_rate": mip_finished / float(task_count),
                "completion_gap_tasks": completed - mip_finished,
                "mip_primary_optimum_proven_by_full_completion": (
                    mip_finished == task_count
                ),
                "gene_primary_optimal": (
                    completed == task_count and mip_finished == task_count
                ),
                "gene_makespan": result["makespan"],
                "mip_makespan": mip["makespan"],
                "gene_distance": result["distance"],
                "mip_distance": mip["distance"],
                "gene_runtime_s": result["runtime_s"],
                "mip_runtime_s": mip["runtime_s"],
            }
            rows.append(row)
            print(
                "V20_PUBLIC_ONTIME " + json.dumps(row, ensure_ascii=False),
                flush=True,
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> Path:
    checkpoint_data, records = _load_checkpoint_records(Path(args.checkpoint))
    tt_record = _select_record(records, "total_time", args.total_time_gene_id)
    ontime_record = _select_record(
        records, "on_time_completed_tasks", args.on_time_gene_id
    )
    genes = [
        ("total_time", tt_record.gene),
        ("on_time", ontime_record.gene),
    ]
    device = _actual_device(args.device)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "checkpoint_completed_round": checkpoint_data.get("completed_round"),
        "device": device,
        "frozen_genes": {
            "total_time": tt_record.record_id,
            "on_time": ontime_record.record_id,
        },
        "mTSPLib": {
            "objective": "closed-depot min-max route length",
            "service_time": 0,
            "deadline": "neutral constant feature",
            "priority": "uniform neutral feature",
            "primary_gene": "total_time",
            "cross_specialist_gene": "on_time",
        },
        "TWPC_MRTA": {
            "objective": "maximize tasks completed within hard time windows",
            "public_reference": "MIP with 120 s solver limit",
            "constraints_enforced": [
                "robot starts",
                "published distance matrix",
                "earliest start time",
                "time-window completion deadline",
                "service duration",
                "precedence",
            ],
            "priority": "uniform neutral feature because benchmark has no priority",
            "primary_gene": "on_time",
            "cross_specialist_gene": "total_time",
        },
        "retraining": False,
        "gene_bank_feedback": False,
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("V20_PUBLIC_PROTOCOL " + json.dumps(protocol, ensure_ascii=False), flush=True)

    rows = []
    if args.mode in {"all", "tt"}:
        rows.extend(run_mtsplib(args, genes, run_dir, cache_dir, device))
    if args.mode in {"all", "ontime"}:
        rows.extend(run_twpc(args, genes, run_dir, cache_dir, device))

    _write_csv(run_dir / "summary.csv", rows)
    (run_dir / "summary.json").write_text(
        json.dumps({"protocol": PROTOCOL, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"V20_PUBLIC_RUN_DIR={run_dir}", flush=True)
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--cache-dir",
        default="runs/gene_mrta_v20/public_benchmark_cache",
    )
    p.add_argument("--mode", choices=("all", "tt", "ontime"), default="all")
    p.add_argument("--twpc-samples", type=int, default=10)
    p.add_argument("--total-time-gene-id")
    p.add_argument("--on-time-gene-id")
    p.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.twpc_samples <= 0 or args.twpc_samples > 10:
        raise ValueError("twpc-samples must be in [1, 10]")
    run(args)


if __name__ == "__main__":
    main()
