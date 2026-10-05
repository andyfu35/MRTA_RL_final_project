from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum
from marl2d.gene_mrta_v115.scaling import scale_config
from marl2d.gene_mrta_v117.oracle import (
    solve_global_completion_optimum,
    solve_global_deadline_optimum,
    solve_global_path_efficiency_optimum,
    solve_global_priority_optimum,
)


BANK_VERSION = "v117_stage_a_oracle_bank_v3_all_exact_base_axes"


def _write_bank_payload(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    with temp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.flush()
        os.fsync(
            handle.fileno()
        )
    temp.replace(path)


def _load_resume_rows(
    path: Path,
    *,
    robots: int,
    tasks: int,
    seed_base: int,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )
    if data.get("version") != BANK_VERSION:
        raise ValueError(
            "Existing oracle bank has incompatible version"
        )
    if (
        int(data.get("robots", -1)) != robots
        or int(data.get("tasks", -1)) != tasks
        or int(data.get("seed_base", -1)) != seed_base
    ):
        raise ValueError(
            "Existing oracle bank does not match requested formal protocol"
        )

    required = {
        "completion_optimum",
        "time_optimum",
        "path_efficiency_optimum",
        "priority_optimum",
        "deadline_optimum",
    }
    rows: dict[
        int,
        dict[str, Any],
    ] = {}
    for row in data.get(
        "rows",
        [],
    ):
        if required.issubset(
            row
        ):
            rows[
                int(row["seed"])
            ] = row
    return rows


def build_stage_a_oracle_bank(
    *,
    robots: int,
    tasks: int,
    world_count: int,
    seed_base: int,
    time_limit: float | None,
    output_path: Path,
    solver_display: bool = False,
) -> dict[str, Any]:
    if world_count <= 0:
        raise ValueError("world_count must be positive")

    config = scale_config(
        robots,
        tasks,
        base=EnvConfig(),
        preserve_spatial_density=True,
    )

    resume_rows = _load_resume_rows(
        output_path,
        robots=robots,
        tasks=tasks,
        seed_base=seed_base,
    )
    rows_by_seed: dict[
        int,
        dict[str, Any],
    ] = dict(resume_rows)

    for world_index in range(world_count):
        seed = int(seed_base + world_index)
        if seed in rows_by_seed:
            print(
                f"V117_ORACLE_RESUME world={world_index + 1}/{world_count} "
                f"seed={seed} status=skip_completed",
                flush=True,
            )
            continue
        print(
            f"V117_ORACLE world={world_index + 1}/{world_count} "
            f"seed={seed} case={robots}R/{tasks}T",
            flush=True,
        )
        world = generate_world(config, seed)

        completion_oracle = (
            solve_global_completion_optimum(
                world,
                config,
                time_limit=time_limit,
                solver_display=solver_display,
            )
        )
        if (
            not completion_oracle.optimal
            or completion_oracle.completion is None
        ):
            raise RuntimeError(
                f"Completion oracle failed exact proof for seed={seed}: "
                f"status={completion_oracle.status} "
                f"message={completion_oracle.message}"
            )

        path_oracle = (
            solve_global_path_efficiency_optimum(
                world,
                config,
                time_limit=time_limit,
                solver_display=solver_display,
            )
        )
        if (
            not path_oracle.optimal
            or path_oracle.path_efficiency is None
        ):
            raise RuntimeError(
                f"Path-efficiency oracle failed exact proof for seed={seed}: "
                f"status={path_oracle.status} message={path_oracle.message}"
            )

        time_oracle = solve_global_time_optimum(
            world,
            config,
            time_limit=time_limit,
            solver_display=solver_display,
        )
        if not time_oracle.optimal or time_oracle.time_optimality is None:
            raise RuntimeError(
                f"Time oracle failed exact proof for seed={seed}: "
                f"status={time_oracle.status} message={time_oracle.message}"
            )

        deadline_oracle = solve_global_deadline_optimum(
            world,
            config,
            time_limit=time_limit,
            solver_display=solver_display,
        )
        if (
            not deadline_oracle.optimal
            or deadline_oracle.deadline_satisfaction is None
        ):
            raise RuntimeError(
                f"Deadline oracle failed exact proof for seed={seed}: "
                f"status={deadline_oracle.status} message={deadline_oracle.message}"
            )

        priority_oracle = solve_global_priority_optimum(
            world,
            config,
            time_limit=time_limit,
            solver_display=solver_display,
        )
        if (
            not priority_oracle.optimal
            or priority_oracle.priority_satisfaction is None
        ):
            raise RuntimeError(
                f"Priority oracle failed exact proof for seed={seed}: "
                f"status={priority_oracle.status} "
                f"message={priority_oracle.message}"
            )

        row = {
            "world_index": world_index,
            "seed": seed,
            "completion_optimum": float(
                completion_oracle.completion
            ),
            "time_optimum": float(
                time_oracle.time_optimality
            ),
            "path_efficiency_optimum": float(
                path_oracle.path_efficiency
            ),
            "priority_optimum": float(
                priority_oracle.priority_satisfaction
            ),
            "deadline_optimum": float(
                deadline_oracle.deadline_satisfaction
            ),
            "time_completed_tasks": int(
                time_oracle.completed_tasks or 0
            ),
            "priority_completed_tasks": int(
                priority_oracle.completed_tasks or 0
            ),
            "completion_oracle_seconds": float(
                completion_oracle.solve_seconds
            ),
            "time_oracle_seconds": float(
                time_oracle.solve_seconds
            ),
            "path_oracle_seconds": float(
                path_oracle.solve_seconds
            ),
            "deadline_oracle_seconds": float(
                deadline_oracle.solve_seconds
            ),
            "priority_oracle_seconds": float(
                priority_oracle.solve_seconds
            ),
        }
        rows_by_seed[seed] = row
        ordered_rows = [
            rows_by_seed[
                seed_base + index
            ]
            for index in range(
                world_count
            )
            if (
                seed_base + index
                in rows_by_seed
            )
        ]
        progress_payload = {
            "version": BANK_VERSION,
            "robots": int(robots),
            "tasks": int(tasks),
            "world_count": int(world_count),
            "seed_base": int(seed_base),
            "config": asdict(config),
            "rows": ordered_rows,
        }
        _write_bank_payload(
            output_path,
            progress_payload,
        )
        print(
            "V117_ORACLE_RESULT "
            + json.dumps(row),
            flush=True,
        )

    rows = [
        rows_by_seed[
            seed_base + index
        ]
        for index in range(
            world_count
        )
        if (
            seed_base + index
            in rows_by_seed
        )
    ]
    if len(rows) != world_count:
        raise RuntimeError(
            "Oracle bank incomplete after build"
        )

    payload = {
        "version": BANK_VERSION,
        "robots": int(robots),
        "tasks": int(tasks),
        "world_count": int(world_count),
        "seed_base": int(seed_base),
        "config": asdict(config),
        "rows": rows,
    }

    _write_bank_payload(
        output_path,
        payload,
    )
    print(
        f"V117_ORACLE_BANK={output_path}",
        flush=True,
    )
    return payload


def load_stage_a_oracle_bank(
    path: Path,
) -> tuple[
    EnvConfig,
    list,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )
    if data.get("version") != BANK_VERSION:
        raise ValueError(
            "Unsupported V1.17 oracle bank version"
        )

    config = EnvConfig(
        **data["config"]
    )
    rows = list(data["rows"])
    worlds = [
        generate_world(
            config,
            int(row["seed"]),
        )
        for row in rows
    ]
    completion_optima = [
        float(row["completion_optimum"])
        for row in rows
    ]
    time_optima = [
        float(row["time_optimum"])
        for row in rows
    ]
    path_efficiency_optima = [
        float(row["path_efficiency_optimum"])
        for row in rows
    ]
    priority_optima = [
        float(row["priority_optimum"])
        for row in rows
    ]
    deadline_optima = [
        float(row["deadline_optimum"])
        for row in rows
    ]
    return (
        config,
        worlds,
        completion_optima,
        time_optima,
        path_efficiency_optima,
        priority_optima,
        deadline_optima,
    )


def _optional_seconds(
    value: str,
) -> float | None:
    text = value.strip().lower()
    if text in {
        "none",
        "unlimited",
        "inf",
        "infinite",
    }:
        return None
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError(
            "time-limit must be positive or unlimited"
        )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--robots", type=int, default=4)
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--world-count", type=int, default=32)
    p.add_argument("--seed-base", type=int, default=117_000_000)
    p.add_argument(
        "--time-limit",
        type=_optional_seconds,
        default=300.0,
    )
    p.add_argument(
        "--solver-display",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    build_stage_a_oracle_bank(
        robots=args.robots,
        tasks=args.tasks,
        world_count=args.world_count,
        seed_base=args.seed_base,
        time_limit=args.time_limit,
        output_path=Path(args.output),
        solver_display=args.solver_display,
    )


if __name__ == "__main__":
    main()
