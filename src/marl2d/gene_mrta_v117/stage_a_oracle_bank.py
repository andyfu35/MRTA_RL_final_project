from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum
from marl2d.gene_mrta_v115.scaling import scale_config
from marl2d.gene_mrta_v117.oracle import solve_global_priority_optimum


BANK_VERSION = "v117_stage_a_oracle_bank_v1"


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

    rows: list[dict[str, Any]] = []
    for world_index in range(world_count):
        seed = int(seed_base + world_index)
        print(
            f"V117_ORACLE world={world_index + 1}/{world_count} "
            f"seed={seed} case={robots}R/{tasks}T",
            flush=True,
        )
        world = generate_world(config, seed)

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
            "time_optimum": float(
                time_oracle.time_optimality
            ),
            "priority_optimum": float(
                priority_oracle.priority_satisfaction
            ),
            "time_completed_tasks": int(
                time_oracle.completed_tasks or 0
            ),
            "priority_completed_tasks": int(
                priority_oracle.completed_tasks or 0
            ),
            "time_oracle_seconds": float(
                time_oracle.solve_seconds
            ),
            "priority_oracle_seconds": float(
                priority_oracle.solve_seconds
            ),
        }
        rows.append(row)
        print(
            "V117_ORACLE_RESULT "
            + json.dumps(row),
            flush=True,
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
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
    time_optima = [
        float(row["time_optimum"])
        for row in rows
    ]
    priority_optima = [
        float(row["priority_optimum"])
        for row in rows
    ]
    return (
        config,
        worlds,
        time_optima,
        priority_optima,
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
