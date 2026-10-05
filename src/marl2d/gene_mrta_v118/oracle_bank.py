from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    generate_world,
)
from marl2d.gene_mrta_v115.scaling import (
    scale_config,
)
from marl2d.gene_mrta_v117.oracle import (
    solve_global_completion_optimum,
)
from marl2d.gene_mrta_v118.oracle import (
    OBJECTIVES,
    solve_all_complete_optimum,
)


BANK_VERSION = (
    "v118_feasibility_first_oracle_bank_v1"
)


def feasibility_first_config(
    robots: int,
    tasks: int,
) -> EnvConfig:
    base = scale_config(
        robots,
        tasks,
        base=EnvConfig(),
        preserve_spatial_density=True,
    )
    tasks_per_robot = (
        float(tasks)
        / float(robots)
    )
    load_scale = max(
        tasks_per_robot / 5.0,
        0.5,
    )
    horizon = (
        180.0
        * load_scale
    )
    battery_capacity = (
        350.0
        * load_scale
    )
    return replace(
        base,
        episode_time=float(
            horizon
        ),
        service_time_min=2.0,
        service_time_max=18.0,
        deadline_min=float(
            0.30
            * horizon
        ),
        deadline_max=float(
            horizon
        ),
        battery_capacity=float(
            battery_capacity
        ),
        initial_battery_min=float(
            0.72
            * battery_capacity
        ),
        initial_battery_max=float(
            battery_capacity
        ),
    )


def _atomic_write(
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
    temp.replace(
        path
    )


def build_oracle_bank(
    *,
    robots: int,
    tasks: int,
    world_count: int,
    seed_base: int,
    time_limit: float | None,
    output_path: Path,
    solver_display: bool = False,
    max_candidates: int = 10000,
) -> dict[str, Any]:
    if world_count <= 0:
        raise ValueError(
            "world_count must be positive"
        )

    config = (
        feasibility_first_config(
            robots,
            tasks,
        )
    )

    if output_path.exists():
        existing = json.loads(
            output_path.read_text(
                encoding="utf-8",
            )
        )
        if (
            existing.get(
                "version"
            )
            != BANK_VERSION
        ):
            raise ValueError(
                "Existing V1.18 bank has incompatible version"
            )
        if (
            int(
                existing.get(
                    "robots",
                    -1,
                )
            )
            != robots
            or int(
                existing.get(
                    "tasks",
                    -1,
                )
            )
            != tasks
            or int(
                existing.get(
                    "seed_base",
                    -1,
                )
            )
            != seed_base
        ):
            raise ValueError(
                "Existing V1.18 bank does not match protocol"
            )
        rows = list(
            existing.get(
                "rows",
                [],
            )
        )
        candidate_offset = int(
            existing.get(
                "next_candidate_offset",
                0,
            )
        )
    else:
        rows = []
        candidate_offset = 0

    accepted_seeds = {
        int(
            row["seed"]
        )
        for row in rows
    }

    while (
        len(rows)
        < world_count
    ):
        if (
            candidate_offset
            >= max_candidates
        ):
            raise RuntimeError(
                "Could not find enough all-complete worlds within max_candidates"
            )

        seed = int(
            seed_base
            + candidate_offset
        )
        candidate_offset += 1

        if seed in accepted_seeds:
            continue

        print(
            "V118_WORLD_CANDIDATE "
            f"accepted={len(rows)}/{world_count} "
            f"seed={seed}",
            flush=True,
        )

        world = generate_world(
            config,
            seed,
        )
        completion = (
            solve_global_completion_optimum(
                world,
                config,
                time_limit=time_limit,
                solver_display=(
                    solver_display
                ),
            )
        )
        if (
            not completion.optimal
            or completion.completion
            is None
        ):
            raise RuntimeError(
                "Completion feasibility proof failed: "
                f"seed={seed} status={completion.status}"
            )

        if (
            completion.completion
            < 1.0 - 1e-9
        ):
            print(
                "V118_WORLD_REJECT "
                f"seed={seed} "
                f"completion_star={completion.completion:.6f}",
                flush=True,
            )
            continue

        objective_results = {}
        for objective in OBJECTIVES:
            result = (
                solve_all_complete_optimum(
                    world,
                    config,
                    objective=objective,
                    time_limit=(
                        time_limit
                    ),
                    solver_display=(
                        solver_display
                    ),
                )
            )
            if (
                not result.optimal
                or result.score
                is None
                or result.completed_tasks
                != tasks
            ):
                raise RuntimeError(
                    "All-complete oracle failed: "
                    f"seed={seed} objective={objective} "
                    f"status={result.status}"
                )
            objective_results[
                objective
            ] = result

        row = {
            "world_index": len(
                rows
            ),
            "seed": seed,
            "completion_optimum": 1.0,
            "time_optimum": float(
                objective_results[
                    "time"
                ].score
            ),
            "path_efficiency_optimum": float(
                objective_results[
                    "path_efficiency"
                ].score
            ),
            "priority_service_optimum": float(
                objective_results[
                    "priority_service"
                ].score
            ),
            "deadline_optimum": float(
                objective_results[
                    "deadline"
                ].score
            ),
            "completion_oracle_seconds": float(
                completion.solve_seconds
            ),
            "time_oracle_seconds": float(
                objective_results[
                    "time"
                ].solve_seconds
            ),
            "path_oracle_seconds": float(
                objective_results[
                    "path_efficiency"
                ].solve_seconds
            ),
            "priority_service_oracle_seconds": float(
                objective_results[
                    "priority_service"
                ].solve_seconds
            ),
            "deadline_oracle_seconds": float(
                objective_results[
                    "deadline"
                ].solve_seconds
            ),
        }
        rows.append(
            row
        )
        accepted_seeds.add(
            seed
        )

        payload = {
            "version": (
                BANK_VERSION
            ),
            "protocol": (
                "feasibility_first_all_tasks_required"
            ),
            "robots": int(
                robots
            ),
            "tasks": int(
                tasks
            ),
            "world_count": int(
                world_count
            ),
            "seed_base": int(
                seed_base
            ),
            "next_candidate_offset": int(
                candidate_offset
            ),
            "config": asdict(
                config
            ),
            "rows": rows,
        }
        _atomic_write(
            output_path,
            payload,
        )
        print(
            "V118_WORLD_ACCEPT "
            + json.dumps(
                row
            ),
            flush=True,
        )

    payload = {
        "version": (
            BANK_VERSION
        ),
        "protocol": (
            "feasibility_first_all_tasks_required"
        ),
        "robots": int(
            robots
        ),
        "tasks": int(
            tasks
        ),
        "world_count": int(
            world_count
        ),
        "seed_base": int(
            seed_base
        ),
        "next_candidate_offset": int(
            candidate_offset
        ),
        "config": asdict(
            config
        ),
        "rows": rows,
    }
    _atomic_write(
        output_path,
        payload,
    )
    print(
        f"V118_ORACLE_BANK={output_path}",
        flush=True,
    )
    return payload


def load_oracle_bank(
    path: Path,
) -> tuple[
    EnvConfig,
    list,
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
    if (
        data.get(
            "version"
        )
        != BANK_VERSION
    ):
        raise ValueError(
            "Unsupported V1.18 oracle bank version"
        )

    config = EnvConfig(
        **data[
            "config"
        ]
    )
    rows = list(
        data["rows"]
    )
    worlds = [
        generate_world(
            config,
            int(
                row["seed"]
            ),
        )
        for row in rows
    ]
    return (
        config,
        worlds,
        [
            float(
                row[
                    "time_optimum"
                ]
            )
            for row in rows
        ],
        [
            float(
                row[
                    "path_efficiency_optimum"
                ]
            )
            for row in rows
        ],
        [
            float(
                row[
                    "priority_service_optimum"
                ]
            )
            for row in rows
        ],
        [
            float(
                row[
                    "deadline_optimum"
                ]
            )
            for row in rows
        ],
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
    result = float(
        value
    )
    if result <= 0.0:
        raise argparse.ArgumentTypeError(
            "time-limit must be positive or unlimited"
        )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--robots",
        type=int,
        default=4,
    )
    p.add_argument(
        "--tasks",
        type=int,
        default=20,
    )
    p.add_argument(
        "--world-count",
        type=int,
        default=16,
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=118_100_000,
    )
    p.add_argument(
        "--time-limit",
        type=_optional_seconds,
        default=300.0,
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=10000,
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
    build_oracle_bank(
        robots=args.robots,
        tasks=args.tasks,
        world_count=(
            args.world_count
        ),
        seed_base=(
            args.seed_base
        ),
        time_limit=(
            args.time_limit
        ),
        output_path=Path(
            args.output
        ),
        solver_display=(
            args.solver_display
        ),
        max_candidates=(
            args.max_candidates
        ),
    )


if __name__ == "__main__":
    main()
