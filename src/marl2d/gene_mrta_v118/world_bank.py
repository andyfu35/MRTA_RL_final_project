from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    World,
    generate_world,
)
from marl2d.gene_mrta_v115.scaling import (
    scale_config,
)


EPS = 1e-12
BANK_VERSION = (
    "v118_constructive_feasible_world_bank_v2"
)


@dataclass(frozen=True)
class FeasibleWitness:
    routes: tuple[tuple[int, ...], ...]
    finish_times: tuple[float, ...]
    travel_energy: tuple[float, ...]
    batteries: tuple[float, ...]


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


def _construct_witness(
    world: World,
    config: EnvConfig,
    *,
    battery_reserve_fraction: float,
) -> FeasibleWitness | None:
    R = config.num_robots
    N = config.num_tasks
    per_robot_cap = int(
        math.ceil(
            N / max(R, 1)
        )
    )

    routes: list[
        list[int]
    ] = [
        []
        for _ in range(R)
    ]
    tail_node = np.arange(
        R,
        dtype=np.int64,
    )
    tail_time = np.zeros(
        R,
        dtype=np.float64,
    )
    travel_energy = np.zeros(
        R,
        dtype=np.float64,
    )
    remaining = set(
        range(N)
    )

    while remaining:
        candidates: list[
            tuple[
                float,
                float,
                int,
                int,
                float,
            ]
        ] = []

        for robot in range(R):
            if (
                len(
                    routes[robot]
                )
                >= per_robot_cap
            ):
                continue

            node = int(
                tail_node[
                    robot
                ]
            )
            for task in remaining:
                distance = float(
                    world.path_to_tasks[
                        node,
                        task,
                    ]
                )
                if not math.isfinite(
                    distance
                ):
                    continue

                finish = float(
                    tail_time[
                        robot
                    ]
                    + distance
                    / config.robot_speed
                    + float(
                        world.task_service_times[
                            task
                        ]
                    )
                )
                energy = float(
                    travel_energy[
                        robot
                    ]
                    + distance
                    * config.energy_per_distance
                )
                reserve = (
                    battery_reserve_fraction
                    * config.battery_capacity
                )

                if (
                    finish
                    > config.episode_time
                    + EPS
                    or energy
                    + reserve
                    > config.battery_capacity
                    + EPS
                ):
                    continue

                # Earliest projected finish first. A tiny route-length term
                # keeps the constructive witness balanced without becoming an
                # optimization objective used by the Gene Bank.
                key = (
                    finish
                    + 1e-6
                    * len(
                        routes[
                            robot
                        ]
                    ),
                    energy,
                    robot,
                    task,
                    distance,
                )
                candidates.append(
                    key
                )

        if not candidates:
            return None

        (
            _finish_key,
            _energy_key,
            robot,
            task,
            distance,
        ) = min(
            candidates
        )

        tail_time[
            robot
        ] += (
            distance
            / config.robot_speed
            + float(
                world.task_service_times[
                    task
                ]
            )
        )
        travel_energy[
            robot
        ] += (
            distance
            * config.energy_per_distance
        )
        routes[
            robot
        ].append(
            task
        )
        tail_node[
            robot
        ] = (
            R + task
        )
        remaining.remove(
            task
        )

    assigned = sorted(
        task
        for route in routes
        for task in route
    )
    if assigned != list(
        range(N)
    ):
        raise RuntimeError(
            "Constructive witness did not assign every task exactly once"
        )

    reserve = (
        battery_reserve_fraction
        * config.battery_capacity
    )
    required = (
        travel_energy
        + reserve
    )
    if np.any(
        required
        > config.battery_capacity
        + EPS
    ):
        return None

    batteries = np.maximum(
        world.robot_initial_batteries,
        required,
    )
    if np.any(
        batteries
        > config.battery_capacity
        + EPS
    ):
        return None

    return FeasibleWitness(
        routes=tuple(
            tuple(
                route
            )
            for route in routes
        ),
        finish_times=tuple(
            float(
                value
            )
            for value in tail_time
        ),
        travel_energy=tuple(
            float(
                value
            )
            for value in travel_energy
        ),
        batteries=tuple(
            float(
                value
            )
            for value in batteries
        ),
    )


def construct_feasible_world(
    config: EnvConfig,
    seed: int,
    *,
    battery_reserve_fraction: float = 0.05,
) -> tuple[
    World,
    FeasibleWitness,
] | None:
    raw = generate_world(
        config,
        seed,
    )
    witness = _construct_witness(
        raw,
        config,
        battery_reserve_fraction=(
            battery_reserve_fraction
        ),
    )
    if witness is None:
        return None

    world = World(
        robot_positions=(
            raw.robot_positions.copy()
        ),
        robot_initial_batteries=np.asarray(
            witness.batteries,
            dtype=np.float64,
        ),
        task_positions=(
            raw.task_positions.copy()
        ),
        task_service_times=(
            raw.task_service_times.copy()
        ),
        task_priorities=(
            raw.task_priorities.copy()
        ),
        task_deadlines=(
            raw.task_deadlines.copy()
        ),
        obstacles=(
            raw.obstacles.copy()
        ),
        path_to_tasks=(
            raw.path_to_tasks.copy()
        ),
    )
    return (
        world,
        witness,
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
        path.suffix
        + ".tmp"
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


def build_world_bank(
    *,
    robots: int,
    tasks: int,
    world_count: int,
    seed_base: int,
    output_path: Path,
    max_candidates: int = 10000,
    battery_reserve_fraction: float = 0.05,
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

    rows: list[
        dict[str, Any]
    ] = []
    candidate_offset = 0

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
                "Existing constructive bank has incompatible version"
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
                "Existing constructive bank does not match protocol"
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

    while len(
        rows
    ) < world_count:
        if (
            candidate_offset
            >= max_candidates
        ):
            raise RuntimeError(
                "Could not construct enough feasible worlds"
            )

        seed = int(
            seed_base
            + candidate_offset
        )
        candidate_offset += 1

        built = (
            construct_feasible_world(
                config,
                seed,
                battery_reserve_fraction=(
                    battery_reserve_fraction
                ),
            )
        )
        if built is None:
            print(
                "V118_FAST_WORLD_REJECT "
                f"seed={seed}",
                flush=True,
            )
            continue

        _world, witness = built
        row = {
            "world_index": len(
                rows
            ),
            "seed": seed,
            "completion_witness": 1.0,
            "witness_routes": [
                list(
                    route
                )
                for route
                in witness.routes
            ],
            "witness_finish_times": list(
                witness.finish_times
            ),
            "witness_travel_energy": list(
                witness.travel_energy
            ),
            "witness_batteries": list(
                witness.batteries
            ),
        }
        rows.append(
            row
        )

        payload = {
            "version": (
                BANK_VERSION
            ),
            "protocol": (
                "constructive_feasibility_raw_capability_v2"
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
            "battery_reserve_fraction": float(
                battery_reserve_fraction
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
            "V118_FAST_WORLD_ACCEPT "
            + json.dumps(
                {
                    "world_index": row[
                        "world_index"
                    ],
                    "seed": seed,
                    "max_witness_finish": max(
                        witness.finish_times
                    ),
                    "max_witness_energy": max(
                        witness.travel_energy
                    ),
                }
            ),
            flush=True,
        )

    payload = {
        "version": (
            BANK_VERSION
        ),
        "protocol": (
            "constructive_feasibility_raw_capability_v2"
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
        "battery_reserve_fraction": float(
            battery_reserve_fraction
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
        f"V118_FAST_WORLD_BANK={output_path}",
        flush=True,
    )
    return payload


def load_world_bank(
    path: Path,
) -> tuple[
    EnvConfig,
    list[World],
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
            "Unsupported constructive V1.18 world bank version"
        )

    config = EnvConfig(
        **data[
            "config"
        ]
    )
    reserve = float(
        data.get(
            "battery_reserve_fraction",
            0.05,
        )
    )
    worlds: list[
        World
    ] = []

    for row in data[
        "rows"
    ]:
        built = (
            construct_feasible_world(
                config,
                int(
                    row[
                        "seed"
                    ]
                ),
                battery_reserve_fraction=(
                    reserve
                ),
            )
        )
        if built is None:
            raise RuntimeError(
                "Saved constructive world no longer regenerates as feasible"
            )
        world, witness = built
        expected_routes = tuple(
            tuple(
                int(task)
                for task in route
            )
            for route
            in row[
                "witness_routes"
            ]
        )
        if (
            witness.routes
            != expected_routes
        ):
            raise RuntimeError(
                "Constructive witness regeneration mismatch"
            )
        worlds.append(
            world
        )

    return (
        config,
        worlds,
    )


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
        default=100,
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=117_100_000,
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=10000,
    )
    p.add_argument(
        "--battery-reserve-fraction",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    build_world_bank(
        robots=args.robots,
        tasks=args.tasks,
        world_count=(
            args.world_count
        ),
        seed_base=(
            args.seed_base
        ),
        output_path=Path(
            args.output
        ),
        max_candidates=(
            args.max_candidates
        ),
        battery_reserve_fraction=(
            args.battery_reserve_fraction
        ),
    )


if __name__ == "__main__":
    main()
