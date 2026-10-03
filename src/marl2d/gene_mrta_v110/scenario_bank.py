from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, World, generate_world
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum


SCENARIO_BANK_VERSION = "v110_diverse_100_v1"
DEFAULT_SEED_BASE = 95_000_000
DEFAULT_CANDIDATES = 500
DEFAULT_WORLD_COUNT = 100


def _pairwise_mean_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= 1:
        return 0.0
    delta = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    upper = distances[np.triu_indices(points.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def world_descriptor(
    world: World,
    config: EnvConfig,
) -> np.ndarray:
    """
    Policy-independent scenario descriptor.

    The descriptor intentionally uses only world structure. No trained Gene
    score enters scenario selection, avoiding policy-conditioned seed mining.
    """
    R = config.num_robots
    T = config.num_tasks

    robot_task_euclidean = np.linalg.norm(
        world.robot_positions[:, None, :]
        - world.task_positions[None, :, :],
        axis=-1,
    )
    start_paths = world.path_to_tasks[:R, :]
    finite = np.isfinite(start_paths)

    start_finishes = (
        start_paths / config.robot_speed
        + world.task_service_times[None, :]
    )
    start_energy = (
        start_paths * config.energy_per_distance
    )
    feasible = (
        finite
        & (
            start_finishes
            <= config.episode_time + 1e-12
        )
        & (
            start_energy
            <= world.robot_initial_batteries[:, None]
            + 1e-12
        )
    )
    task_degree = np.sum(feasible, axis=0)

    euclidean_safe = np.maximum(
        robot_task_euclidean,
        config.grid_resolution,
    )
    detour = np.where(
        finite,
        start_paths / euclidean_safe,
        np.nan,
    )
    finite_detour = detour[np.isfinite(detour)]

    obstacle_area = 0.0
    if world.obstacles.size:
        widths = (
            world.obstacles[:, 2]
            - world.obstacles[:, 0]
        )
        heights = (
            world.obstacles[:, 3]
            - world.obstacles[:, 1]
        )
        obstacle_area = float(
            np.sum(widths * heights)
            / max(
                config.world_size
                * config.world_size,
                1e-12,
            )
        )

    return np.asarray(
        [
            float(
                np.mean(
                    world.robot_initial_batteries
                )
                / config.battery_capacity
            ),
            float(
                np.std(
                    world.robot_initial_batteries
                )
                / config.battery_capacity
            ),
            float(
                np.mean(
                    world.task_service_times
                )
                / config.episode_time
            ),
            float(
                np.std(
                    world.task_service_times
                )
                / config.episode_time
            ),
            float(
                np.mean(
                    world.task_deadlines
                )
                / config.episode_time
            ),
            float(
                np.std(
                    world.task_deadlines
                )
                / config.episode_time
            ),
            float(
                np.mean(
                    world.task_priorities
                )
            ),
            float(
                _pairwise_mean_distance(
                    world.task_positions
                )
                / config.diagonal
            ),
            float(
                np.mean(
                    robot_task_euclidean
                )
                / config.diagonal
            ),
            float(
                np.mean(
                    finite_detour
                )
                if finite_detour.size
                else 10.0
            ),
            float(
                np.mean(feasible)
            ),
            float(
                np.mean(task_degree == 0)
            ),
            float(
                np.mean(task_degree == 1)
            ),
            obstacle_area,
            float(
                np.mean(
                    np.isfinite(
                        world.path_to_tasks
                    )
                )
            ),
        ],
        dtype=np.float64,
    )


def select_diverse_indices(
    descriptors: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Deterministic farthest-point coverage in standardized descriptor space.
    """
    x = np.asarray(
        descriptors,
        dtype=np.float64,
    )
    if x.ndim != 2:
        raise ValueError(
            "descriptors must be 2D"
        )
    if count <= 0:
        raise ValueError(
            "count must be positive"
        )
    if count > x.shape[0]:
        raise ValueError(
            "count cannot exceed candidates"
        )

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = (
        x - mean[None, :]
    ) / np.maximum(
        std[None, :],
        1e-12,
    )

    centroid_distance = np.linalg.norm(
        z,
        axis=1,
    )
    first = int(
        np.argmax(
            centroid_distance
        )
    )

    selected = [first]
    min_distance = np.linalg.norm(
        z - z[first][None, :],
        axis=1,
    )
    min_distance[first] = -np.inf

    while len(selected) < count:
        idx = int(
            np.argmax(
                min_distance
            )
        )
        selected.append(idx)
        distance = np.linalg.norm(
            z - z[idx][None, :],
            axis=1,
        )
        min_distance = np.minimum(
            min_distance,
            distance,
        )
        min_distance[
            np.asarray(
                selected,
                dtype=np.int64,
            )
        ] = -np.inf

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def _atomic_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temp.replace(path)


def _validate_seed_range(
    seed_base: int,
    candidate_count: int,
) -> None:
    low = int(seed_base)
    high = low + int(candidate_count) - 1
    protected = (
        (98_000_000, 98_000_099),
        (99_000_000, 99_000_099),
    )
    for p_low, p_high in protected:
        overlaps = not (
            high < p_low
            or low > p_high
        )
        if overlaps:
            raise ValueError(
                "V1.10 scenario candidates may not overlap "
                f"protected range {p_low}-{p_high}"
            )


def build_scenario_bank(
    args: argparse.Namespace,
) -> Path:
    if args.worlds != DEFAULT_WORLD_COUNT:
        raise ValueError(
            "V1.10 mating bank is frozen at 100 worlds"
        )
    _validate_seed_range(
        args.seed_base,
        args.candidates,
    )

    config = EnvConfig()
    output_dir = Path(
        args.output_dir
    )
    worlds_dir = (
        output_dir / "worlds"
    )
    worlds_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor_cache = (
        output_dir
        / "candidate_descriptors.json"
    )

    if descriptor_cache.exists():
        candidate_rows = json.loads(
            descriptor_cache.read_text(
                encoding="utf-8"
            )
        )["candidates"]
    else:
        candidate_rows: list[
            dict[str, object]
        ] = []
        for idx in range(
            args.candidates
        ):
            seed = (
                args.seed_base + idx
            )
            world = generate_world(
                config,
                seed,
            )
            descriptor = (
                world_descriptor(
                    world,
                    config,
                )
            )
            candidate_rows.append(
                {
                    "seed": seed,
                    "descriptor": (
                        descriptor.tolist()
                    ),
                }
            )
            if (
                (idx + 1) % 25 == 0
                or idx
                == args.candidates - 1
            ):
                print(
                    f"DESCRIPTORS "
                    f"{idx+1}/{args.candidates}"
                )

        _atomic_json(
            descriptor_cache,
            {
                "version": (
                    SCENARIO_BANK_VERSION
                ),
                "seed_base": (
                    args.seed_base
                ),
                "candidate_count": (
                    args.candidates
                ),
                "candidates": (
                    candidate_rows
                ),
            },
        )

    descriptors = np.asarray(
        [
            row["descriptor"]
            for row in candidate_rows
        ],
        dtype=np.float64,
    )
    selected_ids = (
        select_diverse_indices(
            descriptors,
            args.worlds,
        )
    )
    selected_rows = [
        candidate_rows[int(idx)]
        for idx in selected_ids
    ]

    solved_rows: list[
        dict[str, object]
    ] = []

    for rank, row in enumerate(
        selected_rows,
        start=1,
    ):
        seed = int(row["seed"])
        world_file = (
            worlds_dir
            / f"world_{seed}.json"
        )

        cached = None
        if world_file.exists():
            try:
                cached = json.loads(
                    world_file.read_text(
                        encoding="utf-8"
                    )
                )
            except json.JSONDecodeError:
                cached = None

        if (
            cached is not None
            and bool(
                cached.get(
                    "optimal",
                    False,
                )
            )
        ):
            solved = cached
            source = "CACHED"
        else:
            world = generate_world(
                config,
                seed,
            )
            oracle = (
                solve_global_time_optimum(
                    world,
                    config,
                    time_limit=(
                        args.time_limit
                    ),
                )
            )
            attempts = 1
            if (
                not oracle.optimal
                and args.retry_time_limit
                > args.time_limit
            ):
                oracle = (
                    solve_global_time_optimum(
                        world,
                        config,
                        time_limit=(
                            args.retry_time_limit
                        ),
                    )
                )
                attempts = 2

            solved = {
                "seed": seed,
                "descriptor": (
                    row["descriptor"]
                ),
                "time_optimum": (
                    oracle.time_optimality
                ),
                "optimal": bool(
                    oracle.optimal
                ),
                "mip_gap": (
                    oracle.mip_gap
                ),
                "solve_seconds": (
                    oracle.solve_seconds
                ),
                "oracle_attempts": (
                    attempts
                ),
            }
            _atomic_json(
                world_file,
                solved,
            )
            source = "NEW"

        solved_rows.append(
            solved
        )
        print(
            f"[{rank:03d}/"
            f"{args.worlds}] "
            f"{source} seed={seed} "
            f"optimal="
            f"{solved.get('optimal')} "
            f"T*="
            f"{solved.get('time_optimum')}"
        )

    not_optimal = [
        row
        for row in solved_rows
        if not bool(
            row.get(
                "optimal",
                False,
            )
        )
    ]
    if not_optimal:
        raise RuntimeError(
            "Scenario bank incomplete: "
            f"{len(not_optimal)} worlds "
            "do not have a proven optimum. "
            "Rerun to retry cached failures."
        )

    payload = {
        "version": (
            SCENARIO_BANK_VERSION
        ),
        "selection": {
            "policy_conditioned": False,
            "method": (
                "standardized_descriptor_"
                "farthest_point_sampling"
            ),
            "seed_base": (
                args.seed_base
            ),
            "candidate_count": (
                args.candidates
            ),
            "world_count": (
                args.worlds
            ),
            "protected_ranges": [
                "98,000,000-98,000,099",
                "99,000,000-99,000,099",
            ],
        },
        "environment": asdict(
            config
        ),
        "worlds": solved_rows,
    }

    bank_path = (
        output_dir
        / "scenario_bank_100.json"
    )
    _atomic_json(
        bank_path,
        payload,
    )

    print(
        f"SCENARIO_BANK={bank_path}"
    )
    return bank_path


def load_scenario_bank(
    path: Path,
    config: EnvConfig,
) -> tuple[
    list[World],
    np.ndarray,
    list[int],
]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    rows = data["worlds"]
    if len(rows) != 100:
        raise ValueError(
            "Expected exactly 100 mating scenarios"
        )

    for row in rows:
        if not bool(
            row.get(
                "optimal",
                False,
            )
        ):
            raise ValueError(
                "Every mating scenario needs "
                "a proven exact optimum"
            )

    seeds = [
        int(row["seed"])
        for row in rows
    ]
    worlds = [
        generate_world(
            config,
            seed,
        )
        for seed in seeds
    ]
    stars = np.asarray(
        [
            float(
                row[
                    "time_optimum"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return worlds, stars, seeds


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seed-base",
        type=int,
        default=(
            DEFAULT_SEED_BASE
        ),
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=(
            DEFAULT_CANDIDATES
        ),
    )
    p.add_argument(
        "--worlds",
        type=int,
        default=(
            DEFAULT_WORLD_COUNT
        ),
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
    )
    p.add_argument(
        "--retry-time-limit",
        type=float,
        default=900.0,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "gene_mrta_v110_scenario_bank"
        ),
    )
    return p


def main() -> None:
    build_scenario_bank(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
