from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, World, generate_world
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum


SCENARIO_BANK_VERSION = "v110_diverse_100_v2_frozen"
DEFAULT_SEED_BASE = 95_000_000
DEFAULT_CANDIDATES = 500
DEFAULT_WORLD_COUNT = 100

# Frozen after the policy-independent 500-candidate descriptor coverage pass.
# Original slot 2 seed 95,000,034 could not prove exact optimality after
# 300 s + 900 s (MIP gap ~= 0.56185), so it was permanently replaced by
# the nearest unused descriptor candidate 95,000,442, which proved exact.
FROZEN_SCENARIO_SEEDS = (
    95000284,
    95000442,
    95000291,
    95000377,
    95000263,
    95000027,
    95000423,
    95000236,
    95000417,
    95000413,
    95000205,
    95000271,
    95000099,
    95000276,
    95000285,
    95000168,
    95000030,
    95000209,
    95000143,
    95000007,
    95000084,
    95000001,
    95000307,
    95000432,
    95000157,
    95000459,
    95000125,
    95000248,
    95000037,
    95000477,
    95000390,
    95000057,
    95000492,
    95000065,
    95000428,
    95000326,
    95000332,
    95000247,
    95000071,
    95000415,
    95000212,
    95000123,
    95000145,
    95000244,
    95000042,
    95000393,
    95000046,
    95000265,
    95000450,
    95000314,
    95000427,
    95000196,
    95000437,
    95000117,
    95000400,
    95000443,
    95000047,
    95000246,
    95000028,
    95000012,
    95000070,
    95000293,
    95000172,
    95000176,
    95000031,
    95000447,
    95000295,
    95000353,
    95000151,
    95000160,
    95000155,
    95000380,
    95000050,
    95000426,
    95000105,
    95000224,
    95000290,
    95000025,
    95000468,
    95000252,
    95000122,
    95000272,
    95000269,
    95000270,
    95000364,
    95000335,
    95000256,
    95000207,
    95000144,
    95000347,
    95000456,
    95000190,
    95000039,
    95000049,
    95000408,
    95000098,
    95000202,
    95000211,
    95000040,
    95000136,
)

FROZEN_FAILED_ORIGINAL_SEED = 95_000_034
FROZEN_REPLACEMENT_SEED = 95_000_442
FROZEN_REPLACEMENT_DESCRIPTOR_DISTANCE = 2.843834


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



def _solve_scenario_candidate(
    row: dict[str, object],
    *,
    config: EnvConfig,
    worlds_dir: Path,
    time_limit: float,
    retry_time_limit: float,
) -> tuple[dict[str, object], str]:
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

    if cached is not None:
        if bool(
            cached.get(
                "optimal",
                False,
            )
        ):
            return cached, "CACHED"

        # A previous full solve already failed to prove exact optimality.
        # Do not burn the same 300+900 second budget again on reruns.
        if int(
            cached.get(
                "oracle_attempts",
                0,
            )
        ) >= 2:
            return cached, "KNOWN_FAILED"

    world = generate_world(
        config,
        seed,
    )
    print(
        f"SOLVING seed={seed} "
        f"limit={time_limit:g}s",
        flush=True,
    )
    oracle = solve_global_time_optimum(
        world,
        config,
        time_limit=time_limit,
    )
    attempts = 1

    if (
        not oracle.optimal
        and retry_time_limit
        > time_limit
    ):
        print(
            f"RETRY seed={seed} "
            f"limit={retry_time_limit:g}s",
            flush=True,
        )
        oracle = solve_global_time_optimum(
            world,
            config,
            time_limit=retry_time_limit,
        )
        attempts = 2

    solved = {
        "seed": seed,
        "descriptor": row["descriptor"],
        "time_optimum": (
            oracle.time_optimality
        ),
        "optimal": bool(
            oracle.optimal
        ),
        "mip_gap": oracle.mip_gap,
        "solve_seconds": (
            oracle.solve_seconds
        ),
        "oracle_attempts": attempts,
    }
    _atomic_json(
        world_file,
        solved,
    )
    return solved, "NEW"


def _standardized_descriptors(
    descriptors: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        descriptors,
        dtype=np.float64,
    )
    mean = np.mean(
        x,
        axis=0,
    )
    std = np.std(
        x,
        axis=0,
    )
    return (
        x - mean[None, :]
    ) / np.maximum(
        std[None, :],
        1e-12,
    )


def _replacement_candidate_order(
    descriptors: np.ndarray,
    *,
    target_index: int,
    excluded_indices: set[int],
) -> list[tuple[int, float]]:
    z = _standardized_descriptors(
        descriptors
    )
    distances = np.linalg.norm(
        z - z[target_index][None, :],
        axis=1,
    )

    candidates = [
        (
            idx,
            float(
                distances[idx]
            ),
        )
        for idx in range(
            z.shape[0]
        )
        if idx not in excluded_indices
    ]
    candidates.sort(
        key=lambda item: (
            item[1],
            item[0],
        )
    )
    return candidates


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

    # The 100-world development set is now frozen. The descriptor-selection
    # machinery remains in this module for audit/reproduction, but build-bank
    # no longer changes membership after the freeze.
    seed_to_row = {
        int(row["seed"]): row
        for row in candidate_rows
    }
    missing = [
        seed
        for seed in FROZEN_SCENARIO_SEEDS
        if seed not in seed_to_row
    ]
    if missing:
        raise RuntimeError(
            "Frozen V1.10 seeds are missing from "
            f"the candidate cache: {missing}"
        )
    if len(FROZEN_SCENARIO_SEEDS) != args.worlds:
        raise RuntimeError(
            "Frozen V1.10 scenario set must contain "
            f"exactly {args.worlds} unique worlds."
        )
    if len(set(FROZEN_SCENARIO_SEEDS)) != args.worlds:
        raise RuntimeError(
            "Frozen V1.10 scenario set contains duplicate seeds."
        )

    selected_rows = [
        seed_to_row[seed]
        for seed in FROZEN_SCENARIO_SEEDS
    ]

    solved_rows: list[
        dict[str, object]
    ] = []

    for rank, row in enumerate(
        selected_rows,
        start=1,
    ):
        seed = int(row["seed"])
        solved, source = (
            _solve_scenario_candidate(
                row,
                config=config,
                worlds_dir=worlds_dir,
                time_limit=(
                    args.time_limit
                ),
                retry_time_limit=(
                    args.retry_time_limit
                ),
            )
        )

        print(
            f"[{rank:03d}/"
            f"{args.worlds}] "
            f"{source} seed={seed} "
            f"optimal="
            f"{solved.get('optimal')} "
            f"T*="
            f"{solved.get('time_optimum')} "
            f"gap="
            f"{solved.get('mip_gap')}",
            flush=True,
        )

        if not bool(
            solved.get(
                "optimal",
                False,
            )
        ):
            raise RuntimeError(
                "Frozen V1.10 scenario bank requires "
                "a proven exact optimum for every world. "
                f"Seed {seed} is not exact."
            )

        solved_rows.append(solved)

    payload = {
        "version": (
            SCENARIO_BANK_VERSION
        ),
        "selection": {
            "policy_conditioned": False,
            "method": (
                "frozen_standardized_descriptor_"
                "coverage"
            ),
            "frozen": True,
            "freeze_note": (
                "Slot 2 seed 95000034 was permanently "
                "replaced by nearest unused candidate "
                "95000442 after exact MILP failed to "
                "prove optimality within 300s + 900s."
            ),
            "failed_original_seed": (
                FROZEN_FAILED_ORIGINAL_SEED
            ),
            "replacement_seed": (
                FROZEN_REPLACEMENT_SEED
            ),
            "replacement_descriptor_distance": (
                FROZEN_REPLACEMENT_DESCRIPTOR_DISTANCE
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
            "automatic_exact_replacement": False,
            "frozen_scenario_seeds": list(
                FROZEN_SCENARIO_SEEDS
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
