from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


FIXED_WORLD_SEEDS: tuple[int, ...] = (
    76411066, 20914714, 48464897, 48747897, 78427502, 36459867, 85032859,
    41298699, 80265923, 93935430, 80428412, 83137887, 14312654, 24459496,
    11540772, 41585173, 25635750, 20785042, 44541239, 71966566, 87239350,
    67288471, 59467204, 39221858, 10748959, 21529777, 31943953, 76927726,
    28016269, 38472493, 27188234, 32291850, 97018234, 71445721, 18045601,
    13925305, 99507849, 23098986, 68532008, 18453899, 34512546, 46646799,
    26531260, 61965194, 75017088, 40593702, 31334399, 57898148, 40450881,
    15166032, 78474883, 76563710, 10005663, 94556590, 42801358, 97899269,
    38111921, 23112969, 26523958, 15407303, 65135457, 76587331, 32764118,
    15429500, 67278671, 91293211, 57147349, 25716984, 81221219, 54402251,
    87848086, 15092607, 51408569, 49371622, 74786017, 34828591, 86123894,
    76725638, 30688859, 84288947, 16137152, 27396211, 24997701, 72243566,
    14519661, 80843695, 24264912, 76301632, 96434001, 40801047, 74861251,
    50409639, 28951210, 82384530, 19476354, 19740558, 18807391, 69660961,
    88892650, 89668964,
)


@dataclass(frozen=True)
class WorldConfig:
    world_size: float = 100.0
    robot_min: int = 5
    robot_max: int = 20
    task_min: int = 10
    task_max: int = 100
    robot_speed: float = 5.0
    priority_min: int = 1
    priority_max: int = 10
    service_time_min: float = 1.0
    service_time_max: float = 12.0
    deadline_slack_min: float = 1.10
    deadline_slack_max: float = 1.35
    deadline_additive_slack_min: float = 2.0
    deadline_additive_slack_max: float = 8.0

    @property
    def diagonal(self) -> float:
        return math.sqrt(2.0) * self.world_size


@dataclass(frozen=True)
class World:
    seed: int
    robot_positions: np.ndarray
    task_positions: np.ndarray
    task_priorities: np.ndarray
    task_deadlines: np.ndarray
    task_service_times: np.ndarray
    baseline_completion_times: np.ndarray
    baseline_makespan: float

    @property
    def robot_count(self) -> int:
        return int(self.robot_positions.shape[0])

    @property
    def task_count(self) -> int:
        return int(self.task_positions.shape[0])

    def task_features(self) -> np.ndarray:
        """Raw input schema: [x, y, priority, deadline, service_time]."""
        return np.column_stack(
            [
                self.task_positions,
                self.task_priorities,
                self.task_deadlines,
                self.task_service_times,
            ]
        ).astype(np.float32, copy=False)

    def initial_robot_features(self) -> np.ndarray:
        """Raw schema: [x, y, accumulated_distance, estimated_finish_time]."""
        zeros = np.zeros((self.robot_count, 2), dtype=np.float32)
        return np.concatenate(
            [self.robot_positions.astype(np.float32, copy=False), zeros],
            axis=1,
        )


def _feasible_baseline_completion_times(
    robot_positions: np.ndarray,
    task_positions: np.ndarray,
    service_times: np.ndarray,
    robot_speed: float,
) -> tuple[np.ndarray, float]:
    """Construct one feasible schedule used only to make deadlines sane."""
    robot_tail = np.asarray(robot_positions, dtype=np.float64).copy()
    task_positions = np.asarray(task_positions, dtype=np.float64)
    service_times = np.asarray(service_times, dtype=np.float64)
    finish = np.zeros(robot_tail.shape[0], dtype=np.float64)
    remaining = np.ones(task_positions.shape[0], dtype=bool)
    completion = np.zeros(task_positions.shape[0], dtype=np.float64)

    for _ in range(task_positions.shape[0]):
        distances = np.linalg.norm(
            robot_tail[:, None, :] - task_positions[None, :, :],
            axis=-1,
        )
        projected = (
            finish[:, None]
            + distances / float(robot_speed)
            + service_times[None, :]
        )
        projected[:, ~remaining] = np.inf
        flat = int(np.argmin(projected))
        robot = flat // task_positions.shape[0]
        task = flat % task_positions.shape[0]
        value = float(projected[robot, task])
        completion[task] = value
        finish[robot] = value
        robot_tail[robot] = task_positions[task]
        remaining[task] = False

    return completion, float(np.max(finish))


def generate_world(seed: int, config: WorldConfig | None = None) -> World:
    config = config or WorldConfig()
    rng = np.random.default_rng(int(seed))

    robot_count = int(rng.integers(config.robot_min, config.robot_max + 1))
    task_count = int(rng.integers(config.task_min, config.task_max + 1))
    robot_positions = rng.uniform(
        0.0, config.world_size, size=(robot_count, 2)
    ).astype(np.float64)
    task_positions = rng.uniform(
        0.0, config.world_size, size=(task_count, 2)
    ).astype(np.float64)
    task_priorities = rng.integers(
        config.priority_min,
        config.priority_max + 1,
        size=task_count,
    ).astype(np.float64)
    task_service_times = rng.uniform(
        config.service_time_min,
        config.service_time_max,
        size=task_count,
    ).astype(np.float64)

    baseline_completion, baseline_makespan = _feasible_baseline_completion_times(
        robot_positions,
        task_positions,
        task_service_times,
        config.robot_speed,
    )
    multiplicative_slack = rng.uniform(
        config.deadline_slack_min,
        config.deadline_slack_max,
        size=task_count,
    )
    additive_slack = rng.uniform(
        config.deadline_additive_slack_min,
        config.deadline_additive_slack_max,
        size=task_count,
    )
    task_deadlines = baseline_completion * multiplicative_slack + additive_slack

    return World(
        seed=int(seed),
        robot_positions=robot_positions,
        task_positions=task_positions,
        task_priorities=task_priorities,
        task_deadlines=task_deadlines.astype(np.float64),
        task_service_times=task_service_times,
        baseline_completion_times=baseline_completion,
        baseline_makespan=baseline_makespan,
    )


def fixed_worlds(config: WorldConfig | None = None, count: int = 100) -> list[World]:
    if count <= 0 or count > len(FIXED_WORLD_SEEDS):
        raise ValueError(f"count must be in [1, {len(FIXED_WORLD_SEEDS)}]")
    return [generate_world(seed, config) for seed in FIXED_WORLD_SEEDS[:count]]
