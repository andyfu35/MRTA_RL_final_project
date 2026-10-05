from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    World,
    build_world,
)


SCHEMA_VERSION = "gene_mrta_v117_world_v1"


@dataclass(frozen=True)
class RobotSpec:
    robot_id: int
    start_position: tuple[float, float]
    initial_battery: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": int(self.robot_id),
            "start_position": [
                float(self.start_position[0]),
                float(self.start_position[1]),
            ],
            "initial_battery": float(self.initial_battery),
        }


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    position: tuple[float, float]
    service_time: float
    priority: float
    deadline: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "position": [
                float(self.position[0]),
                float(self.position[1]),
            ],
            "service_time": float(self.service_time),
            "priority": float(self.priority),
            "deadline": float(self.deadline),
        }


@dataclass(frozen=True)
class WorldSpec:
    seed: int | None
    robots: tuple[RobotSpec, ...]
    tasks: tuple[TaskSpec, ...]
    obstacles: tuple[
        tuple[float, float, float, float],
        ...,
    ]

    def to_dict(
        self,
        config: EnvConfig,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": (
                None
                if self.seed is None
                else int(self.seed)
            ),
            "global": {
                "world_size": float(config.world_size),
                "episode_horizon": float(config.episode_time),
                "robot_speed": float(config.robot_speed),
                "battery_capacity": float(config.battery_capacity),
                "energy_per_distance": float(config.energy_per_distance),
                "grid_resolution": float(config.grid_resolution),
            },
            "robots": [
                robot.to_dict()
                for robot in self.robots
            ],
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ],
            "obstacles": [
                [
                    float(value)
                    for value in obstacle
                ]
                for obstacle in self.obstacles
            ],
        }


def world_to_spec(
    world: World,
    *,
    seed: int | None = None,
) -> WorldSpec:
    robots = tuple(
        RobotSpec(
            robot_id=i,
            start_position=(
                float(world.robot_positions[i, 0]),
                float(world.robot_positions[i, 1]),
            ),
            initial_battery=float(
                world.robot_initial_batteries[i]
            ),
        )
        for i in range(
            world.robot_positions.shape[0]
        )
    )
    tasks = tuple(
        TaskSpec(
            task_id=j,
            position=(
                float(world.task_positions[j, 0]),
                float(world.task_positions[j, 1]),
            ),
            service_time=float(
                world.task_service_times[j]
            ),
            priority=float(
                world.task_priorities[j]
            ),
            deadline=float(
                world.task_deadlines[j]
            ),
        )
        for j in range(
            world.task_positions.shape[0]
        )
    )
    obstacles = tuple(
        tuple(
            float(value)
            for value in row
        )
        for row in np.asarray(
            world.obstacles,
            dtype=np.float64,
        )
    )
    return WorldSpec(
        seed=seed,
        robots=robots,
        tasks=tasks,
        obstacles=obstacles,
    )


def spec_to_world(
    spec: WorldSpec,
    config: EnvConfig,
) -> World:
    if len(spec.robots) != config.num_robots:
        raise ValueError(
            "Robot count does not match EnvConfig"
        )
    if len(spec.tasks) != config.num_tasks:
        raise ValueError(
            "Task count does not match EnvConfig"
        )

    robot_ids = [
        robot.robot_id
        for robot in spec.robots
    ]
    task_ids = [
        task.task_id
        for task in spec.tasks
    ]
    if robot_ids != list(
        range(config.num_robots)
    ):
        raise ValueError(
            "robot_id must be contiguous and ordered from 0"
        )
    if task_ids != list(
        range(config.num_tasks)
    ):
        raise ValueError(
            "task_id must be contiguous and ordered from 0"
        )

    robot_positions = np.asarray(
        [
            robot.start_position
            for robot in spec.robots
        ],
        dtype=np.float64,
    )
    batteries = np.asarray(
        [
            robot.initial_battery
            for robot in spec.robots
        ],
        dtype=np.float64,
    )
    task_positions = np.asarray(
        [
            task.position
            for task in spec.tasks
        ],
        dtype=np.float64,
    )
    services = np.asarray(
        [
            task.service_time
            for task in spec.tasks
        ],
        dtype=np.float64,
    )
    priorities = np.asarray(
        [
            task.priority
            for task in spec.tasks
        ],
        dtype=np.float64,
    )
    deadlines = np.asarray(
        [
            task.deadline
            for task in spec.tasks
        ],
        dtype=np.float64,
    )
    obstacles = np.asarray(
        spec.obstacles,
        dtype=np.float64,
    )
    if obstacles.size == 0:
        obstacles = np.zeros(
            (0, 4),
            dtype=np.float64,
        )

    return build_world(
        config,
        robot_positions,
        batteries,
        task_positions,
        services,
        priorities,
        deadlines,
        obstacles,
    )
