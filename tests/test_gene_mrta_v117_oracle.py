import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
)
from marl2d.gene_mrta_v117.oracle import (
    solve_global_completion_optimum,
    solve_global_deadline_optimum,
    solve_global_path_efficiency_optimum,
    solve_global_priority_optimum,
)


def test_priority_oracle_selects_highest_value_task():
    config = EnvConfig(
        num_robots=1,
        num_tasks=2,
        robot_speed=100.0,
        service_time_min=9.0,
        service_time_max=9.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=5.0,
        deadline_max=10.0,
        episode_time=10.0,
        obstacle_count=0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        robot_positions=np.asarray(
            [[10.0, 10.0]],
            dtype=np.float64,
        ),
        robot_initial_batteries=np.asarray(
            [100.0],
            dtype=np.float64,
        ),
        task_positions=np.asarray(
            [
                [11.0, 10.0],
                [12.0, 10.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.asarray(
            [9.0, 9.0],
            dtype=np.float64,
        ),
        task_priorities=np.asarray(
            [0.1, 1.0],
            dtype=np.float64,
        ),
        task_deadlines=np.asarray(
            [10.0, 10.0],
            dtype=np.float64,
        ),
        obstacles=np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )

    optimum = solve_global_priority_optimum(
        world,
        config,
        time_limit=30.0,
    )

    assert optimum.optimal
    assert optimum.completed_tasks == 1
    assert optimum.routes == ((1,),)
    assert np.isclose(
        optimum.priority_satisfaction,
        1.0 / 1.1,
    )


def test_completion_oracle_maximizes_task_count():
    config = EnvConfig(
        num_robots=1,
        num_tasks=2,
        robot_speed=100.0,
        service_time_min=9.0,
        service_time_max=9.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=5.0,
        deadline_max=10.0,
        episode_time=10.0,
        obstacle_count=0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        robot_positions=np.asarray(
            [[10.0, 10.0]],
            dtype=np.float64,
        ),
        robot_initial_batteries=np.asarray(
            [100.0],
            dtype=np.float64,
        ),
        task_positions=np.asarray(
            [
                [11.0, 10.0],
                [12.0, 10.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.asarray(
            [9.0, 9.0],
            dtype=np.float64,
        ),
        task_priorities=np.asarray(
            [0.1, 1.0],
            dtype=np.float64,
        ),
        task_deadlines=np.asarray(
            [10.0, 10.0],
            dtype=np.float64,
        ),
        obstacles=np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )

    optimum = solve_global_completion_optimum(
        world,
        config,
        time_limit=30.0,
    )

    assert optimum.optimal
    assert optimum.completed_tasks == 1
    assert np.isclose(
        optimum.completion,
        0.5,
    )


def test_deadline_oracle_counts_only_on_time_tasks():
    config = EnvConfig(
        num_robots=1,
        num_tasks=2,
        robot_speed=100.0,
        service_time_min=6.0,
        service_time_max=6.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=6.1,
        deadline_max=10.0,
        episode_time=20.0,
        obstacle_count=0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        robot_positions=np.asarray(
            [[10.0, 10.0]],
            dtype=np.float64,
        ),
        robot_initial_batteries=np.asarray(
            [100.0],
            dtype=np.float64,
        ),
        task_positions=np.asarray(
            [
                [11.0, 10.0],
                [12.0, 10.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.asarray(
            [6.0, 6.0],
            dtype=np.float64,
        ),
        task_priorities=np.asarray(
            [0.5, 0.5],
            dtype=np.float64,
        ),
        task_deadlines=np.asarray(
            [6.1, 6.1],
            dtype=np.float64,
        ),
        obstacles=np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )

    optimum = solve_global_deadline_optimum(
        world,
        config,
        time_limit=30.0,
    )

    assert optimum.optimal
    assert optimum.completed_tasks == 1
    assert np.isclose(
        optimum.deadline_satisfaction,
        0.5,
    )


def test_path_efficiency_oracle_prefers_shorter_feasible_task():
    config = EnvConfig(
        world_size=100.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=100.0,
        service_time_min=9.0,
        service_time_max=9.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=10.0,
        deadline_max=10.0,
        episode_time=10.0,
        obstacle_count=0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        robot_positions=np.asarray(
            [[10.0, 10.0]],
            dtype=np.float64,
        ),
        robot_initial_batteries=np.asarray(
            [100.0],
            dtype=np.float64,
        ),
        task_positions=np.asarray(
            [
                [11.0, 10.0],
                [60.0, 10.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.asarray(
            [9.0, 9.0],
            dtype=np.float64,
        ),
        task_priorities=np.asarray(
            [0.5, 0.5],
            dtype=np.float64,
        ),
        task_deadlines=np.asarray(
            [10.0, 10.0],
            dtype=np.float64,
        ),
        obstacles=np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )

    optimum = solve_global_path_efficiency_optimum(
        world,
        config,
        time_limit=30.0,
    )

    expected = (
        1.0
        - float(
            world.path_to_tasks[0, 0]
        )
        / config.diagonal
    ) / 2.0

    assert optimum.optimal
    assert optimum.routes == ((0,),)
    assert np.isclose(
        optimum.path_efficiency,
        expected,
    )
