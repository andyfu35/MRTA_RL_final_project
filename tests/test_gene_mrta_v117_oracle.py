import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
)
from marl2d.gene_mrta_v117.oracle import (
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
