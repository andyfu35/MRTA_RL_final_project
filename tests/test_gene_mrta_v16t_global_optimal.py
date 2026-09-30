import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, build_world, generate_world
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum
from marl2d.gene_mrta_v16t.hungarian_benchmark import rollout_world


def test_global_oracle_known_two_task_optimum():
    config = EnvConfig(
        world_size=20.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=4.0,
        priority_min=1.0,
        priority_max=1.0,
        deadline_min=10.0,
        deadline_max=10.0,
        episode_time=10.0,
        obstacle_count=0,
        obstacle_size_min=2.0,
        obstacle_size_max=4.0,
        obstacle_clearance=1.0,
        grid_resolution=1.0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        np.array([[1.0, 1.0]]),
        np.array([100.0]),
        np.array([[2.0, 1.0], [3.0, 1.0]]),
        np.array([4.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([10.0, 10.0]),
        np.zeros((0, 4)),
    )

    result = solve_global_time_optimum(world, config, time_limit=30.0)

    # Optimal route is task 1 then task 0:
    # finish times = 3 and 8 seconds.
    # T = ((1 - 3/10) + (1 - 8/10)) / 2 = 0.45.
    assert result.optimal
    assert np.isclose(result.time_optimality, 0.45, atol=1e-8)
    assert result.completed_tasks == 2
    assert result.routes == ((1, 0),)


def test_global_oracle_is_not_worse_than_event_hungarian():
    config = EnvConfig()
    world = generate_world(config, seed=97_000_000)

    oracle = solve_global_time_optimum(world, config, time_limit=60.0)
    hungarian = rollout_world(
        world,
        config,
        score_mode="path_time",
        matcher="hungarian",
    ).evaluation.time_optimality

    if oracle.optimal:
        assert oracle.time_optimality is not None
        assert oracle.time_optimality + 1e-8 >= hungarian
