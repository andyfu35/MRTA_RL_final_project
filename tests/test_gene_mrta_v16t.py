import numpy as np

from marl2d.gene_mrta_v16.env import EnvConfig as EnvConfigV16
from marl2d.gene_mrta_v16.env import generate_world as generate_world_v16
from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
    evaluate_baseline_on_worlds,
    evaluate_gene,
    generate_world,
)
from marl2d.gene_mrta_v16t.gene import Gene
from marl2d.gene_mrta_v16t.train import AXES


def _one_task_world(distance: float, horizon: float = 10.0):
    config = EnvConfig(
        world_size=20.0,
        num_robots=1,
        num_tasks=1,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=1.0,
        priority_max=1.0,
        deadline_min=horizon,
        deadline_max=horizon,
        episode_time=horizon,
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
        np.array([[1.0, 1.0]], dtype=np.float64),
        np.array([100.0], dtype=np.float64),
        np.array([[1.0 + distance, 1.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([horizon], dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
    )
    return config, world


def test_v16t_preserves_v16_world_for_same_seed():
    old = generate_world_v16(EnvConfigV16(), seed=789)
    new = generate_world(EnvConfig(), seed=789)

    assert np.allclose(old.robot_positions, new.robot_positions)
    assert np.allclose(old.robot_initial_batteries, new.robot_initial_batteries)
    assert np.allclose(old.task_positions, new.task_positions)
    assert np.allclose(old.task_service_times, new.task_service_times)
    assert np.allclose(old.task_priorities, new.task_priorities)
    assert np.allclose(old.task_deadlines, new.task_deadlines)
    assert np.allclose(old.obstacles, new.obstacles)
    assert np.allclose(old.path_to_tasks, new.path_to_tasks)


def test_gene_observation_remains_eight_dimensional():
    gene = Gene(np.zeros(8, dtype=np.float64))
    assert gene.vector().shape == (8,)
    assert len(gene.to_dict()["observation_names"]) == 8


def test_sixth_capability_axis_is_time_optimality():
    assert AXES == (
        "completion",
        "efficiency",
        "priority_satisfaction",
        "deadline_satisfaction",
        "balance",
        "time_optimality",
    )


def test_time_optimality_matches_completion_time_formula():
    config, world = _one_task_world(distance=4.0, horizon=10.0)
    result = evaluate_baseline_on_worlds("shortest_path_time", [world], config)

    # travel=4, service=1, finish=5, contribution=1-5/10=0.5
    assert np.isclose(result.completion, 1.0)
    assert np.isclose(result.time_optimality, 0.5)


def test_earlier_completion_scores_higher_with_same_completion():
    fast_config, fast_world = _one_task_world(distance=2.0, horizon=10.0)
    slow_config, slow_world = _one_task_world(distance=6.0, horizon=10.0)

    fast = evaluate_baseline_on_worlds(
        "shortest_path_time",
        [fast_world],
        fast_config,
    )
    slow = evaluate_baseline_on_worlds(
        "shortest_path_time",
        [slow_world],
        slow_config,
    )

    assert np.isclose(fast.completion, slow.completion)
    assert fast.time_optimality > slow.time_optimality


def test_unfinished_task_contributes_zero_not_fake_speed_reward():
    config, world = _one_task_world(distance=12.0, horizon=10.0)
    result = evaluate_baseline_on_worlds("shortest_path_time", [world], config)

    assert np.isclose(result.completion, 0.0)
    assert np.isclose(result.time_optimality, 0.0)


def test_time_optimality_is_bounded_by_completion():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(
        np.array([-1.0, -1.0, -0.5, 1.0, -1.0, 0.5, -0.2, -0.1])
    )
    result = evaluate_gene(gene, world, config)

    assert 0.0 <= result.time_optimality <= 1.0
    assert result.time_optimality <= result.completion + 1e-12


def test_shortest_path_time_baseline_reports_time_axis():
    config = EnvConfig()
    worlds = [generate_world(config, seed=500 + i) for i in range(2)]
    result = evaluate_baseline_on_worlds("shortest_path_time", worlds, config)

    assert result.time_optimality > 0.0
    assert result.time_optimality <= result.completion + 1e-12
