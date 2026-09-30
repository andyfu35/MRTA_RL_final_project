import numpy as np

from marl2d.gene_mrta_v16.env import (
    EnvConfig,
    build_world,
    evaluate_baseline_on_worlds,
    evaluate_gene,
    generate_world,
)
from marl2d.gene_mrta_v16.gene import Gene
from marl2d.gene_mrta_v15.env import EnvConfig as EnvConfigV15
from marl2d.gene_mrta_v15.env import generate_world as generate_world_v15


def _single_task_battery_world(initial_battery: float):
    config = EnvConfig(
        world_size=20.0,
        num_robots=1,
        num_tasks=1,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=20.0,
        deadline_max=20.0,
        episode_time=20.0,
        obstacle_count=0,
        obstacle_size_min=4.0,
        obstacle_size_max=8.0,
        obstacle_clearance=1.0,
        grid_resolution=1.0,
        battery_capacity=20.0,
        initial_battery_min=0.0,
        initial_battery_max=20.0,
        energy_per_distance=1.0,
    )
    return config, build_world(
        config,
        np.array([[1.0, 10.0]], dtype=np.float64),
        np.array([initial_battery], dtype=np.float64),
        np.array([[11.0, 10.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([20.0], dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
    )


def test_default_battery_configuration_is_moderately_constrained():
    config = EnvConfig()

    assert np.isclose(config.battery_capacity, 70.0)
    assert np.isclose(config.initial_battery_min, 35.0)
    assert np.isclose(config.initial_battery_max, 70.0)
    assert np.isclose(config.energy_per_distance, 1.0)



def test_v16_preserves_v15_nonbattery_world_for_same_seed():
    config15 = EnvConfigV15()
    config16 = EnvConfig()
    old = generate_world_v15(config15, seed=789)
    new = generate_world(config16, seed=789)

    assert np.allclose(old.robot_positions, new.robot_positions)
    assert np.allclose(old.task_positions, new.task_positions)
    assert np.allclose(old.task_service_times, new.task_service_times)
    assert np.allclose(old.task_priorities, new.task_priorities)
    assert np.allclose(old.task_deadlines, new.task_deadlines)
    assert np.allclose(old.obstacles, new.obstacles)
    assert np.allclose(old.path_to_tasks, new.path_to_tasks)


def test_generated_batteries_are_deterministic_and_bounded():
    config = EnvConfig()
    a = generate_world(config, seed=123)
    b = generate_world(config, seed=123)

    assert np.allclose(a.robot_initial_batteries, b.robot_initial_batteries)
    assert np.all(a.robot_initial_batteries >= config.initial_battery_min)
    assert np.all(a.robot_initial_batteries <= config.initial_battery_max)
    assert np.allclose(a.obstacles, b.obstacles)
    assert np.allclose(a.path_to_tasks, b.path_to_tasks)


def test_gene_has_eight_battery_aware_observations():
    gene = Gene(
        np.array([-1.0, -1.0, -0.5, 1.0, -1.0, 0.5, -0.2, -0.1])
    )
    data = gene.to_dict()

    assert gene.vector().shape == (8,)
    assert data["observation_names"] == [
        "euclidean_distance_norm",
        "path_cost_norm",
        "service_time_norm",
        "priority_norm",
        "deadline_remaining_norm",
        "battery_remaining_norm",
        "robot_workload_norm",
        "competition_norm",
    ]


def test_insufficient_battery_blocks_otherwise_feasible_task():
    config, world = _single_task_battery_world(initial_battery=5.0)
    result = evaluate_baseline_on_worlds("shortest_path_time", [world], config)

    assert np.isclose(result.completion, 0.0)
    assert np.isclose(result.energy_consumed, 0.0)
    assert np.isclose(result.mean_final_battery, 5.0)
    assert result.battery_blocked_pair_events > 0.0


def test_travel_energy_is_deducted_from_battery():
    config, world = _single_task_battery_world(initial_battery=12.0)
    result = evaluate_baseline_on_worlds("shortest_path_time", [world], config)

    assert np.isclose(result.completion, 1.0)
    assert np.isclose(result.total_travel, 10.0)
    assert np.isclose(result.energy_consumed, 10.0)
    assert np.isclose(result.mean_final_battery, 2.0)
    assert np.isclose(result.battery_remaining_fraction, 2.0 / 12.0)


def test_max_battery_margin_prefers_robot_with_more_post_task_reserve():
    config = EnvConfig(
        world_size=20.0,
        num_robots=2,
        num_tasks=1,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=20.0,
        deadline_max=20.0,
        episode_time=20.0,
        obstacle_count=0,
        obstacle_size_min=4.0,
        obstacle_size_max=8.0,
        obstacle_clearance=1.0,
        grid_resolution=1.0,
        battery_capacity=20.0,
        initial_battery_min=0.0,
        initial_battery_max=20.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        np.array([[1.0, 10.0], [1.0, 10.0]], dtype=np.float64),
        np.array([12.0, 20.0], dtype=np.float64),
        np.array([[11.0, 10.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([20.0], dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
    )
    result = evaluate_baseline_on_worlds("max_battery_margin", [world], config)

    assert np.isclose(result.completion, 1.0)
    assert np.isclose(result.robot_task_counts[0], 0.0)
    assert np.isclose(result.robot_task_counts[1], 1.0)
    assert np.isclose(result.robot_final_batteries[0], 12.0)
    assert np.isclose(result.robot_final_batteries[1], 10.0)


def test_battery_aware_baseline_is_available_on_random_worlds():
    config = EnvConfig()
    worlds = [generate_world(config, seed=200 + idx) for idx in range(2)]
    result = evaluate_baseline_on_worlds(
        "max_battery_margin",
        worlds,
        config,
    )

    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.battery_remaining_fraction <= 1.0
    assert result.energy_consumed >= 0.0


def test_all_five_capability_axes_and_battery_metrics_are_bounded():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(
        np.array([-1.0, -1.0, -0.5, 1.0, -1.0, 0.5, -0.2, -0.1])
    )
    result = evaluate_gene(gene, world, config)

    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.priority_satisfaction <= 1.0
    assert 0.0 <= result.deadline_satisfaction <= 1.0
    assert 0.0 <= result.balance <= 1.0
    assert result.deadline_satisfaction <= result.completion + 1e-12
    assert result.efficiency <= result.completion + 1e-12
    assert 0.0 <= result.battery_remaining_fraction <= 1.0
    assert result.energy_consumed >= 0.0
    assert min(result.robot_final_batteries) >= -1e-12


def test_unknown_baseline_is_rejected():
    config = EnvConfig()
    world = generate_world(config, seed=456)

    try:
        evaluate_baseline_on_worlds("not_a_policy", [world], config)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown baseline")
