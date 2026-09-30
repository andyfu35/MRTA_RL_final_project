import numpy as np

from marl2d.gene_mrta_v15.env import (
    EnvConfig,
    build_world,
    evaluate_baseline_on_worlds,
    evaluate_gene,
    generate_world,
)
from marl2d.gene_mrta_v15.gene import Gene


def _detour_world():
    config = EnvConfig(
        world_size=20.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=15.0,
        deadline_max=15.0,
        episode_time=15.0,
        obstacle_count=1,
        obstacle_size_min=4.0,
        obstacle_size_max=8.0,
        obstacle_clearance=1.0,
        grid_resolution=1.0,
    )
    robots = np.array([[2.0, 10.0]], dtype=np.float64)
    tasks = np.array(
        [
            [8.0, 10.0],
            [2.0, 18.0],
        ],
        dtype=np.float64,
    )
    services = np.array([1.0, 1.0], dtype=np.float64)
    priorities = np.array([0.1, 1.0], dtype=np.float64)
    deadlines = np.array([15.0, 15.0], dtype=np.float64)
    obstacles = np.array(
        [
            [4.0, 7.0, 7.0, 13.0],
        ],
        dtype=np.float64,
    )
    return config, build_world(
        config,
        robots,
        tasks,
        services,
        priorities,
        deadlines,
        obstacles,
    )



def test_calibrated_defaults_use_comparable_path_scale():
    config = EnvConfig()

    assert config.obstacle_count == 10
    assert np.isclose(config.obstacle_size_min, 12.0)
    assert np.isclose(config.obstacle_size_max, 20.0)
    assert np.isclose(config.path_cost_scale, config.diagonal)


def test_generated_obstacles_and_paths_are_deterministic():
    config = EnvConfig()
    a = generate_world(config, seed=123)
    b = generate_world(config, seed=123)

    assert a.obstacles.shape == (config.obstacle_count, 4)
    assert np.allclose(a.obstacles, b.obstacles)
    assert np.allclose(a.path_to_tasks, b.path_to_tasks)
    assert np.all(np.isfinite(a.path_to_tasks[: config.num_robots]))


def test_gene_has_seven_obstacle_aware_observations():
    gene = Gene(
        np.array([-1.0, -1.0, -0.5, 1.0, -1.0, -0.2, -0.1])
    )
    data = gene.to_dict()

    assert gene.vector().shape == (7,)
    assert data["observation_names"] == [
        "euclidean_distance_norm",
        "path_cost_norm",
        "service_time_norm",
        "priority_norm",
        "deadline_remaining_norm",
        "robot_workload_norm",
        "competition_norm",
    ]


def test_obstacle_can_reverse_euclidean_nearest_choice():
    config, world = _detour_world()

    direct = np.linalg.norm(
        world.task_positions - world.robot_positions[0],
        axis=1,
    )
    paths = world.path_to_tasks[0]

    assert direct[0] < direct[1]
    assert paths[0] > paths[1]

    nearest = evaluate_baseline_on_worlds("nearest", [world], config)
    nearest_path = evaluate_baseline_on_worlds(
        "nearest_path",
        [world],
        config,
    )

    assert np.isclose(nearest.completion, 0.5)
    assert np.isclose(nearest_path.completion, 0.5)
    assert nearest.completed_priority < nearest_path.completed_priority


def test_actual_travel_uses_astar_path_cost():
    config, world = _detour_world()
    nearest = evaluate_baseline_on_worlds("nearest", [world], config)

    assert nearest.total_travel > nearest.total_euclidean_travel
    assert nearest.detour_ratio > 1.0


def test_shortest_path_time_baseline_is_available():
    config = EnvConfig()
    worlds = [generate_world(config, seed=200 + idx) for idx in range(2)]
    result = evaluate_baseline_on_worlds(
        "shortest_path_time",
        worlds,
        config,
    )

    assert 0.0 <= result.completion <= 1.0
    assert result.detour_ratio >= 1.0


def test_all_five_capability_axes_are_bounded():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(
        np.array([-1.0, -1.0, -0.5, 1.0, -1.0, -0.2, -0.1])
    )
    result = evaluate_gene(gene, world, config)

    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.priority_satisfaction <= 1.0
    assert 0.0 <= result.deadline_satisfaction <= 1.0
    assert 0.0 <= result.balance <= 1.0
    assert result.deadline_satisfaction <= result.completion + 1e-12
    assert result.efficiency <= result.completion + 1e-12
    assert result.detour_ratio >= 1.0


def test_unknown_baseline_is_rejected():
    config = EnvConfig()
    world = generate_world(config, seed=456)

    try:
        evaluate_baseline_on_worlds("not_a_policy", [world], config)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown baseline")
