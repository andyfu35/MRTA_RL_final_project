import numpy as np

from marl2d.gene_mrta_v13.env import (
    EnvConfig,
    World,
    evaluate_baseline_on_worlds,
    evaluate_gene,
    generate_world,
)
from marl2d.gene_mrta_v13.gene import Gene


def test_generated_priorities_are_deterministic_and_bounded():
    config = EnvConfig(priority_min=0.1, priority_max=1.0)
    a = generate_world(config, seed=123)
    b = generate_world(config, seed=123)

    assert np.allclose(a.robot_positions, b.robot_positions)
    assert np.allclose(a.task_positions, b.task_positions)
    assert np.allclose(a.task_service_times, b.task_service_times)
    assert np.allclose(a.task_priorities, b.task_priorities)
    assert np.all(a.task_priorities >= 0.1)
    assert np.all(a.task_priorities <= 1.0)


def test_gene_has_five_priority_aware_observations():
    gene = Gene(np.array([-1.0, -0.5, 1.0, -0.2, -0.1]))
    data = gene.to_dict()

    assert gene.vector().shape == (5,)
    assert data["observation_names"] == [
        "distance_norm",
        "service_time_norm",
        "priority_norm",
        "robot_workload_norm",
        "competition_norm",
    ]


def test_priority_baselines_prefer_high_value_task():
    config = EnvConfig(
        world_size=10.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=2.0,
        priority_min=0.1,
        priority_max=1.0,
        episode_time=4.0,
    )
    world = World(
        robot_positions=np.array([[0.0, 0.0]], dtype=np.float64),
        task_positions=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.array([2.0, 1.0], dtype=np.float64),
        task_priorities=np.array([0.1, 1.0], dtype=np.float64),
    )

    nearest = evaluate_baseline_on_worlds(
        "nearest",
        [world],
        config,
    )
    highest = evaluate_baseline_on_worlds(
        "highest_priority",
        [world],
        config,
    )
    per_time = evaluate_baseline_on_worlds(
        "priority_per_time",
        [world],
        config,
    )

    assert np.isclose(nearest.completion, 0.5)
    assert np.isclose(highest.completion, 0.5)
    assert np.isclose(per_time.completion, 0.5)
    assert highest.priority_satisfaction > nearest.priority_satisfaction
    assert per_time.priority_satisfaction > nearest.priority_satisfaction


def test_priority_satisfaction_matches_completed_priority_fraction():
    config = EnvConfig(
        world_size=10.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=2.0,
        priority_min=0.1,
        priority_max=1.0,
        episode_time=4.0,
    )
    world = World(
        robot_positions=np.array([[0.0, 0.0]], dtype=np.float64),
        task_positions=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.array([2.0, 1.0], dtype=np.float64),
        task_priorities=np.array([0.1, 1.0], dtype=np.float64),
    )
    result = evaluate_baseline_on_worlds(
        "highest_priority",
        [world],
        config,
    )

    assert np.isclose(result.completed_priority, 1.0)
    assert np.isclose(result.total_priority, 1.1)
    assert np.isclose(result.priority_satisfaction, 1.0 / 1.1)


def test_all_four_capability_axes_are_bounded():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(np.array([-1.0, -0.5, 1.0, -0.2, -0.1]))
    result = evaluate_gene(gene, world, config)

    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.priority_satisfaction <= 1.0
    assert 0.0 <= result.balance <= 1.0
    assert 0.0 <= result.route_efficiency <= 1.0
    assert result.efficiency <= result.completion + 1e-12
    assert np.isclose(
        sum(result.robot_task_counts),
        result.completed_tasks,
    )


def test_unknown_baseline_is_rejected():
    config = EnvConfig()
    world = generate_world(config, seed=456)

    try:
        evaluate_baseline_on_worlds("not_a_policy", [world], config)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown baseline")
