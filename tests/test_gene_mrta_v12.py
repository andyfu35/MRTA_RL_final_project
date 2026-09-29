import numpy as np

from marl2d.gene_mrta_v12.env import (
    EnvConfig,
    World,
    evaluate_gene,
    generate_world,
    jain_fairness,
)
from marl2d.gene_mrta_v12.gene import Gene


def test_generated_service_times_are_deterministic_and_bounded():
    config = EnvConfig(
        service_time_min=2.0,
        service_time_max=15.0,
    )
    a = generate_world(config, seed=123)
    b = generate_world(config, seed=123)

    assert np.allclose(a.robot_positions, b.robot_positions)
    assert np.allclose(a.task_positions, b.task_positions)
    assert np.allclose(a.task_service_times, b.task_service_times)
    assert np.all(a.task_service_times >= 2.0)
    assert np.all(a.task_service_times <= 15.0)


def test_gene_has_four_v12_observations():
    gene = Gene(np.array([-1.0, -0.5, -0.2, 0.1]))
    data = gene.to_dict()
    assert gene.vector().shape == (4,)
    assert data["observation_names"] == [
        "distance_norm",
        "service_time_norm",
        "robot_workload_norm",
        "competition_norm",
    ]


def test_shortest_total_time_can_beat_nearest_when_service_varies():
    config = EnvConfig(
        world_size=10.0,
        num_robots=1,
        num_tasks=3,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=6.0,
        episode_time=7.0,
    )
    world = World(
        robot_positions=np.array([[0.0, 0.0]], dtype=np.float64),
        task_positions=np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.array(
            [6.0, 1.0, 1.0],
            dtype=np.float64,
        ),
    )

    nearest = Gene(
        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )

    raw = np.array(
        [
            -config.diagonal / config.robot_speed,
            -config.service_time_range,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    shortest_total = Gene(raw / np.linalg.norm(raw))

    nearest_result = evaluate_gene(nearest, world, config)
    total_result = evaluate_gene(shortest_total, world, config)

    assert np.isclose(nearest_result.completion, 1.0 / 3.0)
    assert np.isclose(total_result.completion, 2.0 / 3.0)
    assert total_result.completion > nearest_result.completion


def test_balance_is_completion_weighted():
    config = EnvConfig(
        world_size=10.0,
        num_robots=1,
        num_tasks=3,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=6.0,
        episode_time=7.0,
    )
    world = World(
        robot_positions=np.array([[0.0, 0.0]], dtype=np.float64),
        task_positions=np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.array(
            [6.0, 1.0, 1.0],
            dtype=np.float64,
        ),
    )
    nearest = Gene(
        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    result = evaluate_gene(nearest, world, config)

    assert np.isclose(result.balance, result.completion)


def test_metrics_and_reporting_are_consistent():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(np.array([-1.0, -0.5, -0.2, 0.1]))
    result = evaluate_gene(gene, world, config)

    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.balance <= 1.0
    assert 0.0 <= result.route_efficiency <= 1.0
    assert result.efficiency <= result.completion + 1e-12
    assert np.isclose(
        sum(result.robot_task_counts),
        result.completed_tasks,
    )
    assert all(load >= 0.0 for load in result.robot_workloads)


def test_jain_fairness_extremes():
    assert np.isclose(
        jain_fairness(np.array([5.0, 5.0, 5.0, 5.0])),
        1.0,
    )
    assert np.isclose(
        jain_fairness(np.array([20.0, 0.0, 0.0, 0.0])),
        0.25,
    )
