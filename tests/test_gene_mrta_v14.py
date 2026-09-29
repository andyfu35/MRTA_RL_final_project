import numpy as np

from marl2d.gene_mrta_v14.env import (
    EnvConfig,
    World,
    evaluate_baseline_on_worlds,
    evaluate_gene,
    generate_world,
)
from marl2d.gene_mrta_v14.gene import Gene


def _deadline_world() -> tuple[EnvConfig, World]:
    config = EnvConfig(
        world_size=10.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=4.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=3.0,
        deadline_max=10.0,
        episode_time=10.0,
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
        task_service_times=np.array([4.0, 1.0], dtype=np.float64),
        task_priorities=np.array([0.5, 0.5], dtype=np.float64),
        task_deadlines=np.array([10.0, 3.0], dtype=np.float64),
    )
    return config, world


def test_generated_deadlines_are_deterministic_and_bounded():
    config = EnvConfig(deadline_min=25.0, deadline_max=50.0)
    a = generate_world(config, seed=123)
    b = generate_world(config, seed=123)

    assert np.allclose(a.robot_positions, b.robot_positions)
    assert np.allclose(a.task_positions, b.task_positions)
    assert np.allclose(a.task_service_times, b.task_service_times)
    assert np.allclose(a.task_priorities, b.task_priorities)
    assert np.allclose(a.task_deadlines, b.task_deadlines)
    assert np.all(a.task_deadlines >= 25.0)
    assert np.all(a.task_deadlines <= 50.0)


def test_gene_has_six_deadline_aware_observations():
    gene = Gene(np.array([-1.0, -0.5, 1.0, -1.0, -0.2, -0.1]))
    data = gene.to_dict()

    assert gene.vector().shape == (6,)
    assert data["observation_names"] == [
        "distance_norm",
        "service_time_norm",
        "priority_norm",
        "deadline_remaining_norm",
        "robot_workload_norm",
        "competition_norm",
    ]


def test_deadline_baselines_improve_on_time_completion():
    config, world = _deadline_world()

    nearest = evaluate_baseline_on_worlds("nearest", [world], config)
    earliest = evaluate_baseline_on_worlds(
        "earliest_deadline",
        [world],
        config,
    )
    least_laxity = evaluate_baseline_on_worlds(
        "least_laxity",
        [world],
        config,
    )

    assert np.isclose(nearest.completion, 1.0)
    assert np.isclose(earliest.completion, 1.0)
    assert np.isclose(least_laxity.completion, 1.0)
    assert np.isclose(nearest.deadline_satisfaction, 0.5)
    assert np.isclose(earliest.deadline_satisfaction, 1.0)
    assert np.isclose(least_laxity.deadline_satisfaction, 1.0)


def test_soft_deadline_does_not_remove_late_task():
    config, world = _deadline_world()
    nearest = evaluate_baseline_on_worlds("nearest", [world], config)

    assert np.isclose(nearest.completed_tasks, 2.0)
    assert np.isclose(nearest.on_time_tasks, 1.0)
    assert np.isclose(nearest.completion, 1.0)
    assert np.isclose(nearest.deadline_satisfaction, 0.5)


def test_all_five_capability_axes_are_bounded():
    config = EnvConfig()
    world = generate_world(config, seed=321)
    gene = Gene(np.array([-1.0, -0.5, 1.0, -1.0, -0.2, -0.1]))
    result = evaluate_gene(gene, world, config)

    assert 0.0 <= result.completion <= 1.0
    assert 0.0 <= result.efficiency <= 1.0
    assert 0.0 <= result.priority_satisfaction <= 1.0
    assert 0.0 <= result.deadline_satisfaction <= 1.0
    assert 0.0 <= result.balance <= 1.0
    assert result.deadline_satisfaction <= result.completion + 1e-12
    assert result.efficiency <= result.completion + 1e-12


def test_unknown_baseline_is_rejected():
    config = EnvConfig()
    world = generate_world(config, seed=456)

    try:
        evaluate_baseline_on_worlds("not_a_policy", [world], config)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown baseline")
