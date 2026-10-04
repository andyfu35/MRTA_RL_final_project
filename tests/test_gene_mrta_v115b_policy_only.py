import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig
from marl2d.gene_mrta_v115.policy_only import (
    _fit_exponent,
    build_euclidean_world,
)


def test_euclidean_world_has_exact_dense_distances():
    config = EnvConfig(
        world_size=50.0,
        num_robots=3,
        num_tasks=5,
        obstacle_count=0,
    )
    world, seconds = (
        build_euclidean_world(
            config,
            seed=77,
        )
    )

    nodes = np.vstack(
        [
            world.robot_positions,
            world.task_positions,
        ]
    )
    expected = np.linalg.norm(
        nodes[:, None, :]
        - world.task_positions[
            None, :, :
        ],
        axis=-1,
    )

    assert world.path_to_tasks.shape == (
        8,
        5,
    )
    assert np.allclose(
        world.path_to_tasks,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    assert seconds >= 0.0


def test_task_self_distances_are_zero():
    config = EnvConfig(
        num_robots=2,
        num_tasks=4,
        obstacle_count=0,
    )
    world, _ = build_euclidean_world(
        config,
        seed=5,
    )
    task_rows = (
        world.path_to_tasks[
            config.num_robots:,
            :
        ]
    )
    assert np.allclose(
        np.diag(
            task_rows
        ),
        0.0,
        atol=1e-12,
    )


def test_fit_exponent_recovers_quadratic():
    summaries = [
        {
            "tasks": 20,
            "mean_policy_seconds": 4.0,
        },
        {
            "tasks": 40,
            "mean_policy_seconds": 16.0,
        },
        {
            "tasks": 80,
            "mean_policy_seconds": 64.0,
        },
    ]
    exponent = _fit_exponent(
        summaries,
        "mean_policy_seconds",
    )
    assert np.isclose(
        exponent,
        2.0,
        atol=1e-12,
    )
