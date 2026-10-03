import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v113.evolve import (
    _axis_scores,
)
from marl2d.gene_mrta_v113.robust_metrics import (
    rollout_route_tail_robust_metrics,
)
from marl2d.gene_mrta_v113.route_tail import (
    rollout_route_tail_gene,
)


def _world():
    config = EnvConfig(
        world_size=20.0,
        num_robots=2,
        num_tasks=5,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=1.0,
        priority_max=1.0,
        deadline_min=50.0,
        deadline_max=50.0,
        episode_time=50.0,
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
        np.asarray(
            [
                [0.0, 0.0],
                [15.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [100.0, 100.0],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.ones(
            5,
            dtype=np.float64,
        ),
        np.ones(
            5,
            dtype=np.float64,
        ),
        np.full(
            5,
            50.0,
            dtype=np.float64,
        ),
        np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )
    return config, world


def _always_continue_gene():
    vector = np.zeros(
        RouteTailDirectGene.parameter_count(8),
        dtype=np.float64,
    )
    vector[137] = 1.0
    vector[-1] = -100.0
    return RouteTailDirectGene(
        vector,
        hidden_dim=8,
    )


def test_route_tail_robust_time_matches_route_tail_rollout():
    config, world = _world()
    gene = _always_continue_gene()

    rollout = rollout_route_tail_gene(
        gene,
        world,
        config,
    )
    robust = (
        rollout_route_tail_robust_metrics(
            gene,
            world,
            config,
        )
    )

    assert np.isclose(
        robust.time_optimality,
        rollout.evaluation.time_optimality,
    )
    assert (
        0.0
        <= robust.continuation_preservation
        <= 1.0
    )
    assert (
        0.0
        <= robust.fleet_option_reserve
        <= 1.0
    )
    assert np.isclose(
        robust.mean_queue_depth,
        2.5,
    )
    assert np.isclose(
        robust.max_queue_depth,
        5.0,
    )


def test_v113_axis_scoring_uses_exact_t_ceiling_ratio_and_tail():
    tensor = np.zeros(
        (2, 10, 3),
        dtype=np.float64,
    )
    stars = np.ones(
        10,
        dtype=np.float64,
    )

    tensor[0, :, 0] = 0.9
    tensor[1, :, 0] = 0.9
    tensor[1, 0, 0] = 0.4
    tensor[:, :, 1] = 0.8
    tensor[:, :, 2] = 0.7

    scores = _axis_scores(
        tensor,
        stars,
    )

    assert np.isclose(
        scores["mean_time"][0],
        0.9,
    )
    assert np.isclose(
        scores["tail10_time"][0],
        0.9,
    )
    assert np.isclose(
        scores["tail10_time"][1],
        0.4,
    )
    assert np.isclose(
        scores[
            "continuation_preservation"
        ][0],
        0.8,
    )
    assert np.isclose(
        scores[
            "fleet_option_reserve"
        ][0],
        0.7,
    )
