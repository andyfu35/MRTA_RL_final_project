import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, build_world
from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v18.rollout import rollout_direct_gene
from marl2d.gene_mrta_v19.robust_metrics import (
    _future_graph_masses,
    rollout_robust_metrics,
)
from marl2d.gene_mrta_v19.train import (
    AXES,
    _axis_scores,
    _select_axis_archives,
)


def _world():
    config = EnvConfig(
        world_size=20.0,
        num_robots=2,
        num_tasks=3,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=1.0,
        priority_max=1.0,
        deadline_min=20.0,
        deadline_max=20.0,
        episode_time=20.0,
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
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([100.0, 100.0]),
        np.array([[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]]),
        np.ones(3),
        np.ones(3),
        np.full(3, 20.0),
        np.zeros((0, 4)),
    )
    return config, world


def test_robust_metric_replay_preserves_v18_time_score():
    config, world = _world()
    rng = np.random.default_rng(7)
    gene = ConsequenceAwareDirectGene.random(rng, hidden_dim=4)

    original = rollout_direct_gene(gene, world, config).evaluation.time_optimality
    robust = rollout_robust_metrics(gene, world, config)

    assert np.isclose(original, robust.time_optimality)
    assert 0.0 <= robust.continuation_preservation <= 1.0
    assert 0.0 <= robust.fleet_option_reserve <= 1.0


def test_future_graph_masses_are_normalized():
    config, world = _world()
    option, reserve = _future_graph_masses(
        world=world,
        config=config,
        now=0.0,
        current_node_ids=np.array([0, 1], dtype=np.int64),
        battery_remaining=np.array([100.0, 100.0]),
        busy_until=np.zeros(2),
        task_available=np.ones(3, dtype=bool),
    )
    assert 0.0 <= option <= 1.0
    assert 0.0 <= reserve <= 1.0
    assert option > 0.0
    assert reserve > 0.0


def test_axis_scores_are_independent():
    batch = np.array(
        [
            [[0.8, 0.9, 0.5], [0.8, 0.9, 0.5]],
            [[0.7, 0.6, 0.95], [0.7, 0.6, 0.95]],
        ],
        dtype=np.float64,
    )
    hard = np.array(
        [
            [[0.4, 0.9, 0.5]],
            [[0.9, 0.6, 0.95]],
        ],
        dtype=np.float64,
    )
    scores = _axis_scores(
        batch_tensor=batch,
        batch_star=np.array([1.0, 1.0]),
        hard_tensor=hard,
        hard_star=np.array([1.0]),
    )

    assert scores["mean_time"][0] > scores["mean_time"][1]
    assert scores["hard_world_time"][1] > scores["hard_world_time"][0]
    assert (
        scores["continuation_preservation"][0]
        > scores["continuation_preservation"][1]
    )
    assert scores["fleet_option_reserve"][1] > scores["fleet_option_reserve"][0]


def test_each_axis_archive_can_keep_different_specialist():
    rng = np.random.default_rng(9)
    genes = [
        ConsequenceAwareDirectGene.random(rng, hidden_dim=2)
        for _ in range(4)
    ]
    scores = {
        "mean_time": np.array([1.0, 0.0, 0.0, 0.0]),
        "hard_world_time": np.array([0.0, 1.0, 0.0, 0.0]),
        "continuation_preservation": np.array([0.0, 0.0, 1.0, 0.0]),
        "fleet_option_reserve": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    archives = _select_axis_archives(
        genes,
        scores,
        size_per_axis=1,
    )
    assert tuple(archives) == AXES
    assert archives["mean_time"][0].key() == genes[0].key()
    assert archives["hard_world_time"][0].key() == genes[1].key()
    assert archives["continuation_preservation"][0].key() == genes[2].key()
    assert archives["fleet_option_reserve"][0].key() == genes[3].key()
