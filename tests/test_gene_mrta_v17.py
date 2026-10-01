import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, build_world
from marl2d.gene_mrta_v17.capability_oracle import (
    solve_world_capability_ceilings,
)
from marl2d.gene_mrta_v17.direct_gene import DirectAssignmentGene
from marl2d.gene_mrta_v17.rollout import rollout_direct_gene
from marl2d.gene_mrta_v17.train import normalized_capability_scores


def _simple_world():
    config = EnvConfig(
        world_size=20.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=4.0,
        priority_min=1.0,
        priority_max=2.0,
        deadline_min=4.0,
        deadline_max=6.0,
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
        np.array([1.0, 2.0]),
        np.array([6.0, 4.0]),
        np.zeros((0, 4)),
    )
    return config, world


def test_direct_gene_emits_unique_feasible_pairs():
    rng = np.random.default_rng(7)
    vector = rng.normal(
        size=DirectAssignmentGene.parameter_count(4)
    )
    # Make STOP impossible for this uniqueness test.
    vector[-1] = -100.0
    gene = DirectAssignmentGene(vector, hidden_dim=4)
    observations = rng.normal(size=(3, 5, 8))
    eligible = np.ones((3, 5), dtype=bool)
    eligible[1, 2] = False
    free = np.array([True, True, True])
    available = np.array([True, True, True, True, True])

    assignments = gene.assign(
        observations,
        eligible,
        free,
        available,
    )

    assert len(assignments) == 3
    assert len({r for r, _ in assignments}) == 3
    assert len({t for _, t in assignments}) == 3
    assert all(eligible[r, t] for r, t in assignments)


def test_direct_gene_can_stop_without_forced_full_matching():
    rng = np.random.default_rng(11)
    vector = np.zeros(DirectAssignmentGene.parameter_count(4))
    vector[-1] = 100.0
    gene = DirectAssignmentGene(vector, hidden_dim=4)
    observations = rng.normal(size=(3, 5, 8))
    eligible = np.ones((3, 5), dtype=bool)
    assignments = gene.assign(
        observations,
        eligible,
        np.ones(3, dtype=bool),
        np.ones(5, dtype=bool),
    )
    assert assignments == []


def test_direct_rollout_has_no_duplicate_task_within_event():
    config, world = _simple_world()
    gene = DirectAssignmentGene(
        np.zeros(DirectAssignmentGene.parameter_count(4)),
        hidden_dim=4,
    )
    rollout = rollout_direct_gene(gene, world, config)

    for event in rollout.assignment_events:
        robots = [pair[0] for pair in event]
        tasks = [pair[1] for pair in event]
        assert len(robots) == len(set(robots))
        assert len(tasks) == len(set(tasks))
    assert 0.0 <= rollout.evaluation.time_optimality <= 1.0


def test_multi_axis_oracle_known_small_world():
    config, world = _simple_world()
    ceilings, results = solve_world_capability_ceilings(
        world,
        config,
        time_limit=30.0,
    )

    assert ceilings is not None
    assert all(result.optimal for result in results.values())
    assert np.isclose(ceilings.completion, 1.0, atol=1e-8)
    assert np.isclose(
        ceilings.priority_satisfaction,
        1.0,
        atol=1e-8,
    )
    assert np.isclose(
        ceilings.deadline_satisfaction,
        0.5,
        atol=1e-8,
    )
    assert np.isclose(
        ceilings.time_optimality,
        0.45,
        atol=1e-8,
    )
    expected_efficiency = 1.0 - 1.0 / config.diagonal
    assert np.isclose(
        ceilings.efficiency,
        expected_efficiency,
        atol=1e-8,
    )
    assert np.isclose(ceilings.balance_upper_bound, 1.0)


def test_normalized_capability_is_per_world_ratio_then_mean():
    raw = np.array(
        [
            [
                [0.4, 0.3, 0.2, 0.1, 0.4, 0.25],
                [0.3, 0.2, 0.3, 0.2, 0.3, 0.20],
            ]
        ],
        dtype=np.float64,
    )
    ceilings = np.array(
        [
            [0.5, 0.5, 0.4, 0.2, 0.5, 0.5],
            [0.6, 0.4, 0.6, 0.4, 0.6, 0.4],
        ],
        dtype=np.float64,
    )
    ratios, scores = normalized_capability_scores(raw, ceilings)

    expected = raw / ceilings[None, :, :]
    assert np.allclose(ratios, expected)
    assert np.allclose(scores, expected.mean(axis=1))
