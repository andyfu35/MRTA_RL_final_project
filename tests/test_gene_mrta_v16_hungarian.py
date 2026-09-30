import numpy as np

from marl2d.gene_mrta_v16.env import (
    EnvConfig,
    evaluate_gene,
    generate_world,
)
from marl2d.gene_mrta_v16.gene import Gene
from marl2d.gene_mrta_v16.hungarian_benchmark import (
    _greedy_match,
    _hungarian_match,
    rollout_world,
)


def test_hungarian_finds_better_global_batch_matching_than_greedy():
    scores = np.array(
        [
            [9.0, 8.0],
            [8.0, 0.0],
        ],
        dtype=np.float64,
    )
    eligible = np.ones((2, 2), dtype=bool)
    free = np.ones(2, dtype=bool)
    available = np.ones(2, dtype=bool)

    greedy = _greedy_match(scores, eligible, free, available)
    hungarian = _hungarian_match(scores, eligible, free, available)

    greedy_score = sum(scores[r, t] for r, t in greedy)
    hungarian_score = sum(scores[r, t] for r, t in hungarian)

    assert len(greedy) == 2
    assert len(hungarian) == 2
    assert np.isclose(greedy_score, 9.0)
    assert np.isclose(hungarian_score, 16.0)


def test_hungarian_allows_infeasible_robot_to_remain_unmatched():
    scores = np.array(
        [
            [5.0, 4.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    eligible = np.array(
        [
            [True, True],
            [False, False],
        ]
    )
    free = np.ones(2, dtype=bool)
    available = np.ones(2, dtype=bool)

    assignments = _hungarian_match(scores, eligible, free, available)

    assert len(assignments) == 1
    assert assignments[0][0] == 0


def test_unbatched_gene_greedy_matches_v16_reference_evaluator():
    config = EnvConfig()
    world = generate_world(config, seed=12345)
    gene = Gene(
        np.array(
            [-1.0, -1.3, -0.6, 0.5, 0.3, -0.4, 0.2, -0.7],
            dtype=np.float64,
        )
    )

    expected = evaluate_gene(gene, world, config)
    actual = rollout_world(
        world,
        config,
        score_mode="gene",
        matcher="greedy",
        gene=gene,
    ).evaluation

    for metric in (
        "completion",
        "efficiency",
        "priority_satisfaction",
        "deadline_satisfaction",
        "balance",
        "completed_tasks",
        "total_travel",
        "energy_consumed",
    ):
        assert np.isclose(
            getattr(actual, metric),
            getattr(expected, metric),
            rtol=1e-12,
            atol=1e-12,
        ), metric


def test_hungarian_path_time_rollout_is_valid_and_timed():
    config = EnvConfig()
    world = generate_world(config, seed=54321)

    result = rollout_world(
        world,
        config,
        score_mode="path_time",
        matcher="hungarian",
    )

    assert 0.0 <= result.evaluation.completion <= 1.0
    assert 0.0 <= result.evaluation.efficiency <= 1.0
    assert 0.0 <= result.evaluation.priority_satisfaction <= 1.0
    assert 0.0 <= result.evaluation.deadline_satisfaction <= 1.0
    assert 0.0 <= result.evaluation.balance <= 1.0
    assert result.timing.decision_count > 0
    assert result.timing.decision_total_ns > 0
    assert result.timing.matcher_ns > 0
    assert result.wall_ns > 0
