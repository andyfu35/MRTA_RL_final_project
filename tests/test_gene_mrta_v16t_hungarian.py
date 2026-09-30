import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v16t.gene import Gene
from marl2d.gene_mrta_v16t.hungarian_benchmark import (
    _greedy_match,
    _hungarian_match,
    rollout_world,
)


def test_hungarian_batch_matching_beats_greedy_counterexample():
    scores = np.array([[9.0, 8.0], [8.0, 0.0]], dtype=np.float64)
    eligible = np.ones((2, 2), dtype=bool)
    free = np.ones(2, dtype=bool)
    available = np.ones(2, dtype=bool)

    greedy = _greedy_match(scores, eligible, free, available)
    hungarian = _hungarian_match(scores, eligible, free, available)

    greedy_score = sum(scores[r, t] for r, t in greedy)
    hungarian_score = sum(scores[r, t] for r, t in hungarian)

    assert np.isclose(greedy_score, 9.0)
    assert np.isclose(hungarian_score, 16.0)


def test_hungarian_path_time_has_zero_local_time_regret():
    config = EnvConfig()
    world = generate_world(config, seed=12345)

    result = rollout_world(
        world,
        config,
        score_mode="path_time",
        matcher="hungarian",
    )

    assert result.local_hungarian_time_regret_events > 0
    assert np.isclose(
        result.local_hungarian_time_regret_sum,
        0.0,
        atol=1e-12,
    )
    assert result.evaluation.time_optimality <= (
        result.evaluation.completion + 1e-12
    )


def test_gene_greedy_reports_local_hungarian_regret():
    config = EnvConfig()
    world = generate_world(config, seed=54321)
    gene = Gene(
        np.array(
            [-1.0, -1.2, -0.6, 0.3, 0.2, -0.4, 0.1, -0.5],
            dtype=np.float64,
        )
    )

    result = rollout_world(
        world,
        config,
        score_mode="gene",
        matcher="greedy",
        gene=gene,
    )

    assert result.local_hungarian_time_regret_events > 0
    assert result.local_hungarian_time_regret_sum >= -1e-12
    assert 0.0 <= result.evaluation.time_optimality <= 1.0


def test_same_gene_hungarian_is_valid_time_axis_rollout():
    config = EnvConfig()
    world = generate_world(config, seed=22222)
    gene = Gene(np.zeros(8, dtype=np.float64))

    result = rollout_world(
        world,
        config,
        score_mode="gene",
        matcher="hungarian",
        gene=gene,
    )

    assert 0.0 <= result.evaluation.time_optimality <= result.evaluation.completion + 1e-12
    assert result.timing.matcher_ns > 0
