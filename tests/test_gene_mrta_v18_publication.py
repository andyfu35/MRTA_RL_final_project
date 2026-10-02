import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v18.publication_benchmark import (
    PUBLICATION_SEED_BASE,
    PUBLICATION_WORLD_COUNT,
    _aggregate,
    _bootstrap_mean_ci,
    _load_existing_world,
    _paired_stats,
)


def test_publication_seed_range_is_frozen():
    assert PUBLICATION_SEED_BASE == 98_000_000
    assert PUBLICATION_WORLD_COUNT == 100
    assert (
        PUBLICATION_SEED_BASE
        + PUBLICATION_WORLD_COUNT
        - 1
        == 98_000_099
    )


def test_bootstrap_ci_is_deterministic():
    values = np.array([0.90, 0.95, 1.00], dtype=np.float64)
    a = _bootstrap_mean_ci(values, seed=123, reps=2000)
    b = _bootstrap_mean_ci(values, seed=123, reps=2000)
    assert a == b
    assert a[0] <= values.mean() <= a[1]


def test_paired_stats_reports_wins_ties_losses():
    v18 = np.array([0.9, 0.8, 0.7, 0.5], dtype=np.float64)
    base = np.array([0.8, 0.8, 0.6, 0.6], dtype=np.float64)

    stats = _paired_stats(v18, base, seed=321)

    assert stats["wins"] == 2
    assert stats["ties"] == 1
    assert stats["losses"] == 1
    assert np.isclose(stats["mean_delta"], 0.025)


def test_aggregate_uses_only_proven_optimal_worlds():
    rows = [
        {
            "world_seed": 1,
            "optimal": True,
            "v18_retention": 0.98,
            "v17_retention": 0.95,
            "v16to_retention": 0.96,
            "hungarian_retention": 0.94,
            "solve_seconds": 1.0,
        },
        {
            "world_seed": 2,
            "optimal": True,
            "v18_retention": 1.00,
            "v17_retention": 0.97,
            "v16to_retention": 0.98,
            "hungarian_retention": 0.95,
            "solve_seconds": 2.0,
        },
        {
            "world_seed": 3,
            "optimal": False,
            "solve_seconds": 900.0,
        },
    ]

    aggregate = _aggregate(rows)

    assert aggregate["worlds_requested"] == 3
    assert aggregate["worlds_proven_optimal"] == 2
    assert aggregate["worlds_not_proven_optimal"] == 1
    assert np.isclose(
        aggregate["methods"]["v18"]["mean"],
        0.99,
    )
    assert np.isclose(
        aggregate["paired"]["v18_minus_v17"]["mean_delta"],
        0.03,
    )


def test_existing_world_cache_requires_matching_seed(tmp_path: Path):
    path = tmp_path / "world_98000000.json"
    path.write_text(
        json.dumps(
            {
                "world_seed": 98_000_000,
                "optimal": True,
            }
        ),
        encoding="utf-8",
    )

    assert _load_existing_world(path, 98_000_000) is not None
    assert _load_existing_world(path, 98_000_001) is None
