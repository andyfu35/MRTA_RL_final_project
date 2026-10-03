import numpy as np
import pytest

from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v110.recombination import (
    OPERATORS,
    hidden_block_indices,
    offspring_family,
    recombine,
)
from marl2d.gene_mrta_v110.scenario_bank import (
    FROZEN_FAILED_ORIGINAL_SEED,
    FROZEN_REPLACEMENT_SEED,
    FROZEN_SCENARIO_SEEDS,
    _replacement_candidate_order,
    _validate_seed_range,
    select_diverse_indices,
)
from marl2d.gene_mrta_v110.train import (
    GeneRecord,
    _axis_scores,
    _best_by_axis,
    _inheritance_retention,
    _quality_weight,
)


def test_hidden_blocks_partition_policy_vector():
    h = 8
    blocks, global_ids = hidden_block_indices(h)
    all_ids = np.concatenate(
        blocks + [global_ids]
    )
    expected = (
        ConsequenceAwareDirectGene.parameter_count(h)
    )

    assert len(blocks) == h
    assert all(len(block) == 18 for block in blocks)
    assert len(global_ids) == 4
    assert all_ids.size == expected
    assert len(set(all_ids.tolist())) == expected
    assert set(all_ids.tolist()) == set(range(expected))


def test_all_recombination_operators_produce_valid_gene():
    rng = np.random.default_rng(11)
    anchor = ConsequenceAwareDirectGene.random(
        rng,
        hidden_dim=8,
    )
    parent_a = anchor.mutated(
        rng,
        sigma=0.1,
        mutation_rate=0.5,
    )
    parent_b = anchor.mutated(
        rng,
        sigma=0.1,
        mutation_rate=0.5,
    )

    for operator in OPERATORS:
        result = recombine(
            parent_a,
            parent_b,
            anchor,
            rng,
            operator=operator,
        )
        assert result.operator == operator
        assert result.gene.hidden_dim == 8
        assert result.gene.vector_data.shape == (
            ConsequenceAwareDirectGene.parameter_count(8),
        )
        assert np.all(
            np.isfinite(
                result.gene.vector_data
            )
        )


def test_offspring_family_generates_multiple_possibilities():
    rng = np.random.default_rng(12)
    anchor = ConsequenceAwareDirectGene.random(rng)
    parent_a = anchor.mutated(
        rng,
        sigma=0.15,
        mutation_rate=0.7,
    )
    parent_b = anchor.mutated(
        rng,
        sigma=0.15,
        mutation_rate=0.7,
    )

    family = offspring_family(
        parent_a,
        parent_b,
        anchor,
        rng,
        children=4,
    )

    assert len(family) == 4
    assert len(
        {
            child.operator
            for child in family
        }
    ) == 4
    assert len(
        {
            child.gene.key()
            for child in family
        }
    ) == 4


def test_axis_scores_include_worst_ten_percent():
    tensor = np.zeros(
        (2, 10, 3),
        dtype=np.float64,
    )
    tensor[0, :, 0] = 1.0
    tensor[1, :, 0] = 1.0
    tensor[1, 0, 0] = 0.4
    tensor[:, :, 1] = 0.8
    tensor[:, :, 2] = 0.7
    stars = np.ones(10)

    scores = _axis_scores(
        tensor,
        stars,
    )

    assert scores["mean_time"][0] == pytest.approx(1.0)
    assert scores["tail10_time"][0] == pytest.approx(1.0)
    assert scores["tail10_time"][1] == pytest.approx(0.4)
    assert scores["continuation_preservation"][0] == pytest.approx(0.8)
    assert scores["fleet_option_reserve"][0] == pytest.approx(0.7)


def _record(
    name: str,
    scores: dict[str, float],
    capabilities: tuple[str, ...],
) -> GeneRecord:
    rng = np.random.default_rng(
        sum(ord(char) for char in name)
    )
    return GeneRecord(
        record_id=name,
        gene=ConsequenceAwareDirectGene.random(rng),
        capabilities=capabilities,
        scores=scores,
        origin="test",
        generation=0,
    )


def test_quality_weight_is_min_relative_capability_retention():
    a = _record(
        "a",
        {
            "mean_time": 0.98,
            "tail10_time": 0.80,
            "continuation_preservation": 0.70,
            "fleet_option_reserve": 0.70,
        },
        ("mean_time",),
    )
    b = _record(
        "b",
        {
            "mean_time": 0.90,
            "tail10_time": 0.95,
            "continuation_preservation": 0.70,
            "fleet_option_reserve": 0.70,
        },
        ("tail10_time",),
    )
    hybrid = _record(
        "hybrid",
        {
            "mean_time": 0.97,
            "tail10_time": 0.94,
            "continuation_preservation": 0.70,
            "fleet_option_reserve": 0.70,
        },
        ("mean_time", "tail10_time"),
    )

    records = {
        "a": a,
        "b": b,
        "hybrid": hybrid,
    }
    best = _best_by_axis(records)

    q = _quality_weight(
        hybrid,
        best,
    )
    expected = min(
        0.97 / 0.98,
        0.94 / 0.95,
    )
    assert q == pytest.approx(expected)


def test_inheritance_retention_requires_parent_capability_union():
    a = _record(
        "a",
        {
            "mean_time": 0.98,
            "tail10_time": 0.70,
            "continuation_preservation": 0.70,
            "fleet_option_reserve": 0.70,
        },
        ("mean_time",),
    )
    b = _record(
        "b",
        {
            "mean_time": 0.80,
            "tail10_time": 0.96,
            "continuation_preservation": 0.70,
            "fleet_option_reserve": 0.70,
        },
        ("tail10_time",),
    )
    child_scores = {
        "mean_time": 0.97,
        "tail10_time": 0.95,
        "continuation_preservation": 0.70,
        "fleet_option_reserve": 0.70,
    }

    retention = _inheritance_retention(
        child_scores,
        a,
        b,
        ("mean_time", "tail10_time"),
    )

    assert retention["mean_time"] == pytest.approx(0.97 / 0.98)
    assert retention["tail10_time"] == pytest.approx(0.95 / 0.96)
    assert min(retention.values()) > 0.95


def test_diverse_selector_is_deterministic_and_unique():
    rng = np.random.default_rng(1)
    descriptors = rng.normal(
        size=(50, 6)
    )

    first = select_diverse_indices(
        descriptors,
        10,
    )
    second = select_diverse_indices(
        descriptors,
        10,
    )

    assert np.array_equal(first, second)
    assert first.shape == (10,)
    assert len(set(first.tolist())) == 10


def test_protected_seed_ranges_are_rejected():
    _validate_seed_range(
        95_000_000,
        500,
    )

    with pytest.raises(ValueError):
        _validate_seed_range(
            98_000_000,
            100,
        )

    with pytest.raises(ValueError):
        _validate_seed_range(
            99_000_000,
            100,
        )


def test_replacement_order_prefers_nearest_unused_descriptor():
    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [2.0, 2.0],
            [0.2, 0.2],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )
    order = _replacement_candidate_order(
        descriptors,
        target_index=0,
        excluded_indices={0, 1},
    )

    indices = [
        idx
        for idx, _distance in order
    ]
    assert indices[0] == 3
    assert 0 not in indices
    assert 1 not in indices


def test_replacement_order_never_returns_original_selected_set():
    rng = np.random.default_rng(77)
    descriptors = rng.normal(
        size=(30, 5)
    )
    selected = set(
        select_diverse_indices(
            descriptors,
            8,
        ).tolist()
    )
    target = next(
        iter(selected)
    )

    order = _replacement_candidate_order(
        descriptors,
        target_index=target,
        excluded_indices=set(selected),
    )

    returned = {
        idx
        for idx, _distance in order
    }
    assert returned.isdisjoint(
        selected
    )
    assert len(returned) == (
        len(descriptors)
        - len(selected)
    )


def test_frozen_v110_seed_set_is_exactly_100_unique_worlds():
    assert len(FROZEN_SCENARIO_SEEDS) == 100
    assert len(set(FROZEN_SCENARIO_SEEDS)) == 100
    assert FROZEN_FAILED_ORIGINAL_SEED not in FROZEN_SCENARIO_SEEDS
    assert FROZEN_REPLACEMENT_SEED in FROZEN_SCENARIO_SEEDS
    assert FROZEN_SCENARIO_SEEDS[1] == FROZEN_REPLACEMENT_SEED


def test_frozen_v110_seed_set_does_not_touch_protected_ranges():
    for seed in FROZEN_SCENARIO_SEEDS:
        assert not (98_000_000 <= seed <= 98_000_099)
        assert not (99_000_000 <= seed <= 99_000_099)
