import numpy as np
import pytest

from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v110.train import GeneRecord
from marl2d.gene_mrta_v111.discover import (
    _dual_retention,
    _sample_complementary_pairs,
)
from marl2d.gene_mrta_v111.law import (
    LawCoefficients,
    adaptive_delta_child,
    group_features,
    sample_law_candidates,
)


def _record(
    name: str,
    capabilities: tuple[str, ...],
    scores: dict[str, float],
) -> GeneRecord:
    rng = np.random.default_rng(
        sum(ord(char) for char in name)
    )
    return GeneRecord(
        record_id=name,
        gene=ConsequenceAwareDirectGene.random(
            rng
        ),
        capabilities=capabilities,
        scores=scores,
        origin="test",
        generation=0,
    )


def test_law_sampling_is_deterministic_and_contains_center():
    first = sample_law_candidates(
        8,
        seed=17,
    )
    second = sample_law_candidates(
        8,
        seed=17,
    )

    assert [
        law.to_dict()
        for law in first
    ] == [
        law.to_dict()
        for law in second
    ]
    assert first[0].law_id == "law_000"
    assert first[0].beta_norm_ratio == 0.0
    assert first[0].beta_quality_diff == 0.0
    assert first[0].beta_capability_diff == 0.0
    assert first[0].gamma_bias == 0.0
    assert first[0].gamma_cosine == 0.0
    assert first[0].gamma_sign_agreement == 0.0
    assert first[0].gamma_magnitude == 0.0


def test_adaptive_delta_law_is_parent_swap_symmetric():
    rng = np.random.default_rng(91)
    anchor = ConsequenceAwareDirectGene.random(
        rng
    )
    parent_a = anchor.mutated(
        rng,
        sigma=0.22,
        mutation_rate=0.8,
    )
    parent_b = anchor.mutated(
        rng,
        sigma=0.18,
        mutation_rate=0.7,
    )
    law = sample_law_candidates(
        5,
        seed=9,
    )[3]

    child_ab, _ = adaptive_delta_child(
        parent_a,
        parent_b,
        anchor,
        law,
        quality_a=0.98,
        quality_b=0.95,
        capability_count_a=3,
        capability_count_b=2,
    )
    child_ba, _ = adaptive_delta_child(
        parent_b,
        parent_a,
        anchor,
        law,
        quality_a=0.95,
        quality_b=0.98,
        capability_count_a=2,
        capability_count_b=3,
    )

    assert np.allclose(
        child_ab.vector_data,
        child_ba.vector_data,
        atol=1e-12,
        rtol=1e-12,
    )


def test_adaptive_delta_metadata_has_nine_bounded_groups():
    rng = np.random.default_rng(92)
    anchor = ConsequenceAwareDirectGene.random(
        rng
    )
    parent_a = anchor.mutated(
        rng,
        sigma=0.20,
        mutation_rate=1.0,
    )
    parent_b = anchor.mutated(
        rng,
        sigma=0.20,
        mutation_rate=1.0,
    )
    law = LawCoefficients(
        law_id="manual",
        beta_norm_ratio=1.2,
        beta_quality_diff=-0.8,
        beta_capability_diff=0.5,
        gamma_bias=0.2,
        gamma_cosine=1.1,
        gamma_sign_agreement=-0.7,
        gamma_magnitude=0.4,
    )

    _child, metadata = (
        adaptive_delta_child(
            parent_a,
            parent_b,
            anchor,
            law,
            quality_a=0.97,
            quality_b=0.96,
            capability_count_a=2,
            capability_count_b=3,
        )
    )

    groups = metadata["groups"]
    assert len(groups) == 9
    assert all(
        0.0 <= row["alpha"] <= 1.0
        for row in groups
    )
    assert all(
        0.5 <= row["eta"] <= 1.5
        for row in groups
    )


def test_group_features_are_finite():
    rng = np.random.default_rng(93)
    anchor = ConsequenceAwareDirectGene.random(
        rng
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

    rows = group_features(
        parent_a,
        parent_b,
        anchor,
    )

    assert len(rows) == 9
    for row in rows:
        for value in row.values():
            assert np.isfinite(value)


def test_dual_retention_requires_parent_and_ceiling_threshold():
    child = {
        "mean_time": 0.95,
        "tail10_time": 0.91,
        "continuation_preservation": 0.78,
        "fleet_option_reserve": 0.82,
    }
    parent_a = {
        "mean_time": 0.98,
        "tail10_time": 0.88,
        "continuation_preservation": 0.78,
        "fleet_option_reserve": 0.80,
    }
    parent_b = {
        "mean_time": 0.90,
        "tail10_time": 0.95,
        "continuation_preservation": 0.74,
        "fleet_option_reserve": 0.84,
    }
    ceiling = {
        "mean_time": 0.99,
        "tail10_time": 0.96,
        "continuation_preservation": 0.80,
        "fleet_option_reserve": 0.85,
    }

    result = _dual_retention(
        child_scores=child,
        parent_a_scores=parent_a,
        parent_b_scores=parent_b,
        caps_a=(
            "mean_time",
            "continuation_preservation",
        ),
        caps_b=(
            "tail10_time",
            "fleet_option_reserve",
        ),
        ceiling=ceiling,
        threshold=0.95,
    )

    assert len(
        result[
            "required_capabilities"
        ]
    ) == 4
    assert result["accepted"] is False
    assert result[
        "min_dual_retention"
    ] == pytest.approx(
        min(
            0.95 / 0.99,
            0.91 / 0.96,
            0.78 / 0.80,
            0.82 / 0.85,
        )
    )


def test_pair_sampler_requires_both_parents_to_add_capability():
    ceiling = {
        "mean_time": 1.0,
        "tail10_time": 1.0,
        "continuation_preservation": 1.0,
        "fleet_option_reserve": 1.0,
    }
    scores = {
        axis: 0.98
        for axis in ceiling
    }
    records = {
        "m": _record(
            "m",
            ("mean_time",),
            scores,
        ),
        "t": _record(
            "t",
            ("tail10_time",),
            scores,
        ),
        "c": _record(
            "c",
            (
                "continuation_preservation",
            ),
            scores,
        ),
        "mt": _record(
            "mt",
            (
                "mean_time",
                "tail10_time",
            ),
            scores,
        ),
    }
    certified = {
        key: record.capabilities
        for key, record
        in records.items()
    }

    pairs = _sample_complementary_pairs(
        records=records,
        active_ids=list(records),
        certified=certified,
        ceiling=ceiling,
        count=2,
        rng=np.random.default_rng(4),
        q_power=10.0,
        uniform_fraction=0.05,
    )

    for first, second in pairs:
        caps_a = set(
            certified[first]
        )
        caps_b = set(
            certified[second]
        )
        assert caps_a - caps_b
        assert caps_b - caps_a
