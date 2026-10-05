import numpy as np

from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.fusion1 import (
    _choose_pairs,
)
from marl2d.gene_mrta_v117.pareto_bank import (
    analysis_best_by_axis,
    dominates,
    maximin_gene_id,
    pareto_front_ids,
    rebuild_pareto_bank,
)
from marl2d.gene_mrta_v117.stage_a_train import (
    CLEAN_MATING_OPERATORS,
    Record,
    _quality,
    _rebuild_active,
    _record_from_dict,
)


def _record(
    rid: str,
    scores: dict[str, float],
) -> Record:
    gene = RouteTailDirectGene(
        np.zeros(148, dtype=np.float64),
        hidden_dim=8,
    )
    return Record(
        record_id=rid,
        gene=gene,
        scores=scores,
        capabilities=(),
        origin="test",
        generation=0,
    )


def test_stage_a_random_gene_has_148_parameters():
    rng = np.random.default_rng(117)
    gene = RouteTailDirectGene.random(
        rng,
        hidden_dim=8,
        scale=0.35,
    )
    assert gene.vector_data.shape == (148,)


def test_stage_a_archives_keep_conflicting_specialists():
    base = {
        axis: 0.5
        for axis in BASE_AXES
    }
    time_scores = dict(base)
    time_scores["global_time_optimality"] = 0.99
    priority_scores = dict(base)
    priority_scores[
        "global_priority_optimality"
    ] = 0.995

    records = {
        "time": _record(
            "time",
            time_scores,
        ),
        "priority": _record(
            "priority",
            priority_scores,
        ),
    }

    active, archives = _rebuild_active(
        records,
        archive_size=1,
        hybrid_limit=0,
        certification_threshold=0.95,
    )

    assert (
        archives[
            "global_time_optimality"
        ][0]
        == "time"
    )
    assert (
        archives[
            "global_priority_optimality"
        ][0]
        == "priority"
    )
    assert "global_time_optimality" in (
        active["time"].capabilities
    )
    assert "global_priority_optimality" in (
        active["priority"].capabilities
    )


def test_parent_quality_uses_weakest_declared_capability():
    scores = {
        axis: 1.0
        for axis in BASE_AXES
    }
    scores["global_priority_optimality"] = 0.8
    record = _record(
        "hybrid",
        scores,
    )
    record.capabilities = (
        "global_time_optimality",
        "global_priority_optimality",
    )
    best = {
        axis: 1.0
        for axis in BASE_AXES
    }

    assert np.isclose(
        _quality(record, best),
        0.8,
    )


def test_clean_stage_a_mating_uses_no_ancestor_delta_operators():
    assert set(
        CLEAN_MATING_OPERATORS
    ) == {
        "parameter_blend",
        "block_pick",
        "block_blend",
    }
    assert "ancestor_delta" not in CLEAN_MATING_OPERATORS
    assert "ties_delta" not in CLEAN_MATING_OPERATORS
    assert "dare_delta" not in CLEAN_MATING_OPERATORS


def test_archive_membership_is_not_mating_inheritance():
    dominant_scores = {
        axis: 1.0
        for axis in BASE_AXES
    }
    fusion_scores = {
        axis: 0.96
        for axis in BASE_AXES
    }

    dominant = _record(
        "dominant",
        dominant_scores,
    )
    fusion = _record(
        "fusion",
        fusion_scores,
    )
    fusion.origin = "mating"
    fusion.inherited_capabilities = (
        "global_time_optimality",
        "global_priority_optimality",
    )
    fusion.capabilities = (
        "global_time_optimality",
        "global_priority_optimality",
    )

    active, _archives = _rebuild_active(
        {
            "dominant": dominant,
            "fusion": fusion,
        },
        archive_size=1,
        hybrid_limit=1,
        certification_threshold=0.95,
    )

    assert active["fusion"].archive_capabilities == ()
    assert active["fusion"].inherited_capabilities == (
        "global_priority_optimality",
        "global_time_optimality",
    )
    assert active["fusion"].capabilities == (
        "global_priority_optimality",
        "global_time_optimality",
    )


def test_non_inherited_mating_gene_is_not_kept_as_fusion():
    dominant_scores = {
        axis: 1.0
        for axis in BASE_AXES
    }
    weak_scores = {
        axis: 0.96
        for axis in BASE_AXES
    }

    dominant = _record(
        "dominant",
        dominant_scores,
    )
    weak = _record(
        "weak",
        weak_scores,
    )
    weak.origin = "mating"

    active, _archives = _rebuild_active(
        {
            "dominant": dominant,
            "weak": weak,
        },
        archive_size=1,
        hybrid_limit=1,
        certification_threshold=0.95,
    )

    assert "weak" not in active


def test_record_round_trip_preserves_capability_provenance():
    scores = {
        axis: 0.97
        for axis in BASE_AXES
    }
    record = _record(
        "roundtrip",
        scores,
    )
    record.origin = "mating"
    record.archive_capabilities = (
        "global_time_optimality",
    )
    record.inherited_capabilities = (
        "global_priority_optimality",
        "global_time_optimality",
    )
    record.capabilities = (
        "global_priority_optimality",
        "global_time_optimality",
    )
    record.parents = (
        "parent_a",
        "parent_b",
    )
    record.operator = "block_pick"

    restored = _record_from_dict(
        record.to_dict()
    )

    assert restored.record_id == record.record_id
    assert restored.origin == "mating"
    assert restored.archive_capabilities == (
        "global_time_optimality",
    )
    assert restored.inherited_capabilities == (
        "global_priority_optimality",
        "global_time_optimality",
    )
    assert restored.parents == (
        "parent_a",
        "parent_b",
    )
    assert restored.operator == "block_pick"
    assert np.allclose(
        restored.gene.vector_data,
        record.gene.vector_data,
    )


def test_pareto_dominance_uses_all_capability_axes():
    a = {
        axis: 0.90
        for axis in BASE_AXES
    }
    b = dict(a)
    a[
        "global_time_optimality"
    ] = 0.91

    assert dominates(a, b)
    assert not dominates(b, a)

    tradeoff = dict(a)
    tradeoff[
        "global_time_optimality"
    ] = 0.89
    tradeoff[
        "global_priority_optimality"
    ] = 0.95
    assert not dominates(
        a,
        tradeoff,
    )
    assert not dominates(
        tradeoff,
        a,
    )


def test_pareto_front_keeps_conflicting_tradeoffs():
    base = {
        axis: 0.80
        for axis in BASE_AXES
    }

    time_scores = dict(base)
    time_scores[
        "global_time_optimality"
    ] = 0.99

    priority_scores = dict(base)
    priority_scores[
        "global_priority_optimality"
    ] = 0.99

    dominated_scores = {
        axis: 0.70
        for axis in BASE_AXES
    }

    records = {
        "time": _record(
            "time",
            time_scores,
        ),
        "priority": _record(
            "priority",
            priority_scores,
        ),
        "dominated": _record(
            "dominated",
            dominated_scores,
        ),
    }

    front, dominated = (
        pareto_front_ids(
            records
        )
    )

    assert set(front) == {
        "time",
        "priority",
    }
    assert dominated == (
        "dominated",
    )


def test_pareto_rebuild_removes_dominated_gene_without_axis_labels():
    strong_scores = {
        axis: 0.90
        for axis in BASE_AXES
    }
    weak_scores = {
        axis: 0.80
        for axis in BASE_AXES
    }

    records = {
        "strong": _record(
            "strong",
            strong_scores,
        ),
        "weak": _record(
            "weak",
            weak_scores,
        ),
    }

    rebuilt = rebuild_pareto_bank(
        records,
        max_size=16,
        epsilon=0.001,
    )

    assert set(
        rebuilt.records
    ) == {
        "strong",
    }
    assert "weak" in (
        rebuilt.dominated_ids
    )


def test_pareto_epsilon_cell_collapses_near_duplicate_tradeoff():
    base = {
        axis: 0.90
        for axis in BASE_AXES
    }
    nearby = dict(base)
    nearby[
        "global_time_optimality"
    ] = 0.901
    nearby[
        "global_priority_optimality"
    ] = 0.899

    records = {
        "base": _record(
            "base",
            base,
        ),
        "nearby": _record(
            "nearby",
            nearby,
        ),
    }

    rebuilt = rebuild_pareto_bank(
        records,
        max_size=16,
        epsilon=0.01,
    )

    assert len(
        rebuilt.records
    ) == 1
    assert len(
        rebuilt.epsilon_duplicate_ids
    ) == 1


def test_pareto_analysis_views_do_not_control_bank_membership():
    base = {
        axis: 0.80
        for axis in BASE_AXES
    }

    time_scores = dict(base)
    time_scores[
        "global_time_optimality"
    ] = 0.99

    balanced_scores = {
        axis: 0.90
        for axis in BASE_AXES
    }

    records = {
        "time": _record(
            "time",
            time_scores,
        ),
        "balanced": _record(
            "balanced",
            balanced_scores,
        ),
    }

    rebuilt = rebuild_pareto_bank(
        records,
        max_size=16,
        epsilon=0.001,
    )
    assert set(
        rebuilt.records
    ) == {
        "time",
        "balanced",
    }

    best = analysis_best_by_axis(
        rebuilt.records
    )
    assert best[
        "global_time_optimality"
    ] == "time"
    assert (
        maximin_gene_id(
            rebuilt.records
        )
        == "balanced"
    )


def test_pareto_mating_pairs_use_capability_space_distance():
    records = {}

    a_scores = {
        axis: 0.80
        for axis in BASE_AXES
    }
    a_scores[
        "global_time_optimality"
    ] = 0.99

    b_scores = {
        axis: 0.80
        for axis in BASE_AXES
    }
    b_scores[
        "global_priority_optimality"
    ] = 0.99

    c_scores = dict(
        a_scores
    )
    c_scores[
        "global_time_optimality"
    ] = 0.98

    for rid, scores in (
        ("a", a_scores),
        ("b", b_scores),
        ("c", c_scores),
    ):
        records[rid] = _record(
            rid,
            scores,
        )

    pairs = _choose_pairs(
        records,
        pair_count=1,
        rng=np.random.default_rng(
            11702
        ),
    )

    assert set(
        pairs[0]
    ) == {
        "a",
        "b",
    }
