import numpy as np

from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.fusion1 import (
    PROGRESSIVE_OPERATORS,
    _choose_pairs,
    _passed_axes,
)
from marl2d.gene_mrta_v117.unseen_audit import (
    _candidate_ids,
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


def test_fusion1_prioritizes_complementary_capability_union():
    scores = {
        axis: 0.95
        for axis in BASE_AXES
    }

    time_gene = _record(
        "time_gene",
        scores,
    )
    time_gene.capabilities = (
        "global_time_optimality",
        "global_deadline_optimality",
    )
    time_gene.archive_capabilities = (
        "global_time_optimality",
        "global_deadline_optimality",
    )

    priority_gene = _record(
        "priority_gene",
        scores,
    )
    priority_gene.capabilities = (
        "global_priority_optimality",
        "global_path_efficiency",
    )
    priority_gene.archive_capabilities = (
        "global_priority_optimality",
        "global_path_efficiency",
    )

    completion_gene = _record(
        "completion_gene",
        scores,
    )
    completion_gene.capabilities = (
        "global_completion_optimality",
        "workload_balance",
    )
    completion_gene.archive_capabilities = (
        "global_completion_optimality",
        "workload_balance",
    )

    records = {
        record.record_id: record
        for record in (
            time_gene,
            priority_gene,
            completion_gene,
        )
    }

    pairs = _choose_pairs(
        records,
        pair_count=3,
        archive_size=3,
        rng=np.random.default_rng(11701),
    )

    assert pairs
    assert any(
        (
            "global_time_optimality"
            in union
            and "global_priority_optimality"
            in union
        )
        for _a, _b, union in pairs
    )


def test_unseen_audit_candidate_set_keeps_specialists_and_fusions():
    scores = {
        axis: 0.90
        for axis in BASE_AXES
    }
    records = {}

    for index, axis in enumerate(
        BASE_AXES
    ):
        record = _record(
            f"specialist_{index}",
            scores,
        )
        record.scores = dict(scores)
        record.scores[axis] = 0.99
        record.capabilities = (
            axis,
        )
        record.archive_capabilities = (
            axis,
        )
        records[
            record.record_id
        ] = record

    fusion = _record(
        "fusion",
        {
            axis: 0.96
            for axis in BASE_AXES
        },
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
    records[
        fusion.record_id
    ] = fusion

    ids, roles = _candidate_ids(
        records,
        archive_size=1,
        hybrid_count=4,
    )

    assert "fusion" in ids
    assert "fusion" in roles["fusion"]
    assert any(
        "specialist:global_time_optimality"
        in role
        for role in roles.values()
    )


def test_progressive_fusion_accepts_partial_inheritance_without_lowering_gate():
    union = tuple(BASE_AXES[:4])
    parent_a = {
        axis: 1.0
        for axis in BASE_AXES
    }
    parent_b = dict(parent_a)
    ceiling = dict(parent_a)

    child = {
        axis: 0.80
        for axis in BASE_AXES
    }
    child[union[0]] = 0.97
    child[union[1]] = 0.96
    child[union[2]] = 0.94
    child[union[3]] = 0.99

    inherited = _passed_axes(
        child,
        union,
        parent_a,
        parent_b,
        ceiling,
        threshold=0.95,
    )

    assert inherited == tuple(
        sorted(
            (
                union[0],
                union[1],
                union[3],
            )
        )
    )


def test_progressive_fusion_uses_parent_preserving_operators():
    assert set(
        PROGRESSIVE_OPERATORS
    ) == {
        "sparse_block_graft",
        "sparse_block_blend",
        "near_parent_blend",
    }
