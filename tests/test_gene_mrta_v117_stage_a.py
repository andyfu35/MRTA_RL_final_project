import numpy as np

from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.stage_a_train import (
    Record,
    _quality,
    _rebuild_active,
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
