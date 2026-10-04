import numpy as np

from marl2d.gene_mrta_v18.direct_gene import (
    ConsequenceAwareDirectGene,
)
from marl2d.gene_mrta_v113.evolve import (
    AXES,
    GeneRecord,
    _best_by_axis,
)
from marl2d.gene_mrta_v1142.assay import (
    _axis_pair_stats,
    _center_rule,
    _dominance,
    _make_child,
    _pair_key,
    build_pair_manifest,
)


def _record(
    record_id: str,
    rng: np.random.Generator,
    capabilities,
    score_shift: float,
) -> GeneRecord:
    return GeneRecord(
        record_id=record_id,
        gene=ConsequenceAwareDirectGene.random(
            rng
        ),
        capabilities=tuple(
            capabilities
        ),
        scores={
            axis: 0.8
            + score_shift
            + idx * 0.001
            for idx, axis
            in enumerate(
                AXES
            )
        },
        origin="test",
        generation=0,
    )


def test_pair_key_is_unordered_and_rejects_self_pair():
    assert _pair_key(
        "b",
        "a",
    ) == (
        "a",
        "b",
    )
    assert _pair_key(
        "a",
        "b",
    ) == (
        "a",
        "b",
    )

    try:
        _pair_key(
            "a",
            "a",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "self pair must fail"
        )


def test_pair_manifest_is_unique_and_deterministic():
    rng = np.random.default_rng(
        17
    )
    caps = [
        ("mean_time",),
        ("tail10_time",),
        (
            "continuation_preservation",
        ),
        (
            "fleet_option_reserve",
        ),
        (
            "mean_time",
            "tail10_time",
        ),
        (
            "continuation_preservation",
            "fleet_option_reserve",
        ),
        tuple(AXES),
        (
            "mean_time",
            "fleet_option_reserve",
        ),
    ]
    records = {
        f"g{idx}": _record(
            f"g{idx}",
            rng,
            cap,
            idx * 0.002,
        )
        for idx, cap
        in enumerate(caps)
    }
    ids = list(
        records
    )
    best = _best_by_axis(
        records,
        ids,
    )

    first = build_pair_manifest(
        count=12,
        active_ids=ids,
        records=records,
        best=best,
        rng=np.random.default_rng(
            77
        ),
        parent_q_power=10.0,
        uniform_fraction=0.05,
    )
    second = build_pair_manifest(
        count=12,
        active_ids=ids,
        records=records,
        best=best,
        rng=np.random.default_rng(
            77
        ),
        parent_q_power=10.0,
        uniform_fraction=0.05,
    )

    assert first == second
    assert len(first) == 12
    assert len(
        set(first)
    ) == 12
    assert all(
        a < b
        for a, b
        in first
    )


def test_center_rule_is_exact_parent_midpoint():
    rng = np.random.default_rng(
        9
    )
    anchor = (
        ConsequenceAwareDirectGene.random(
            rng
        )
    )
    parent_a = GeneRecord(
        record_id="a",
        gene=ConsequenceAwareDirectGene.random(
            rng
        ),
        capabilities=tuple(
            AXES
        ),
        scores={
            axis: 0.9
            for axis
            in AXES
        },
        origin="test",
        generation=0,
    )
    parent_b = GeneRecord(
        record_id="b",
        gene=ConsequenceAwareDirectGene.random(
            rng
        ),
        capabilities=tuple(
            AXES
        ),
        scores={
            axis: 0.91
            for axis
            in AXES
        },
        origin="test",
        generation=0,
    )
    best = {
        axis: 0.91
        for axis
        in AXES
    }

    child, _ = _make_child(
        rule=_center_rule(),
        parent_a=parent_a,
        parent_b=parent_b,
        anchor=anchor,
        best=best,
    )

    expected = 0.5 * (
        parent_a.gene.vector_data
        + parent_b.gene.vector_data
    )
    assert np.allclose(
        child.vector_data,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_axis_pair_stats_counts_wins_ties_losses():
    result = _axis_pair_stats(
        [
            1.0,
            2.0,
            3.0,
        ],
        [
            0.5,
            2.0,
            4.0,
        ],
        tolerance=1e-12,
    )

    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert np.isclose(
        result[
            "mean_delta"
        ],
        -1.0 / 6.0,
    )


def test_dominance_is_axiswise_not_scalarized():
    base = {
        axis: 0.8
        for axis
        in AXES
    }

    adaptive = dict(base)
    adaptive[
        "mean_time"
    ] = 0.9
    assert _dominance(
        adaptive,
        base,
        tolerance=1e-12,
    ) == "adaptive"

    mixed = dict(base)
    mixed[
        "mean_time"
    ] = 0.9
    mixed[
        "tail10_time"
    ] = 0.7
    assert _dominance(
        mixed,
        base,
        tolerance=1e-12,
    ) == "neither"

    assert _dominance(
        base,
        base,
        tolerance=1e-12,
    ) == "tie"
