import numpy as np

from marl2d.gene_mrta_v120.bank import BankRecord
from marl2d.gene_mrta_v120.capabilities import (
    GeneAssessment,
    InstanceAssessment,
)
from marl2d.gene_mrta_v121.train import (
    build_round_summary,
    parent_probabilities,
    parser,
)


def _assessment(small, medium, *, overall=None, worst=0.5):
    if overall is None:
        overall = 0.5 * (small + medium)
    rows = (
        InstanceAssessment(
            instance_id="small",
            benchmark_set="S",
            size_band="small",
            vertex_count=100,
            robot_count=3,
            reference_kind="best_known",
            reference_value=100.0,
            success=True,
            completion=1.0,
            objective=100.0 / small,
            reference_retention=small,
            gap_to_reference=(1.0 / small) - 1.0,
            beat_bks=False,
        ),
        InstanceAssessment(
            instance_id="medium",
            benchmark_set="S",
            size_band="medium",
            vertex_count=500,
            robot_count=5,
            reference_kind="best_known",
            reference_value=100.0,
            success=True,
            completion=1.0,
            objective=100.0 / medium,
            reference_retention=medium,
            gap_to_reference=(1.0 / medium) - 1.0,
            beat_bks=False,
        ),
    )
    return GeneAssessment(
        success=True,
        worst_completion=1.0,
        mean_completion=1.0,
        success_instances=2,
        instance_count=2,
        scores={
            "retention_small": small,
            "retention_medium": medium,
        },
        overall_reference_retention=overall,
        worst_reference_retention=worst,
        exact_matches=0,
        bks_improvements=0,
        instances=rows,
    )


def _record(rid, small, medium, *, overall=None, parent=None):
    if overall is None:
        overall = 0.5 * (small + medium)
    return BankRecord(
        record_id=rid,
        vector_data=(0.0,),
        hidden_dim=8,
        scores={
            "retention_small": small,
            "retention_medium": medium,
        },
        overall_retention=overall,
        worst_retention=min(small, medium),
        generation=0,
        origin="test",
        parents=(() if parent is None else (parent,)),
    )


def test_v121_formal_defaults_are_1000_worlds_34_instances_50_rounds():
    args = parser().parse_args(["--run-dir", "runs/test"])
    assert args.worlds_per_round == 1000
    assert args.expected_instance_count == 34
    assert args.rounds == 50
    assert args.candidate_k == 32


def test_v121_parent_sampling_is_equal_axis_total_score_squared():
    records = {
        "a": _record("a", 0.5, 0.5),
        "b": _record("b", 1.0, 1.0),
    }
    ids, probs = parent_probabilities(
        records,
        ("retention_small", "retention_medium"),
    )
    assert ids == ["a", "b"]
    # Scores are 0.5 and 1.0. Squared weights are 0.25 and 1.0.
    assert np.allclose(probs, [0.2, 0.8])


def test_v121_round_summary_uses_all_worlds_and_all_fixed_instances():
    records = [
        _record("a", 0.6, 0.4),
        _record("b", 0.8, 0.6),
        _record("c", 1.0, 0.8),
    ]
    assessments = [
        _assessment(0.6, 0.4),
        _assessment(0.8, 0.6),
        _assessment(1.0, 0.8),
    ]
    bank = {
        "b": records[1],
        "c": records[2],
    }
    summary = build_round_summary(
        round_index=0,
        worlds_per_round=3,
        instance_count=34,
        axes=("retention_small", "retention_medium"),
        candidate_records=records,
        candidate_assessments=assessments,
        parent_records=[None, None, None],
        bank_records=bank,
        bank_before=0,
        dominated_removed=1,
        epsilon_removed=0,
        crowding_removed=0,
        previous_summary=None,
    )
    assert summary["rollouts_this_round"] == 102
    assert summary["successful_worlds"] == 3
    assert np.isclose(
        summary["population_axis_stats"]["retention_small"]["mean"],
        0.8,
    )
    assert np.isclose(
        summary["population_axis_stats"]["retention_medium"]["mean"],
        0.6,
    )
    assert summary["population_axis_mean_delta_from_previous_round"] == {
        "retention_small": None,
        "retention_medium": None,
    }


def test_v121_round_summary_tracks_population_mean_improvement():
    previous = {
        "population_axis_stats": {
            "retention_small": {"mean": 0.50},
            "retention_medium": {"mean": 0.40},
        }
    }
    records = [
        _record("a", 0.60, 0.50),
        _record("b", 0.80, 0.70),
    ]
    assessments = [
        _assessment(0.60, 0.50),
        _assessment(0.80, 0.70),
    ]
    summary = build_round_summary(
        round_index=1,
        worlds_per_round=2,
        instance_count=34,
        axes=("retention_small", "retention_medium"),
        candidate_records=records,
        candidate_assessments=assessments,
        parent_records=[
            _record("pa", 0.55, 0.45),
            _record("pb", 0.75, 0.65),
        ],
        bank_records={"b": records[1]},
        bank_before=1,
        dominated_removed=2,
        epsilon_removed=0,
        crowding_removed=0,
        previous_summary=previous,
    )
    delta = summary["population_axis_mean_delta_from_previous_round"]
    assert np.isclose(delta["retention_small"], 0.20)
    assert np.isclose(delta["retention_medium"], 0.20)
    diag = summary["parent_child_diagnostic"]
    assert diag["overall_win_tie_loss"]["win"] == 2
    assert diag["overall_win_tie_loss"]["loss"] == 0


def test_v121_round_summary_rejects_partial_world_batch():
    records = [
        _record("a", 0.6, 0.5),
        _record("b", 0.7, 0.6),
    ]
    assessments = [
        _assessment(0.6, 0.5),
        _assessment(0.7, 0.6),
    ]

    try:
        build_round_summary(
            round_index=0,
            worlds_per_round=3,
            instance_count=34,
            axes=("retention_small", "retention_medium"),
            candidate_records=records,
            candidate_assessments=assessments,
            parent_records=[None, None],
            bank_records={},
            bank_before=0,
            dominated_removed=0,
            epsilon_removed=0,
            crowding_removed=0,
            previous_summary=None,
        )
    except ValueError as exc:
        assert "requested number of worlds" in str(exc)
    else:
        raise AssertionError("Partial round must not be accepted")
