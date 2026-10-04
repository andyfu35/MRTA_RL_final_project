import numpy as np

from marl2d.gene_mrta_v116.milp_policy_scaling import (
    _case_summary,
    _is_completed_row,
    _parse_optional_float,
    gap_metrics,
    milp_problem_size,
)


def test_milp_problem_size_formula():
    size = milp_problem_size(
        4,
        20,
    )

    assert (
        size[
            "milp_binary_variables"
        ]
        == 1680
    )
    assert (
        size[
            "milp_continuous_variables"
        ]
        == 80
    )
    assert (
        size[
            "milp_total_variables"
        ]
        == 1760
    )
    assert (
        size[
            "milp_approx_constraints"
        ]
        == 1868
    )


def test_exact_gap_metrics():
    result = gap_metrics(
        policy_score=0.9,
        milp_incumbent_score=1.0,
        milp_upper_bound_score=1.0,
        optimal=True,
    )

    assert np.isclose(
        result["exact_retention"],
        0.9,
    )
    assert np.isclose(
        result["exact_relative_gap"],
        0.1,
    )
    assert np.isclose(
        result["exact_absolute_gap"],
        0.1,
    )
    assert np.isclose(
        result[
            "policy_retention_lower_bound"
        ],
        0.9,
    )


def test_timeout_bound_is_not_exact_gap():
    result = gap_metrics(
        policy_score=0.72,
        milp_incumbent_score=0.70,
        milp_upper_bound_score=0.80,
        optimal=False,
    )

    assert result[
        "exact_retention"
    ] is None
    assert result[
        "exact_relative_gap"
    ] is None
    assert np.isclose(
        result[
            "policy_retention_lower_bound"
        ],
        0.9,
    )
    assert np.isclose(
        result[
            "policy_minus_milp_incumbent"
        ],
        0.02,
    )
    assert np.isclose(
        result[
            "policy_to_milp_incumbent_ratio"
        ],
        0.72 / 0.70,
    )


def test_unlimited_time_limit_parser():
    assert _parse_optional_float("unlimited") is None
    assert _parse_optional_float("none") is None
    assert np.isclose(
        _parse_optional_float("300"),
        300.0,
    )


def test_resume_requires_optimal_for_exact_unlimited():
    timed_out = {
        "status": "ok",
        "milp_optimal": False,
    }
    proven = {
        "status": "ok",
        "milp_optimal": True,
    }

    assert _is_completed_row(
        timed_out,
        exact_required=False,
    )
    assert not _is_completed_row(
        timed_out,
        exact_required=True,
    )
    assert _is_completed_row(
        proven,
        exact_required=True,
    )


def test_case_summary_reports_worst_gap_and_runtime_seed():
    rows = [
        {
            "case": "6R_30T",
            "robots": 6,
            "tasks": 30,
            "status": "ok",
            "milp_optimal": True,
            "seed": 101,
            "common_preprocess_seconds": 0.1,
            "policy_seconds": 0.01,
            "milp_total_seconds": 10.0,
            "milp_solver_seconds": 9.9,
            "milp_over_policy_time_ratio": 1000.0,
            "policy_time_optimality": 0.29,
            "exact_retention": 0.97,
            "exact_relative_gap": 0.03,
            "exact_absolute_gap": 0.009,
            "policy_retention_lower_bound": 0.97,
            "policy_to_milp_incumbent_ratio": 0.97,
            "milp_incumbent_score": 0.299,
            "milp_binary_variables": 5580,
            "milp_continuous_variables": 180,
            "milp_total_variables": 5760,
            "milp_approx_constraints": 5982,
        },
        {
            "case": "6R_30T",
            "robots": 6,
            "tasks": 30,
            "status": "ok",
            "milp_optimal": True,
            "seed": 102,
            "common_preprocess_seconds": 0.1,
            "policy_seconds": 0.02,
            "milp_total_seconds": 20.0,
            "milp_solver_seconds": 19.9,
            "milp_over_policy_time_ratio": 1000.0,
            "policy_time_optimality": 0.27,
            "exact_retention": 0.90,
            "exact_relative_gap": 0.10,
            "exact_absolute_gap": 0.03,
            "policy_retention_lower_bound": 0.90,
            "policy_to_milp_incumbent_ratio": 0.90,
            "milp_incumbent_score": 0.30,
            "milp_binary_variables": 5580,
            "milp_continuous_variables": 180,
            "milp_total_variables": 5760,
            "milp_approx_constraints": 5982,
        },
    ]

    summary = _case_summary(rows)

    assert np.isclose(summary["max_exact_relative_gap"], 0.10)
    assert np.isclose(summary["max_exact_absolute_gap"], 0.03)
    assert summary["worst_exact_gap_seed"] == 102
    assert np.isclose(summary["max_milp_total_seconds"], 20.0)
    assert summary["slowest_milp_seed"] == 102
    assert np.isclose(summary["max_policy_seconds"], 0.02)
    assert summary["slowest_policy_seed"] == 102
