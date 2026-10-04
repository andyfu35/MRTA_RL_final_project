from marl2d.gene_mrta_v1143.analyze import (
    _parent_context,
    _transition,
    _wtl,
)


def _row(delta):
    return {
        "adaptive_minus_center": {
            "mean_time": delta,
            "tail10_time": delta,
            "continuation_preservation": delta,
            "fleet_option_reserve": delta,
        }
    }


def test_wtl_counts():
    result = _wtl(
        [_row(0.2), _row(0.0), _row(-0.3)],
        "mean_time",
        tolerance=1e-12,
    )
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1


def test_parent_context_is_unordered():
    row_a = {
        "parent_a_capabilities": ["mean_time"],
        "parent_b_capabilities": ["tail10_time", "mean_time"],
    }
    row_b = {
        "parent_a_capabilities": ["tail10_time", "mean_time"],
        "parent_b_capabilities": ["mean_time"],
    }
    assert _parent_context(row_a) == _parent_context(row_b)


def test_transition_labels():
    assert _transition(True, False) == "rescue"
    assert _transition(False, True) == "loss"
    assert _transition(True, True) == "both"
    assert _transition(False, False) == "neither"
