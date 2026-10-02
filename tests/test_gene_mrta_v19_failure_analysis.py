import numpy as np

from marl2d.gene_mrta_v18.bottom10_failure_analysis import (
    _candidate_axes,
    _future_graph_snapshot,
    _select_bottom_rows,
)
from marl2d.gene_mrta_v16t.env import EnvConfig, build_world


def _simple_world():
    config = EnvConfig(
        world_size=20.0,
        num_robots=2,
        num_tasks=3,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=1.0,
        priority_max=1.0,
        deadline_min=20.0,
        deadline_max=20.0,
        episode_time=20.0,
        obstacle_count=0,
        obstacle_size_min=2.0,
        obstacle_size_max=4.0,
        obstacle_clearance=1.0,
        grid_resolution=1.0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([100.0, 100.0]),
        np.array([[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]]),
        np.ones(3),
        np.ones(3),
        np.full(3, 20.0),
        np.zeros((0, 4)),
    )
    return config, world


def test_select_bottom_rows_uses_v18_retention():
    publication = {
        "rows": [
            {"world_seed": 3, "optimal": True, "v18_retention": 0.95},
            {"world_seed": 1, "optimal": True, "v18_retention": 0.80},
            {"world_seed": 2, "optimal": True, "v18_retention": 0.90},
            {"world_seed": 4, "optimal": False, "v18_retention": 0.10},
        ]
    }
    rows = _select_bottom_rows(publication, 2)
    assert [row["world_seed"] for row in rows] == [1, 2]


def test_future_graph_snapshot_reports_reachability_and_option_mass():
    config, world = _simple_world()
    snapshot = _future_graph_snapshot(
        world=world,
        config=config,
        now=0.0,
        current_node_ids=np.array([0, 1], dtype=np.int64),
        battery_remaining=np.array([100.0, 100.0]),
        busy_until=np.zeros(2),
        task_available=np.ones(3, dtype=bool),
    )
    assert snapshot["available_tasks"] == 3
    assert snapshot["edge_count"] > 0
    assert 0.0 <= snapshot["edge_density"] <= 1.0
    assert 0.0 <= snapshot["reachable_task_ratio"] <= 1.0
    assert snapshot["option_mass"] > 0.0
    assert len(snapshot["task_owner_counts"]) == 3
    assert len(snapshot["robot_option_counts"]) == 2


def test_candidate_axes_follow_observed_failure_flags():
    analyses = [
        {
            "mechanism_flags": {
                "tags": [
                    "continuation_collapse",
                    "fleet_reserve_risk",
                ]
            }
        },
        {
            "mechanism_flags": {
                "tags": [
                    "immediate_future_imbalance",
                    "hard_for_all",
                ]
            }
        },
    ]
    axes = _candidate_axes(analyses)
    names = {item["axis"] for item in axes}
    assert "continuation_preservation" in names
    assert "fleet_option_reserve" in names
    assert "immediate_time_capture" in names
    assert "hard_world_time_specialist" in names
