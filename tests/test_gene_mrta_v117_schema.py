import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    Evaluation,
    generate_world,
)
from marl2d.gene_mrta_v117.capabilities import (
    BASE_AXES,
    base_scores_for_evaluation,
)
from marl2d.gene_mrta_v117.schema import (
    world_to_spec,
    spec_to_world,
)


def test_world_schema_round_trip_preserves_task_data():
    config = EnvConfig(
        num_robots=2,
        num_tasks=5,
        obstacle_count=0,
    )
    world = generate_world(
        config,
        seed=117000001,
    )
    spec = world_to_spec(
        world,
        seed=117000001,
    )
    rebuilt = spec_to_world(
        spec,
        config,
    )

    assert spec.seed == 117000001
    assert len(spec.robots) == 2
    assert len(spec.tasks) == 5
    assert np.allclose(
        rebuilt.robot_positions,
        world.robot_positions,
    )
    assert np.allclose(
        rebuilt.robot_initial_batteries,
        world.robot_initial_batteries,
    )
    assert np.allclose(
        rebuilt.task_positions,
        world.task_positions,
    )
    assert np.allclose(
        rebuilt.task_service_times,
        world.task_service_times,
    )
    assert np.allclose(
        rebuilt.task_priorities,
        world.task_priorities,
    )
    assert np.allclose(
        rebuilt.task_deadlines,
        world.task_deadlines,
    )
    assert np.allclose(
        rebuilt.path_to_tasks,
        world.path_to_tasks,
    )


def test_stage_a_axes_cover_known_objectives_without_battery_loophole():
    evaluation = Evaluation(
        completion=0.8,
        efficiency=0.6,
        priority_satisfaction=0.7,
        deadline_satisfaction=0.5,
        balance=0.75,
        time_optimality=0.24,
        route_efficiency=0.75,
        completed_tasks=16.0,
        completed_priority=7.0,
        total_priority=10.0,
        on_time_tasks=10.0,
        total_travel=100.0,
        total_euclidean_travel=90.0,
        detour_ratio=1.1,
        mean_initial_battery=50.0,
        mean_final_battery=30.0,
        battery_remaining_fraction=0.6,
        energy_consumed=80.0,
        battery_blocked_pair_events=2.0,
        robot_task_counts=(4.0, 4.0, 4.0, 4.0),
        robot_workloads=(20.0, 20.0, 20.0, 20.0),
        robot_final_batteries=(30.0, 30.0, 30.0, 30.0),
    )

    scores = base_scores_for_evaluation(
        evaluation,
        completion_optimum=0.8,
        time_optimum=0.25,
        priority_optimum=0.875,
    )

    assert tuple(scores) == BASE_AXES
    assert np.isclose(
        scores["global_completion_optimality"],
        1.0,
    )
    assert np.isclose(
        scores["global_time_optimality"],
        0.96,
    )
    assert np.isclose(
        scores["path_efficiency"],
        0.6,
    )
    assert np.isclose(
        scores["global_priority_optimality"],
        0.8,
    )
    assert np.isclose(
        scores["deadline_satisfaction"],
        0.5,
    )
    assert np.isclose(
        scores["workload_balance"],
        0.9375,
    )
    assert "battery_remaining" not in scores
    assert "energy_consumed" not in scores


def test_global_capability_one_means_axis_optimum():
    evaluation = Evaluation(
        completion=1.0,
        efficiency=0.8,
        priority_satisfaction=0.9,
        deadline_satisfaction=0.7,
        balance=0.85,
        time_optimality=0.3,
        route_efficiency=0.8,
        completed_tasks=20.0,
        completed_priority=9.0,
        total_priority=10.0,
        on_time_tasks=14.0,
        total_travel=80.0,
        total_euclidean_travel=75.0,
        detour_ratio=1.066,
        mean_initial_battery=50.0,
        mean_final_battery=30.0,
        battery_remaining_fraction=0.6,
        energy_consumed=80.0,
        battery_blocked_pair_events=0.0,
        robot_task_counts=(5.0, 5.0, 5.0, 5.0),
        robot_workloads=(20.0, 20.0, 20.0, 20.0),
        robot_final_batteries=(30.0, 30.0, 30.0, 30.0),
    )

    scores = base_scores_for_evaluation(
        evaluation,
        completion_optimum=1.0,
        time_optimum=0.3,
        priority_optimum=0.9,
    )

    assert np.isclose(
        scores["global_completion_optimality"],
        1.0,
    )
    assert np.isclose(
        scores["global_time_optimality"],
        1.0,
    )
    assert np.isclose(
        scores["global_priority_optimality"],
        1.0,
    )
