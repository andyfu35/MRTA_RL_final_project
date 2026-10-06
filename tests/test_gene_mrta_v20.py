import numpy as np

from marl2d.gene_mrta_v20.bank import (
    BankRecord,
    DEFAULT_RESOLUTIONS,
    dominates,
    rebuild_bank,
)
from marl2d.gene_mrta_v20.gene import SetAssignmentGene
from marl2d.gene_mrta_v20.rollout import rollout_gene_batch
from marl2d.gene_mrta_v20.suite import (
    FIXED_WORLD_SEEDS,
    WorldConfig,
    fixed_worlds,
    generate_world,
)


def test_fixed_suite_has_100_unique_seeds_and_requested_ranges():
    assert len(FIXED_WORLD_SEEDS) == 100
    assert len(set(FIXED_WORLD_SEEDS)) == 100
    worlds = fixed_worlds(count=100)
    assert all(
        5 <= w.robot_count <= 20
        for w in worlds
    )
    assert all(
        10 <= w.task_count <= 100
        for w in worlds
    )


def test_task_and_robot_input_schema_are_5d_and_4d():
    world = generate_world(
        FIXED_WORLD_SEEDS[0]
    )
    assert (
        world.task_features().shape
        == (world.task_count, 5)
    )
    assert (
        world.initial_robot_features().shape
        == (world.robot_count, 4)
    )
    assert np.allclose(
        world.task_features()[:, 2],
        world.task_priorities,
    )
    assert np.allclose(
        world.task_features()[:, 3],
        world.task_deadlines,
    )
    assert np.allclose(
        world.task_features()[:, 4],
        world.task_service_times,
    )


def test_generated_deadlines_are_guaranteed_feasible_by_construction():
    for world in fixed_worlds(count=100):
        assert np.all(
            world.task_deadlines
            > world.baseline_completion_times
        )


def test_gene_parameter_count_is_fixed_when_robot_and_task_counts_change():
    assert (
        SetAssignmentGene.parameter_count(8)
        == 139
    )
    a = generate_world(
        FIXED_WORLD_SEEDS[0]
    )
    b = generate_world(
        FIXED_WORLD_SEEDS[1]
    )
    assert (
        a.robot_count,
        a.task_count,
    ) != (
        b.robot_count,
        b.task_count,
    )
    rng = np.random.default_rng(7)
    gene = SetAssignmentGene.random(
        rng,
        hidden_dim=8,
    )
    assert gene.vector_data.shape == (139,)


def test_rollout_completes_every_task_and_returns_raw_axes():
    config = WorldConfig()
    world = generate_world(
        FIXED_WORLD_SEEDS[2],
        config,
    )
    gene = SetAssignmentGene.random(
        np.random.default_rng(9),
        hidden_dim=8,
    )
    row = rollout_gene_batch(
        [gene],
        world,
        config=config,
        device="cpu",
    )[0]
    assert (
        row.all_tasks_completed
        == world.task_count
    )
    assert row.total_time > 0.0
    assert row.priority >= 0.0
    assert (
        0.0
        <= row.on_time_completed_tasks
        <= world.task_count
    )
    assert np.isclose(
        row.deadline_completion_rate,
        row.on_time_completed_tasks
        / world.task_count,
    )


def _record(
    record_id,
    scores,
):
    vector = np.zeros(
        SetAssignmentGene.parameter_count(8)
    )
    vector[0] = float(
        ord(record_id[0])
    )
    gene = SetAssignmentGene(
        vector,
        hidden_dim=8,
    )
    return BankRecord(
        record_id=record_id,
        gene=gene,
        scores=scores,
        round_index=0,
        parent_id=None,
    )


def test_pareto_mixed_directions_without_axis_bounds():
    a = _record(
        "a",
        {
            "total_time": 10.0,
            "priority": 4.0,
            "on_time_completed_tasks": 20.0,
        },
    )
    b = _record(
        "b",
        {
            "total_time": 12.0,
            "priority": 5.0,
            "on_time_completed_tasks": 19.0,
        },
    )
    c = _record(
        "c",
        {
            "total_time": 9.0,
            "priority": 8.0,
            "on_time_completed_tasks": 21.0,
        },
    )
    assert dominates(a, b)
    assert not dominates(a, c)
    kept = rebuild_bank([a, b, c])
    assert set(kept) == {"a", "c"}



def test_epsilon_grid_merges_near_identical_nonchampion_cells():
    time_priority_champion = _record(
        "h",
        {
            "total_time": 50.0,
            "priority": 19.0,
            "on_time_completed_tasks": 39.0,
        },
    )
    completion_champion = _record(
        "i",
        {
            "total_time": 70.0,
            "priority": 30.0,
            "on_time_completed_tasks": 50.0,
        },
    )
    a = _record(
        "d",
        {
            "total_time": 60.01,
            "priority": 20.09,
            "on_time_completed_tasks": 40.01,
        },
    )
    b = _record(
        "e",
        {
            "total_time": 60.49,
            "priority": 20.01,
            "on_time_completed_tasks": 40.49,
        },
    )
    kept = rebuild_bank(
        [
            time_priority_champion,
            completion_champion,
            a,
            b,
        ],
        resolutions=DEFAULT_RESOLUTIONS,
    )
    assert not (
        "d" in kept
        and "e" in kept
    )


def test_epsilon_grid_always_preserves_exact_axis_champions():
    rows = [
        _record(
            "f",
            {
                "total_time": 60.01,
                "priority": 20.09,
                "on_time_completed_tasks": 40.01,
            },
        ),
        _record(
            "g",
            {
                "total_time": 60.49,
                "priority": 20.01,
                "on_time_completed_tasks": 40.49,
            },
        ),
    ]
    kept = rebuild_bank(
        rows,
        resolutions=DEFAULT_RESOLUTIONS,
    )
    assert "f" in kept
    assert "g" in kept
