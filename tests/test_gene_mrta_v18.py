import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
)
from marl2d.gene_mrta_v17.direct_gene import (
    DirectAssignmentGene as V17DirectAssignmentGene,
)
from marl2d.gene_mrta_v17.rollout import (
    _build_observations as build_v17_observations,
    rollout_direct_gene as rollout_v17,
)

from marl2d.gene_mrta_v18.direct_gene import (
    ConsequenceAwareDirectGene,
)
from marl2d.gene_mrta_v18.rollout import (
    _build_observations as build_v18_observations,
    rollout_direct_gene as rollout_v18,
)


def _world():
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
        np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [100.0, 100.0],
            dtype=np.float64,
        ),
        np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [9.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.ones(
            3,
            dtype=np.float64,
        ),
        np.ones(
            3,
            dtype=np.float64,
        ),
        np.full(
            3,
            20.0,
            dtype=np.float64,
        ),
        np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )

    return config, world


def test_v18_parameter_count_is_148_for_hidden8():
    assert (
        ConsequenceAwareDirectGene.parameter_count(8)
        == 148
    )


def test_lifted_v17_gene_preserves_logits_and_stop():
    rng = np.random.default_rng(7)

    v17 = V17DirectAssignmentGene.random(
        rng,
        hidden_dim=4,
    )
    v18 = ConsequenceAwareDirectGene.from_v17(
        v17
    )

    base = rng.normal(
        size=(3, 5, 8)
    )
    consequences = rng.normal(
        size=(3, 5, 4)
    )
    extended = np.concatenate(
        [base, consequences],
        axis=-1,
    )

    eligible = np.ones(
        (3, 5),
        dtype=bool,
    )
    eligible[1, 2] = False

    rows = np.array(
        [True, True, True]
    )
    cols = np.array(
        [True, True, True, True, True]
    )

    old_logits, old_stop = v17.action_logits(
        base,
        eligible,
        rows,
        cols,
        step=1,
    )
    new_logits, new_stop = v18.action_logits(
        extended,
        eligible,
        rows,
        cols,
        step=1,
    )

    assert np.allclose(
        old_logits,
        new_logits,
        equal_nan=True,
    )
    assert np.isclose(
        old_stop,
        new_stop,
    )


def test_v18_first_eight_features_exactly_match_v17():
    config, world = _world()

    robot_positions = world.robot_positions.copy()
    current_node_ids = np.arange(
        config.num_robots,
        dtype=np.int64,
    )
    battery = world.robot_initial_batteries.copy()
    workloads = np.zeros(
        config.num_robots,
        dtype=np.float64,
    )
    busy = np.zeros(
        config.num_robots,
        dtype=np.float64,
    )
    available = np.ones(
        config.num_tasks,
        dtype=bool,
    )

    old = build_v17_observations(
        world=world,
        config=config,
        robot_positions=robot_positions,
        current_node_ids=current_node_ids,
        battery_remaining=battery,
        robot_workloads=workloads,
        busy_until=busy,
        task_available=available,
    )
    new = build_v18_observations(
        world=world,
        config=config,
        robot_positions=robot_positions,
        current_node_ids=current_node_ids,
        battery_remaining=battery,
        robot_workloads=workloads,
        busy_until=busy,
        task_available=available,
    )

    assert np.allclose(
        old[2],
        new[2][..., :8],
    )
    assert np.array_equal(
        old[3],
        new[3],
    )


def test_v18_consequence_features_have_expected_semantics():
    config, world = _world()

    result = build_v18_observations(
        world=world,
        config=config,
        robot_positions=world.robot_positions.copy(),
        current_node_ids=np.arange(
            config.num_robots,
            dtype=np.int64,
        ),
        battery_remaining=world.robot_initial_batteries.copy(),
        robot_workloads=np.zeros(
            config.num_robots,
            dtype=np.float64,
        ),
        busy_until=np.zeros(
            config.num_robots,
            dtype=np.float64,
        ),
        task_available=np.ones(
            config.num_tasks,
            dtype=bool,
        ),
    )

    observations = result[2]
    eligible = result[3]
    path_lengths = result[4]

    assert observations.shape == (
        2,
        3,
        12,
    )
    assert np.all(
        observations[..., 8:] >= 0.0
    )
    assert np.all(
        observations[..., 8:] <= 1.0
    )

    # R0 -> T0 leaves enough time/battery for at least one next task.
    assert eligible[0, 0]
    assert observations[0, 0, 8] > 0.0
    assert observations[0, 0, 9] > 0.0

    expected_residual = (
        100.0
        - path_lengths[0, 0]
    ) / 100.0
    assert np.isclose(
        observations[0, 0, 11],
        expected_residual,
    )

    # T2 is much more valuable/scarce for R1 than T0, so R0 taking T2
    # should carry greater cross-robot opportunity cost.
    assert (
        observations[0, 2, 10]
        > observations[0, 0, 10]
    )


def test_lifted_v18_rollout_exactly_preserves_v17_policy():
    config, world = _world()
    rng = np.random.default_rng(17)

    v17 = V17DirectAssignmentGene.random(
        rng,
        hidden_dim=4,
    )
    v18 = ConsequenceAwareDirectGene.from_v17(
        v17
    )

    old = rollout_v17(
        v17,
        world,
        config,
    )
    new = rollout_v18(
        v18,
        world,
        config,
    )

    assert (
        old.assignment_events
        == new.assignment_events
    )
    assert np.isclose(
        old.evaluation.time_optimality,
        new.evaluation.time_optimality,
    )
    assert np.isclose(
        old.evaluation.completion,
        new.evaluation.completion,
    )
    assert np.isclose(
        old.evaluation.total_travel,
        new.evaluation.total_travel,
    )
