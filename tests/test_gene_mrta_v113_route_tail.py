import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
)
from marl2d.gene_mrta_v18.direct_gene import (
    ConsequenceAwareDirectGene,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v113.route_tail import (
    build_route_tail_observations,
    plan_route_tails,
    rollout_route_tail_gene,
)


def _simple_world():
    config = EnvConfig(
        world_size=20.0,
        num_robots=2,
        num_tasks=5,
        robot_speed=1.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=1.0,
        priority_max=1.0,
        deadline_min=50.0,
        deadline_max=50.0,
        episode_time=50.0,
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
        np.asarray(
            [
                [0.0, 0.0],
                [15.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [100.0, 100.0],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.ones(
            5,
            dtype=np.float64,
        ),
        np.ones(
            5,
            dtype=np.float64,
        ),
        np.full(
            5,
            50.0,
            dtype=np.float64,
        ),
        np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )
    return config, world


def _always_continue_gene():
    vector = np.zeros(
        RouteTailDirectGene.parameter_count(8),
        dtype=np.float64,
    )

    # Decoder bias is index 137 for hidden_dim=8.
    vector[137] = 1.0

    # STOP bias is the final scalar. Make STOP impossible while a feasible
    # pair exists.
    vector[-1] = -100.0

    return RouteTailDirectGene(
        vector,
        hidden_dim=8,
    )


def test_v113_keeps_same_148_parameter_layout():
    assert (
        RouteTailDirectGene.parameter_count(8)
        == 148
    )

    rng = np.random.default_rng(17)
    old = ConsequenceAwareDirectGene.random(
        rng
    )
    lifted = RouteTailDirectGene.from_v18(
        old
    )
    assert np.array_equal(
        lifted.vector_data,
        old.vector_data,
    )


def test_route_tail_observation_uses_virtual_tail_time_and_node():
    config, world = _simple_world()

    tail_positions = (
        world.robot_positions.copy()
    )
    tail_node_ids = np.arange(
        2,
        dtype=np.int64,
    )
    tail_times = np.zeros(
        2,
        dtype=np.float64,
    )
    battery = (
        world.robot_initial_batteries.copy()
    )
    workloads = np.zeros(
        2,
        dtype=np.float64,
    )
    available = np.ones(
        5,
        dtype=bool,
    )

    first = build_route_tail_observations(
        world=world,
        config=config,
        tail_positions=tail_positions,
        tail_node_ids=tail_node_ids,
        tail_times=tail_times,
        battery_remaining=battery,
        robot_workloads=workloads,
        task_available=available,
    )

    path0 = first[2][0, 0]
    finish0 = (
        path0
        / config.robot_speed
        + world.task_service_times[0]
    )

    available[0] = False
    tail_positions[0] = (
        world.task_positions[0]
    )
    tail_node_ids[0] = (
        config.num_robots
        + 0
    )
    tail_times[0] = finish0
    battery[0] -= (
        path0
        * config.energy_per_distance
    )
    workloads[0] += finish0

    second = build_route_tail_observations(
        world=world,
        config=config,
        tail_positions=tail_positions,
        tail_node_ids=tail_node_ids,
        tail_times=tail_times,
        battery_remaining=battery,
        robot_workloads=workloads,
        task_available=available,
    )

    # R0 -> T1 must now be evaluated from T0, not from R0's initial pose.
    expected = float(
        world.path_to_tasks[
            config.num_robots,
            1,
        ]
    )
    assert np.isclose(
        second[2][0, 1],
        expected,
    )

    expected_deadline_remaining = (
        world.task_deadlines[1]
        - finish0
    ) / config.episode_time
    assert np.isclose(
        second[0][0, 1, 4],
        expected_deadline_remaining,
    )


def test_one_robot_can_receive_multiple_ordered_tasks_in_one_plan():
    config, world = _simple_world()
    gene = _always_continue_gene()

    plan = plan_route_tails(
        gene,
        world,
        config,
    )

    assert plan.remaining_tasks == ()
    assert plan.routes[0] == (
        0,
        1,
        2,
        3,
        4,
    )
    assert plan.routes[1] == ()
    assert len(
        plan.selection_sequence
    ) == 5

    finishes = [
        item.finish_time
        for item
        in plan.selection_sequence
    ]
    assert finishes == sorted(
        finishes
    )
    for previous, current in zip(
        plan.selection_sequence,
        plan.selection_sequence[1:],
    ):
        assert np.isclose(
            current.start_time,
            previous.finish_time,
        )


def test_route_tail_rollout_counts_all_tasks_in_robot_queue():
    config, world = _simple_world()
    rollout = rollout_route_tail_gene(
        _always_continue_gene(),
        world,
        config,
    )

    assert (
        rollout.evaluation.completed_tasks
        == 5.0
    )
    assert (
        rollout.evaluation.completion
        == 1.0
    )
    assert (
        rollout.evaluation.robot_task_counts
        == (5.0, 0.0)
    )
    assert max(
        rollout.plan.final_tail_times
    ) <= config.episode_time
    assert min(
        rollout.plan.final_batteries
    ) >= 0.0
