import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    build_world,
)
from marl2d.gene_mrta_v113.route_tail import (
    RouteTailPlan,
    RouteTailStep,
)
from marl2d.gene_mrta_v117.stage_a_train import (
    Record,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v118.capabilities import (
    BASE_AXES,
    priority_service_score,
)
from marl2d.gene_mrta_v118.oracle import (
    solve_all_complete_optimum,
)
from marl2d.gene_mrta_v118.world_bank import (
    construct_feasible_world,
    feasibility_first_config,
)
from marl2d.gene_mrta_v118.pareto_bank import (
    rebuild_pareto_bank,
)


def _two_task_world():
    config = EnvConfig(
        world_size=100.0,
        num_robots=1,
        num_tasks=2,
        robot_speed=10.0,
        service_time_min=1.0,
        service_time_max=1.0,
        priority_min=0.1,
        priority_max=1.0,
        deadline_min=2.0,
        deadline_max=20.0,
        episode_time=20.0,
        obstacle_count=0,
        battery_capacity=100.0,
        initial_battery_min=100.0,
        initial_battery_max=100.0,
        energy_per_distance=1.0,
    )
    world = build_world(
        config,
        robot_positions=np.asarray(
            [[0.0, 0.0]],
            dtype=np.float64,
        ),
        robot_initial_batteries=np.asarray(
            [100.0],
            dtype=np.float64,
        ),
        task_positions=np.asarray(
            [
                [20.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=np.float64,
        ),
        task_service_times=np.asarray(
            [1.0, 1.0],
            dtype=np.float64,
        ),
        task_priorities=np.asarray(
            [0.1, 1.0],
            dtype=np.float64,
        ),
        task_deadlines=np.asarray(
            [20.0, 20.0],
            dtype=np.float64,
        ),
        obstacles=np.zeros(
            (0, 4),
            dtype=np.float64,
        ),
    )
    return config, world


def test_v118_axes_exclude_completion_gate():
    assert (
        "global_completion_optimality"
        not in BASE_AXES
    )
    assert BASE_AXES == (
        "time_earliness",
        "path_efficiency",
        "priority_service",
        "deadline_satisfaction",
        "workload_balance",
    )


def test_v118_priority_service_rewards_important_task_earlier():
    config, world = (
        _two_task_world()
    )

    high_first = RouteTailPlan(
        routes=((1, 0),),
        selection_sequence=(
            RouteTailStep(
                step=0,
                robot=0,
                task=1,
                start_time=0.0,
                finish_time=2.0,
                path_distance=10.0,
                euclidean_distance=10.0,
                service_time=1.0,
                energy_used=10.0,
            ),
            RouteTailStep(
                step=1,
                robot=0,
                task=0,
                start_time=2.0,
                finish_time=4.0,
                path_distance=10.0,
                euclidean_distance=10.0,
                service_time=1.0,
                energy_used=10.0,
            ),
        ),
        stopped_by_policy=False,
        remaining_tasks=(),
        final_tail_times=(4.0,),
        final_batteries=(80.0,),
        final_workloads=(4.0,),
        battery_blocked_pair_events=0.0,
    )

    low_first = RouteTailPlan(
        routes=((0, 1),),
        selection_sequence=(
            RouteTailStep(
                step=0,
                robot=0,
                task=0,
                start_time=0.0,
                finish_time=2.0,
                path_distance=20.0,
                euclidean_distance=20.0,
                service_time=1.0,
                energy_used=20.0,
            ),
            RouteTailStep(
                step=1,
                robot=0,
                task=1,
                start_time=2.0,
                finish_time=4.0,
                path_distance=10.0,
                euclidean_distance=10.0,
                service_time=1.0,
                energy_used=10.0,
            ),
        ),
        stopped_by_policy=False,
        remaining_tasks=(),
        final_tail_times=(4.0,),
        final_batteries=(70.0,),
        final_workloads=(4.0,),
        battery_blocked_pair_events=0.0,
    )

    assert (
        priority_service_score(
            high_first,
            world,
            config,
        )
        >
        priority_service_score(
            low_first,
            world,
            config,
        )
    )


def test_v118_all_complete_oracle_assigns_every_task():
    config, world = (
        _two_task_world()
    )
    result = (
        solve_all_complete_optimum(
            world,
            config,
            objective="time",
            time_limit=30.0,
        )
    )
    assert result.optimal
    assert result.completed_tasks == 2
    assert sum(
        len(route)
        for route in result.routes
    ) == 2


def test_v118_priority_oracle_prefers_high_priority_early():
    config, world = (
        _two_task_world()
    )
    result = (
        solve_all_complete_optimum(
            world,
            config,
            objective=(
                "priority_service"
            ),
            time_limit=30.0,
        )
    )
    assert result.optimal
    assert result.completed_tasks == 2
    assert result.routes == (
        (1, 0),
    )


def test_v118_feasibility_config_has_enough_total_service_horizon():
    config = (
        feasibility_first_config(
            4,
            20,
        )
    )
    assert (
        config.episode_time
        > 50.0
    )
    assert (
        config.battery_capacity
        > 70.0
    )
    assert (
        config.deadline_max
        <= config.episode_time
    )


def test_v118_pareto_bank_has_no_completion_axis():
    gene = RouteTailDirectGene(
        np.zeros(
            148,
            dtype=np.float64,
        ),
        hidden_dim=8,
    )
    scores_a = {
        axis: 0.9
        for axis in BASE_AXES
    }
    scores_b = {
        axis: 0.8
        for axis in BASE_AXES
    }
    records = {
        "a": Record(
            record_id="a",
            gene=gene,
            scores=scores_a,
            capabilities=(),
            origin="test",
            generation=0,
        ),
        "b": Record(
            record_id="b",
            gene=gene,
            scores=scores_b,
            capabilities=(),
            origin="test",
            generation=0,
        ),
    }
    bank = rebuild_pareto_bank(
        records,
        max_size=16,
        epsilon=0.001,
    )
    assert set(
        bank.records
    ) == {
        "a",
    }


def test_v118_constructive_world_has_explicit_all_task_witness():
    config = feasibility_first_config(
        4,
        20,
    )

    built = None
    for offset in range(20):
        built = construct_feasible_world(
            config,
            117100000 + offset,
            battery_reserve_fraction=0.05,
        )
        if built is not None:
            break

    assert built is not None
    world, witness = built

    assigned = sorted(
        task
        for route in witness.routes
        for task in route
    )
    assert assigned == list(
        range(
            config.num_tasks
        )
    )
    assert max(
        witness.finish_times
    ) <= (
        config.episode_time
        + 1e-9
    )
    assert all(
        used
        <= battery + 1e-9
        for used, battery in zip(
            witness.travel_energy,
            world.robot_initial_batteries,
            strict=True,
        )
    )


def test_v118_capability_axes_are_raw_zero_to_one_metrics():
    assert all(
        "optimality"
        not in axis
        for axis in BASE_AXES
    )
    assert (
        "global_completion_optimality"
        not in BASE_AXES
    )
