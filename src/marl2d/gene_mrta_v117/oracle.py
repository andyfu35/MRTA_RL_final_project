from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from marl2d.gene_mrta_v16t.env import EnvConfig, World
from marl2d.gene_mrta_v16t.global_optimal_core import _milp_options


@dataclass(frozen=True)
class _LinearRouteOptimum:
    optimal: bool
    status: int
    message: str
    score: float | None
    completed_tasks: int | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]
    mip_node_count: int | None = None


@dataclass(frozen=True)
class GlobalPriorityOptimum:
    optimal: bool
    status: int
    message: str
    priority_satisfaction: float | None
    completed_tasks: int | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]
    mip_node_count: int | None = None


@dataclass(frozen=True)
class GlobalCompletionOptimum:
    optimal: bool
    status: int
    message: str
    completion: float | None
    completed_tasks: int | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]
    mip_node_count: int | None = None


@dataclass(frozen=True)
class GlobalDeadlineOptimum:
    optimal: bool
    status: int
    message: str
    deadline_satisfaction: float | None
    completed_tasks: int | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]
    mip_node_count: int | None = None


@dataclass(frozen=True)
class GlobalPathEfficiencyOptimum:
    optimal: bool
    status: int
    message: str
    path_efficiency: float | None
    completed_tasks: int | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]
    mip_node_count: int | None = None


def _path_reward(
    distance: float,
    config: EnvConfig,
) -> float:
    return float(
        1.0
        - np.clip(
            distance
            / max(config.diagonal, 1e-12),
            0.0,
            1.0,
        )
    )


def _solve_linear_route_objective(
    world: World,
    config: EnvConfig,
    *,
    objective: str,
    time_limit: float | None,
    solver_display: bool,
) -> _LinearRouteOptimum:
    if objective not in {
        "completion",
        "priority",
        "deadline",
        "path_efficiency",
    }:
        raise ValueError(
            f"Unsupported route objective: {objective}"
        )

    R = config.num_robots
    N = config.num_tasks
    H = config.episode_time

    total_priority = max(
        float(
            np.sum(
                world.task_priorities
            )
        ),
        1e-12,
    )

    c: list[float] = []
    vlb: list[float] = []
    vub: list[float] = []
    integ: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    clb: list[float] = []
    cub: list[float] = []

    def var(
        obj: float = 0.0,
        lo: float = 0.0,
        hi: float = math.inf,
        integer: bool = False,
    ) -> int:
        k = len(c)
        c.append(float(obj))
        vlb.append(float(lo))
        vub.append(float(hi))
        integ.append(
            1 if integer else 0
        )
        return k

    def con(
        coeff: dict[int, float],
        lo: float = -math.inf,
        hi: float = math.inf,
    ) -> None:
        q = len(clb)
        for k, value in coeff.items():
            rows.append(q)
            cols.append(k)
            vals.append(float(value))
        clb.append(float(lo))
        cub.append(float(hi))

    xs = np.empty(
        (R, N),
        dtype=int,
    )
    xa = np.full(
        (R, N, N),
        -1,
        dtype=int,
    )
    y = np.empty(
        (R, N),
        dtype=int,
    )
    t = np.empty(
        (R, N),
        dtype=int,
    )

    for r in range(R):
        battery = float(
            world.robot_initial_batteries[r]
        )
        for j in range(N):
            d = float(
                world.path_to_tasks[r, j]
            )
            feasible = (
                math.isfinite(d)
                and (
                    d / config.robot_speed
                    + world.task_service_times[j]
                    <= H + 1e-12
                )
                and (
                    d * config.energy_per_distance
                    <= battery + 1e-12
                )
            )
            arc_obj = 0.0
            if (
                objective == "path_efficiency"
                and math.isfinite(d)
            ):
                arc_obj = (
                    -_path_reward(
                        d,
                        config,
                    )
                    / N
                )
            xs[r, j] = var(
                obj=arc_obj,
                hi=(
                    1.0
                    if feasible
                    else 0.0
                ),
                integer=True,
            )

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(
                    world.path_to_tasks[
                        R + i,
                        j,
                    ]
                )
                feasible = (
                    math.isfinite(d)
                    and (
                        d / config.robot_speed
                        + world.task_service_times[j]
                        <= H + 1e-12
                    )
                    and (
                        d * config.energy_per_distance
                        <= battery + 1e-12
                    )
                )
                arc_obj = 0.0
                if (
                    objective
                    == "path_efficiency"
                    and math.isfinite(d)
                ):
                    arc_obj = (
                        -_path_reward(
                            d,
                            config,
                        )
                        / N
                    )
                xa[r, i, j] = var(
                    obj=arc_obj,
                    hi=(
                        1.0
                        if feasible
                        else 0.0
                    ),
                    integer=True,
                )

    for r in range(R):
        for j in range(N):
            y_obj = 0.0
            if objective in {
                "completion",
                "deadline",
            }:
                y_obj = -1.0 / N
            elif objective == "priority":
                y_obj = (
                    -float(
                        world.task_priorities[j]
                    )
                    / total_priority
                )

            y[r, j] = var(
                obj=y_obj,
                hi=1.0,
                integer=True,
            )
            t[r, j] = var(
                hi=H,
            )

    for r in range(R):
        con(
            {
                int(xs[r, j]): 1.0
                for j in range(N)
            },
            hi=1.0,
        )

    for r in range(R):
        for j in range(N):
            z = {
                int(y[r, j]): 1.0,
                int(xs[r, j]): -1.0,
            }
            for i in range(N):
                if i != j:
                    z[
                        int(
                            xa[r, i, j]
                        )
                    ] = -1.0
            con(
                z,
                lo=0.0,
                hi=0.0,
            )

    for r in range(R):
        for i in range(N):
            z = {
                int(y[r, i]): -1.0
            }
            for j in range(N):
                if i != j:
                    z[
                        int(
                            xa[r, i, j]
                        )
                    ] = 1.0
            con(
                z,
                hi=0.0,
            )

    for j in range(N):
        con(
            {
                int(y[r, j]): 1.0
                for r in range(R)
            },
            hi=1.0,
        )

    for r in range(R):
        for j in range(N):
            latest = (
                float(
                    world.task_deadlines[j]
                )
                if objective == "deadline"
                else H
            )
            con(
                {
                    int(t[r, j]): 1.0,
                    int(y[r, j]): -latest,
                },
                hi=0.0,
            )

    finite = world.path_to_tasks[
        np.isfinite(
            world.path_to_tasks
        )
    ]
    max_path = (
        float(np.max(finite))
        if finite.size
        else 0.0
    )
    big_m = (
        H
        + max_path
        / config.robot_speed
        + float(
            np.max(
                world.task_service_times
            )
        )
        + 1.0
    )

    for r in range(R):
        for j in range(N):
            d = float(
                world.path_to_tasks[
                    r,
                    j,
                ]
            )
            if math.isfinite(d):
                duration = (
                    d
                    / config.robot_speed
                    + float(
                        world.task_service_times[
                            j
                        ]
                    )
                )
                con(
                    {
                        int(t[r, j]): 1.0,
                        int(xs[r, j]): -big_m,
                    },
                    lo=duration - big_m,
                )

    for r in range(R):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(
                    world.path_to_tasks[
                        R + i,
                        j,
                    ]
                )
                if math.isfinite(d):
                    duration = (
                        d
                        / config.robot_speed
                        + float(
                            world.task_service_times[
                                j
                            ]
                        )
                    )
                    con(
                        {
                            int(t[r, j]): 1.0,
                            int(t[r, i]): -1.0,
                            int(
                                xa[r, i, j]
                            ): -big_m,
                        },
                        lo=duration - big_m,
                    )

    for r in range(R):
        z: dict[
            int,
            float,
        ] = {}
        for j in range(N):
            d = float(
                world.path_to_tasks[
                    r,
                    j,
                ]
            )
            if math.isfinite(d):
                z[
                    int(xs[r, j])
                ] = (
                    d
                    * config.energy_per_distance
                )

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(
                    world.path_to_tasks[
                        R + i,
                        j,
                    ]
                )
                if math.isfinite(d):
                    z[
                        int(
                            xa[r, i, j]
                        )
                    ] = (
                        d
                        * config.energy_per_distance
                    )

        con(
            z,
            hi=float(
                world.robot_initial_batteries[
                    r
                ]
            ),
        )

    matrix = coo_matrix(
        (
            vals,
            (rows, cols),
        ),
        shape=(
            len(clb),
            len(c),
        ),
        dtype=np.float64,
    ).tocsr()

    started = perf_counter()
    res = milp(
        c=np.asarray(c),
        integrality=np.asarray(
            integ
        ),
        bounds=Bounds(
            np.asarray(vlb),
            np.asarray(vub),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(clb),
            np.asarray(cub),
        ),
        options=_milp_options(
            time_limit=time_limit,
            solver_display=(
                solver_display
            ),
        ),
    )
    elapsed = (
        perf_counter()
        - started
    )

    optimal = (
        int(res.status) == 0
        and res.x is not None
    )
    if (
        time_limit is None
        and not optimal
    ):
        raise RuntimeError(
            "Unlimited exact MILP "
            f"for objective={objective} "
            "terminated without an "
            "optimal proof: "
            f"status={int(res.status)} "
            f"message={res.message}"
        )

    score = (
        float(-res.fun)
        if res.fun is not None
        else None
    )
    completed = None
    routes: list[
        tuple[int, ...]
    ] = []

    if res.x is not None:
        sol = np.asarray(
            res.x
        )
        completed = int(
            round(
                sum(
                    sol[
                        int(y[r, j])
                    ]
                    for r in range(R)
                    for j in range(N)
                )
            )
        )

        for r in range(R):
            first = [
                j
                for j in range(N)
                if sol[
                    int(xs[r, j])
                ] > 0.5
            ]
            if not first:
                routes.append(())
                continue

            route = [
                first[0]
            ]
            while True:
                i = route[-1]
                nxt = [
                    j
                    for j in range(N)
                    if (
                        j != i
                        and sol[
                            int(
                                xa[r, i, j]
                            )
                        ] > 0.5
                    )
                ]
                if not nxt:
                    break
                if nxt[0] in route:
                    raise RuntimeError(
                        "cycle in exact "
                        f"{objective} route"
                    )
                route.append(
                    nxt[0]
                )
            routes.append(
                tuple(route)
            )
    else:
        routes = [
            ()
            for _ in range(R)
        ]

    if (
        optimal
        and score is not None
    ):
        rebuilt = 0.0
        if objective == "priority":
            selected = sum(
                float(
                    world.task_priorities[j]
                )
                for route in routes
                for j in route
            )
            rebuilt = (
                selected
                / total_priority
            )

        elif objective in {
            "completion",
            "deadline",
        }:
            rebuilt = (
                sum(
                    len(route)
                    for route in routes
                )
                / N
            )

        elif objective == "path_efficiency":
            total = 0.0
            for r, route in enumerate(
                routes
            ):
                node = r
                for j in route:
                    d = float(
                        world.path_to_tasks[
                            node,
                            j,
                        ]
                    )
                    total += _path_reward(
                        d,
                        config,
                    )
                    node = R + j
            rebuilt = total / N

        if abs(
            rebuilt - score
        ) > 1e-6:
            raise RuntimeError(
                f"{objective} objective "
                "mismatch: "
                f"solver={score}, "
                f"rebuilt={rebuilt}"
            )

    raw_nodes = getattr(
        res,
        "mip_node_count",
        None,
    )

    return _LinearRouteOptimum(
        optimal=optimal,
        status=int(res.status),
        message=str(res.message),
        score=score,
        completed_tasks=completed,
        solve_seconds=float(
            elapsed
        ),
        mip_gap=(
            float(res.mip_gap)
            if getattr(
                res,
                "mip_gap",
                None,
            ) is not None
            else None
        ),
        routes=tuple(routes),
        mip_node_count=(
            int(raw_nodes)
            if raw_nodes is not None
            else None
        ),
    )


def solve_global_priority_optimum(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> GlobalPriorityOptimum:
    result = _solve_linear_route_objective(
        world,
        config,
        objective="priority",
        time_limit=time_limit,
        solver_display=solver_display,
    )
    return GlobalPriorityOptimum(
        optimal=result.optimal,
        status=result.status,
        message=result.message,
        priority_satisfaction=result.score,
        completed_tasks=result.completed_tasks,
        solve_seconds=result.solve_seconds,
        mip_gap=result.mip_gap,
        routes=result.routes,
        mip_node_count=result.mip_node_count,
    )


def solve_global_completion_optimum(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> GlobalCompletionOptimum:
    result = _solve_linear_route_objective(
        world,
        config,
        objective="completion",
        time_limit=time_limit,
        solver_display=solver_display,
    )
    return GlobalCompletionOptimum(
        optimal=result.optimal,
        status=result.status,
        message=result.message,
        completion=result.score,
        completed_tasks=result.completed_tasks,
        solve_seconds=result.solve_seconds,
        mip_gap=result.mip_gap,
        routes=result.routes,
        mip_node_count=result.mip_node_count,
    )


def solve_global_deadline_optimum(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> GlobalDeadlineOptimum:
    result = _solve_linear_route_objective(
        world,
        config,
        objective="deadline",
        time_limit=time_limit,
        solver_display=solver_display,
    )
    return GlobalDeadlineOptimum(
        optimal=result.optimal,
        status=result.status,
        message=result.message,
        deadline_satisfaction=result.score,
        completed_tasks=result.completed_tasks,
        solve_seconds=result.solve_seconds,
        mip_gap=result.mip_gap,
        routes=result.routes,
        mip_node_count=result.mip_node_count,
    )


def solve_global_path_efficiency_optimum(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> GlobalPathEfficiencyOptimum:
    result = _solve_linear_route_objective(
        world,
        config,
        objective="path_efficiency",
        time_limit=time_limit,
        solver_display=solver_display,
    )
    return GlobalPathEfficiencyOptimum(
        optimal=result.optimal,
        status=result.status,
        message=result.message,
        path_efficiency=result.score,
        completed_tasks=result.completed_tasks,
        solve_seconds=result.solve_seconds,
        mip_gap=result.mip_gap,
        routes=result.routes,
        mip_node_count=result.mip_node_count,
    )
