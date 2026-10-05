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


def solve_global_priority_optimum(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> GlobalPriorityOptimum:
    R = config.num_robots
    N = config.num_tasks
    H = config.episode_time
    total_priority = float(
        np.sum(world.task_priorities)
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
        integ.append(1 if integer else 0)
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

    xs = np.empty((R, N), dtype=int)
    xa = np.full((R, N, N), -1, dtype=int)
    y = np.empty((R, N), dtype=int)
    t = np.empty((R, N), dtype=int)

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
            xs[r, j] = var(
                hi=1.0 if feasible else 0.0,
                integer=True,
            )

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(
                    world.path_to_tasks[R + i, j]
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
                xa[r, i, j] = var(
                    hi=1.0 if feasible else 0.0,
                    integer=True,
                )

    priority_scale = max(
        total_priority,
        1e-12,
    )
    for r in range(R):
        for j in range(N):
            y[r, j] = var(
                obj=(
                    -float(
                        world.task_priorities[j]
                    )
                    / priority_scale
                ),
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
                        int(xa[r, i, j])
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
                        int(xa[r, i, j])
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
            con(
                {
                    int(t[r, j]): 1.0,
                    int(y[r, j]): -H,
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
        + max_path / config.robot_speed
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
                world.path_to_tasks[r, j]
            )
            if math.isfinite(d):
                duration = (
                    d / config.robot_speed
                    + float(
                        world.task_service_times[j]
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
                        d / config.robot_speed
                        + float(
                            world.task_service_times[j]
                        )
                    )
                    con(
                        {
                            int(t[r, j]): 1.0,
                            int(t[r, i]): -1.0,
                            int(xa[r, i, j]): -big_m,
                        },
                        lo=duration - big_m,
                    )

    for r in range(R):
        z: dict[int, float] = {}
        for j in range(N):
            d = float(
                world.path_to_tasks[r, j]
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
                        int(xa[r, i, j])
                    ] = (
                        d
                        * config.energy_per_distance
                    )
        con(
            z,
            hi=float(
                world.robot_initial_batteries[r]
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
        integrality=np.asarray(integ),
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
            solver_display=solver_display,
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
            "Unlimited priority MILP terminated "
            "without an optimal proof: "
            f"status={int(res.status)} "
            f"message={res.message}"
        )

    score = (
        float(-res.fun)
        if res.fun is not None
        else None
    )
    routes: list[
        tuple[int, ...]
    ] = []
    completed = None

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
                        "cycle in priority MILP route"
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
        selected_priority = 0.0
        for route in routes:
            for j in route:
                selected_priority += float(
                    world.task_priorities[j]
                )
        rebuilt = (
            selected_priority
            / priority_scale
        )
        if abs(
            rebuilt - score
        ) > 1e-6:
            raise RuntimeError(
                "priority objective mismatch: "
                f"solver={score}, "
                f"rebuilt={rebuilt}"
            )

    raw_nodes = getattr(
        res,
        "mip_node_count",
        None,
    )

    return GlobalPriorityOptimum(
        optimal=optimal,
        status=int(res.status),
        message=str(res.message),
        priority_satisfaction=score,
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
