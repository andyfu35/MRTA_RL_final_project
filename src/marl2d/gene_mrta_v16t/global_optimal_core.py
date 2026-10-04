from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .env import EnvConfig, World


@dataclass(frozen=True)
class GlobalTimeOptimum:
    optimal: bool
    status: int
    message: str
    time_optimality: float | None
    completed_tasks: int | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]
    mip_node_count: int | None = None
    time_optimality_upper_bound: float | None = None


def _milp_options(
    *,
    time_limit: float | None,
    solver_display: bool,
) -> dict[str, float | bool]:
    options: dict[str, float | bool] = {
        "disp": bool(solver_display),
        "presolve": True,
        "mip_rel_gap": 0.0,
    }
    if time_limit is not None:
        if float(time_limit) <= 0.0:
            raise ValueError("time_limit must be positive or None")
        options["time_limit"] = float(time_limit)
    return options


def solve_global_time_optimum(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> GlobalTimeOptimum:
    R, N, H = config.num_robots, config.num_tasks, config.episode_time

    c: list[float] = []
    vlb: list[float] = []
    vub: list[float] = []
    integ: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    clb: list[float] = []
    cub: list[float] = []

    def var(obj=0.0, lo=0.0, hi=math.inf, integer=False):
        k = len(c)
        c.append(float(obj)); vlb.append(float(lo)); vub.append(float(hi))
        integ.append(1 if integer else 0)
        return k

    def con(coeff: dict[int, float], lo=-math.inf, hi=math.inf):
        q = len(clb)
        for k, v in coeff.items():
            rows.append(q); cols.append(k); vals.append(float(v))
        clb.append(float(lo)); cub.append(float(hi))

    xs = np.empty((R, N), dtype=int)
    xa = np.full((R, N, N), -1, dtype=int)
    y = np.empty((R, N), dtype=int)
    t = np.empty((R, N), dtype=int)

    for r in range(R):
        B = float(world.robot_initial_batteries[r])
        for j in range(N):
            d = float(world.path_to_tasks[r, j])
            ok = (
                math.isfinite(d)
                and d / config.robot_speed + world.task_service_times[j] <= H + 1e-12
                and d * config.energy_per_distance <= B + 1e-12
            )
            xs[r, j] = var(hi=1.0 if ok else 0.0, integer=True)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(world.path_to_tasks[R + i, j])
                ok = (
                    math.isfinite(d)
                    and d / config.robot_speed + world.task_service_times[j] <= H + 1e-12
                    and d * config.energy_per_distance <= B + 1e-12
                )
                xa[r, i, j] = var(hi=1.0 if ok else 0.0, integer=True)

    for r in range(R):
        for j in range(N):
            y[r, j] = var(obj=-1.0 / N, hi=1.0, integer=True)
            t[r, j] = var(obj=1.0 / (H * N), hi=H)

    for r in range(R):
        con({int(xs[r, j]): 1.0 for j in range(N)}, hi=1.0)

    for r in range(R):
        for j in range(N):
            z = {int(y[r, j]): 1.0, int(xs[r, j]): -1.0}
            for i in range(N):
                if i != j:
                    z[int(xa[r, i, j])] = -1.0
            con(z, lo=0.0, hi=0.0)

    for r in range(R):
        for i in range(N):
            z = {int(y[r, i]): -1.0}
            for j in range(N):
                if i != j:
                    z[int(xa[r, i, j])] = 1.0
            con(z, hi=0.0)

    for j in range(N):
        con({int(y[r, j]): 1.0 for r in range(R)}, hi=1.0)

    for r in range(R):
        for j in range(N):
            con({int(t[r, j]): 1.0, int(y[r, j]): -H}, hi=0.0)

    finite = world.path_to_tasks[np.isfinite(world.path_to_tasks)]
    max_path = float(np.max(finite)) if finite.size else 0.0
    M = H + max_path / config.robot_speed + float(
        np.max(world.task_service_times)
    ) + 1.0

    for r in range(R):
        for j in range(N):
            d = float(world.path_to_tasks[r, j])
            if math.isfinite(d):
                duration = d / config.robot_speed + float(world.task_service_times[j])
                con(
                    {int(t[r, j]): 1.0, int(xs[r, j]): -M},
                    lo=duration - M,
                )

    for r in range(R):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(world.path_to_tasks[R + i, j])
                if math.isfinite(d):
                    duration = d / config.robot_speed + float(world.task_service_times[j])
                    con(
                        {
                            int(t[r, j]): 1.0,
                            int(t[r, i]): -1.0,
                            int(xa[r, i, j]): -M,
                        },
                        lo=duration - M,
                    )

    for r in range(R):
        z: dict[int, float] = {}
        for j in range(N):
            d = float(world.path_to_tasks[r, j])
            if math.isfinite(d):
                z[int(xs[r, j])] = d * config.energy_per_distance
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(world.path_to_tasks[R + i, j])
                if math.isfinite(d):
                    z[int(xa[r, i, j])] = d * config.energy_per_distance
        con(z, hi=float(world.robot_initial_batteries[r]))

    A = coo_matrix(
        (vals, (rows, cols)),
        shape=(len(clb), len(c)),
        dtype=np.float64,
    ).tocsr()
    start = perf_counter()
    res = milp(
        c=np.asarray(c),
        integrality=np.asarray(integ),
        bounds=Bounds(np.asarray(vlb), np.asarray(vub)),
        constraints=LinearConstraint(A, np.asarray(clb), np.asarray(cub)),
        options=_milp_options(
            time_limit=time_limit,
            solver_display=solver_display,
        ),
    )
    elapsed = perf_counter() - start

    optimal = int(res.status) == 0 and res.x is not None
    if time_limit is None and not optimal:
        raise RuntimeError(
            "Unlimited exact MILP terminated without an optimal proof: "
            f"status={int(res.status)} message={res.message}"
        )

    score = float(-res.fun) if res.fun is not None else None
    routes: list[tuple[int, ...]] = []
    completed = None

    if res.x is not None:
        sol = np.asarray(res.x)
        completed = int(round(sum(sol[int(y[r, j])] for r in range(R) for j in range(N))))
        for r in range(R):
            first = [j for j in range(N) if sol[int(xs[r, j])] > 0.5]
            if not first:
                routes.append(())
                continue
            route = [first[0]]
            while True:
                i = route[-1]
                nxt = [j for j in range(N) if j != i and sol[int(xa[r, i, j])] > 0.5]
                if not nxt:
                    break
                if nxt[0] in route:
                    raise RuntimeError("cycle in MILP route")
                route.append(nxt[0])
            routes.append(tuple(route))
    else:
        routes = [() for _ in range(R)]

    if optimal and score is not None:
        rebuilt = 0.0
        for r, route in enumerate(routes):
            now = 0.0
            node = r
            energy = 0.0
            for j in route:
                d = float(world.path_to_tasks[node, j])
                energy += d * config.energy_per_distance
                now += d / config.robot_speed + float(world.task_service_times[j])
                if now > H + 1e-7 or energy > world.robot_initial_batteries[r] + 1e-7:
                    raise RuntimeError("MILP route violates fixed environment")
                rebuilt += 1.0 - now / H
                node = R + j
        rebuilt /= N
        if abs(rebuilt - score) > 1e-6:
            raise RuntimeError(f"objective mismatch: solver={score}, rebuilt={rebuilt}")

    raw_dual_bound = getattr(
        res,
        "mip_dual_bound",
        None,
    )
    upper_bound = None
    if raw_dual_bound is not None:
        raw_dual_bound = float(
            raw_dual_bound
        )
        if math.isfinite(
            raw_dual_bound
        ):
            upper_bound = float(
                -raw_dual_bound
            )

    raw_node_count = getattr(
        res,
        "mip_node_count",
        None,
    )
    node_count = (
        int(raw_node_count)
        if raw_node_count is not None
        else None
    )

    return GlobalTimeOptimum(
        optimal=optimal,
        status=int(res.status),
        message=str(res.message),
        time_optimality=score,
        completed_tasks=completed,
        solve_seconds=float(elapsed),
        mip_gap=float(res.mip_gap) if getattr(res, "mip_gap", None) is not None else None,
        routes=tuple(routes),
        mip_node_count=node_count,
        time_optimality_upper_bound=upper_bound,
    )
