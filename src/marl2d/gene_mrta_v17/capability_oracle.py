from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from marl2d.gene_mrta_v16t.env import EnvConfig, World


EXACT_AXES = (
    "completion",
    "efficiency",
    "priority_satisfaction",
    "deadline_satisfaction",
    "time_optimality",
)


@dataclass(frozen=True)
class CapabilityOracleResult:
    axis: str
    optimal: bool
    status: int
    message: str
    value: float | None
    solve_seconds: float
    mip_gap: float | None
    routes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class WorldCapabilityCeilings:
    completion: float
    efficiency: float
    priority_satisfaction: float
    deadline_satisfaction: float
    balance_upper_bound: float
    time_optimality: float
    exact_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "completion": self.completion,
            "efficiency": self.efficiency,
            "priority_satisfaction": self.priority_satisfaction,
            "deadline_satisfaction": self.deadline_satisfaction,
            "balance": self.balance_upper_bound,
            "time_optimality": self.time_optimality,
            "exact_axes": list(self.exact_axes),
            "balance_semantics": (
                "analytical upper bound B<=completion_star because "
                "historical balance=completion*Jain(workload), Jain<=1"
            ),
        }


class _Builder:
    def __init__(self) -> None:
        self.c: list[float] = []
        self.vlb: list[float] = []
        self.vub: list[float] = []
        self.integrality: list[int] = []
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.clb: list[float] = []
        self.cub: list[float] = []

    def var(
        self,
        *,
        objective: float = 0.0,
        lower: float = 0.0,
        upper: float = math.inf,
        integer: bool = False,
    ) -> int:
        idx = len(self.c)
        self.c.append(float(objective))
        self.vlb.append(float(lower))
        self.vub.append(float(upper))
        self.integrality.append(1 if integer else 0)
        return idx

    def con(
        self,
        coeffs: dict[int, float],
        *,
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row = len(self.clb)
        for col, value in coeffs.items():
            if value == 0.0:
                continue
            self.rows.append(row)
            self.cols.append(col)
            self.vals.append(float(value))
        self.clb.append(float(lower))
        self.cub.append(float(upper))

    def matrices(self):
        matrix = coo_matrix(
            (self.vals, (self.rows, self.cols)),
            shape=(len(self.clb), len(self.c)),
            dtype=np.float64,
        ).tocsr()
        return (
            np.asarray(self.c, dtype=np.float64),
            np.asarray(self.integrality, dtype=np.int32),
            Bounds(
                np.asarray(self.vlb, dtype=np.float64),
                np.asarray(self.vub, dtype=np.float64),
            ),
            LinearConstraint(
                matrix,
                np.asarray(self.clb, dtype=np.float64),
                np.asarray(self.cub, dtype=np.float64),
            ),
        )


def _route_value(
    routes: tuple[tuple[int, ...], ...],
    world: World,
    config: EnvConfig,
    axis: str,
) -> float:
    R = config.num_robots
    N = config.num_tasks
    H = config.episode_time
    total_priority = float(np.sum(world.task_priorities))
    completed = 0
    priority = 0.0
    on_time = 0
    efficiency = 0.0
    time_score = 0.0

    for r, route in enumerate(routes):
        now = 0.0
        node = r
        energy = 0.0
        for task in route:
            d = float(world.path_to_tasks[node, task])
            energy += d * config.energy_per_distance
            now += d / config.robot_speed + float(world.task_service_times[task])
            if now > H + 1e-6:
                raise RuntimeError("oracle route violates episode horizon")
            if energy > world.robot_initial_batteries[r] + 1e-6:
                raise RuntimeError("oracle route violates battery")
            completed += 1
            priority += float(world.task_priorities[task])
            on_time += int(now <= world.task_deadlines[task] + 1e-9)
            efficiency += 1.0 - np.clip(d / config.diagonal, 0.0, 1.0)
            time_score += 1.0 - now / H
            node = R + task

    values = {
        "completion": completed / N,
        "efficiency": efficiency / N,
        "priority_satisfaction": priority / max(total_priority, 1e-12),
        "deadline_satisfaction": on_time / N,
        "time_optimality": time_score / N,
    }
    return float(values[axis])


def solve_capability_oracle(
    world: World,
    config: EnvConfig,
    axis: str,
    *,
    time_limit: float = 300.0,
) -> CapabilityOracleResult:
    if axis not in EXACT_AXES:
        raise ValueError(f"unsupported exact oracle axis: {axis}")

    R = config.num_robots
    N = config.num_tasks
    H = float(config.episode_time)
    b = _Builder()

    xs = np.empty((R, N), dtype=np.int64)
    xa = np.full((R, N, N), -1, dtype=np.int64)
    y = np.empty((R, N), dtype=np.int64)
    finish = np.empty((R, N), dtype=np.int64)
    on_time = (
        np.empty((R, N), dtype=np.int64)
        if axis == "deadline_satisfaction"
        else None
    )

    total_priority = max(float(np.sum(world.task_priorities)), 1e-12)

    for r in range(R):
        battery = float(world.robot_initial_batteries[r])
        for j in range(N):
            d = float(world.path_to_tasks[r, j])
            duration = d / config.robot_speed + float(world.task_service_times[j])
            energy = d * config.energy_per_distance
            feasible = (
                math.isfinite(d)
                and duration <= H + 1e-12
                and energy <= battery + 1e-12
            )
            reward = 0.0
            if axis == "efficiency" and math.isfinite(d):
                reward = (
                    1.0 - float(np.clip(d / config.diagonal, 0.0, 1.0))
                ) / N
            xs[r, j] = b.var(
                objective=-reward,
                upper=1.0 if feasible else 0.0,
                integer=True,
            )

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(world.path_to_tasks[R + i, j])
                duration = d / config.robot_speed + float(world.task_service_times[j])
                energy = d * config.energy_per_distance
                feasible = (
                    math.isfinite(d)
                    and duration <= H + 1e-12
                    and energy <= battery + 1e-12
                )
                reward = 0.0
                if axis == "efficiency" and math.isfinite(d):
                    reward = (
                        1.0 - float(np.clip(d / config.diagonal, 0.0, 1.0))
                    ) / N
                xa[r, i, j] = b.var(
                    objective=-reward,
                    upper=1.0 if feasible else 0.0,
                    integer=True,
                )

    for r in range(R):
        for j in range(N):
            objective = 0.0
            if axis == "completion":
                objective = -1.0 / N
            elif axis == "priority_satisfaction":
                objective = -float(world.task_priorities[j]) / total_priority
            elif axis == "time_optimality":
                objective = -1.0 / N
            y[r, j] = b.var(
                objective=objective,
                upper=1.0,
                integer=True,
            )
            finish_obj = (
                1.0 / (H * N)
                if axis == "time_optimality"
                else 0.0
            )
            finish[r, j] = b.var(
                objective=finish_obj,
                upper=H,
            )
            if on_time is not None:
                on_time[r, j] = b.var(
                    objective=-1.0 / N,
                    upper=1.0,
                    integer=True,
                )

    for r in range(R):
        b.con(
            {int(xs[r, j]): 1.0 for j in range(N)},
            upper=1.0,
        )

    for r in range(R):
        for j in range(N):
            coeffs = {
                int(y[r, j]): 1.0,
                int(xs[r, j]): -1.0,
            }
            for i in range(N):
                if i != j:
                    coeffs[int(xa[r, i, j])] = -1.0
            b.con(coeffs, lower=0.0, upper=0.0)

    for r in range(R):
        for i in range(N):
            coeffs = {int(y[r, i]): -1.0}
            for j in range(N):
                if i != j:
                    coeffs[int(xa[r, i, j])] = 1.0
            b.con(coeffs, upper=0.0)

    for j in range(N):
        b.con(
            {int(y[r, j]): 1.0 for r in range(R)},
            upper=1.0,
        )

    for r in range(R):
        for j in range(N):
            b.con(
                {
                    int(finish[r, j]): 1.0,
                    int(y[r, j]): -H,
                },
                upper=0.0,
            )

    finite = world.path_to_tasks[np.isfinite(world.path_to_tasks)]
    max_path = float(np.max(finite)) if finite.size else 0.0
    max_service = float(np.max(world.task_service_times))
    big_m = H + max_path / config.robot_speed + max_service + 1.0

    for r in range(R):
        for j in range(N):
            d = float(world.path_to_tasks[r, j])
            if math.isfinite(d):
                duration = d / config.robot_speed + float(
                    world.task_service_times[j]
                )
                b.con(
                    {
                        int(finish[r, j]): 1.0,
                        int(xs[r, j]): -big_m,
                    },
                    lower=duration - big_m,
                )

    for r in range(R):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(world.path_to_tasks[R + i, j])
                if math.isfinite(d):
                    duration = d / config.robot_speed + float(
                        world.task_service_times[j]
                    )
                    b.con(
                        {
                            int(finish[r, j]): 1.0,
                            int(finish[r, i]): -1.0,
                            int(xa[r, i, j]): -big_m,
                        },
                        lower=duration - big_m,
                    )

    for r in range(R):
        coeffs: dict[int, float] = {}
        for j in range(N):
            d = float(world.path_to_tasks[r, j])
            if math.isfinite(d):
                coeffs[int(xs[r, j])] = d * config.energy_per_distance
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(world.path_to_tasks[R + i, j])
                if math.isfinite(d):
                    coeffs[int(xa[r, i, j])] = d * config.energy_per_distance
        b.con(
            coeffs,
            upper=float(world.robot_initial_batteries[r]),
        )

    if on_time is not None:
        for r in range(R):
            for j in range(N):
                q = int(on_time[r, j])
                b.con(
                    {
                        q: 1.0,
                        int(y[r, j]): -1.0,
                    },
                    upper=0.0,
                )
                # q=1 -> finish <= deadline; q=0 relaxes by big-M.
                b.con(
                    {
                        int(finish[r, j]): 1.0,
                        q: big_m,
                    },
                    upper=float(world.task_deadlines[j]) + big_m,
                )

    c, integrality, bounds, constraints = b.matrices()
    start = perf_counter()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "disp": False,
            "presolve": True,
            "mip_rel_gap": 0.0,
            "time_limit": float(time_limit),
        },
    )
    elapsed = perf_counter() - start

    optimal = int(result.status) == 0 and result.x is not None
    routes: list[tuple[int, ...]] = []
    if result.x is not None:
        sol = np.asarray(result.x)
        for r in range(R):
            first = [j for j in range(N) if sol[int(xs[r, j])] > 0.5]
            if not first:
                routes.append(())
                continue
            route = [first[0]]
            while True:
                i = route[-1]
                nxt = [
                    j
                    for j in range(N)
                    if j != i and sol[int(xa[r, i, j])] > 0.5
                ]
                if not nxt:
                    break
                if nxt[0] in route:
                    raise RuntimeError("cycle in oracle route")
                route.append(nxt[0])
            routes.append(tuple(route))
    else:
        routes = [() for _ in range(R)]

    value = None
    if optimal:
        value = _route_value(
            tuple(routes),
            world,
            config,
            axis,
        )
        if result.fun is not None:
            solver_value = float(-result.fun)
            if abs(value - solver_value) > 1e-6:
                raise RuntimeError(
                    f"{axis} objective mismatch: solver={solver_value}, route={value}"
                )

    return CapabilityOracleResult(
        axis=axis,
        optimal=optimal,
        status=int(result.status),
        message=str(result.message),
        value=value,
        solve_seconds=float(elapsed),
        mip_gap=(
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        routes=tuple(routes),
    )


def solve_world_capability_ceilings(
    world: World,
    config: EnvConfig,
    *,
    time_limit: float = 300.0,
) -> tuple[WorldCapabilityCeilings | None, dict[str, CapabilityOracleResult]]:
    results = {
        axis: solve_capability_oracle(
            world,
            config,
            axis,
            time_limit=time_limit,
        )
        for axis in EXACT_AXES
    }
    if not all(result.optimal and result.value is not None for result in results.values()):
        return None, results

    completion_star = float(results["completion"].value)
    ceilings = WorldCapabilityCeilings(
        completion=completion_star,
        efficiency=float(results["efficiency"].value),
        priority_satisfaction=float(results["priority_satisfaction"].value),
        deadline_satisfaction=float(results["deadline_satisfaction"].value),
        balance_upper_bound=completion_star,
        time_optimality=float(results["time_optimality"].value),
        exact_axes=EXACT_AXES,
    )
    return ceilings, results
