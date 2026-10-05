from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from marl2d.gene_mrta_v16t.env import EnvConfig, World
from marl2d.gene_mrta_v16t.global_optimal_core import _milp_options


EPS = 1e-12
OBJECTIVES = (
    "time",
    "path_efficiency",
    "priority_service",
    "deadline",
)


@dataclass(frozen=True)
class AllCompleteOptimum:
    objective: str
    optimal: bool
    status: int
    message: str
    score: float | None
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
            / max(
                config.diagonal,
                EPS,
            ),
            0.0,
            1.0,
        )
    )


def _route_scores(
    routes: tuple[
        tuple[int, ...],
        ...,
    ],
    world: World,
    config: EnvConfig,
) -> dict[str, float]:
    R = config.num_robots
    N = config.num_tasks
    H = config.episode_time
    total_priority = max(
        float(
            np.sum(
                world.task_priorities
            )
        ),
        EPS,
    )

    time_sum = 0.0
    priority_time_sum = 0.0
    path_sum = 0.0
    on_time = 0

    for r, route in enumerate(
        routes
    ):
        now = 0.0
        energy = 0.0
        node = r
        for task in route:
            distance = float(
                world.path_to_tasks[
                    node,
                    task,
                ]
            )
            if not math.isfinite(
                distance
            ):
                raise RuntimeError(
                    "Exact route contains non-finite path"
                )

            energy += (
                distance
                * config.energy_per_distance
            )
            now += (
                distance
                / config.robot_speed
                + float(
                    world.task_service_times[
                        task
                    ]
                )
            )
            if (
                now
                > H + 1e-7
                or energy
                > float(
                    world.robot_initial_batteries[
                        r
                    ]
                )
                + 1e-7
            ):
                raise RuntimeError(
                    "Exact route violates feasibility"
                )

            earliness = (
                1.0
                - float(
                    np.clip(
                        now
                        / max(
                            H,
                            EPS,
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
            time_sum += earliness
            priority_time_sum += (
                float(
                    world.task_priorities[
                        task
                    ]
                )
                * earliness
            )
            path_sum += (
                _path_reward(
                    distance,
                    config,
                )
            )
            if (
                now
                <= float(
                    world.task_deadlines[
                        task
                    ]
                )
                + EPS
            ):
                on_time += 1

            node = R + task

    completed = sum(
        len(route)
        for route in routes
    )
    if completed != N:
        raise RuntimeError(
            "All-complete oracle rebuilt an incomplete route"
        )

    return {
        "time": (
            time_sum
            / max(
                N,
                1,
            )
        ),
        "path_efficiency": (
            path_sum
            / max(
                N,
                1,
            )
        ),
        "priority_service": (
            priority_time_sum
            / total_priority
        ),
        "deadline": (
            float(on_time)
            / max(
                N,
                1,
            )
        ),
    }


def solve_all_complete_optimum(
    world: World,
    config: EnvConfig,
    *,
    objective: str,
    time_limit: float | None = 300.0,
    solver_display: bool = False,
) -> AllCompleteOptimum:
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unsupported objective: {objective}"
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
        EPS,
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
        index = len(c)
        c.append(float(obj))
        vlb.append(float(lo))
        vub.append(float(hi))
        integ.append(
            1 if integer else 0
        )
        return index

    def con(
        coeff: dict[int, float],
        lo: float = -math.inf,
        hi: float = math.inf,
    ) -> None:
        row = len(clb)
        for index, value in (
            coeff.items()
        ):
            rows.append(row)
            cols.append(index)
            vals.append(
                float(value)
            )
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
    ontime = (
        np.empty(
            (R, N),
            dtype=int,
        )
        if objective == "deadline"
        else None
    )

    for r in range(R):
        battery = float(
            world.robot_initial_batteries[
                r
            ]
        )
        for j in range(N):
            distance = float(
                world.path_to_tasks[
                    r,
                    j,
                ]
            )
            feasible = (
                math.isfinite(
                    distance
                )
                and (
                    distance
                    / config.robot_speed
                    + float(
                        world.task_service_times[
                            j
                        ]
                    )
                    <= H + EPS
                )
                and (
                    distance
                    * config.energy_per_distance
                    <= battery + EPS
                )
            )
            arc_obj = 0.0
            if (
                objective
                == "path_efficiency"
                and math.isfinite(
                    distance
                )
            ):
                arc_obj = (
                    -_path_reward(
                        distance,
                        config,
                    )
                    / max(
                        N,
                        1,
                    )
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
                distance = float(
                    world.path_to_tasks[
                        R + i,
                        j,
                    ]
                )
                feasible = (
                    math.isfinite(
                        distance
                    )
                    and (
                        distance
                        / config.robot_speed
                        + float(
                            world.task_service_times[
                                j
                            ]
                        )
                        <= H + EPS
                    )
                    and (
                        distance
                        * config.energy_per_distance
                        <= battery + EPS
                    )
                )
                arc_obj = 0.0
                if (
                    objective
                    == "path_efficiency"
                    and math.isfinite(
                        distance
                    )
                ):
                    arc_obj = (
                        -_path_reward(
                            distance,
                            config,
                        )
                        / max(
                            N,
                            1,
                        )
                    )
                xa[
                    r,
                    i,
                    j,
                ] = var(
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
            y[r, j] = var(
                hi=1.0,
                integer=True,
            )

            time_obj = 0.0
            if objective == "time":
                time_obj = (
                    1.0
                    / (
                        max(H, EPS)
                        * max(
                            N,
                            1,
                        )
                    )
                )
            elif (
                objective
                == "priority_service"
            ):
                time_obj = (
                    float(
                        world.task_priorities[
                            j
                        ]
                    )
                    / (
                        max(
                            H,
                            EPS,
                        )
                        * total_priority
                    )
                )

            t[r, j] = var(
                obj=time_obj,
                hi=H,
            )
            if ontime is not None:
                ontime[
                    r,
                    j,
                ] = var(
                    obj=(
                        -1.0
                        / max(
                            N,
                            1,
                        )
                    ),
                    hi=1.0,
                    integer=True,
                )

    # At most one route may leave each robot start.
    for r in range(R):
        con(
            {
                int(
                    xs[r, j]
                ): 1.0
                for j in range(N)
            },
            hi=1.0,
        )

    # Assigned task has exactly one incoming edge for its robot.
    for r in range(R):
        for j in range(N):
            coeff = {
                int(
                    y[r, j]
                ): 1.0,
                int(
                    xs[r, j]
                ): -1.0,
            }
            for i in range(N):
                if i != j:
                    coeff[
                        int(
                            xa[
                                r,
                                i,
                                j,
                            ]
                        )
                    ] = -1.0
            con(
                coeff,
                lo=0.0,
                hi=0.0,
            )

    # Assigned task has at most one outgoing edge.
    for r in range(R):
        for i in range(N):
            coeff = {
                int(
                    y[r, i]
                ): -1.0
            }
            for j in range(N):
                if i != j:
                    coeff[
                        int(
                            xa[
                                r,
                                i,
                                j,
                            ]
                        )
                    ] = 1.0
            con(
                coeff,
                hi=0.0,
            )

    # Feasibility-first protocol: every task MUST be assigned once.
    for j in range(N):
        con(
            {
                int(
                    y[r, j]
                ): 1.0
                for r in range(R)
            },
            lo=1.0,
            hi=1.0,
        )

    for r in range(R):
        for j in range(N):
            # Unassigned robot/task time is zero.
            con(
                {
                    int(
                        t[r, j]
                    ): 1.0,
                    int(
                        y[r, j]
                    ): -H,
                },
                hi=0.0,
            )

    finite = world.path_to_tasks[
        np.isfinite(
            world.path_to_tasks
        )
    ]
    max_path = (
        float(
            np.max(
                finite
            )
        )
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
            distance = float(
                world.path_to_tasks[
                    r,
                    j,
                ]
            )
            if math.isfinite(
                distance
            ):
                duration = (
                    distance
                    / config.robot_speed
                    + float(
                        world.task_service_times[
                            j
                        ]
                    )
                )
                con(
                    {
                        int(
                            t[r, j]
                        ): 1.0,
                        int(
                            xs[r, j]
                        ): -big_m,
                    },
                    lo=(
                        duration
                        - big_m
                    ),
                )

    for r in range(R):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                distance = float(
                    world.path_to_tasks[
                        R + i,
                        j,
                    ]
                )
                if math.isfinite(
                    distance
                ):
                    duration = (
                        distance
                        / config.robot_speed
                        + float(
                            world.task_service_times[
                                j
                            ]
                        )
                    )
                    con(
                        {
                            int(
                                t[r, j]
                            ): 1.0,
                            int(
                                t[r, i]
                            ): -1.0,
                            int(
                                xa[
                                    r,
                                    i,
                                    j,
                                ]
                            ): -big_m,
                        },
                        lo=(
                            duration
                            - big_m
                        ),
                    )

    if ontime is not None:
        for r in range(R):
            for j in range(N):
                # ontime can only be 1 for the robot that owns task j.
                con(
                    {
                        int(
                            ontime[
                                r,
                                j,
                            ]
                        ): 1.0,
                        int(
                            y[r, j]
                        ): -1.0,
                    },
                    hi=0.0,
                )
                # z=1 -> completion time <= deadline.
                con(
                    {
                        int(
                            t[r, j]
                        ): 1.0,
                        int(
                            ontime[
                                r,
                                j,
                            ]
                        ): big_m,
                    },
                    hi=(
                        float(
                            world.task_deadlines[
                                j
                            ]
                        )
                        + big_m
                    ),
                )

    for r in range(R):
        energy_coeff: dict[
            int,
            float,
        ] = {}
        for j in range(N):
            distance = float(
                world.path_to_tasks[
                    r,
                    j,
                ]
            )
            if math.isfinite(
                distance
            ):
                energy_coeff[
                    int(
                        xs[r, j]
                    )
                ] = (
                    distance
                    * config.energy_per_distance
                )

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                distance = float(
                    world.path_to_tasks[
                        R + i,
                        j,
                    ]
                )
                if math.isfinite(
                    distance
                ):
                    energy_coeff[
                        int(
                            xa[
                                r,
                                i,
                                j,
                            ]
                        )
                    ] = (
                        distance
                        * config.energy_per_distance
                    )

        con(
            energy_coeff,
            hi=float(
                world.robot_initial_batteries[
                    r
                ]
            ),
        )

    matrix = coo_matrix(
        (
            vals,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            len(clb),
            len(c),
        ),
        dtype=np.float64,
    ).tocsr()

    started = perf_counter()
    result = milp(
        c=np.asarray(
            c,
            dtype=np.float64,
        ),
        integrality=np.asarray(
            integ,
            dtype=np.int8,
        ),
        bounds=Bounds(
            np.asarray(
                vlb,
                dtype=np.float64,
            ),
            np.asarray(
                vub,
                dtype=np.float64,
            ),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(
                clb,
                dtype=np.float64,
            ),
            np.asarray(
                cub,
                dtype=np.float64,
            ),
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
        int(
            result.status
        )
        == 0
        and result.x is not None
    )
    if (
        time_limit is None
        and not optimal
    ):
        raise RuntimeError(
            "Unlimited all-complete MILP terminated without exact proof: "
            f"objective={objective} status={int(result.status)} "
            f"message={result.message}"
        )

    routes: list[
        tuple[int, ...]
    ] = []
    completed = None

    if result.x is not None:
        solution = np.asarray(
            result.x
        )
        completed = int(
            round(
                sum(
                    solution[
                        int(
                            y[r, j]
                        )
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
                if (
                    solution[
                        int(
                            xs[r, j]
                        )
                    ]
                    > 0.5
                )
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
                        and solution[
                            int(
                                xa[
                                    r,
                                    i,
                                    j,
                                ]
                            )
                        ]
                        > 0.5
                    )
                ]
                if not nxt:
                    break
                if nxt[0] in route:
                    raise RuntimeError(
                        "Cycle in exact all-complete route"
                    )
                route.append(
                    nxt[0]
                )
            routes.append(
                tuple(
                    route
                )
            )
    else:
        routes = [
            ()
            for _ in range(R)
        ]

    score = None
    if optimal:
        route_scores = (
            _route_scores(
                tuple(
                    routes
                ),
                world,
                config,
            )
        )
        score = float(
            route_scores[
                objective
            ]
        )

        if result.fun is not None:
            if objective in {
                "time",
                "priority_service",
            }:
                solver_score = (
                    1.0
                    - float(
                        result.fun
                    )
                )
            else:
                solver_score = (
                    -float(
                        result.fun
                    )
                )
            if abs(
                solver_score
                - score
            ) > 1e-6:
                raise RuntimeError(
                    f"{objective} objective mismatch: "
                    f"solver={solver_score} rebuilt={score}"
                )

    raw_nodes = getattr(
        result,
        "mip_node_count",
        None,
    )

    return AllCompleteOptimum(
        objective=objective,
        optimal=optimal,
        status=int(
            result.status
        ),
        message=str(
            result.message
        ),
        score=score,
        completed_tasks=completed,
        solve_seconds=float(
            elapsed
        ),
        mip_gap=(
            float(
                result.mip_gap
            )
            if getattr(
                result,
                "mip_gap",
                None,
            )
            is not None
            else None
        ),
        routes=tuple(
            routes
        ),
        mip_node_count=(
            int(
                raw_nodes
            )
            if raw_nodes
            is not None
            else None
        ),
    )
