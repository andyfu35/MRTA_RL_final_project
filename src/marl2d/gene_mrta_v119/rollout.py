from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from .benchmark import (
    MTRPDInstance,
)


EPS = 1e-12


@dataclass(frozen=True)
class MTRPDStep:
    step: int
    robot: int
    task: int
    from_node: int
    to_node: int
    segment_distance: float
    arrival_time: float
    used_distance: float
    return_distance: float

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "step": self.step,
            "robot": self.robot,
            "task": self.task,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "segment_distance": (
                self.segment_distance
            ),
            "arrival_time": (
                self.arrival_time
            ),
            "used_distance": (
                self.used_distance
            ),
            "return_distance": (
                self.return_distance
            ),
        }


@dataclass(frozen=True)
class MTRPDRollout:
    success: bool
    completion: float
    completed_tasks: int
    task_count: int
    total_latency: float
    route_lengths: tuple[
        float,
        ...,
    ]
    routes: tuple[
        tuple[
            int,
            ...,
        ],
        ...,
    ]
    selection_sequence: tuple[
        MTRPDStep,
        ...,
    ]
    stopped_by_policy: bool
    remaining_tasks: tuple[
        int,
        ...,
    ]

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "success": self.success,
            "completion": (
                self.completion
            ),
            "completed_tasks": (
                self.completed_tasks
            ),
            "task_count": (
                self.task_count
            ),
            "total_latency": (
                self.total_latency
            ),
            "route_lengths": list(
                self.route_lengths
            ),
            "routes": [
                list(
                    route
                )
                for route in self.routes
            ],
            "selection_sequence": [
                step.to_dict()
                for step in (
                    self.selection_sequence
                )
            ],
            "stopped_by_policy": (
                self.stopped_by_policy
            ),
            "remaining_tasks": list(
                self.remaining_tasks
            ),
        }


def _eligible_matrix(
    instance: MTRPDInstance,
    tail_nodes: np.ndarray,
    used_distance: np.ndarray,
    task_available: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    R = instance.robot_count
    T = instance.task_count

    task_nodes = np.arange(
        1,
        T + 1,
        dtype=np.int64,
    )
    segment = (
        instance.distance_matrix[
            tail_nodes[:, None],
            task_nodes[None, :],
        ]
    )
    return_to_depot = (
        instance.distance_matrix[
            task_nodes,
            0,
        ][
            None,
            :
        ]
    )
    projected_used = (
        used_distance[
            :,
            None,
        ]
        + segment
    )
    route_if_stop = (
        projected_used
        + return_to_depot
    )
    eligible = (
        task_available[
            None,
            :
        ]
        & np.isfinite(
            segment
        )
        & (
            route_if_stop
            <= instance.route_limit
            + EPS
        )
    )
    return (
        eligible,
        segment,
        projected_used,
        return_to_depot,
    )


def _build_observations(
    instance: MTRPDInstance,
    *,
    tail_nodes: np.ndarray,
    used_distance: np.ndarray,
    task_available: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    R = instance.robot_count
    T = instance.task_count

    (
        eligible,
        segment,
        projected_used,
        return_to_depot,
    ) = _eligible_matrix(
        instance,
        tail_nodes,
        used_distance,
        task_available,
    )

    scale = max(
        float(
            instance.route_limit
        ),
        1.0,
    )
    distance_norm = np.clip(
        segment / scale,
        0.0,
        1.0,
    )
    budget_remaining = np.clip(
        (
            instance.route_limit
            - used_distance
        )
        / scale,
        0.0,
        1.0,
    )
    workload = np.clip(
        used_distance
        / scale,
        0.0,
        1.0,
    )

    # Candidate arrival at a customer equals cumulative travel distance
    # because MTRPD has zero service time.
    arrivals = projected_used

    competition = np.zeros(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )
    for task in range(T):
        owners = np.flatnonzero(
            eligible[
                :,
                task
            ]
        )
        denom = max(
            len(
                owners
            )
            - 1,
            1,
        )
        for robot in owners:
            competition[
                robot,
                task
            ] = (
                np.sum(
                    arrivals[
                        owners,
                        task
                    ]
                    < arrivals[
                        robot,
                        task
                    ]
                )
                / denom
            )

    future_reachability = np.zeros(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )
    future_best_time = np.zeros(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )
    residual_budget = np.zeros(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )

    task_nodes = np.arange(
        1,
        T + 1,
        dtype=np.int64,
    )
    available_count = int(
        np.sum(
            task_available
        )
    )

    for robot in range(R):
        for task in range(T):
            if not eligible[
                robot,
                task
            ]:
                continue

            node = task + 1
            used_after = float(
                projected_used[
                    robot,
                    task
                ]
            )
            residual_budget[
                robot,
                task
            ] = np.clip(
                (
                    instance.route_limit
                    - used_after
                    - float(
                        return_to_depot[
                            0,
                            task
                        ]
                    )
                )
                / scale,
                0.0,
                1.0,
            )

            remaining = (
                task_available.copy()
            )
            remaining[
                task
            ] = False
            denominator = max(
                available_count - 1,
                1,
            )
            next_segment = (
                instance.distance_matrix[
                    node,
                    task_nodes,
                ]
            )
            next_return = (
                instance.distance_matrix[
                    task_nodes,
                    0,
                ]
            )
            next_used = (
                used_after
                + next_segment
            )
            feasible_after = (
                remaining
                & (
                    next_used
                    + next_return
                    <= instance.route_limit
                    + EPS
                )
            )
            future_reachability[
                robot,
                task
            ] = (
                float(
                    np.sum(
                        feasible_after
                    )
                )
                / float(
                    denominator
                )
            )
            if np.any(
                feasible_after
            ):
                best_arrival = float(
                    np.min(
                        next_used[
                            feasible_after
                        ]
                    )
                )
                future_best_time[
                    robot,
                    task
                ] = float(
                    np.clip(
                        1.0
                        - best_arrival
                        / scale,
                        0.0,
                        1.0,
                    )
                )

    # Opportunity cost: how attractive this task is to another feasible
    # robot, scarcity-weighted by that robot's remaining options.
    utility = np.where(
        eligible,
        1.0
        - np.clip(
            arrivals
            / scale,
            0.0,
            1.0,
        ),
        0.0,
    )
    option_count = np.sum(
        eligible,
        axis=1,
    )
    scarcity = (
        utility
        / np.maximum(
            option_count[
                :,
                None,
            ],
            1,
        )
    )
    opportunity_cost = np.zeros(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )
    for owner in range(R):
        other = np.ones(
            R,
            dtype=bool,
        )
        other[
            owner
        ] = False
        if np.any(
            other
        ):
            opportunity_cost[
                owner
            ] = np.max(
                scarcity[
                    other
                ],
                axis=0,
            )

    zeros = np.zeros(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )
    ones = np.ones(
        (
            R,
            T,
        ),
        dtype=np.float64,
    )

    observations = np.stack(
        [
            distance_norm,  # Euclidean distance
            distance_norm,  # benchmark path distance
            zeros,  # service time = 0
            zeros,  # equal task priority
            ones,  # no deadline pressure
            np.broadcast_to(
                budget_remaining[
                    :,
                    None,
                ],
                (
                    R,
                    T,
                ),
            ),
            np.broadcast_to(
                workload[
                    :,
                    None,
                ],
                (
                    R,
                    T,
                ),
            ),
            competition,
            np.clip(
                future_reachability,
                0.0,
                1.0,
            ),
            np.clip(
                future_best_time,
                0.0,
                1.0,
            ),
            np.clip(
                opportunity_cost,
                0.0,
                1.0,
            ),
            np.clip(
                residual_budget,
                0.0,
                1.0,
            ),
        ],
        axis=-1,
    )

    return (
        observations,
        eligible,
        segment,
    )


def rollout_gene(
    gene: RouteTailDirectGene,
    instance: MTRPDInstance,
) -> MTRPDRollout:
    instance.validate()

    R = instance.robot_count
    T = instance.task_count

    tail_nodes = np.zeros(
        R,
        dtype=np.int64,
    )
    used_distance = np.zeros(
        R,
        dtype=np.float64,
    )
    task_available = np.ones(
        T,
        dtype=bool,
    )
    row_open = np.ones(
        R,
        dtype=bool,
    )

    routes: list[
        list[int]
    ] = [
        []
        for _ in range(R)
    ]
    sequence: list[
        MTRPDStep
    ] = []
    total_latency = 0.0
    stopped_by_policy = False

    for step in range(T):
        if not np.any(
            task_available
        ):
            break

        (
            observations,
            eligible,
            segment,
        ) = _build_observations(
            instance,
            tail_nodes=tail_nodes,
            used_distance=(
                used_distance
            ),
            task_available=(
                task_available
            ),
        )

        if not np.any(
            eligible
        ):
            break

        logits, stop_logit = (
            gene.action_logits(
                observations,
                eligible,
                row_open,
                task_available,
                step,
            )
        )
        flat = int(
            np.argmax(
                logits
            )
        )
        best = float(
            logits.flat[
                flat
            ]
        )
        if not math.isfinite(
            best
        ):
            break
        if (
            stop_logit
            >= best
        ):
            stopped_by_policy = True
            break

        robot = (
            flat // T
        )
        task = (
            flat % T
        )
        if not eligible[
            robot,
            task
        ]:
            raise RuntimeError(
                "MTRPD decoder selected infeasible pair"
            )

        from_node = int(
            tail_nodes[
                robot
            ]
        )
        to_node = (
            task + 1
        )
        distance = float(
            segment[
                robot,
                task
            ]
        )
        used = float(
            used_distance[
                robot
            ]
            + distance
        )
        return_distance = float(
            instance.distance_matrix[
                to_node,
                0,
            ]
        )
        if (
            used
            + return_distance
            > instance.route_limit
            + EPS
        ):
            raise RuntimeError(
                "MTRPD route-limit invariant violated"
            )

        total_latency += used
        sequence.append(
            MTRPDStep(
                step=step,
                robot=robot,
                task=task,
                from_node=from_node,
                to_node=to_node,
                segment_distance=(
                    distance
                ),
                arrival_time=used,
                used_distance=used,
                return_distance=(
                    return_distance
                ),
            )
        )
        routes[
            robot
        ].append(
            task
        )
        task_available[
            task
        ] = False
        tail_nodes[
            robot
        ] = to_node
        used_distance[
            robot
        ] = used

    route_lengths = np.asarray(
        used_distance,
        dtype=np.float64,
    ).copy()
    for robot in range(R):
        route_lengths[
            robot
        ] += float(
            instance.distance_matrix[
                int(
                    tail_nodes[
                        robot
                    ]
                ),
                0,
            ]
        )

    completed = int(
        len(
            sequence
        )
    )
    success = bool(
        completed == T
        and np.all(
            route_lengths
            <= instance.route_limit
            + EPS
        )
    )
    completion = float(
        completed
        / max(
            T,
            1,
        )
    )

    return MTRPDRollout(
        success=success,
        completion=completion,
        completed_tasks=completed,
        task_count=T,
        total_latency=float(
            total_latency
        ),
        route_lengths=tuple(
            float(
                value
            )
            for value in route_lengths
        ),
        routes=tuple(
            tuple(
                route
            )
            for route in routes
        ),
        selection_sequence=tuple(
            sequence
        ),
        stopped_by_policy=(
            stopped_by_policy
        ),
        remaining_tasks=tuple(
            int(
                task
            )
            for task in np.flatnonzero(
                task_available
            )
        ),
    )
