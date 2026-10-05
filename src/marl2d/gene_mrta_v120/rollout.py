from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .benchmark import (
    MinMaxMTSPInstance,
    edge_distance,
)
from .gene import (
    ScalableRouteTailGene,
)


EPS = 1e-12


@dataclass(frozen=True)
class MinMaxRollout:
    success: bool
    completion: float
    completed_tasks: int
    task_count: int
    objective: float
    route_lengths: tuple[float, ...]
    routes: tuple[tuple[int, ...], ...]
    candidate_k: int

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "completion": self.completion,
            "completed_tasks": self.completed_tasks,
            "task_count": self.task_count,
            "objective": self.objective,
            "route_lengths": list(self.route_lengths),
            "routes": [list(route) for route in self.routes],
            "candidate_k": self.candidate_k,
        }


def _geometry_scales(
    instance: MinMaxMTSPInstance,
) -> tuple[float, float]:
    coords = instance.coordinates
    span = np.ptp(coords, axis=0)
    euclidean_scale = max(
        float(np.linalg.norm(span)),
        1.0,
    )

    if instance.edge_weight_type == "ATT":
        edge_scale = max(
            euclidean_scale / math.sqrt(10.0) + 1.0,
            1.0,
        )
    else:
        edge_scale = euclidean_scale

    expected_tasks = max(
        2,
        int(math.ceil(instance.task_count / instance.robot_count)),
    )
    route_scale = max(
        edge_scale * expected_tasks,
        edge_scale,
        1.0,
    )
    return edge_scale, route_scale


def _nearest_remaining(
    tree: cKDTree,
    coordinates: np.ndarray,
    tail_node: int,
    remaining: np.ndarray,
    *,
    count: int,
) -> list[int]:
    total = int(coordinates.shape[0])
    remaining_count = int(np.sum(remaining))
    if remaining_count <= count:
        return [
            int(node)
            for node in np.flatnonzero(remaining)
        ]

    query_k = min(
        total,
        max(
            count + 1,
            count * 2,
        ),
    )
    while True:
        _, indices = tree.query(
            coordinates[tail_node],
            k=query_k,
        )
        values = np.atleast_1d(indices)
        selected: list[int] = []
        for raw in values:
            node = int(raw)
            if (
                node > 0
                and remaining[node]
            ):
                selected.append(node)
                if len(selected) >= count:
                    return selected

        if query_k >= total:
            break
        query_k = min(
            total,
            query_k * 2,
        )

    return [
        int(node)
        for node in np.flatnonzero(remaining)
    ][:count]


def _candidate_nodes(
    instance: MinMaxMTSPInstance,
    tree: cKDTree,
    tail_nodes: np.ndarray,
    remaining: np.ndarray,
    *,
    candidate_k: int,
) -> np.ndarray:
    remaining_nodes = np.flatnonzero(remaining)
    if (
        len(remaining_nodes)
        <= candidate_k * max(2, instance.robot_count)
    ):
        return remaining_nodes.astype(np.int64)

    union: set[int] = set()
    for tail in np.unique(tail_nodes):
        union.update(
            _nearest_remaining(
                tree,
                instance.coordinates,
                int(tail),
                remaining,
                count=candidate_k,
            )
        )

    if not union:
        raise RuntimeError(
            f"{instance.instance_id}: no candidate task while tasks remain"
        )
    return np.asarray(
        sorted(union),
        dtype=np.int64,
    )


def _observations(
    instance: MinMaxMTSPInstance,
    *,
    tail_nodes: np.ndarray,
    route_lengths: np.ndarray,
    candidate_nodes: np.ndarray,
    used_robot: np.ndarray,
    edge_scale: float,
    route_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    R = instance.robot_count
    C = int(candidate_nodes.shape[0])

    tail_positions = instance.coordinates[tail_nodes]
    candidate_positions = instance.coordinates[candidate_nodes]

    delta = (
        tail_positions[:, None, :]
        - candidate_positions[None, :, :]
    )
    geometric = np.linalg.norm(
        delta,
        axis=-1,
    )
    benchmark_distance = edge_distance(
        instance,
        tail_positions[:, None, :],
        candidate_positions[None, :, :],
    )

    projected = (
        route_lengths[:, None]
        + benchmark_distance
    )
    return_distance = edge_distance(
        instance,
        candidate_positions,
        np.broadcast_to(
            instance.depot,
            candidate_positions.shape,
        ),
    )
    projected_with_return = (
        projected
        + return_distance[None, :]
    )

    geometric_norm = np.clip(
        geometric / max(edge_scale, EPS),
        0.0,
        1.0,
    )
    edge_norm = np.clip(
        benchmark_distance / max(edge_scale, EPS),
        0.0,
        1.0,
    )

    current_work = np.clip(
        route_lengths / max(route_scale, EPS),
        0.0,
        1.0,
    )
    work_remaining = np.clip(
        1.0 - current_work,
        0.0,
        1.0,
    )

    competition = np.zeros(
        (R, C),
        dtype=np.float64,
    )
    for col in range(C):
        values = projected_with_return[:, col]
        for robot in range(R):
            competition[robot, col] = float(
                np.sum(values < values[robot] - EPS)
                / max(R - 1, 1)
            )

    utility = (
        1.0
        - np.clip(
            projected_with_return
            / max(route_scale, EPS),
            0.0,
            1.0,
        )
    )
    opportunity_cost = np.zeros(
        (R, C),
        dtype=np.float64,
    )
    for robot in range(R):
        if R <= 1:
            continue
        other = np.ones(R, dtype=bool)
        other[robot] = False
        opportunity_cost[robot] = np.max(
            utility[other],
            axis=0,
        )

    zeros = np.zeros(
        (R, C),
        dtype=np.float64,
    )
    ones = np.ones(
        (R, C),
        dtype=np.float64,
    )
    projected_utility = np.clip(
        1.0
        - projected / max(route_scale, EPS),
        0.0,
        1.0,
    )
    residual_route = np.clip(
        1.0
        - projected_with_return / max(route_scale, EPS),
        0.0,
        1.0,
    )

    observations = np.stack(
        [
            geometric_norm,
            edge_norm,
            zeros,  # service time: none
            zeros,  # priority: all cities equal
            ones,   # deadline: none
            np.broadcast_to(
                work_remaining[:, None],
                (R, C),
            ),
            np.broadcast_to(
                current_work[:, None],
                (R, C),
            ),
            competition,
            ones,  # every remaining city is reachable in complete graph
            projected_utility,
            opportunity_cost,
            residual_route,
        ],
        axis=-1,
    )

    # Minmax-mTSP requires every salesman to visit at least one city.
    # Until all robots have a first city, only unused robots are legal rows.
    if np.any(~used_robot):
        row_open = ~used_robot
    else:
        row_open = np.ones(R, dtype=bool)

    return observations, row_open


def rollout_gene(
    gene: ScalableRouteTailGene,
    instance: MinMaxMTSPInstance,
    *,
    candidate_k: int = 32,
) -> MinMaxRollout:
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    R = instance.robot_count
    T = instance.task_count
    if T < R:
        raise ValueError(
            f"{instance.instance_id}: fewer tasks than robots"
        )

    tree = cKDTree(instance.coordinates)

    remaining = np.zeros(
        instance.vertex_count,
        dtype=bool,
    )
    remaining[1:] = True

    tail_nodes = np.zeros(
        R,
        dtype=np.int64,
    )
    route_lengths = np.zeros(
        R,
        dtype=np.float64,
    )
    used_robot = np.zeros(
        R,
        dtype=bool,
    )
    routes: list[list[int]] = [
        []
        for _ in range(R)
    ]

    edge_scale, route_scale = _geometry_scales(instance)

    completed = 0
    for step in range(T):
        candidate_nodes = _candidate_nodes(
            instance,
            tree,
            tail_nodes,
            remaining,
            candidate_k=candidate_k,
        )
        if candidate_nodes.size == 0:
            break

        observations, row_open = _observations(
            instance,
            tail_nodes=tail_nodes,
            route_lengths=route_lengths,
            candidate_nodes=candidate_nodes,
            used_robot=used_robot,
            edge_scale=edge_scale,
            route_scale=route_scale,
        )

        C = int(candidate_nodes.shape[0])
        eligible = np.broadcast_to(
            row_open[:, None],
            (R, C),
        ).copy()
        col_open = np.ones(
            C,
            dtype=bool,
        )

        logits, _stop_logit = gene.action_logits_total(
            observations,
            eligible,
            row_open,
            col_open,
            step=step,
            total_tasks=T,
        )
        flat = int(np.argmax(logits))
        best = float(logits.flat[flat])
        if not math.isfinite(best):
            break

        robot = flat // C
        col = flat % C
        node = int(candidate_nodes[col])

        segment = float(
            edge_distance(
                instance,
                instance.coordinates[tail_nodes[robot]],
                instance.coordinates[node],
            )
        )
        route_lengths[robot] += segment
        tail_nodes[robot] = node
        remaining[node] = False
        used_robot[robot] = True
        routes[robot].append(node)
        completed += 1

    # Close every route at the common depot.
    for robot in range(R):
        if used_robot[robot]:
            route_lengths[robot] += float(
                edge_distance(
                    instance,
                    instance.coordinates[tail_nodes[robot]],
                    instance.depot,
                )
            )

    success = bool(
        completed == T
        and np.all(used_robot)
        and not np.any(remaining[1:])
    )
    objective = (
        float(np.max(route_lengths))
        if success
        else float("inf")
    )

    return MinMaxRollout(
        success=success,
        completion=float(completed / max(T, 1)),
        completed_tasks=int(completed),
        task_count=int(T),
        objective=objective,
        route_lengths=tuple(
            float(value)
            for value in route_lengths
        ),
        routes=tuple(
            tuple(route)
            for route in routes
        ),
        candidate_k=int(candidate_k),
    )
