from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch

from marl2d.gene_mrta_v120.benchmark import (
    MinMaxMTSPInstance,
    edge_distance,
)
from marl2d.gene_mrta_v120.gene import ScalableRouteTailGene
from marl2d.gene_mrta_v120.rollout import _geometry_scales


EPS = 1e-12
OBS_DIM = 12


@dataclass(frozen=True)
class BatchRolloutResult:
    objectives: np.ndarray
    route_lengths: np.ndarray
    chosen_robots: np.ndarray
    chosen_nodes: np.ndarray
    success: np.ndarray

    @property
    def world_count(self) -> int:
        return int(self.objectives.shape[0])

    def routes_for_world(self, world: int, robot_count: int) -> tuple[tuple[int, ...], ...]:
        routes: list[list[int]] = [[] for _ in range(robot_count)]
        for robot, node in zip(
            self.chosen_robots[world],
            self.chosen_nodes[world],
        ):
            routes[int(robot)].append(int(node))
        return tuple(tuple(route) for route in routes)


@dataclass
class MPSInstanceCache:
    instance: MinMaxMTSPInstance
    distance64: np.ndarray
    geometric: torch.Tensor
    edge: torch.Tensor
    neighbor_order: torch.Tensor
    edge_scale: float
    route_scale: float
    device: torch.device


def mps_available() -> bool:
    return bool(
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _distance_matrix(instance: MinMaxMTSPInstance) -> np.ndarray:
    coords = instance.coordinates
    a = coords[:, None, :]
    b = coords[None, :, :]
    return np.asarray(
        edge_distance(instance, a, b),
        dtype=np.float64,
    )


def _geometric_matrix(instance: MinMaxMTSPInstance) -> np.ndarray:
    coords = np.asarray(instance.coordinates, dtype=np.float64)
    delta = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def _neighbor_order(instance: MinMaxMTSPInstance) -> np.ndarray:
    """Match V1.20 cKDTree geometric candidate ordering, once per instance."""
    coords = np.asarray(instance.coordinates, dtype=np.float64)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=instance.vertex_count)
    indices = np.asarray(indices, dtype=np.int64)

    rows: list[np.ndarray] = []
    for row in indices:
        task_nodes = row[row > 0]
        if task_nodes.shape[0] != instance.task_count:
            raise RuntimeError(
                f"{instance.instance_id}: invalid precomputed neighbor order"
            )
        rows.append(task_nodes)
    return np.stack(rows, axis=0)


def prepare_instance(
    instance: MinMaxMTSPInstance,
    *,
    device: torch.device | str = "mps",
) -> MPSInstanceCache:
    device = torch.device(device)
    distance64 = _distance_matrix(instance)
    geometric64 = _geometric_matrix(instance)
    neighbor64 = _neighbor_order(instance)
    edge_scale, route_scale = _geometry_scales(instance)

    return MPSInstanceCache(
        instance=instance,
        distance64=distance64,
        geometric=torch.as_tensor(
            geometric64,
            dtype=torch.float32,
            device=device,
        ),
        edge=torch.as_tensor(
            distance64,
            dtype=torch.float32,
            device=device,
        ),
        neighbor_order=torch.as_tensor(
            neighbor64,
            dtype=torch.long,
            device=device,
        ),
        edge_scale=float(edge_scale),
        route_scale=float(route_scale),
        device=device,
    )


def _unpack_gene_batch(
    genes: Sequence[ScalableRouteTailGene],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not genes:
        raise ValueError("At least one Gene is required")
    hidden = int(genes[0].hidden_dim)
    if any(int(gene.hidden_dim) != hidden for gene in genes):
        raise ValueError("All Genes in a batch must use the same hidden_dim")

    vectors = torch.as_tensor(
        np.stack(
            [
                np.asarray(gene.vector_data, dtype=np.float32)
                for gene in genes
            ],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    batch = int(vectors.shape[0])
    offset = 0

    w_pair = vectors[
        :, offset : offset + OBS_DIM * hidden
    ].reshape(batch, OBS_DIM, hidden)
    offset += OBS_DIM * hidden

    b_pair = vectors[:, offset : offset + hidden]
    offset += hidden

    w_decode = vectors[:, offset : offset + 4 * hidden + 1]
    offset += 4 * hidden + 1

    b_decode = vectors[:, offset]
    return w_pair, b_pair, w_decode, b_decode


def _candidate_nodes(
    cache: MPSInstanceCache,
    tail_nodes: torch.Tensor,
    remaining: torch.Tensor,
    *,
    remaining_count: int,
    candidate_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return sorted candidate node IDs plus a validity mask.

    Semantics match V1.20:
    - when few tasks remain, use all remaining tasks;
    - otherwise union the nearest K remaining tasks for every current route
      tail, then sort the union by node ID.
    """
    B, R = tail_nodes.shape
    N = cache.instance.vertex_count
    T = cache.instance.task_count
    device = cache.device

    if remaining_count <= candidate_k * max(2, R):
        selected = remaining
        cmax = remaining_count
    else:
        ranking = cache.neighbor_order[tail_nodes]  # [B,R,T]
        rem_expanded = remaining[:, None, :].expand(B, R, N)
        ranked_remaining = torch.gather(
            rem_expanded,
            2,
            ranking,
        )

        rank_positions = torch.arange(
            T,
            dtype=torch.float32,
            device=device,
        ).view(1, 1, T)
        masked_positions = torch.where(
            ranked_remaining,
            rank_positions,
            torch.full_like(rank_positions, float(T + 1)),
        )
        nearest_positions = torch.topk(
            masked_positions,
            k=candidate_k,
            dim=2,
            largest=False,
            sorted=False,
        ).indices
        nearest_nodes = torch.gather(
            ranking,
            2,
            nearest_positions,
        )

        selected_i = torch.zeros(
            (B, N),
            dtype=torch.int32,
            device=device,
        )
        selected_i.scatter_(
            1,
            nearest_nodes.reshape(B, -1),
            1,
        )
        selected = selected_i > 0
        cmax = min(remaining_count, R * candidate_k)

    node_values = torch.arange(
        N,
        dtype=torch.float32,
        device=device,
    ).view(1, N).expand(B, N)
    keys = torch.where(
        selected,
        node_values,
        torch.full_like(node_values, float(N)),
    )
    values = torch.topk(
        keys,
        k=cmax,
        dim=1,
        largest=False,
        sorted=True,
    ).values
    valid = values < float(N)
    candidates = torch.where(
        valid,
        values,
        torch.zeros_like(values),
    ).to(torch.long)
    return candidates, valid


def _pair_lookup(
    flat_matrix: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    n: int,
) -> torch.Tensor:
    index = left * n + right
    return flat_matrix[index]


def _official_objectives(
    cache: MPSInstanceCache,
    chosen_robots: np.ndarray,
    chosen_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    B, T = chosen_nodes.shape
    R = cache.instance.robot_count
    rows = np.arange(B)

    tails = np.zeros((B, R), dtype=np.int64)
    lengths = np.zeros((B, R), dtype=np.float64)

    for step in range(T):
        robots = chosen_robots[:, step]
        nodes = chosen_nodes[:, step]
        previous = tails[rows, robots]
        segment = cache.distance64[previous, nodes]
        lengths[rows, robots] += segment
        tails[rows, robots] = nodes

    for robot in range(R):
        lengths[:, robot] += cache.distance64[tails[:, robot], 0]

    objectives = np.max(lengths, axis=1)
    return objectives, lengths


@torch.inference_mode()
def rollout_gene_batch_mps(
    genes: Sequence[ScalableRouteTailGene],
    cache: MPSInstanceCache,
    *,
    candidate_k: int = 32,
) -> BatchRolloutResult:
    if not genes:
        raise ValueError("At least one Gene is required")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    instance = cache.instance
    B = len(genes)
    R = instance.robot_count
    N = instance.vertex_count
    T = instance.task_count
    device = cache.device

    w_pair, b_pair, w_decode, b_decode = _unpack_gene_batch(
        genes,
        device=device,
    )
    hidden = int(w_pair.shape[-1])

    tail_nodes = torch.zeros((B, R), dtype=torch.long, device=device)
    route_lengths = torch.zeros((B, R), dtype=torch.float32, device=device)
    used_robot = torch.zeros((B, R), dtype=torch.bool, device=device)
    remaining = torch.ones((B, N), dtype=torch.bool, device=device)
    remaining[:, 0] = False

    chosen_robots = torch.empty((B, T), dtype=torch.long, device=device)
    chosen_nodes = torch.empty((B, T), dtype=torch.long, device=device)

    geometric_flat = cache.geometric.reshape(-1)
    edge_flat = cache.edge.reshape(-1)
    edge_scale = max(float(cache.edge_scale), EPS)
    route_scale = max(float(cache.route_scale), EPS)

    for step in range(T):
        remaining_count = T - step
        candidates, candidate_valid = _candidate_nodes(
            cache,
            tail_nodes,
            remaining,
            remaining_count=remaining_count,
            candidate_k=candidate_k,
        )
        C = int(candidates.shape[1])

        tail_grid = tail_nodes[:, :, None].expand(B, R, C)
        cand_grid = candidates[:, None, :].expand(B, R, C)

        geometric = _pair_lookup(
            geometric_flat,
            tail_grid,
            cand_grid,
            N,
        )
        benchmark_distance = _pair_lookup(
            edge_flat,
            tail_grid,
            cand_grid,
            N,
        )

        projected = route_lengths[:, :, None] + benchmark_distance
        return_distance = _pair_lookup(
            edge_flat,
            candidates,
            torch.zeros_like(candidates),
            N,
        )
        projected_with_return = projected + return_distance[:, None, :]

        geometric_norm = torch.clamp(
            geometric / edge_scale,
            0.0,
            1.0,
        )
        edge_norm = torch.clamp(
            benchmark_distance / edge_scale,
            0.0,
            1.0,
        )
        current_work = torch.clamp(
            route_lengths / route_scale,
            0.0,
            1.0,
        )
        work_remaining = torch.clamp(
            1.0 - current_work,
            0.0,
            1.0,
        )

        if R <= 1:
            competition = torch.zeros_like(projected_with_return)
            opportunity_cost = torch.zeros_like(projected_with_return)
        else:
            comparisons = (
                projected_with_return[:, :, None, :]
                < projected_with_return[:, None, :, :] - EPS
            )
            competition = (
                comparisons.sum(dim=1, dtype=torch.float32)
                / float(R - 1)
            )

            utility = 1.0 - torch.clamp(
                projected_with_return / route_scale,
                0.0,
                1.0,
            )
            top_values, top_indices = torch.topk(
                utility,
                k=2,
                dim=1,
                largest=True,
                sorted=True,
            )
            robot_ids = torch.arange(
                R,
                device=device,
                dtype=torch.long,
            ).view(1, R, 1)
            owner = top_indices[:, 0, :][:, None, :]
            opportunity_cost = torch.where(
                owner == robot_ids,
                top_values[:, 1, :][:, None, :],
                top_values[:, 0, :][:, None, :],
            )

        projected_utility = torch.clamp(
            1.0 - projected / route_scale,
            0.0,
            1.0,
        )
        residual_route = torch.clamp(
            1.0 - projected_with_return / route_scale,
            0.0,
            1.0,
        )

        zero = torch.zeros_like(projected)
        one = torch.ones_like(projected)
        observations = torch.stack(
            [
                geometric_norm,
                edge_norm,
                zero,
                zero,
                one,
                work_remaining[:, :, None].expand(B, R, C),
                current_work[:, :, None].expand(B, R, C),
                competition,
                one,
                projected_utility,
                opportunity_cost,
                residual_route,
            ],
            dim=-1,
        )

        if step < R:
            row_open = ~used_robot
        else:
            row_open = torch.ones(
                (B, R),
                dtype=torch.bool,
                device=device,
            )
        active = (
            row_open[:, :, None]
            & candidate_valid[:, None, :]
        )

        pair_h = torch.tanh(
            torch.einsum("brco,boh->brch", observations, w_pair)
            + b_pair[:, None, None, :]
        )
        mask_f = active.to(torch.float32)
        masked_h = pair_h * mask_f[:, :, :, None]

        global_h = (
            masked_h.sum(dim=(1, 2))
            / torch.clamp(
                mask_f.sum(dim=(1, 2))[:, None],
                min=1.0,
            )
        )
        robot_h = (
            masked_h.sum(dim=2)
            / torch.clamp(
                mask_f.sum(dim=2)[:, :, None],
                min=1.0,
            )
        )
        task_h = (
            masked_h.sum(dim=1)
            / torch.clamp(
                mask_f.sum(dim=1)[:, :, None],
                min=1.0,
            )
        )

        h = hidden
        logits = (
            (pair_h * w_decode[:, None, None, 0:h]).sum(dim=-1)
            + (
                global_h[:, None, None, :]
                * w_decode[:, None, None, h : 2 * h]
            ).sum(dim=-1)
            + (
                robot_h[:, :, None, :]
                * w_decode[:, None, None, 2 * h : 3 * h]
            ).sum(dim=-1)
            + (
                task_h[:, None, :, :]
                * w_decode[:, None, None, 3 * h : 4 * h]
            ).sum(dim=-1)
            + (float(step) / max(1, T))
            * w_decode[:, None, None, 4 * h]
            + b_decode[:, None, None]
        )
        logits = logits.masked_fill(~active, float("-inf"))

        flat = torch.argmax(logits.reshape(B, -1), dim=1)
        robot = torch.div(flat, C, rounding_mode="floor")
        col = flat % C
        node = torch.gather(
            candidates,
            1,
            col[:, None],
        ).squeeze(1)

        chosen_robots[:, step] = robot
        chosen_nodes[:, step] = node

        current_tail = torch.gather(
            tail_nodes,
            1,
            robot[:, None],
        ).squeeze(1)
        segment = _pair_lookup(
            edge_flat,
            current_tail,
            node,
            N,
        )
        route_lengths.scatter_add_(
            1,
            robot[:, None],
            segment[:, None],
        )
        tail_nodes.scatter_(
            1,
            robot[:, None],
            node[:, None],
        )
        remaining.scatter_(
            1,
            node[:, None],
            False,
        )
        used_robot.scatter_(
            1,
            robot[:, None],
            True,
        )

    # One synchronization per full instance batch rather than per decoder step.
    chosen_robots_np = chosen_robots.cpu().numpy()
    chosen_nodes_np = chosen_nodes.cpu().numpy()

    objectives, official_lengths = _official_objectives(
        cache,
        chosen_robots_np,
        chosen_nodes_np,
    )
    success = np.ones(B, dtype=bool)

    return BatchRolloutResult(
        objectives=objectives,
        route_lengths=official_lengths,
        chosen_robots=chosen_robots_np,
        chosen_nodes=chosen_nodes_np,
        success=success,
    )
