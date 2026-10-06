from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .gene import ROBOT_DIM, TASK_DIM, SetAssignmentGene
from .suite import World, WorldConfig


AXES = ("total_time", "priority", "completed_tasks")
AXIS_DIRECTIONS = {
    "total_time": "min",
    "priority": "min",
    "completed_tasks": "max",
}


@dataclass(frozen=True)
class WorldMetrics:
    seed: int
    total_time: float
    priority: float
    completed_tasks: float
    total_tasks: int
    deadline_completion_rate: float
    all_tasks_completed: int
    total_distance: float

    def axes(self) -> dict[str, float]:
        return {
            "total_time": self.total_time,
            "priority": self.priority,
            "completed_tasks": self.completed_tasks,
        }

    def to_dict(self) -> dict[str, float | int]:
        return {
            "seed": self.seed,
            "total_time": self.total_time,
            "priority": self.priority,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "deadline_completion_rate": self.deadline_completion_rate,
            "all_tasks_completed": self.all_tasks_completed,
            "total_distance": self.total_distance,
        }


@dataclass(frozen=True)
class GeneEvaluation:
    total_time: float
    priority: float
    completed_tasks: float
    deadline_completion_rate: float
    all_tasks_completed: float
    total_distance: float
    per_seed: tuple[WorldMetrics, ...]

    def axes(self) -> dict[str, float]:
        return {
            "total_time": self.total_time,
            "priority": self.priority,
            "completed_tasks": self.completed_tasks,
        }


def _device_name(device: str) -> str:
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _unpack_gene_batch(
    genes: Sequence[SetAssignmentGene],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if not genes:
        raise ValueError("At least one Gene is required")
    hidden = genes[0].hidden_dim
    if any(g.hidden_dim != hidden for g in genes):
        raise ValueError("All genes in a batch must share hidden_dim")

    vectors = torch.as_tensor(
        np.stack([g.vector_data for g in genes], axis=0),
        dtype=torch.float32,
        device=device,
    )
    g = vectors.shape[0]
    offset = 0

    task_w = vectors[:, offset : offset + TASK_DIM * hidden].reshape(
        g, TASK_DIM, hidden
    )
    offset += TASK_DIM * hidden
    task_b = vectors[:, offset : offset + hidden]
    offset += hidden

    robot_w = vectors[:, offset : offset + ROBOT_DIM * hidden].reshape(
        g, ROBOT_DIM, hidden
    )
    offset += ROBOT_DIM * hidden
    robot_b = vectors[:, offset : offset + hidden]
    offset += hidden

    decoder_dim = SetAssignmentGene.decoder_dim(hidden)
    decoder_w = vectors[:, offset : offset + decoder_dim]
    offset += decoder_dim
    decoder_b = vectors[:, offset]

    return task_w, task_b, robot_w, robot_b, decoder_w, decoder_b


def _normalized_world_tensors(
    world: World,
    config: WorldConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    time_scale = max(float(world.baseline_makespan), 1.0)
    distance_scale = max(
        time_scale * float(config.robot_speed),
        config.diagonal,
        1.0,
    )
    priority_scale = max(float(np.mean(world.task_priorities)), 1.0)

    task = world.task_features().astype(np.float32, copy=True)
    task[:, 0:2] /= float(config.world_size)
    task[:, 2] /= priority_scale
    task[:, 3] /= time_scale
    task[:, 4] /= time_scale

    robot = world.initial_robot_features().astype(np.float32, copy=True)
    robot[:, 0:2] /= float(config.world_size)
    robot[:, 2] /= distance_scale
    robot[:, 3] /= time_scale

    return (
        torch.as_tensor(task, dtype=torch.float32, device=device),
        torch.as_tensor(robot, dtype=torch.float32, device=device),
        torch.as_tensor(
            world.task_positions,
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            world.task_priorities,
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            world.task_deadlines,
            dtype=torch.float32,
            device=device,
        ),
    )


def rollout_gene_batch(
    genes: Sequence[SetAssignmentGene],
    world: World,
    config: WorldConfig | None = None,
    device: str = "auto",
) -> list[WorldMetrics]:
    config = config or WorldConfig()
    device_obj = torch.device(_device_name(device))
    gene_count = len(genes)
    if gene_count == 0:
        return []

    task_input, _robot_input0, task_positions, priorities, deadlines = (
        _normalized_world_tensors(world, config, device_obj)
    )
    service_times = torch.as_tensor(
        world.task_service_times,
        dtype=torch.float32,
        device=device_obj,
    )
    initial_robot_positions = torch.as_tensor(
        world.robot_positions,
        dtype=torch.float32,
        device=device_obj,
    )

    task_w, task_b, robot_w, robot_b, decoder_w, decoder_b = (
        _unpack_gene_batch(genes, device_obj)
    )
    hidden = genes[0].hidden_dim
    r_count = world.robot_count
    t_count = world.task_count

    task_h = torch.tanh(
        torch.einsum("nf,gfh->gnh", task_input, task_w)
        + task_b[:, None, :]
    )

    robot_positions = initial_robot_positions[None, :, :].expand(
        gene_count, -1, -1
    ).clone()
    accumulated_distance = torch.zeros(
        (gene_count, r_count),
        dtype=torch.float32,
        device=device_obj,
    )
    finish_time = torch.zeros_like(accumulated_distance)
    available = torch.ones(
        (gene_count, t_count),
        dtype=torch.bool,
        device=device_obj,
    )
    completion_times = torch.zeros(
        (gene_count, t_count),
        dtype=torch.float32,
        device=device_obj,
    )

    time_scale = max(float(world.baseline_makespan), 1.0)
    distance_scale = max(
        time_scale * float(config.robot_speed),
        config.diagonal,
        1.0,
    )
    gene_ids = torch.arange(gene_count, device=device_obj)

    for step in range(t_count):
        robot_input = torch.empty(
            (gene_count, r_count, ROBOT_DIM),
            dtype=torch.float32,
            device=device_obj,
        )
        robot_input[..., 0:2] = (
            robot_positions / float(config.world_size)
        )
        robot_input[..., 2] = (
            accumulated_distance / distance_scale
        )
        robot_input[..., 3] = finish_time / time_scale
        robot_h = torch.tanh(
            torch.einsum(
                "grf,gfh->grh",
                robot_input,
                robot_w,
            )
            + robot_b[:, None, :]
        )

        mask_f = available.to(torch.float32)
        task_mean = (
            torch.sum(task_h * mask_f[..., None], dim=1)
            / torch.clamp(
                torch.sum(mask_f, dim=1, keepdim=True),
                min=1.0,
            )
        )
        neg_inf = torch.full_like(task_h, float("-inf"))
        task_max = torch.where(
            available[..., None],
            task_h,
            neg_inf,
        ).max(dim=1).values
        robot_mean = robot_h.mean(dim=1)
        robot_max = robot_h.max(dim=1).values

        pair_distance = torch.linalg.vector_norm(
            robot_positions[:, :, None, :]
            - task_positions[None, None, :, :],
            dim=-1,
        )
        distance_norm = pair_distance / float(config.diagonal)
        progress = float(step) / max(t_count - 1, 1)

        o = 0
        w_robot_local = decoder_w[:, o : o + hidden]
        o += hidden
        w_task_local = decoder_w[:, o : o + hidden]
        o += hidden
        w_task_mean = decoder_w[:, o : o + hidden]
        o += hidden
        w_task_max = decoder_w[:, o : o + hidden]
        o += hidden
        w_robot_mean = decoder_w[:, o : o + hidden]
        o += hidden
        w_robot_max = decoder_w[:, o : o + hidden]
        o += hidden
        w_distance = decoder_w[:, o]
        o += 1
        w_progress = decoder_w[:, o]

        score = (
            torch.einsum(
                "grh,gh->gr",
                robot_h,
                w_robot_local,
            )[:, :, None]
            + torch.einsum(
                "gth,gh->gt",
                task_h,
                w_task_local,
            )[:, None, :]
            + torch.einsum(
                "gh,gh->g",
                task_mean,
                w_task_mean,
            )[:, None, None]
            + torch.einsum(
                "gh,gh->g",
                task_max,
                w_task_max,
            )[:, None, None]
            + torch.einsum(
                "gh,gh->g",
                robot_mean,
                w_robot_mean,
            )[:, None, None]
            + torch.einsum(
                "gh,gh->g",
                robot_max,
                w_robot_max,
            )[:, None, None]
            + distance_norm * w_distance[:, None, None]
            + progress * w_progress[:, None, None]
            + decoder_b[:, None, None]
        )
        score = torch.where(
            available[:, None, :],
            score,
            float("-inf"),
        )
        flat = torch.argmax(
            score.reshape(gene_count, -1),
            dim=1,
        )
        chosen_robot = flat // t_count
        chosen_task = flat % t_count

        selected_distance = pair_distance[
            gene_ids,
            chosen_robot,
            chosen_task,
        ]
        selected_service = service_times[chosen_task]
        new_finish = (
            finish_time[gene_ids, chosen_robot]
            + selected_distance / float(config.robot_speed)
            + selected_service
        )
        finish_time[gene_ids, chosen_robot] = new_finish
        accumulated_distance[
            gene_ids,
            chosen_robot,
        ] += selected_distance
        robot_positions[
            gene_ids,
            chosen_robot,
        ] = task_positions[chosen_task]
        completion_times[
            gene_ids,
            chosen_task,
        ] = new_finish
        available[
            gene_ids,
            chosen_task,
        ] = False

    total_time = finish_time.max(dim=1).values
    total_distance = accumulated_distance.sum(dim=1)

    order = torch.argsort(completion_times, dim=1)
    ranks = torch.empty_like(completion_times)
    rank_values = torch.arange(
        t_count,
        dtype=torch.float32,
        device=device_obj,
    )
    ranks.scatter_(
        1,
        order,
        rank_values[None, :].expand(gene_count, -1),
    )
    priority_cost = (
        torch.sum(
            ranks * priorities[None, :],
            dim=1,
        )
        / torch.clamp(
            torch.sum(priorities),
            min=1e-6,
        )
    )

    deadline_active = deadlines > 0.0
    on_time = (
        ~deadline_active[None, :]
    ) | (
        completion_times
        <= deadlines[None, :] + 1e-6
    )
    completed_tasks = (
        on_time.to(torch.float32).sum(dim=1)
    )
    deadline_rate = completed_tasks / float(t_count)

    arrays = [
        total_time.detach().cpu().numpy(),
        priority_cost.detach().cpu().numpy(),
        completed_tasks.detach().cpu().numpy(),
        deadline_rate.detach().cpu().numpy(),
        total_distance.detach().cpu().numpy(),
    ]

    return [
        WorldMetrics(
            seed=world.seed,
            total_time=float(arrays[0][i]),
            priority=float(arrays[1][i]),
            completed_tasks=float(arrays[2][i]),
            total_tasks=t_count,
            deadline_completion_rate=float(arrays[3][i]),
            all_tasks_completed=t_count,
            total_distance=float(arrays[4][i]),
        )
        for i in range(gene_count)
    ]


def evaluate_population(
    genes: Sequence[SetAssignmentGene],
    worlds: Sequence[World],
    config: WorldConfig | None = None,
    device: str = "auto",
    gene_batch_size: int = 32,
) -> list[GeneEvaluation]:
    if not genes:
        return []
    if not worlds:
        raise ValueError("At least one world is required")
    config = config or WorldConfig()

    per_gene: list[list[WorldMetrics]] = [
        [] for _ in genes
    ]
    for world in worlds:
        for start in range(
            0,
            len(genes),
            gene_batch_size,
        ):
            stop = min(
                start + gene_batch_size,
                len(genes),
            )
            rows = rollout_gene_batch(
                genes[start:stop],
                world,
                config=config,
                device=device,
            )
            for local, row in enumerate(rows):
                per_gene[start + local].append(row)

    out: list[GeneEvaluation] = []
    for rows in per_gene:
        out.append(
            GeneEvaluation(
                total_time=float(
                    np.mean(
                        [x.total_time for x in rows]
                    )
                ),
                priority=float(
                    np.mean(
                        [x.priority for x in rows]
                    )
                ),
                completed_tasks=float(
                    np.mean(
                        [x.completed_tasks for x in rows]
                    )
                ),
                deadline_completion_rate=float(
                    np.mean(
                        [
                            x.deadline_completion_rate
                            for x in rows
                        ]
                    )
                ),
                all_tasks_completed=float(
                    np.mean(
                        [x.all_tasks_completed for x in rows]
                    )
                ),
                total_distance=float(
                    np.mean(
                        [x.total_distance for x in rows]
                    )
                ),
                per_seed=tuple(rows),
            )
        )
    return out
