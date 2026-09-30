from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np


SQRT2 = math.sqrt(2.0)
_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
)


def grid_shape(world_size: float, resolution: float) -> tuple[int, int]:
    if resolution <= 0.0:
        raise ValueError("grid resolution must be positive")
    n = max(1, int(math.ceil(world_size / resolution)))
    return n, n


def point_to_cell(
    point: np.ndarray,
    world_size: float,
    resolution: float,
) -> tuple[int, int]:
    rows, cols = grid_shape(world_size, resolution)
    x = float(np.clip(point[0], 0.0, np.nextafter(world_size, 0.0)))
    y = float(np.clip(point[1], 0.0, np.nextafter(world_size, 0.0)))
    col = min(cols - 1, max(0, int(x / resolution)))
    row = min(rows - 1, max(0, int(y / resolution)))
    return row, col


def cell_center(cell: tuple[int, int], resolution: float) -> np.ndarray:
    row, col = cell
    return np.array(
        [(col + 0.5) * resolution, (row + 0.5) * resolution],
        dtype=np.float64,
    )


def build_occupancy(
    world_size: float,
    resolution: float,
    obstacles: np.ndarray,
) -> np.ndarray:
    rows, cols = grid_shape(world_size, resolution)
    occupancy = np.zeros((rows, cols), dtype=bool)
    if obstacles.size == 0:
        return occupancy

    xs = (np.arange(cols, dtype=np.float64) + 0.5) * resolution
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)

    for x0, y0, x1, y1 in np.asarray(obstacles, dtype=np.float64):
        occupancy |= (
            (xx >= x0)
            & (xx <= x1)
            & (yy >= y0)
            & (yy <= y1)
        )
    return occupancy


def _diagonal_allowed(
    occupancy: np.ndarray,
    row: int,
    col: int,
    next_row: int,
    next_col: int,
) -> bool:
    if row == next_row or col == next_col:
        return True
    return (
        not occupancy[row, next_col]
        and not occupancy[next_row, col]
    )


def _octile(
    a: tuple[int, int],
    b: tuple[int, int],
    resolution: float,
) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diagonal = min(dr, dc)
    straight = max(dr, dc) - diagonal
    return resolution * (SQRT2 * diagonal + straight)


def astar_path_length_cells(
    occupancy: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    resolution: float,
) -> float:
    if start == goal:
        return 0.0
    if occupancy[start] or occupancy[goal]:
        return math.inf

    rows, cols = occupancy.shape
    best_g = np.full((rows, cols), np.inf, dtype=np.float64)
    best_g[start] = 0.0
    heap: list[tuple[float, float, int, int]] = [
        (_octile(start, goal, resolution), 0.0, start[0], start[1])
    ]

    while heap:
        _, g, row, col = heapq.heappop(heap)
        if g > best_g[row, col] + 1e-12:
            continue
        if (row, col) == goal:
            return float(g)

        for dr, dc, step_units in _NEIGHBORS:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if occupancy[nr, nc]:
                continue
            if not _diagonal_allowed(occupancy, row, col, nr, nc):
                continue

            ng = g + step_units * resolution
            if ng + 1e-12 >= best_g[nr, nc]:
                continue
            best_g[nr, nc] = ng
            h = _octile((nr, nc), goal, resolution)
            heapq.heappush(heap, (ng + h, ng, nr, nc))

    return math.inf


def astar_path_length(
    occupancy: np.ndarray,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    world_size: float,
    resolution: float,
) -> float:
    start = point_to_cell(start_point, world_size, resolution)
    goal = point_to_cell(goal_point, world_size, resolution)
    direct = float(np.linalg.norm(goal_point - start_point))
    if start == goal:
        return direct

    grid_cost = astar_path_length_cells(
        occupancy,
        start,
        goal,
        resolution,
    )
    if not math.isfinite(grid_cost):
        return math.inf

    free_grid_cost = _octile(start, goal, resolution)
    obstacle_detour = max(0.0, grid_cost - free_grid_cost)
    return direct + obstacle_detour


def points_are_connected(
    occupancy: np.ndarray,
    points: np.ndarray,
    world_size: float,
    resolution: float,
) -> bool:
    if len(points) <= 1:
        return True

    cells = [
        point_to_cell(point, world_size, resolution)
        for point in np.asarray(points, dtype=np.float64)
    ]
    if any(occupancy[cell] for cell in cells):
        return False

    start = cells[0]
    targets = set(cells[1:])
    visited = {start}
    queue = deque([start])
    rows, cols = occupancy.shape

    while queue and targets:
        row, col = queue.popleft()
        targets.discard((row, col))
        for dr, dc, _ in _NEIGHBORS:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            nxt = (nr, nc)
            if nxt in visited or occupancy[nxt]:
                continue
            if not _diagonal_allowed(occupancy, row, col, nr, nc):
                continue
            visited.add(nxt)
            queue.append(nxt)

    return not targets


def precompute_path_to_tasks(
    robot_positions: np.ndarray,
    task_positions: np.ndarray,
    obstacles: np.ndarray,
    world_size: float,
    resolution: float,
) -> np.ndarray:
    robot_positions = np.asarray(robot_positions, dtype=np.float64)
    task_positions = np.asarray(task_positions, dtype=np.float64)
    nodes = np.vstack([robot_positions, task_positions])
    robot_count = robot_positions.shape[0]
    task_count = task_positions.shape[0]
    occupancy = build_occupancy(world_size, resolution, obstacles)

    out = np.full((nodes.shape[0], task_count), np.inf, dtype=np.float64)
    cache: dict[tuple[int, int], float] = {}

    for source_idx in range(nodes.shape[0]):
        for task_idx in range(task_count):
            target_idx = robot_count + task_idx
            if source_idx == target_idx:
                out[source_idx, task_idx] = 0.0
                continue

            key = (
                min(source_idx, target_idx),
                max(source_idx, target_idx),
            )
            if key not in cache:
                cache[key] = astar_path_length(
                    occupancy,
                    nodes[source_idx],
                    nodes[target_idx],
                    world_size,
                    resolution,
                )
            out[source_idx, task_idx] = cache[key]

    return out
