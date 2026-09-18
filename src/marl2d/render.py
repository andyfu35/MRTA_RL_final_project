from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import AGENT_IDS
from .env import VectorArena2D
from .exchange import PolicySet
from .policy import ActorCritic, load_model_snapshot
from .two_runner import TWO_RUNNER_IDS
from .two_runner_env import TwoRunnerArena2D


def _world_to_pixel(x: float, y: float, width: float, height: float, image_size: tuple[int, int]) -> tuple[int, int]:
    w, h = image_size
    return int(round(x / width * w)), int(round(h - y / height * h))


def _radius_to_pixel(radius: float, width: float, image_width: int) -> int:
    return max(2, int(round(radius / width * image_width)))


def _rotated_rectangle(center_x: float, center_y: float, length: float, width: float, theta: float) -> list[tuple[float, float]]:
    half_l, half_w = length * 0.5, width * 0.5
    c, s = math.cos(theta), math.sin(theta)
    points: list[tuple[float, float]] = []
    for local_x, local_y in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
        points.append((center_x + c * local_x - s * local_y, center_y + s * local_x + c * local_y))
    return points


def _robot_geometry(x: float, y: float, theta: float, body_length: float, body_width: float, wheel_length: float, wheel_width: float) -> dict[str, Any]:
    c, s = math.cos(theta), math.sin(theta)
    lateral = body_width * 0.5 + wheel_width * 0.55
    left_center = (x - s * lateral, y + c * lateral)
    right_center = (x + s * lateral, y - c * lateral)
    return {
        "body": _rotated_rectangle(x, y, body_length, body_width, theta),
        "wheels": [
            _rotated_rectangle(left_center[0], left_center[1], wheel_length, wheel_width, theta),
            _rotated_rectangle(right_center[0], right_center[1], wheel_length, wheel_width, theta),
        ],
        "heading_tip": (x + c * body_length * 0.5, y + s * body_length * 0.5),
    }


def _world_polygon_to_pixels(points: list[tuple[float, float]], width: float, height: float, image_size: tuple[int, int]) -> list[tuple[int, int]]:
    return [_world_to_pixel(px, py, width, height, image_size) for px, py in points]


def _draw_frame(env: VectorArena2D, step: int, status: str) -> Image.Image:
    image_width = 900
    image_height = max(500, int(round(image_width * env.height / env.width)))
    image_size = (image_width, image_height)
    image = Image.new("RGB", image_size, "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, image_size[0] - 1, image_size[1] - 1), outline="#111827", width=4)

    gx, gy = _world_to_pixel(float(env.goal[0]), float(env.goal[1]), env.width, env.height, image_size)
    gr = _radius_to_pixel(env.goal_radius, env.width, image_size[0])
    draw.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill="#dcfce7", outline="#15803d", width=3)
    draw.text((gx - 15, gy - 6), "GOAL", fill="#14532d", font=font)

    for ox, oy, obstacle_width, obstacle_height in env.obstacles[0]:
        p0 = _world_to_pixel(float(ox - obstacle_width * 0.5), float(oy - obstacle_height * 0.5), env.width, env.height, image_size)
        p1 = _world_to_pixel(float(ox + obstacle_width * 0.5), float(oy + obstacle_height * 0.5), env.width, env.height, image_size)
        left, right = sorted((p0[0], p1[0]))
        top, bottom = sorted((p0[1], p1[1]))
        draw.rectangle((left, top, right, bottom), fill="#94a3b8", outline="#334155", width=2)

    colors = ("#2563eb", "#60a5fa", "#dc2626", "#fb7185")
    labels = ("R0", "R1", "B0", "B1")
    for i in range(4):
        x, y, theta = (float(v) for v in env.state[0, i])
        geometry = _robot_geometry(x, y, theta, env.body_length, env.body_width, env.wheel_length, env.wheel_width)
        for wheel in geometry["wheels"]:
            draw.polygon(_world_polygon_to_pixels(wheel, env.width, env.height, image_size), fill="#111827", outline="#030712")
        body_pixels = _world_polygon_to_pixels(geometry["body"], env.width, env.height, image_size)
        draw.polygon(body_pixels, fill=colors[i], outline="#111827")
        draw.line(body_pixels + [body_pixels[0]], fill="#111827", width=2)
        px, py = _world_to_pixel(x, y, env.width, env.height, image_size)
        hx, hy = _world_to_pixel(float(geometry["heading_tip"][0]), float(geometry["heading_tip"][1]), env.width, env.height, image_size)
        draw.line((px, py, hx, hy), fill="#ffffff", width=3)
        draw.ellipse((hx - 3, hy - 3, hx + 3, hy + 3), fill="#ffffff", outline="#111827")
        draw.text((px - 8, py - 24), labels[i], fill="#111827", font=font)

    draw.rectangle((8, 8, 270, 36), fill="#ffffff", outline="#cbd5e1")
    draw.text((14, 15), f"step={step}   {status}   obstacles={env.obstacle_count}", fill="#111827", font=font)
    return image


def render_policy_set_gif(cfg: dict[str, Any], policy_set: PolicySet, output_path: str | Path, max_steps: int | None = None, device: str = "cpu") -> dict[str, int | bool]:
    torch_device = torch.device(device)
    models: dict[str, ActorCritic] = {}
    for agent_id in AGENT_IDS:
        model = ActorCritic(21, 2, cfg["ppo"]["hidden_sizes"]).to(torch_device)
        load_model_snapshot(model, policy_set[agent_id])
        model.eval()
        models[agent_id] = model

    env = VectorArena2D(1, cfg["environment"], cfg["reward"], seed=int(cfg.get("seed", 0)) + 999)
    obs = env.reset(seed=int(cfg.get("seed", 0)) + 999)
    limit = int(max_steps if max_steps is not None else cfg["environment"]["max_steps"])
    frames = [_draw_frame(env, 0, "start")]
    goal_reached = False
    timed_out = False
    steps_taken = 0

    for step in range(1, limit + 1):
        actions = np.zeros((1, 4, 2), dtype=np.float32)
        for idx, agent_id in enumerate(AGENT_IDS):
            obs_tensor = torch.from_numpy(obs[:, idx, :]).to(torch_device)
            with torch.no_grad():
                action = models[agent_id].deterministic(obs_tensor)
            actions[:, idx, :] = action.cpu().numpy()
        obs, _rewards, done, info = env.step(actions)
        goal_reached = bool(info["goal_reached"][0])
        timed_out = bool(info["timeout"][0])
        status = "goal" if goal_reached else ("timeout" if timed_out else "running")
        frames.append(_draw_frame(env, step, status))
        steps_taken = step
        if bool(done[0]):
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=max(40, int(float(cfg["environment"]["dt"]) * 1000)), loop=0, optimize=False)
    return {"frames": len(frames), "steps": steps_taken, "goal_reached": goal_reached, "timeout": timed_out}


def _load_two_runner_models(
    cfg: dict[str, Any],
    policy_set: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, ActorCritic]:
    models: dict[str, ActorCritic] = {}
    for agent_id in TWO_RUNNER_IDS:
        model = ActorCritic(18, 2, cfg["ppo"]["hidden_sizes"]).to(device)
        load_model_snapshot(model, policy_set[agent_id])
        model.eval()
        models[agent_id] = model
    return models


def _run_two_runner_episode(
    cfg: dict[str, Any],
    models: dict[str, ActorCritic],
    *,
    seed: int,
    device: torch.device,
    max_steps: int | None = None,
    capture: bool = False,
):
    env = TwoRunnerArena2D(1, cfg["environment"], cfg["two_runner_reward"], seed=int(seed))
    obs = env.observe()
    limit = int(max_steps if max_steps is not None else cfg["environment"]["max_steps"])
    any_collision = False
    frames_data = []
    if capture:
        frames_data.append(
            {
                "step": 0,
                "state": env.state[0].copy(),
                "alive": env.alive[0].copy(),
                "collision": env.collision[0].copy(),
                "status": "start",
            }
        )

    final_info = None
    for step in range(1, limit + 1):
        alive_before = env.alive[0].copy()
        actions = np.zeros((1, 2, 2), dtype=np.float32)
        for idx, agent_id in enumerate(TWO_RUNNER_IDS):
            if not alive_before[idx]:
                continue
            with torch.no_grad():
                action = models[agent_id].deterministic(
                    torch.from_numpy(obs[:, idx, :]).to(device)
                )
            actions[:, idx, :] = action.cpu().numpy()

        obs, _rewards, done, info = env.step(actions)
        any_collision = any_collision or bool(np.asarray(info["new_death"])[0].any())
        final_info = info
        if capture:
            if bool(info["team_success"][0]):
                status = "goal"
            elif bool(info["both_dead"][0]):
                status = "both_dead"
            elif bool(info["timeout"][0]):
                status = "timeout"
            else:
                status = "running"
            frames_data.append(
                {
                    "step": step,
                    "state": env.state[0].copy(),
                    "alive": env.alive[0].copy(),
                    "collision": env.collision[0].copy(),
                    "status": status,
                }
            )
        if bool(done[0]):
            break

    if final_info is None:
        raise RuntimeError("two-runner render episode produced no simulator step")

    summary = {
        "seed": int(seed),
        "steps": int(env.steps[0]),
        "team_success": bool(final_info["team_success"][0]),
        "any_collision": bool(any_collision),
        "both_dead": bool(final_info["both_dead"][0]),
        "timeout": bool(final_info["timeout"][0]),
        "goal_reached": [bool(x) for x in final_info["goal_reached"][0].tolist()],
    }
    return env, frames_data, summary


def find_two_runner_demo_seed(
    cfg: dict[str, Any],
    policy_set: dict[str, dict[str, torch.Tensor]],
    *,
    seed_start: int,
    search_episodes: int = 200,
    device: str = "cpu",
    max_steps: int | None = None,
) -> tuple[int, dict[str, Any]]:
    if int(search_episodes) <= 0:
        raise ValueError("search_episodes must be positive")
    torch_device = torch.device(device)
    models = _load_two_runner_models(cfg, policy_set, torch_device)
    first_success: tuple[int, dict[str, Any]] | None = None

    for offset in range(int(search_episodes)):
        seed = int(seed_start) + offset
        _env, _frames, summary = _run_two_runner_episode(
            cfg,
            models,
            seed=seed,
            device=torch_device,
            max_steps=max_steps,
            capture=False,
        )
        if summary["team_success"]:
            if first_success is None:
                first_success = (seed, summary)
            if not summary["any_collision"]:
                return seed, summary

    if first_success is not None:
        return first_success
    raise RuntimeError(
        f"No deterministic successful episode found in seeds "
        f"{int(seed_start)}..{int(seed_start) + int(search_episodes) - 1}"
    )


def _draw_two_runner_frame(
    env: TwoRunnerArena2D,
    cfg: dict[str, Any],
    frame_data: dict[str, Any],
    trails: list[list[tuple[float, float]]],
    *,
    seed: int,
) -> Image.Image:
    image_width = 800
    image_height = max(500, int(round(image_width * env.height / env.width)))
    image_size = (image_width, image_height)
    image = Image.new("RGB", image_size, "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, image_size[0] - 1, image_size[1] - 1), outline="#111827", width=4)

    gx, gy = _world_to_pixel(float(env.goal[0]), float(env.goal[1]), env.width, env.height, image_size)
    gr = _radius_to_pixel(env.goal_radius, env.width, image_size[0])
    draw.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill="#dcfce7", outline="#15803d", width=3)
    draw.text((gx - 15, gy - 6), "GOAL", fill="#14532d", font=font)

    for ox, oy, obstacle_width, obstacle_height in env.obstacles[0]:
        p0 = _world_to_pixel(
            float(ox - obstacle_width * 0.5),
            float(oy - obstacle_height * 0.5),
            env.width,
            env.height,
            image_size,
        )
        p1 = _world_to_pixel(
            float(ox + obstacle_width * 0.5),
            float(oy + obstacle_height * 0.5),
            env.width,
            env.height,
            image_size,
        )
        left, right = sorted((p0[0], p1[0]))
        top, bottom = sorted((p0[1], p1[1]))
        draw.rectangle((left, top, right, bottom), fill="#94a3b8", outline="#334155", width=2)

    trail_colors = ("#93c5fd", "#67e8f9")
    for idx, trail in enumerate(trails):
        if len(trail) >= 2:
            pixels = [
                _world_to_pixel(x, y, env.width, env.height, image_size)
                for x, y in trail
            ]
            draw.line(pixels, fill=trail_colors[idx], width=3)

    state = np.asarray(frame_data["state"], dtype=np.float32)
    alive = np.asarray(frame_data["alive"], dtype=bool)
    colors = ("#2563eb", "#0891b2")
    labels = ("R0", "R1")
    body_length = float(cfg["environment"].get("body_length", 0.70))
    body_width = float(cfg["environment"].get("body_width", 0.44))
    wheel_length = float(cfg["environment"].get("wheel_length", 0.34))
    wheel_width = float(cfg["environment"].get("wheel_width", 0.10))

    for idx in range(2):
        x, y, theta = (float(v) for v in state[idx])
        geometry = _robot_geometry(
            x, y, theta, body_length, body_width, wheel_length, wheel_width
        )
        for wheel in geometry["wheels"]:
            draw.polygon(
                _world_polygon_to_pixels(wheel, env.width, env.height, image_size),
                fill="#111827",
                outline="#030712",
            )
        body_pixels = _world_polygon_to_pixels(
            geometry["body"], env.width, env.height, image_size
        )
        fill = colors[idx] if alive[idx] else "#9ca3af"
        draw.polygon(body_pixels, fill=fill, outline="#111827")
        draw.line(body_pixels + [body_pixels[0]], fill="#111827", width=2)
        px, py = _world_to_pixel(x, y, env.width, env.height, image_size)
        hx, hy = _world_to_pixel(
            float(geometry["heading_tip"][0]),
            float(geometry["heading_tip"][1]),
            env.width,
            env.height,
            image_size,
        )
        draw.line((px, py, hx, hy), fill="#ffffff", width=3)
        draw.text(
            (px - 12, py - 24),
            f"{labels[idx]}{' DEAD' if not alive[idx] else ''}",
            fill="#111827",
            font=font,
        )

    status = str(frame_data["status"])
    step = int(frame_data["step"])
    draw.rectangle((8, 8, 350, 42), fill="#ffffff", outline="#cbd5e1")
    draw.text(
        (14, 15),
        f"seed={seed}  step={step}  status={status}",
        fill="#111827",
        font=font,
    )
    return image


def render_two_runner_gif(
    cfg: dict[str, Any],
    policy_set: dict[str, dict[str, torch.Tensor]],
    output_path: str | Path,
    *,
    seed: int,
    max_steps: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    torch_device = torch.device(device)
    models = _load_two_runner_models(cfg, policy_set, torch_device)
    env, frames_data, summary = _run_two_runner_episode(
        cfg,
        models,
        seed=int(seed),
        device=torch_device,
        max_steps=max_steps,
        capture=True,
    )

    trails: list[list[tuple[float, float]]] = [[], []]
    frames: list[Image.Image] = []
    for frame_data in frames_data:
        state = np.asarray(frame_data["state"], dtype=np.float32)
        for idx in range(2):
            trails[idx].append((float(state[idx, 0]), float(state[idx, 1])))
        frames.append(
            _draw_two_runner_frame(
                env,
                cfg,
                frame_data,
                trails,
                seed=int(seed),
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(float(cfg["environment"]["dt"]) * 1000)),
        loop=0,
        optimize=False,
    )
    summary["frames"] = len(frames)
    summary["output"] = str(output_path)
    return summary
