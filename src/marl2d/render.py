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


def _world_to_pixel(x: float, y: float, width: float, height: float, image_size: tuple[int, int]) -> tuple[int, int]:
    w, h = image_size
    px = int(round(x / width * w))
    py = int(round(h - y / height * h))
    return px, py


def _radius_to_pixel(radius: float, width: float, image_width: int) -> int:
    return max(2, int(round(radius / width * image_width)))


def _draw_frame(env: VectorArena2D, step: int, status: str) -> Image.Image:
    image_size = (800, 480)
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = env.width
    height = env.height

    draw.rectangle((0, 0, image_size[0] - 1, image_size[1] - 1), outline="black", width=3)

    gx, gy = _world_to_pixel(float(env.goal[0]), float(env.goal[1]), width, height, image_size)
    gr = _radius_to_pixel(env.goal_radius, width, image_size[0])
    draw.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill="#d8f3dc", outline="#2d6a4f", width=3)
    draw.text((gx - 14, gy - 6), "GOAL", fill="black", font=font)

    for ox, oy, radius in env.obstacles:
        px, py = _world_to_pixel(float(ox), float(oy), width, height, image_size)
        pr = _radius_to_pixel(float(radius), width, image_size[0])
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill="#adb5bd", outline="#495057", width=2)

    colors = ("#1d4ed8", "#2563eb", "#dc2626", "#ef4444")
    labels = ("R0", "R1", "B0", "B1")
    rr = _radius_to_pixel(env.robot_radius, width, image_size[0])
    for i in range(4):
        x, y, theta = (float(v) for v in env.state[0, i])
        px, py = _world_to_pixel(x, y, width, height, image_size)
        draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=colors[i], outline="black", width=2)
        arrow_len = rr * 2.2
        ex = px + arrow_len * math.cos(theta)
        ey = py - arrow_len * math.sin(theta)
        draw.line((px, py, ex, ey), fill="black", width=3)
        draw.text((px - 7, py - rr - 13), labels[i], fill="black", font=font)

    draw.rectangle((5, 5, 245, 31), fill="white", outline="#cccccc")
    draw.text((10, 10), f"step={step}  {status}", fill="black", font=font)
    return image


def render_policy_set_gif(
    cfg: dict[str, Any],
    policy_set: PolicySet,
    output_path: str | Path,
    max_steps: int | None = None,
    device: str = "cpu",
) -> dict[str, int | bool]:
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
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(float(cfg["environment"]["dt"]) * 1000)),
        loop=0,
        optimize=False,
    )
    return {
        "frames": len(frames),
        "steps": steps_taken,
        "goal_reached": goal_reached,
        "timeout": timed_out,
    }
