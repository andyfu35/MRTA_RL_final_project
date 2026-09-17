from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import AGENT_IDS


def validate_config(cfg: dict[str, Any]) -> None:
    required = {"environment", "reward", "training", "collection_profiles", "ppo", "network"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    samples = int(cfg["training"]["samples_per_update"])
    if samples <= 0:
        raise ValueError("training.samples_per_update must be > 0")

    profiles = cfg["collection_profiles"]
    if set(profiles) != set(AGENT_IDS):
        raise ValueError(f"collection_profiles must contain exactly {AGENT_IDS}")

    for agent_id in AGENT_IDS:
        profile = profiles[agent_id]
        n = int(profile["parallel_envs"])
        steps = int(profile["rollout_steps"])
        batches = int(profile["batches"])
        if min(n, steps, batches) <= 0:
            raise ValueError(f"{agent_id} collection values must be > 0")
        actual = n * steps * batches
        if actual != samples:
            raise ValueError(
                f"{agent_id} collects {actual} samples/update, expected {samples}; "
                "parallel_envs * rollout_steps * batches must match samples_per_update"
            )

    env = cfg["environment"]
    if int(env["lidar_rays"]) != 8:
        raise ValueError("This prototype fixes lidar_rays=8 so observation size remains 21")
    if float(env["dt"]) <= 0 or float(env["wheel_base"]) <= 0 or float(env["wheel_radius"]) <= 0:
        raise ValueError("dt, wheel_base and wheel_radius must be positive")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(cfg)
    return cfg
