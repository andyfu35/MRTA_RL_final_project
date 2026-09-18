from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import AGENT_IDS
from .two_runner import TWO_RUNNER_IDS


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
    if float(env["max_wheel_linear_speed"]) <= 0:
        raise ValueError("max_wheel_linear_speed must be positive")
    if float(env["width"]) <= 0 or float(env["height"]) <= 0:
        raise ValueError("environment width and height must be positive")

    obstacles = env.get("obstacles", {})
    if not isinstance(obstacles, dict):
        raise ValueError("environment.obstacles must be a mapping")
    count = int(obstacles.get("count", 0))
    if count < 0:
        raise ValueError("environment.obstacles.count must be >= 0")
    if count > 0:
        required_obstacle_keys = {"min_width", "max_width", "min_height", "max_height"}
        missing_obstacle_keys = required_obstacle_keys - set(obstacles)
        if missing_obstacle_keys:
            raise ValueError(f"Missing obstacle settings: {sorted(missing_obstacle_keys)}")
        if not 0 < float(obstacles["min_width"]) <= float(obstacles["max_width"]):
            raise ValueError("Obstacle width range is invalid")
        if not 0 < float(obstacles["min_height"]) <= float(obstacles["max_height"]):
            raise ValueError("Obstacle height range is invalid")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(cfg)
    return cfg


def validate_single_runner_config(cfg: dict[str, Any]) -> None:
    required = {"environment", "single_runner_reward", "training", "collection", "ppo", "evaluation"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing single-runner config sections: {sorted(missing)}")

    env = cfg["environment"]
    if int(env["lidar_rays"]) != 8:
        raise ValueError("Single-runner prototype fixes lidar_rays=8 so observation size remains 15")
    if float(env["dt"]) <= 0 or float(env["wheel_base"]) <= 0 or float(env["wheel_radius"]) <= 0:
        raise ValueError("dt, wheel_base and wheel_radius must be positive")
    if float(env["max_wheel_linear_speed"]) <= 0:
        raise ValueError("max_wheel_linear_speed must be positive")

    reward = cfg["single_runner_reward"]
    mode = str(reward["mode"]).upper()
    if mode not in {"R0", "R1", "R2", "R3"}:
        raise ValueError("single_runner_reward.mode must be one of R0, R1, R2, R3")
    if float(reward["safety_distance"]) <= 0:
        raise ValueError("single_runner_reward.safety_distance must be positive")
    if float(reward["safety_scale"]) < 0:
        raise ValueError("single_runner_reward.safety_scale must be >= 0")

    samples = int(cfg["training"]["samples_per_update"])
    if samples <= 0:
        raise ValueError("training.samples_per_update must be > 0")
    collection = cfg["collection"]
    parallel_envs = int(collection["parallel_envs"])
    rollout_steps = int(collection["rollout_steps"])
    batches = int(collection["batches"])
    if min(parallel_envs, rollout_steps, batches) <= 0:
        raise ValueError("single-runner collection values must be > 0")
    actual = parallel_envs * rollout_steps * batches
    if actual != samples:
        raise ValueError(
            f"single runner collects {actual} samples/update, expected {samples}; "
            "parallel_envs * rollout_steps * batches must match samples_per_update"
        )

    evaluation = cfg["evaluation"]
    if int(evaluation["episodes"]) <= 0:
        raise ValueError("evaluation.episodes must be positive")

    validation = cfg.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError("validation must be a mapping")
        if int(validation["every"]) <= 0:
            raise ValueError("validation.every must be positive")
        if int(validation["episodes"]) <= 0:
            raise ValueError("validation.episodes must be positive")
        int(validation["seed_start"])


def load_single_runner_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_single_runner_config(cfg)
    return cfg


def validate_two_runner_config(cfg: dict[str, Any]) -> None:
    required = {
        "environment",
        "two_runner_reward",
        "training",
        "collection_profiles",
        "ppo",
        "validation",
        "evaluation",
        "final_test",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing two-runner config sections: {sorted(missing)}")

    env = cfg["environment"]
    if int(env["lidar_rays"]) != 8:
        raise ValueError("Two-runner prototype fixes lidar_rays=8 so observation size remains 18")
    for key in ("dt", "wheel_base", "wheel_radius", "max_wheel_linear_speed", "width", "height"):
        if float(env[key]) <= 0:
            raise ValueError(f"environment.{key} must be positive")

    obstacles = env.get("obstacles", {})
    if not isinstance(obstacles, dict):
        raise ValueError("environment.obstacles must be a mapping")
    if int(obstacles.get("count", 0)) < 0:
        raise ValueError("environment.obstacles.count must be >= 0")

    reward = cfg["two_runner_reward"]
    required_reward = {
        "team_success_bonus",
        "collision_penalty",
        "timeout_penalty",
        "self_progress_scale",
        "team_progress_scale",
        "step_penalty",
        "safety_distance",
        "safety_scale",
    }
    missing_reward = required_reward - set(reward)
    if missing_reward:
        raise ValueError(f"Missing two-runner reward settings: {sorted(missing_reward)}")
    if float(reward["safety_distance"]) <= 0:
        raise ValueError("two_runner_reward.safety_distance must be positive")
    if float(reward["safety_scale"]) < 0:
        raise ValueError("two_runner_reward.safety_scale must be >= 0")

    target_kl = cfg["ppo"].get("target_kl")
    if target_kl is not None and float(target_kl) <= 0.0:
        raise ValueError("ppo.target_kl must be > 0 when configured")

    samples = int(cfg["training"]["samples_per_update"])
    if samples <= 0:
        raise ValueError("training.samples_per_update must be positive")
    profiles = cfg["collection_profiles"]
    if set(profiles) != set(TWO_RUNNER_IDS):
        raise ValueError(f"collection_profiles must contain exactly {TWO_RUNNER_IDS}")
    for agent_id in TWO_RUNNER_IDS:
        if int(profiles[agent_id]["parallel_envs"]) <= 0:
            raise ValueError(f"{agent_id}.parallel_envs must be positive")

    validation = cfg["validation"]
    if int(validation["every"]) <= 0:
        raise ValueError("validation.every must be positive")
    for section in ("validation", "evaluation", "final_test"):
        entry = cfg[section]
        if int(entry["episodes"]) <= 0:
            raise ValueError(f"{section}.episodes must be positive")
        int(entry["seed_start"])


def load_two_runner_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_two_runner_config(cfg)
    return cfg
