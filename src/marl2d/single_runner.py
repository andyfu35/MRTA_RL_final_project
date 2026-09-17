from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .policy import ActorCritic, load_model_snapshot, snapshot_model
from .ppo import RolloutBatch, ppo_update
from .single_runner_env import SingleRunnerArena2D


def compute_path_efficiency(direct_distance: float, actual_distance: float, success: bool) -> float | None:
    if not success or actual_distance <= 0.0:
        return None
    return float(np.clip(float(direct_distance) / float(actual_distance), 0.0, 1.0))


def validation_is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Prefer validation success, then lower collision rate, then higher reward."""
    if best is None:
        return True
    candidate_key = (
        float(candidate["success_rate"]),
        -float(candidate["collision_rate"]),
        float(candidate["mean_episode_reward"]),
    )
    best_key = (
        float(best["success_rate"]),
        -float(best["collision_rate"]),
        float(best["mean_episode_reward"]),
    )
    return candidate_key > best_key


def _snapshot_collector_env(env: SingleRunnerArena2D) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in (
        "base_seed",
        "episode_counts",
        "map_seeds",
        "state",
        "prev_action",
        "collision",
        "steps",
        "obstacles",
    ):
        value = getattr(env, name)
        state[name] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    state["rng_state"] = copy.deepcopy(env.rng.bit_generator.state)
    return state


def _restore_collector_env(env: SingleRunnerArena2D, state: dict[str, Any]) -> None:
    env.base_seed = int(state["base_seed"])
    for name in (
        "episode_counts",
        "map_seeds",
        "state",
        "prev_action",
        "collision",
        "steps",
        "obstacles",
    ):
        target = getattr(env, name)
        target[...] = np.asarray(state[name], dtype=target.dtype)
    env.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_values[t] * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    return advantages, advantages + values


class SingleRunnerTrainer:
    """PPO trainer for Experiment 1: one runner navigating random box maps."""

    obs_dim = SingleRunnerArena2D.observation_dim
    action_dim = SingleRunnerArena2D.action_dim

    def __init__(self, cfg: dict[str, Any], output_dir: str | Path, device: str = "cpu") -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        seed = int(cfg.get("seed", 0))
        np.random.seed(seed)
        torch.manual_seed(seed)
        ppo_cfg = cfg["ppo"]
        self.model = ActorCritic(self.obs_dim, self.action_dim, ppo_cfg["hidden_sizes"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(ppo_cfg["learning_rate"]))
        self.current_round = 0
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._collector_envs: list[SingleRunnerArena2D] = []
        self._collector_obs: list[np.ndarray] = []
        self.best_validation: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, torch.Tensor]:
        return snapshot_model(self.model)

    def _ensure_collectors(self) -> None:
        """Create vector worlds once so unfinished episodes survive PPO update boundaries."""
        if self._collector_envs:
            return

        profile = self.cfg["collection"]
        n_envs = int(profile["parallel_envs"])
        batches = int(profile["batches"])
        base_seed = int(self.cfg.get("seed", 0))

        for batch_index in range(batches):
            # Keep sequential collector batches on disjoint seed ranges. Within each
            # vector env, VectorArena2D derives each world's map seed from world id
            # and episode count, so only a completed world advances to a new map.
            env_seed = base_seed + batch_index * 1_000_000_000
            env = SingleRunnerArena2D(
                n_envs,
                self.cfg["environment"],
                self.cfg["single_runner_reward"],
                seed=env_seed,
            )
            self._collector_envs.append(env)
            self._collector_obs.append(env.observe())

    def _collect_round(self, round_index: int) -> tuple[RolloutBatch, dict[str, float | int]]:
        profile = self.cfg["collection"]
        n_envs = int(profile["parallel_envs"])
        rollout_steps = int(profile["rollout_steps"])
        batches = int(profile["batches"])
        expected_samples = int(self.cfg["training"]["samples_per_update"])
        actual_samples = n_envs * rollout_steps * batches
        if actual_samples != expected_samples:
            raise ValueError(
                f"single runner collection produces {actual_samples} samples, expected {expected_samples}"
            )

        self._ensure_collectors()
        # Action sampling remains deterministic/reproducible for a given training
        # seed and PPO round, while map seeds are driven only by world/episode.
        torch.manual_seed(int(self.cfg.get("seed", 0)) + int(round_index) * 1_000_000)

        obs_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        log_prob_parts: list[np.ndarray] = []
        return_parts: list[np.ndarray] = []
        advantage_parts: list[np.ndarray] = []
        reward_sum = 0.0
        collision_count = 0
        timeout_count = 0
        episode_count = 0
        success_count = 0
        seen_map_seeds: set[int] = set()

        gamma = float(self.cfg["ppo"]["gamma"])
        gae_lambda = float(self.cfg["ppo"]["gae_lambda"])

        for batch_index in range(batches):
            env = self._collector_envs[batch_index]
            obs = self._collector_obs[batch_index]
            seen_map_seeds.update(int(seed) for seed in env.map_seeds.tolist())

            obs_buf = np.zeros((rollout_steps, n_envs, self.obs_dim), dtype=np.float32)
            action_buf = np.zeros((rollout_steps, n_envs, self.action_dim), dtype=np.float32)
            log_prob_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            value_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            next_value_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            reward_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_steps, n_envs), dtype=bool)

            for t in range(rollout_steps):
                obs_tensor = torch.from_numpy(obs).to(self.device)
                with torch.no_grad():
                    action, log_prob, value = self.model.sample(obs_tensor)
                action_np = action.cpu().numpy()
                next_obs, rewards, done, info = env.step(action_np)

                obs_buf[t] = obs
                action_buf[t] = action_np
                log_prob_buf[t] = log_prob.cpu().numpy()
                value_buf[t] = value.cpu().numpy()
                reward_buf[t] = rewards
                done_buf[t] = done
                reward_sum += float(rewards.sum())

                # Terminal categories are mutually exclusive and follow the same
                # precedence as the environment terminal reward: goal > collision > timeout.
                goal_terminal = done & info["goal_reached"]
                collision_terminal = done & info["collision"] & ~goal_terminal
                timeout_terminal = done & info["timeout"] & ~goal_terminal & ~collision_terminal
                success_count += int(goal_terminal.sum())
                collision_count += int(collision_terminal.sum())
                timeout_count += int(timeout_terminal.sum())
                episode_count += int(done.sum())

                with torch.no_grad():
                    next_obs_tensor = torch.from_numpy(next_obs).to(self.device)
                    next_value_buf[t] = self.model.value(next_obs_tensor).cpu().numpy()

                if np.any(done):
                    # Reset only completed worlds. Every unfinished world keeps its
                    # pose, step counter, obstacle map, and episode seed across PPO rounds.
                    env.reset_indices(done)
                    seen_map_seeds.update(int(seed) for seed in env.map_seeds[done].tolist())
                    next_obs = env.observe()
                obs = next_obs

            self._collector_obs[batch_index] = obs
            advantages, returns = _gae(
                reward_buf,
                value_buf,
                next_value_buf,
                done_buf,
                gamma,
                gae_lambda,
            )
            obs_parts.append(obs_buf.reshape(-1, self.obs_dim))
            action_parts.append(action_buf.reshape(-1, self.action_dim))
            log_prob_parts.append(log_prob_buf.reshape(-1))
            return_parts.append(returns.reshape(-1))
            advantage_parts.append(advantages.reshape(-1))

        observations_np = np.concatenate(obs_parts)
        actions_np = np.concatenate(action_parts)
        observations = torch.from_numpy(observations_np).to(self.device)
        batch = RolloutBatch(
            observations=observations,
            actions=torch.from_numpy(actions_np).to(self.device),
            old_log_probs=torch.from_numpy(np.concatenate(log_prob_parts)).to(self.device),
            returns=torch.from_numpy(np.concatenate(return_parts)).to(self.device),
            advantages=torch.from_numpy(np.concatenate(advantage_parts)).to(self.device),
        )
        metrics: dict[str, float | int] = {
            "samples": int(observations.shape[0]),
            "mean_step_reward": reward_sum / max(1, expected_samples),
            # Preserve the historical per-sample collision metric for old plots.
            "collision_rate": collision_count / max(1, expected_samples),
            "episodes": int(episode_count),
            "completed_episodes": int(episode_count),
            "successes": int(success_count),
            "goals": int(success_count),
            "collisions": int(collision_count),
            "timeouts": int(timeout_count),
            "success_rate": success_count / max(1, episode_count),
            "parallel_envs": n_envs,
            "unique_map_seeds": len(seen_map_seeds),
            "stochastic_action_std": float(np.std(actions_np)),
        }
        return batch, metrics

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        checkpoint_path = Path(path) if path is not None else self.output_dir / "latest.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.current_round,
                "model": self.snapshot(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "collector_states": [_snapshot_collector_env(env) for env in self._collector_envs],
                "collector_obs": [obs.copy() for obs in self._collector_obs],
                "best_validation": copy.deepcopy(self.best_validation),
                "training_state_version": 1,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def resume_from_checkpoint(self, path: str | Path) -> str:
        data = torch.load(Path(path), map_location=self.device, weights_only=False)
        if int(data.get("obs_dim", self.obs_dim)) != self.obs_dim or int(
            data.get("action_dim", self.action_dim)
        ) != self.action_dim:
            raise ValueError("Resume checkpoint observation/action dimensions do not match this trainer")

        checkpoint_cfg = data.get("config", {})
        checkpoint_mode = str(checkpoint_cfg.get("single_runner_reward", {}).get("mode", "")).upper()
        current_mode = str(self.cfg.get("single_runner_reward", {}).get("mode", "")).upper()
        if checkpoint_mode and current_mode and checkpoint_mode != current_mode:
            raise ValueError(
                f"Resume reward mode mismatch: checkpoint={checkpoint_mode}, current={current_mode}"
            )

        load_model_snapshot(self.model, data["model"])
        self.current_round = int(data["version"])
        self.best_validation = copy.deepcopy(data.get("best_validation"))

        has_full_state = (
            "optimizer" in data
            and "collector_states" in data
            and "collector_obs" in data
            and int(data.get("training_state_version", 0)) >= 1
        )
        if not has_full_state:
            self._collector_envs = []
            self._collector_obs = []
            return "legacy"

        self.optimizer.load_state_dict(data["optimizer"])
        collector_states = list(data["collector_states"])
        collector_obs = list(data["collector_obs"])
        expected_batches = int(self.cfg["collection"]["batches"])
        if len(collector_states) != expected_batches or len(collector_obs) != expected_batches:
            raise ValueError("Resume checkpoint collector batch count does not match current config")

        self._collector_envs = []
        self._collector_obs = []
        self._ensure_collectors()
        for env, env_state in zip(self._collector_envs, collector_states):
            _restore_collector_env(env, env_state)
        self._collector_obs = [np.asarray(obs, dtype=np.float32).copy() for obs in collector_obs]
        return "full"

    def run(self, rounds: int | None = None) -> list[dict[str, float | int]]:
        if rounds is None:
            rounds = int(self.cfg["training"]["rounds"])
        rounds = int(rounds)
        if rounds <= 0:
            return []

        records: list[dict[str, float | int]] = []
        checkpoint_every = max(1, int(self.cfg["training"].get("checkpoint_every", 1)))
        for _ in range(rounds):
            self.model.train()
            batch, collection_metrics = self._collect_round(self.current_round)
            update_metrics = ppo_update(self.model, self.optimizer, batch, self.cfg["ppo"])
            self.current_round += 1
            record: dict[str, float | int] = {
                "round": self.current_round,
                **collection_metrics,
                **update_metrics,
            }

            is_best = False
            validation_cfg = self.cfg.get("validation")
            if validation_cfg:
                validate_every = int(validation_cfg["every"])
                if self.current_round % validate_every == 0:
                    summary = evaluate_single_runner(
                        self.cfg,
                        self.snapshot(),
                        episodes=int(validation_cfg["episodes"]),
                        seed_start=int(validation_cfg["seed_start"]),
                        device=str(self.device),
                    )
                    record.update(
                        {
                            "val_episodes": int(summary["episodes"]),
                            "val_success_rate": float(summary["success_rate"]),
                            "val_collision_rate": float(summary["collision_rate"]),
                            "val_timeout_rate": float(summary["timeout_rate"]),
                            "val_mean_episode_reward": float(summary["mean_episode_reward"]),
                        }
                    )
                    candidate = {
                        "round": self.current_round,
                        "success_rate": float(summary["success_rate"]),
                        "collision_rate": float(summary["collision_rate"]),
                        "mean_episode_reward": float(summary["mean_episode_reward"]),
                    }
                    if validation_is_better(candidate, self.best_validation):
                        self.best_validation = candidate
                        is_best = True

            records.append(record)
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.save_checkpoint()
            if is_best:
                self.save_checkpoint(self.output_dir / "best.pt")
            if self.current_round % checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / f"round_{self.current_round:05d}.pt")
        return records


def load_single_runner_checkpoint(
    path: str | Path,
) -> tuple[int, dict[str, torch.Tensor], dict[str, Any]]:
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return int(data["version"]), data["model"], data["config"]


def evaluate_single_runner(
    cfg: dict[str, Any],
    snapshot: dict[str, torch.Tensor],
    episodes: int,
    seed_start: int,
    device: str = "cpu",
) -> dict[str, Any]:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    torch_device = torch.device(device)
    model = ActorCritic(
        SingleRunnerArena2D.observation_dim,
        SingleRunnerArena2D.action_dim,
        cfg["ppo"]["hidden_sizes"],
    ).to(torch_device)
    load_model_snapshot(model, snapshot)
    model.eval()

    seeds = [int(seed_start) + i for i in range(episodes)]
    successes = 0
    collision_counts: list[int] = []
    collision_episodes = 0
    success_times: list[float] = []
    efficiencies: list[float] = []
    episode_rewards: list[float] = []
    minimum_clearances: list[float] = []

    for episode_seed in seeds:
        env = SingleRunnerArena2D(
            1,
            cfg["environment"],
            cfg["single_runner_reward"],
            seed=episode_seed,
        )
        obs = env.reset(seed=episode_seed)
        start = env.runner_state[0, :2].copy()
        direct_distance = float(np.linalg.norm(start - env.goal))
        actual_distance = 0.0
        collisions = 0
        success = False
        steps = 0
        episode_reward = 0.0
        min_clearance = float("inf")

        for step in range(1, int(cfg["environment"]["max_steps"]) + 1):
            previous = env.runner_state[0, :2].copy()
            with torch.no_grad():
                action = model.deterministic(torch.from_numpy(obs).to(torch_device)).cpu().numpy()
            obs, reward, done, info = env.step(action)
            current = env.runner_state[0, :2].copy()
            actual_distance += float(np.linalg.norm(current - previous))
            collisions += int(bool(info["collision"][0]))
            episode_reward += float(reward[0])
            min_clearance = min(min_clearance, float(info["min_clearance"][0]))
            steps = step
            if bool(done[0]):
                success = bool(info["goal_reached"][0])
                break

        collision_counts.append(collisions)
        collision_episodes += int(collisions > 0)
        episode_rewards.append(episode_reward)
        minimum_clearances.append(min_clearance)
        if success:
            successes += 1
            success_times.append(steps * float(cfg["environment"]["dt"]))
            efficiency = compute_path_efficiency(direct_distance, actual_distance, success=True)
            if efficiency is not None:
                efficiencies.append(efficiency)

    timeouts = episodes - successes - collision_episodes
    return {
        "episodes": episodes,
        "seeds": seeds,
        "successes": successes,
        "success_rate": successes / episodes,
        "collision_rate": collision_episodes / episodes,
        "timeouts": timeouts,
        "timeout_rate": timeouts / episodes,
        "mean_time_to_goal_s": float(np.mean(success_times)) if success_times else None,
        "mean_collisions": float(np.mean(collision_counts)),
        "mean_path_efficiency": float(np.mean(efficiencies)) if efficiencies else None,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_min_clearance_m": float(np.mean(minimum_clearances)),
    }
