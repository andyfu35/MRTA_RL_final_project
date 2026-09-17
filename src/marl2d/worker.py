from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import AGENT_IDS
from .env import VectorArena2D
from .policy import ActorCritic, load_model_snapshot, snapshot_model
from .ppo import RolloutBatch, ppo_update


class AgentWorker:
    """Owns one trainable PPO policy and emulates one future training computer."""

    def __init__(self, agent_id: str, cfg: dict[str, Any], device: str = "cpu") -> None:
        if agent_id not in AGENT_IDS:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        self.agent_id = agent_id
        self.agent_index = AGENT_IDS.index(agent_id)
        self.cfg = cfg
        self.device = torch.device(device)
        ppo_cfg = cfg["ppo"]
        self.model = ActorCritic(21, 2, ppo_cfg["hidden_sizes"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(ppo_cfg["learning_rate"]))

    def _models_from_policy_set(
        self, policy_set: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, ActorCritic]:
        if set(policy_set) != set(AGENT_IDS):
            raise ValueError("policy_set must contain all four agents")
        models: dict[str, ActorCritic] = {}
        hidden_sizes = self.cfg["ppo"]["hidden_sizes"]
        for agent_id in AGENT_IDS:
            model = ActorCritic(21, 2, hidden_sizes).to(self.device)
            load_model_snapshot(model, policy_set[agent_id])
            model.eval()
            models[agent_id] = model
        return models

    @staticmethod
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
        returns = advantages + values
        return advantages, returns

    def collect_round(
        self,
        policy_set: dict[str, dict[str, torch.Tensor]],
        round_index: int,
    ) -> tuple[RolloutBatch, dict[str, float]]:
        profile = self.cfg["collection_profiles"][self.agent_id]
        n_envs = int(profile["parallel_envs"])
        rollout_steps = int(profile["rollout_steps"])
        batches = int(profile["batches"])
        expected_samples = int(self.cfg["training"]["samples_per_update"])
        actual_samples = n_envs * rollout_steps * batches
        if actual_samples != expected_samples:
            raise ValueError(
                f"{self.agent_id} profile produces {actual_samples} samples, expected {expected_samples}"
            )

        base_seed = int(self.cfg.get("seed", 0)) + self.agent_index * 100_000 + int(round_index) * 1_000
        torch.manual_seed(base_seed)
        models = self._models_from_policy_set(policy_set)
        target_model = models[self.agent_id]

        obs_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        log_prob_parts: list[np.ndarray] = []
        return_parts: list[np.ndarray] = []
        advantage_parts: list[np.ndarray] = []

        reward_sum = 0.0
        collision_count = 0
        episode_count = 0
        success_count = 0

        gamma = float(self.cfg["ppo"]["gamma"])
        gae_lambda = float(self.cfg["ppo"]["gae_lambda"])

        for batch_index in range(batches):
            env_seed = base_seed + batch_index * 37
            env = VectorArena2D(
                n_envs,
                self.cfg["environment"],
                self.cfg["reward"],
                seed=env_seed,
            )
            obs = env.reset(seed=env_seed)

            obs_buf = np.zeros((rollout_steps, n_envs, 21), dtype=np.float32)
            action_buf = np.zeros((rollout_steps, n_envs, 2), dtype=np.float32)
            log_prob_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            value_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            next_value_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            reward_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_steps, n_envs), dtype=bool)

            for t in range(rollout_steps):
                all_actions = np.zeros((n_envs, 4, 2), dtype=np.float32)
                for idx, agent_id in enumerate(AGENT_IDS):
                    obs_tensor = torch.from_numpy(obs[:, idx, :]).to(self.device)
                    with torch.no_grad():
                        action, log_prob, value = models[agent_id].sample(obs_tensor)
                    all_actions[:, idx, :] = action.cpu().numpy()
                    if idx == self.agent_index:
                        obs_buf[t] = obs[:, idx, :]
                        action_buf[t] = action.cpu().numpy()
                        log_prob_buf[t] = log_prob.cpu().numpy()
                        value_buf[t] = value.cpu().numpy()

                next_obs, rewards, done, info = env.step(all_actions)
                reward_buf[t] = rewards[:, self.agent_index]
                done_buf[t] = done
                reward_sum += float(rewards[:, self.agent_index].sum())
                collision_count += int(info["collision"][:, self.agent_index].sum())
                episode_count += int(done.sum())
                success_count += int((done & info["goal_reached"]).sum())

                with torch.no_grad():
                    next_obs_tensor = torch.from_numpy(next_obs[:, self.agent_index, :]).to(self.device)
                    next_value_buf[t] = target_model.value(next_obs_tensor).cpu().numpy()

                if np.any(done):
                    env.reset_indices(done)
                    next_obs = env.observe()
                obs = next_obs

            advantages, returns = self._gae(
                reward_buf,
                value_buf,
                next_value_buf,
                done_buf,
                gamma,
                gae_lambda,
            )
            obs_parts.append(obs_buf.reshape(-1, 21))
            action_parts.append(action_buf.reshape(-1, 2))
            log_prob_parts.append(log_prob_buf.reshape(-1))
            return_parts.append(returns.reshape(-1))
            advantage_parts.append(advantages.reshape(-1))

        observations = torch.from_numpy(np.concatenate(obs_parts, axis=0)).to(self.device)
        actions = torch.from_numpy(np.concatenate(action_parts, axis=0)).to(self.device)
        old_log_probs = torch.from_numpy(np.concatenate(log_prob_parts, axis=0)).to(self.device)
        returns = torch.from_numpy(np.concatenate(return_parts, axis=0)).to(self.device)
        advantages = torch.from_numpy(np.concatenate(advantage_parts, axis=0)).to(self.device)

        batch = RolloutBatch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
        )
        stats = {
            "samples": int(observations.shape[0]),
            "mean_step_reward": reward_sum / max(1, expected_samples),
            "collision_rate": collision_count / max(1, expected_samples),
            "episodes": int(episode_count),
            "successes": int(success_count),
        }
        return batch, stats

    def train_round(
        self,
        policy_set: dict[str, dict[str, torch.Tensor]],
        round_index: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        load_model_snapshot(self.model, policy_set[self.agent_id])
        self.model.train()
        batch, collection_metrics = self.collect_round(policy_set, round_index)
        update_metrics = ppo_update(self.model, self.optimizer, batch, self.cfg["ppo"])
        metrics: dict[str, float] = {**collection_metrics, **update_metrics}
        return snapshot_model(self.model), metrics
