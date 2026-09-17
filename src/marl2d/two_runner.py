from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .policy import ActorCritic, load_model_snapshot, snapshot_model
from .ppo import RolloutBatch, ppo_update
from .two_runner_env import TwoRunnerArena2D

TWO_RUNNER_IDS = ("runner_0", "runner_1")


@dataclass
class TargetTransition:
    observation: np.ndarray
    action: np.ndarray
    old_log_prob: float
    reward: float
    value: float
    next_value: float


@dataclass
class FinalizedSample:
    observation: np.ndarray
    action: np.ndarray
    old_log_prob: float
    reward: float
    value: float
    next_value: float
    done: bool
    advantage: float
    return_value: float


def finalize_target_trajectory(
    transitions: list[TargetTransition],
    *,
    delayed_team_success: bool,
    team_success_bonus: float,
    gamma: float,
    gae_lambda: float,
) -> list[FinalizedSample]:
    if not transitions:
        return []
    rewards = [float(item.reward) for item in transitions]
    if delayed_team_success:
        rewards[-1] += float(team_success_bonus)

    advantages = [0.0] * len(transitions)
    gae = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        done = index == len(transitions) - 1
        not_done = 0.0 if done else 1.0
        next_value = 0.0 if done else float(item.next_value)
        delta = rewards[index] + float(gamma) * next_value * not_done - float(item.value)
        gae = delta + float(gamma) * float(gae_lambda) * not_done * gae
        advantages[index] = gae

    finalized: list[FinalizedSample] = []
    for index, item in enumerate(transitions):
        done = index == len(transitions) - 1
        advantage = float(advantages[index])
        finalized.append(
            FinalizedSample(
                observation=np.asarray(item.observation, dtype=np.float32).copy(),
                action=np.asarray(item.action, dtype=np.float32).copy(),
                old_log_prob=float(item.old_log_prob),
                reward=float(rewards[index]),
                value=float(item.value),
                next_value=0.0 if done else float(item.next_value),
                done=done,
                advantage=advantage,
                return_value=advantage + float(item.value),
            )
        )
    return finalized


class TwoRunnerWorker:
    obs_dim = 18
    action_dim = 2

    def __init__(
        self,
        agent_id: str,
        cfg: dict[str, Any],
        device: str = "cpu",
        env_factory: Callable[..., Any] = TwoRunnerArena2D,
    ) -> None:
        if agent_id not in TWO_RUNNER_IDS:
            raise ValueError(f"Unknown two-runner agent_id: {agent_id}")
        self.agent_id = agent_id
        self.agent_index = TWO_RUNNER_IDS.index(agent_id)
        self.cfg = cfg
        self.device = torch.device(device)
        self.env_factory = env_factory
        ppo_cfg = cfg["ppo"]
        self.model = ActorCritic(self.obs_dim, self.action_dim, ppo_cfg["hidden_sizes"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(ppo_cfg["learning_rate"]))
        self._collector_env: Any | None = None
        self._collector_obs: np.ndarray | None = None
        self.pending_trajectories: list[list[TargetTransition]] = []
        self.finalized_queue: deque[FinalizedSample] = deque()

    def _ensure_collector(self) -> None:
        if self._collector_env is not None:
            return
        n_envs = int(self.cfg["collection_profiles"][self.agent_id]["parallel_envs"])
        base_seed = int(self.cfg.get("seed", 0)) + self.agent_index * 100_000
        self._collector_env = self.env_factory(
            n_envs,
            self.cfg["environment"],
            self.cfg["two_runner_reward"],
            seed=base_seed,
        )
        self._collector_obs = self._collector_env.observe()
        self.pending_trajectories = [[] for _ in range(n_envs)]

    def _models_from_policy_set(self, policy_set: dict[str, dict[str, torch.Tensor]]) -> dict[str, ActorCritic]:
        if set(policy_set) != set(TWO_RUNNER_IDS):
            raise ValueError(f"policy_set must contain exactly {TWO_RUNNER_IDS}")
        result: dict[str, ActorCritic] = {}
        hidden = self.cfg["ppo"]["hidden_sizes"]
        for agent_id in TWO_RUNNER_IDS:
            model = ActorCritic(self.obs_dim, self.action_dim, hidden).to(self.device)
            load_model_snapshot(model, policy_set[agent_id])
            model.eval()
            result[agent_id] = model
        return result

    def _finalize_env(self, env_index: int, delayed_team_success: bool) -> int:
        trajectory = self.pending_trajectories[env_index]
        finalized = finalize_target_trajectory(
            trajectory,
            delayed_team_success=bool(delayed_team_success),
            team_success_bonus=float(self.cfg["two_runner_reward"]["team_success_bonus"]),
            gamma=float(self.cfg["ppo"]["gamma"]),
            gae_lambda=float(self.cfg["ppo"]["gae_lambda"]),
        )
        self.finalized_queue.extend(finalized)
        self.pending_trajectories[env_index] = []
        return len(finalized)

    def collect_round(
        self,
        frozen_policy_set: dict[str, dict[str, torch.Tensor]],
        round_index: int,
    ) -> tuple[RolloutBatch, dict[str, float | int]]:
        expected_samples = int(self.cfg["training"]["samples_per_update"])
        if expected_samples <= 0:
            raise ValueError("training.samples_per_update must be positive")
        self._ensure_collector()
        assert self._collector_env is not None and self._collector_obs is not None
        env = self._collector_env
        models = self._models_from_policy_set(frozen_policy_set)
        target_model = models[self.agent_id]
        torch.manual_seed(int(self.cfg.get("seed", 0)) + self.agent_index * 100_000 + int(round_index) * 1_000_000)

        simulator_steps = 0
        completed_team_episodes = 0
        target_deaths = 0
        target_goal_contributions = 0
        delayed_team_credits = 0
        unique_map_seeds: set[int] = set()

        while len(self.finalized_queue) < expected_samples:
            obs = self._collector_obs
            alive_before = np.asarray(env.alive, dtype=bool).copy()
            n_envs = int(env.num_envs)
            all_actions = np.zeros((n_envs, 2, 2), dtype=np.float32)
            target_actions = np.zeros((n_envs, 2), dtype=np.float32)
            target_log_probs = np.zeros(n_envs, dtype=np.float32)
            target_values = np.zeros(n_envs, dtype=np.float32)

            for agent_index, agent_id in enumerate(TWO_RUNNER_IDS):
                alive_mask = alive_before[:, agent_index]
                if not np.any(alive_mask):
                    continue
                obs_tensor = torch.from_numpy(obs[alive_mask, agent_index, :]).to(self.device)
                with torch.no_grad():
                    action, log_prob, value = models[agent_id].sample(obs_tensor)
                action_np = action.cpu().numpy()
                all_actions[alive_mask, agent_index, :] = action_np
                if agent_index == self.agent_index:
                    target_actions[alive_mask] = action_np
                    target_log_probs[alive_mask] = log_prob.cpu().numpy()
                    target_values[alive_mask] = value.cpu().numpy()

            next_obs, rewards, done, info = env.step(all_actions)
            simulator_steps += n_envs
            unique_map_seeds.update(int(x) for x in np.asarray(info.get("map_seed", env.map_seeds)).tolist())

            with torch.no_grad():
                next_values = target_model.value(
                    torch.from_numpy(next_obs[:, self.agent_index, :]).to(self.device)
                ).cpu().numpy()

            target_alive_before = np.asarray(info["alive_before"], dtype=bool)[:, self.agent_index]
            for env_index in np.flatnonzero(target_alive_before):
                self.pending_trajectories[int(env_index)].append(
                    TargetTransition(
                        observation=obs[env_index, self.agent_index, :],
                        action=target_actions[env_index],
                        old_log_prob=float(target_log_probs[env_index]),
                        reward=float(rewards[env_index, self.agent_index]),
                        value=float(target_values[env_index]),
                        next_value=float(next_values[env_index]),
                    )
                )

            new_death = np.asarray(info["new_death"], dtype=bool)[:, self.agent_index]
            target_deaths += int(new_death.sum())
            goal_reached = np.asarray(info["goal_reached"], dtype=bool)[:, self.agent_index]
            target_goal_contributions += int(goal_reached.sum())

            for env_index in np.flatnonzero(done):
                env_index = int(env_index)
                completed_team_episodes += 1
                team_success = bool(np.asarray(info["team_success"])[env_index])
                target_was_alive = bool(np.asarray(info["alive_before"])[env_index, self.agent_index])
                delayed = team_success and not target_was_alive
                if delayed:
                    delayed_team_credits += 1
                self._finalize_env(env_index, delayed_team_success=delayed)

            if np.any(done):
                env.reset_indices(done)
                next_obs = env.observe()
            self._collector_obs = next_obs

        consumed = [self.finalized_queue.popleft() for _ in range(expected_samples)]
        observations = torch.from_numpy(np.stack([x.observation for x in consumed])).to(self.device)
        actions = torch.from_numpy(np.stack([x.action for x in consumed])).to(self.device)
        old_log_probs = torch.tensor([x.old_log_prob for x in consumed], dtype=torch.float32, device=self.device)
        returns = torch.tensor([x.return_value for x in consumed], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([x.advantage for x in consumed], dtype=torch.float32, device=self.device)
        batch = RolloutBatch(observations, actions, old_log_probs, returns, advantages)
        metrics: dict[str, float | int] = {
            "samples": expected_samples,
            "simulator_steps": simulator_steps,
            "completed_team_episodes": completed_team_episodes,
            "target_deaths": target_deaths,
            "target_goal_contributions": target_goal_contributions,
            "delayed_team_credits": delayed_team_credits,
            "queued_surplus_samples": len(self.finalized_queue),
            "unique_map_seeds": len(unique_map_seeds),
            "stochastic_action_std": float(actions.detach().cpu().numpy().std()),
        }
        return batch, metrics

    def train_round(self, frozen_policy_set, round_index: int):
        load_model_snapshot(self.model, frozen_policy_set[self.agent_id])
        self.model.train()
        batch, collection_metrics = self.collect_round(frozen_policy_set, round_index)
        update_metrics = ppo_update(self.model, self.optimizer, batch, self.cfg["ppo"])
        return snapshot_model(self.model), {**collection_metrics, **update_metrics}
