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


def finalize_target_fragment(
    transitions: list[TargetTransition],
    *,
    terminal: bool,
    gamma: float,
    gae_lambda: float,
) -> list[FinalizedSample]:
    """Finalize one on-policy PPO fragment.

    Terminal fragments stop value propagation at the last transition.
    Fixed-horizon fragments keep the last transition non-terminal and
    bootstrap from its stored next_value.
    """
    if not transitions:
        return []

    advantages = [0.0] * len(transitions)
    gae = 0.0
    last_index = len(transitions) - 1
    for index in range(last_index, -1, -1):
        item = transitions[index]
        terminal_step = bool(terminal and index == last_index)
        not_done = 0.0 if terminal_step else 1.0
        next_value = 0.0 if terminal_step else float(item.next_value)
        delta = (
            float(item.reward)
            + float(gamma) * next_value * not_done
            - float(item.value)
        )
        gae = (
            delta
            + float(gamma) * float(gae_lambda) * not_done * gae
        )
        advantages[index] = gae

    finalized: list[FinalizedSample] = []
    for index, item in enumerate(transitions):
        terminal_step = bool(terminal and index == last_index)
        advantage = float(advantages[index])
        finalized.append(
            FinalizedSample(
                observation=np.asarray(
                    item.observation, dtype=np.float32
                ).copy(),
                action=np.asarray(item.action, dtype=np.float32).copy(),
                old_log_prob=float(item.old_log_prob),
                reward=float(item.reward),
                value=float(item.value),
                next_value=(
                    0.0 if terminal_step else float(item.next_value)
                ),
                done=terminal_step,
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
        self.finalized_outcomes: deque[str] = deque()
        self.finalized_trajectory_ids: deque[int] = deque()

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

    def _finalize_env(
        self,
        env_index: int,
        delayed_team_success: bool,
        *,
        outcome: str,
        trajectory_id: int,
    ) -> int:
        trajectory = self.pending_trajectories[env_index]
        finalized = finalize_target_trajectory(
            trajectory,
            delayed_team_success=bool(delayed_team_success),
            team_success_bonus=float(self.cfg["two_runner_reward"]["team_success_bonus"]),
            gamma=float(self.cfg["ppo"]["gamma"]),
            gae_lambda=float(self.cfg["ppo"]["gae_lambda"]),
        )
        self.finalized_queue.extend(finalized)
        self.finalized_outcomes.extend([str(outcome)] * len(finalized))
        self.finalized_trajectory_ids.extend([int(trajectory_id)] * len(finalized))
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

        # A PPO round must never consume samples generated by an older frozen
        # policy set. Round boundaries are therefore clean: pending trajectories
        # are drained before the previous round returns, and surplus samples are
        # discarded instead of carried into the next update.
        if (
            self.finalized_queue
            or self.finalized_outcomes
            or self.finalized_trajectory_ids
            or any(self.pending_trajectories)
        ):
            raise RuntimeError("two-runner collector crossed a PPO round boundary with stale samples")

        models = self._models_from_policy_set(frozen_policy_set)
        target_model = models[self.agent_id]
        round_seed = int(self.cfg.get("seed", 0)) + self.agent_index * 100_000 + int(round_index) * 1_000_000
        torch.manual_seed(round_seed)

        simulator_steps = 0
        completed_team_episodes = 0
        target_deaths = 0
        target_goal_contributions = 0
        delayed_team_credits = 0
        unique_map_seeds: set[int] = set()
        outcome_episode_counts = {"success": 0, "timeout": 0, "both_dead": 0, "other": 0}
        outcome_trajectory_lengths: dict[str, list[int]] = {
            "success": [],
            "timeout": [],
            "both_dead": [],
            "other": [],
        }
        trajectory_serial = 0
        draining = False
        active_episode_mask = np.ones(int(env.num_envs), dtype=bool)

        while True:
            obs = self._collector_obs
            alive_before = np.asarray(env.alive, dtype=bool).copy()
            n_envs = int(env.num_envs)
            step_mask = active_episode_mask if draining else np.ones(n_envs, dtype=bool)
            all_actions = np.zeros((n_envs, 2, 2), dtype=np.float32)
            target_actions = np.zeros((n_envs, 2), dtype=np.float32)
            target_log_probs = np.zeros(n_envs, dtype=np.float32)
            target_values = np.zeros(n_envs, dtype=np.float32)

            for agent_index, agent_id in enumerate(TWO_RUNNER_IDS):
                alive_mask = alive_before[:, agent_index] & step_mask
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

            target_alive_before = (
                np.asarray(info["alive_before"], dtype=bool)[:, self.agent_index] & step_mask
            )
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

            new_death = np.asarray(info["new_death"], dtype=bool)[:, self.agent_index] & step_mask
            target_deaths += int(new_death.sum())
            goal_reached = np.asarray(info["goal_reached"], dtype=bool)[:, self.agent_index] & step_mask
            target_goal_contributions += int(goal_reached.sum())

            effective_done = np.asarray(done, dtype=bool) & step_mask
            for env_index in np.flatnonzero(effective_done):
                env_index = int(env_index)
                completed_team_episodes += 1
                team_success = bool(np.asarray(info["team_success"])[env_index])
                both_dead = bool(np.asarray(info["both_dead"])[env_index])
                timed_out = bool(np.asarray(info["timeout"])[env_index])
                if team_success:
                    outcome = "success"
                elif both_dead:
                    outcome = "both_dead"
                elif timed_out:
                    outcome = "timeout"
                else:
                    outcome = "other"
                target_was_alive = bool(np.asarray(info["alive_before"])[env_index, self.agent_index])
                delayed = team_success and not target_was_alive
                if delayed:
                    delayed_team_credits += 1
                trajectory_length = len(self.pending_trajectories[env_index])
                outcome_episode_counts[outcome] += 1
                outcome_trajectory_lengths[outcome].append(trajectory_length)
                self._finalize_env(
                    env_index,
                    delayed_team_success=delayed,
                    outcome=outcome,
                    trajectory_id=trajectory_serial,
                )
                trajectory_serial += 1

            if not draining and len(self.finalized_queue) >= expected_samples:
                # Do not launch any new episodes after the round has enough
                # finalized data. Existing episodes are allowed to terminate so
                # delayed team credit becomes known under this same policy set.
                draining = True
                active_episode_mask = ~effective_done
                if not np.any(active_episode_mask):
                    self._collector_obs = next_obs
                    break
            elif draining:
                active_episode_mask &= ~effective_done
                if not np.any(active_episode_mask):
                    self._collector_obs = next_obs
                    break
            else:
                if np.any(effective_done):
                    env.reset_indices(effective_done)
                    next_obs = env.observe()

            self._collector_obs = next_obs

        if any(self.pending_trajectories):
            raise RuntimeError("drain-at-boundary ended with unfinished target trajectories")
        if len(self.finalized_queue) < expected_samples:
            raise RuntimeError("collector drained without enough finalized samples")

        sample_pool = list(self.finalized_queue)
        outcome_pool = list(self.finalized_outcomes)
        trajectory_id_pool = list(self.finalized_trajectory_ids)
        pool_size = len(sample_pool)
        if not (pool_size == len(outcome_pool) == len(trajectory_id_pool)):
            raise RuntimeError("two-runner diagnostic sidecars are not aligned with finalized samples")
        selection_rng = np.random.default_rng(round_seed + 1)
        selected_indices = selection_rng.choice(
            pool_size, size=expected_samples, replace=False
        )
        consumed = [sample_pool[int(index)] for index in selected_indices]
        selected_outcomes = [outcome_pool[int(index)] for index in selected_indices]
        selected_trajectory_ids = [trajectory_id_pool[int(index)] for index in selected_indices]
        discarded_surplus = pool_size - expected_samples
        self.finalized_queue.clear()
        self.finalized_outcomes.clear()
        self.finalized_trajectory_ids.clear()

        # Every pre-boundary episode is terminal now. Start the next round from
        # fresh episodes only after the current frozen-policy sample pool is sealed.
        reset_mask = np.ones(int(env.num_envs), dtype=bool)
        env.reset_indices(reset_mask)
        self._collector_obs = env.observe()

        observations = torch.from_numpy(np.stack([x.observation for x in consumed])).to(self.device)
        actions = torch.from_numpy(np.stack([x.action for x in consumed])).to(self.device)
        old_log_probs = torch.tensor([x.old_log_prob for x in consumed], dtype=torch.float32, device=self.device)
        returns = torch.tensor([x.return_value for x in consumed], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([x.advantage for x in consumed], dtype=torch.float32, device=self.device)
        batch = RolloutBatch(observations, actions, old_log_probs, returns, advantages)
        success_pool_mask = np.asarray([x == "success" for x in outcome_pool], dtype=bool)
        success_selected_mask = np.asarray([x == "success" for x in selected_outcomes], dtype=bool)
        selected_advantages = np.asarray([x.advantage for x in consumed], dtype=np.float64)

        def _mean(values: list[int] | np.ndarray) -> float:
            return float(np.mean(values)) if len(values) else 0.0

        def _masked_mean_std(mask: np.ndarray) -> tuple[float, float]:
            values = selected_advantages[mask]
            if values.size == 0:
                return 0.0, 0.0
            return float(values.mean()), float(values.std())

        success_adv_mean, success_adv_std = _masked_mean_std(success_selected_mask)
        failure_adv_mean, failure_adv_std = _masked_mean_std(~success_selected_mask)
        success_lengths = outcome_trajectory_lengths["success"]
        failure_lengths = (
            outcome_trajectory_lengths["timeout"]
            + outcome_trajectory_lengths["both_dead"]
            + outcome_trajectory_lengths["other"]
        )
        gae_decay = float(self.cfg["ppo"]["gamma"]) * float(self.cfg["ppo"]["gae_lambda"])
        success_credit_weights = [
            gae_decay ** max(int(length) - 1, 0)
            for length in success_lengths
        ]

        metrics: dict[str, float | int] = {
            "samples": expected_samples,
            "simulator_steps": simulator_steps,
            "completed_team_episodes": completed_team_episodes,
            "team_success_episodes": outcome_episode_counts["success"],
            "timeout_episodes": outcome_episode_counts["timeout"],
            "both_dead_episodes": outcome_episode_counts["both_dead"],
            "target_deaths": target_deaths,
            "target_goal_contributions": target_goal_contributions,
            "delayed_team_credits": delayed_team_credits,
            "sample_pool_size": pool_size,
            "sample_pool_success_transitions": int(success_pool_mask.sum()),
            "sample_pool_failure_transitions": int((~success_pool_mask).sum()),
            "sample_pool_success_fraction": float(success_pool_mask.mean()) if pool_size else 0.0,
            "selected_success_transitions": int(success_selected_mask.sum()),
            "selected_failure_transitions": int((~success_selected_mask).sum()),
            "selected_success_fraction": float(success_selected_mask.mean()),
            "selected_unique_trajectories": len(set(selected_trajectory_ids)),
            "selected_success_trajectories": len({
                selected_trajectory_ids[i]
                for i in range(len(selected_trajectory_ids))
                if success_selected_mask[i]
            }),
            "selected_failure_trajectories": len({
                selected_trajectory_ids[i]
                for i in range(len(selected_trajectory_ids))
                if not success_selected_mask[i]
            }),
            "mean_success_target_trajectory_steps": _mean(success_lengths),
            "mean_failure_target_trajectory_steps": _mean(failure_lengths),
            "mean_success_terminal_credit_weight_at_start": _mean(success_credit_weights),
            "selected_success_advantage_mean": success_adv_mean,
            "selected_success_advantage_std": success_adv_std,
            "selected_failure_advantage_mean": failure_adv_mean,
            "selected_failure_advantage_std": failure_adv_std,
            "discarded_surplus_samples": discarded_surplus,
            "queued_surplus_samples": 0,
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


class SharedTwoRunnerCollector:
    """Fixed-horizon shared-world PPO collector for both Runner policies.

    Every round runs the same frozen joint policy for exactly rollout_steps in
    the same parallel worlds. Team episodes may end and reset inside the
    rollout. Episodes still alive at the horizon are truncated only for PPO
    bookkeeping and use critic bootstrap; the physical world state continues
    into the next round, where the newly committed policy immediately acts.
    """

    obs_dim = 18
    action_dim = 2

    def __init__(
        self,
        cfg: dict[str, Any],
        device: str = "cpu",
        env_factory: Callable[..., Any] = TwoRunnerArena2D,
    ) -> None:
        joint_cfg = cfg.get("joint_collection", {})
        if not bool(joint_cfg.get("enabled", False)):
            raise ValueError(
                "SharedTwoRunnerCollector requires joint_collection.enabled=true"
            )
        self.cfg = cfg
        self.device = torch.device(device)
        self.env_factory = env_factory
        self.parallel_envs = int(joint_cfg["parallel_envs"])
        self.rollout_steps = int(joint_cfg["rollout_steps"])
        if self.parallel_envs <= 0:
            raise ValueError("joint_collection.parallel_envs must be positive")
        if self.rollout_steps <= 0:
            raise ValueError("joint_collection.rollout_steps must be positive")

        nominal = int(cfg["training"]["samples_per_update"])
        expected_world_steps = self.parallel_envs * self.rollout_steps
        if nominal != expected_world_steps:
            raise ValueError(
                "training.samples_per_update must equal "
                "joint_collection.parallel_envs * joint_collection.rollout_steps "
                f"({expected_world_steps}) in fixed-horizon mode"
            )

        self._collector_env: Any | None = None
        self._collector_obs: np.ndarray | None = None
        self.pending_trajectories: dict[str, list[list[TargetTransition]]] = {
            aid: [] for aid in TWO_RUNNER_IDS
        }

    def _ensure_collector(self) -> None:
        if self._collector_env is not None:
            return
        base_seed = int(self.cfg.get("seed", 0))
        self._collector_env = self.env_factory(
            self.parallel_envs,
            self.cfg["environment"],
            self.cfg["two_runner_reward"],
            seed=base_seed,
        )
        self._collector_obs = self._collector_env.observe()
        self.pending_trajectories = {
            aid: [[] for _ in range(self.parallel_envs)]
            for aid in TWO_RUNNER_IDS
        }

    def _models_from_policy_set(
        self,
        policy_set: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, ActorCritic]:
        if set(policy_set) != set(TWO_RUNNER_IDS):
            raise ValueError(
                f"policy_set must contain exactly {TWO_RUNNER_IDS}"
            )
        hidden = self.cfg["ppo"]["hidden_sizes"]
        models: dict[str, ActorCritic] = {}
        for aid in TWO_RUNNER_IDS:
            model = ActorCritic(
                self.obs_dim, self.action_dim, hidden
            ).to(self.device)
            load_model_snapshot(model, policy_set[aid])
            model.eval()
            models[aid] = model
        return models

    @staticmethod
    def _team_outcome(info: dict[str, Any], env_index: int) -> str:
        if bool(np.asarray(info["team_success"])[env_index]):
            return "success"
        if bool(np.asarray(info["both_dead"])[env_index]):
            return "both_dead"
        if bool(np.asarray(info["timeout"])[env_index]):
            return "timeout"
        return "other"

    def collect_round(
        self,
        frozen_policy_set: dict[str, dict[str, torch.Tensor]],
        round_index: int,
    ) -> tuple[
        dict[str, RolloutBatch],
        dict[str, dict[str, float | int]],
    ]:
        self._ensure_collector()
        assert self._collector_env is not None
        assert self._collector_obs is not None
        env = self._collector_env

        if any(
            any(self.pending_trajectories[aid])
            for aid in TWO_RUNNER_IDS
        ):
            raise RuntimeError(
                "fixed-horizon collector crossed a round boundary "
                "with an unfinished PPO fragment"
            )

        models = self._models_from_policy_set(frozen_policy_set)
        round_seed = (
            int(self.cfg.get("seed", 0))
            + int(round_index) * 1_000_000
        )
        torch.manual_seed(round_seed)

        finalized: dict[str, list[FinalizedSample]] = {
            aid: [] for aid in TWO_RUNNER_IDS
        }
        finalized_outcomes: dict[str, list[str]] = {
            aid: [] for aid in TWO_RUNNER_IDS
        }
        finalized_trajectory_ids: dict[str, list[int]] = {
            aid: [] for aid in TWO_RUNNER_IDS
        }

        completed_team_episodes = 0
        outcome_episode_counts = {
            "success": 0,
            "timeout": 0,
            "both_dead": 0,
            "other": 0,
        }
        target_deaths = {aid: 0 for aid in TWO_RUNNER_IDS}
        target_goal_contributions = {
            aid: 0 for aid in TWO_RUNNER_IDS
        }
        horizon_bootstrap_fragments = {
            aid: 0 for aid in TWO_RUNNER_IDS
        }
        terminal_fragments = {aid: 0 for aid in TWO_RUNNER_IDS}
        trajectory_lengths: dict[str, dict[str, list[int]]] = {
            aid: {
                "success": [],
                "timeout": [],
                "both_dead": [],
                "death": [],
                "other": [],
                "horizon": [],
            }
            for aid in TWO_RUNNER_IDS
        }
        unique_map_seeds: set[int] = set()
        trajectory_serial = 0

        def finalize_fragment(
            aid: str,
            env_index: int,
            *,
            terminal: bool,
            outcome: str,
        ) -> None:
            nonlocal trajectory_serial
            trajectory = self.pending_trajectories[aid][env_index]
            if not trajectory:
                return
            samples = finalize_target_fragment(
                trajectory,
                terminal=terminal,
                gamma=float(self.cfg["ppo"]["gamma"]),
                gae_lambda=float(self.cfg["ppo"]["gae_lambda"]),
            )
            finalized[aid].extend(samples)
            finalized_outcomes[aid].extend(
                [outcome] * len(samples)
            )
            finalized_trajectory_ids[aid].extend(
                [trajectory_serial] * len(samples)
            )
            trajectory_lengths[aid][outcome].append(
                len(trajectory)
            )
            if terminal:
                terminal_fragments[aid] += 1
            else:
                horizon_bootstrap_fragments[aid] += 1
            self.pending_trajectories[aid][env_index] = []
            trajectory_serial += 1

        for _rollout_step in range(self.rollout_steps):
            obs = self._collector_obs
            alive_before = np.asarray(env.alive, dtype=bool).copy()
            n_envs = int(env.num_envs)

            all_actions = np.zeros(
                (n_envs, 2, 2), dtype=np.float32
            )
            action_cache = {
                aid: np.zeros(
                    (n_envs, 2), dtype=np.float32
                )
                for aid in TWO_RUNNER_IDS
            }
            log_prob_cache = {
                aid: np.zeros(n_envs, dtype=np.float32)
                for aid in TWO_RUNNER_IDS
            }
            value_cache = {
                aid: np.zeros(n_envs, dtype=np.float32)
                for aid in TWO_RUNNER_IDS
            }

            for agent_index, aid in enumerate(TWO_RUNNER_IDS):
                alive_mask = alive_before[:, agent_index]
                if not np.any(alive_mask):
                    continue
                obs_tensor = torch.from_numpy(
                    obs[alive_mask, agent_index, :]
                ).to(self.device)
                with torch.no_grad():
                    action, log_prob, value = models[aid].sample(
                        obs_tensor
                    )
                action_np = action.cpu().numpy()
                all_actions[
                    alive_mask, agent_index, :
                ] = action_np
                action_cache[aid][alive_mask] = action_np
                log_prob_cache[aid][
                    alive_mask
                ] = log_prob.cpu().numpy()
                value_cache[aid][
                    alive_mask
                ] = value.cpu().numpy()

            next_obs, rewards, done, info = env.step(all_actions)
            unique_map_seeds.update(
                int(x)
                for x in np.asarray(
                    info.get("map_seed", env.map_seeds)
                ).tolist()
            )

            next_value_cache: dict[str, np.ndarray] = {}
            with torch.no_grad():
                for agent_index, aid in enumerate(
                    TWO_RUNNER_IDS
                ):
                    next_value_cache[aid] = (
                        models[aid]
                        .value(
                            torch.from_numpy(
                                next_obs[:, agent_index, :]
                            ).to(self.device)
                        )
                        .cpu()
                        .numpy()
                    )

            alive_before_info = np.asarray(
                info["alive_before"], dtype=bool
            )
            new_death = np.asarray(
                info["new_death"], dtype=bool
            )
            goal_reached = np.asarray(
                info["goal_reached"], dtype=bool
            )

            for agent_index, aid in enumerate(TWO_RUNNER_IDS):
                actor_mask = alive_before_info[:, agent_index]
                for env_index in np.flatnonzero(actor_mask):
                    env_index = int(env_index)
                    self.pending_trajectories[aid][
                        env_index
                    ].append(
                        TargetTransition(
                            observation=obs[
                                env_index, agent_index, :
                            ],
                            action=action_cache[aid][env_index],
                            old_log_prob=float(
                                log_prob_cache[aid][env_index]
                            ),
                            reward=float(
                                rewards[
                                    env_index, agent_index
                                ]
                            ),
                            value=float(
                                value_cache[aid][env_index]
                            ),
                            next_value=float(
                                next_value_cache[aid][
                                    env_index
                                ]
                            ),
                        )
                    )

                target_deaths[aid] += int(
                    new_death[:, agent_index].sum()
                )
                target_goal_contributions[aid] += int(
                    goal_reached[:, agent_index].sum()
                )

            # A dead Runner's policy trajectory ends at its collision
            # transition. We deliberately do not add later teammate-success
            # credit to an action sequence that has already terminated.
            for agent_index, aid in enumerate(TWO_RUNNER_IDS):
                for env_index in np.flatnonzero(
                    new_death[:, agent_index]
                ):
                    finalize_fragment(
                        aid,
                        int(env_index),
                        terminal=True,
                        outcome="death",
                    )

            effective_done = np.asarray(done, dtype=bool)
            for env_index in np.flatnonzero(effective_done):
                env_index = int(env_index)
                completed_team_episodes += 1
                outcome = self._team_outcome(
                    info, env_index
                )
                outcome_episode_counts[outcome] += 1
                for aid in TWO_RUNNER_IDS:
                    finalize_fragment(
                        aid,
                        env_index,
                        terminal=True,
                        outcome=outcome,
                    )

            if np.any(effective_done):
                env.reset_indices(effective_done)
                next_obs = env.observe()

            self._collector_obs = next_obs

        # The fixed-horizon cutoff is not an environment terminal. Bootstrap
        # every still-active policy fragment from V(s_{t+1}), clear only the
        # PPO fragment, and keep the physical world state for the next policy.
        for aid in TWO_RUNNER_IDS:
            for env_index in range(int(env.num_envs)):
                finalize_fragment(
                    aid,
                    env_index,
                    terminal=False,
                    outcome="horizon",
                )

        batches: dict[str, RolloutBatch] = {}
        metrics: dict[str, dict[str, float | int]] = {}
        expected_world_transitions = (
            self.parallel_envs * self.rollout_steps
        )

        for aid in TWO_RUNNER_IDS:
            samples = finalized[aid]
            if not samples:
                raise RuntimeError(
                    f"fixed-horizon rollout produced no {aid} actor samples"
                )

            observations = torch.from_numpy(
                np.stack([x.observation for x in samples])
            ).to(self.device)
            actions = torch.from_numpy(
                np.stack([x.action for x in samples])
            ).to(self.device)
            old_log_probs = torch.tensor(
                [x.old_log_prob for x in samples],
                dtype=torch.float32,
                device=self.device,
            )
            returns = torch.tensor(
                [x.return_value for x in samples],
                dtype=torch.float32,
                device=self.device,
            )
            advantages = torch.tensor(
                [x.advantage for x in samples],
                dtype=torch.float32,
                device=self.device,
            )
            batches[aid] = RolloutBatch(
                observations,
                actions,
                old_log_probs,
                returns,
                advantages,
            )

            outcomes = finalized_outcomes[aid]
            trajectory_ids = finalized_trajectory_ids[aid]
            success_mask = np.asarray(
                [x == "success" for x in outcomes],
                dtype=bool,
            )
            selected_advantages = np.asarray(
                [x.advantage for x in samples],
                dtype=np.float64,
            )

            def masked_stats(
                mask: np.ndarray,
            ) -> tuple[float, float]:
                values = selected_advantages[mask]
                if values.size == 0:
                    return 0.0, 0.0
                return (
                    float(values.mean()),
                    float(values.std()),
                )

            success_adv_mean, success_adv_std = (
                masked_stats(success_mask)
            )
            failure_adv_mean, failure_adv_std = (
                masked_stats(~success_mask)
            )
            success_lengths = trajectory_lengths[aid][
                "success"
            ]
            failure_lengths = (
                trajectory_lengths[aid]["timeout"]
                + trajectory_lengths[aid]["both_dead"]
                + trajectory_lengths[aid]["death"]
                + trajectory_lengths[aid]["other"]
            )
            gae_decay = float(
                self.cfg["ppo"]["gamma"]
            ) * float(self.cfg["ppo"]["gae_lambda"])
            success_credit_weights = [
                gae_decay ** max(int(length) - 1, 0)
                for length in success_lengths
            ]

            def mean_or_zero(values: list[int]) -> float:
                return (
                    float(np.mean(values))
                    if values
                    else 0.0
                )

            sample_count = len(samples)
            metrics[aid] = {
                "samples": sample_count,
                "nominal_world_transitions": expected_world_transitions,
                "simulator_steps": expected_world_transitions,
                "rollout_steps": self.rollout_steps,
                "completed_team_episodes": completed_team_episodes,
                "shared_team_episodes": completed_team_episodes,
                "shared_joint_rollout": 1,
                "rollout_policy_version": int(round_index),
                "ppo_data_epochs": int(
                    self.cfg["ppo"]["epochs"]
                ),
                "team_success_episodes": outcome_episode_counts[
                    "success"
                ],
                "timeout_episodes": outcome_episode_counts[
                    "timeout"
                ],
                "both_dead_episodes": outcome_episode_counts[
                    "both_dead"
                ],
                "target_deaths": target_deaths[aid],
                "target_goal_contributions": (
                    target_goal_contributions[aid]
                ),
                "delayed_team_credits": 0,
                "horizon_bootstrap_fragments": (
                    horizon_bootstrap_fragments[aid]
                ),
                "terminal_fragments": terminal_fragments[aid],
                "sample_pool_size": sample_count,
                "sample_pool_success_transitions": int(
                    success_mask.sum()
                ),
                "sample_pool_failure_transitions": int(
                    (~success_mask).sum()
                ),
                "sample_pool_success_fraction": (
                    float(success_mask.mean())
                    if sample_count
                    else 0.0
                ),
                "selected_success_transitions": int(
                    success_mask.sum()
                ),
                "selected_failure_transitions": int(
                    (~success_mask).sum()
                ),
                "selected_success_fraction": (
                    float(success_mask.mean())
                    if sample_count
                    else 0.0
                ),
                "selected_unique_trajectories": len(
                    set(trajectory_ids)
                ),
                "selected_success_trajectories": len(
                    {
                        trajectory_ids[i]
                        for i in range(len(trajectory_ids))
                        if success_mask[i]
                    }
                ),
                "selected_failure_trajectories": len(
                    {
                        trajectory_ids[i]
                        for i in range(len(trajectory_ids))
                        if not success_mask[i]
                    }
                ),
                "mean_success_target_trajectory_steps": (
                    mean_or_zero(success_lengths)
                ),
                "mean_failure_target_trajectory_steps": (
                    mean_or_zero(failure_lengths)
                ),
                "mean_success_terminal_credit_weight_at_start": (
                    mean_or_zero(success_credit_weights)
                ),
                "selected_success_advantage_mean": (
                    success_adv_mean
                ),
                "selected_success_advantage_std": (
                    success_adv_std
                ),
                "selected_failure_advantage_mean": (
                    failure_adv_mean
                ),
                "selected_failure_advantage_std": (
                    failure_adv_std
                ),
                "discarded_surplus_samples": 0,
                "queued_surplus_samples": 0,
                "unique_map_seeds": len(unique_map_seeds),
                "stochastic_action_std": float(
                    actions.detach().cpu().numpy().std()
                ),
            }

        return batches, metrics

    def snapshot_state(self) -> dict[str, Any]:
        self._ensure_collector()
        assert self._collector_env is not None
        assert self._collector_obs is not None
        if any(
            any(self.pending_trajectories[aid])
            for aid in TWO_RUNNER_IDS
        ):
            raise RuntimeError(
                "cannot checkpoint shared collector with unfinished PPO fragment"
            )
        if not hasattr(
            self._collector_env, "snapshot_state"
        ):
            raise ValueError(
                "shared collector environment does not support snapshot_state"
            )
        return {
            "collector_state": self._collector_env.snapshot_state(),
            "collector_obs": self._collector_obs.copy(),
            "pending_trajectories": copy.deepcopy(
                self.pending_trajectories
            ),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self._ensure_collector()
        assert self._collector_env is not None
        if not hasattr(
            self._collector_env, "restore_state"
        ):
            raise ValueError(
                "shared collector environment does not support restore_state"
            )
        self._collector_env.restore_state(
            state["collector_state"]
        )
        self._collector_obs = np.asarray(
            state["collector_obs"], dtype=np.float32
        ).copy()
        self.pending_trajectories = copy.deepcopy(
            state.get(
                "pending_trajectories",
                {
                    aid: [
                        [] for _ in range(self.parallel_envs)
                    ]
                    for aid in TWO_RUNNER_IDS
                },
            )
        )


# --- Experiment 2 trainer / checkpoint / evaluation ---
import copy
import json
from pathlib import Path

from .coordinator import SynchronousCoordinator
from .exchange import MockPolicyExchange
from .policy import expand_observation_snapshot


def _clone_pending(pending):
    return copy.deepcopy(pending)


def _worker_snapshot_training_state(worker: TwoRunnerWorker) -> dict[str, Any]:
    worker._ensure_collector()
    assert worker._collector_env is not None and worker._collector_obs is not None
    if not hasattr(worker._collector_env, "snapshot_state"):
        raise ValueError("collector environment does not support snapshot_state")
    return {
        "optimizer": copy.deepcopy(worker.optimizer.state_dict()),
        "collector_state": worker._collector_env.snapshot_state(),
        "collector_obs": worker._collector_obs.copy(),
        "pending_trajectories": _clone_pending(worker.pending_trajectories),
        "finalized_queue": copy.deepcopy(list(worker.finalized_queue)),
        "finalized_outcomes": list(worker.finalized_outcomes),
        "finalized_trajectory_ids": list(worker.finalized_trajectory_ids),
    }


def _worker_restore_training_state(worker: TwoRunnerWorker, state: dict[str, Any]) -> None:
    worker.optimizer.load_state_dict(state["optimizer"])
    worker._ensure_collector()
    assert worker._collector_env is not None
    if not hasattr(worker._collector_env, "restore_state"):
        raise ValueError("collector environment does not support restore_state")
    worker._collector_env.restore_state(state["collector_state"])
    worker._collector_obs = np.asarray(state["collector_obs"], dtype=np.float32).copy()
    worker.pending_trajectories = _clone_pending(state["pending_trajectories"])
    worker.finalized_queue = deque(copy.deepcopy(state["finalized_queue"]))
    worker.finalized_outcomes = deque(copy.deepcopy(state.get("finalized_outcomes", [])))
    worker.finalized_trajectory_ids = deque(copy.deepcopy(state.get("finalized_trajectory_ids", [])))


def two_runner_validation_is_better(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    collision_limit: float = 0.10,
) -> bool:
    if best is None:
        return True
    limit = float(collision_limit)
    c_safe = float(candidate["any_collision_rate"]) <= limit
    b_safe = float(best["any_collision_rate"]) <= limit
    if c_safe != b_safe:
        return c_safe
    if c_safe:
        c_key = (
            float(candidate["team_success_rate"]),
            -float(candidate["both_dead_rate"]),
            float(candidate["mean_team_episode_reward"]),
        )
        b_key = (
            float(best["team_success_rate"]),
            -float(best["both_dead_rate"]),
            float(best["mean_team_episode_reward"]),
        )
        return c_key > b_key
    c_key = (
        -float(candidate["any_collision_rate"]),
        float(candidate["team_success_rate"]),
        -float(candidate["both_dead_rate"]),
        float(candidate["mean_team_episode_reward"]),
    )
    b_key = (
        -float(best["any_collision_rate"]),
        float(best["team_success_rate"]),
        -float(best["both_dead_rate"]),
        float(best["mean_team_episode_reward"]),
    )
    return c_key > b_key


def evaluate_two_runner(
    cfg: dict[str, Any],
    policy_set: dict[str, dict[str, torch.Tensor]],
    episodes: int,
    seed_start: int,
    device: str = "cpu",
) -> dict[str, Any]:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    dev = torch.device(device)
    hidden = cfg["ppo"]["hidden_sizes"]
    models: dict[str, ActorCritic] = {}
    for aid in TWO_RUNNER_IDS:
        model = ActorCritic(18, 2, hidden).to(dev)
        load_model_snapshot(model, policy_set[aid])
        model.eval()
        models[aid] = model

    successes = timeouts = both_dead_count = any_collision_count = 0
    deaths = np.zeros(2, dtype=np.int64)
    goals = np.zeros(2, dtype=np.int64)
    prior_death_success = 0
    goal_times: list[float] = []
    episode_rewards = np.zeros((episodes, 2), dtype=np.float64)
    min_clearances = np.full((episodes, 2), np.inf, dtype=np.float64)
    efficiencies: list[float] = []
    seeds: list[int] = []

    for episode in range(episodes):
        seed = int(seed_start) + episode
        seeds.append(seed)
        env = TwoRunnerArena2D(1, cfg["environment"], cfg["two_runner_reward"], seed=seed)
        obs = env.observe()
        start_positions = env.state[0, :, :2].copy()
        path_lengths = np.zeros(2, dtype=np.float64)
        ever_dead = np.zeros(2, dtype=bool)
        ep_any_collision = False

        while True:
            alive_before = env.alive[0].copy()
            before = env.state[0, :, :2].copy()
            actions = np.zeros((1, 2, 2), dtype=np.float32)
            for idx, aid in enumerate(TWO_RUNNER_IDS):
                if not alive_before[idx]:
                    continue
                with torch.no_grad():
                    act = models[aid].deterministic(torch.from_numpy(obs[:, idx, :]).to(dev))
                actions[:, idx, :] = act.cpu().numpy()
            next_obs, rewards, done, info = env.step(actions)
            after = env.state[0, :, :2].copy()
            for idx in range(2):
                if alive_before[idx]:
                    path_lengths[idx] += float(np.linalg.norm(after[idx] - before[idx]))
            episode_rewards[episode] += rewards[0]
            min_clearances[episode] = np.minimum(min_clearances[episode], info["min_clearance"][0])
            new_death = info["new_death"][0]
            deaths += new_death.astype(np.int64)
            ever_dead |= new_death
            ep_any_collision = ep_any_collision or bool(new_death.any())
            goals += info["goal_reached"][0].astype(np.int64)
            obs = next_obs

            if bool(done[0]):
                success = bool(info["team_success"][0])
                if success:
                    successes += 1
                    if ever_dead.any():
                        prior_death_success += 1
                        bonus = float(cfg["two_runner_reward"]["team_success_bonus"])
                        for idx in range(2):
                            if ever_dead[idx] and not bool(info["alive_before"][0, idx]):
                                episode_rewards[episode, idx] += bonus
                    goal_times.append(float(env.steps[0]) * float(env.dt))
                    winner_indices = np.flatnonzero(info["goal_reached"][0])
                    if winner_indices.size:
                        winner = int(winner_indices[0])
                        direct = float(np.linalg.norm(env.goal - start_positions[winner]))
                        if path_lengths[winner] > 0:
                            efficiencies.append(float(np.clip(direct / path_lengths[winner], 0.0, 1.0)))
                elif bool(info["both_dead"][0]):
                    both_dead_count += 1
                elif bool(info["timeout"][0]):
                    timeouts += 1
                if ep_any_collision:
                    any_collision_count += 1
                break

    mean_team_reward = float(np.mean(episode_rewards.mean(axis=1)))
    return {
        "episodes": episodes,
        "seeds": seeds,
        "team_successes": int(successes),
        "team_success_rate": successes / episodes,
        "timeouts": int(timeouts),
        "timeout_rate": timeouts / episodes,
        "both_dead_count": int(both_dead_count),
        "both_dead_rate": both_dead_count / episodes,
        "any_collision_count": int(any_collision_count),
        "any_collision_rate": any_collision_count / episodes,
        "runner_0_death_rate": float(deaths[0] / episodes),
        "runner_1_death_rate": float(deaths[1] / episodes),
        "runner_0_goal_count": int(goals[0]),
        "runner_1_goal_count": int(goals[1]),
        "team_success_with_prior_death_count": int(prior_death_success),
        "team_success_with_prior_death_rate": prior_death_success / episodes,
        "mean_time_to_team_goal_s": float(np.mean(goal_times)) if goal_times else None,
        "mean_episode_reward_runner_0": float(np.mean(episode_rewards[:, 0])),
        "mean_episode_reward_runner_1": float(np.mean(episode_rewards[:, 1])),
        "mean_team_episode_reward": mean_team_reward,
        "mean_min_clearance_runner_0_m": float(np.mean(min_clearances[:, 0])),
        "mean_min_clearance_runner_1_m": float(np.mean(min_clearances[:, 1])),
        "mean_winner_path_efficiency": float(np.mean(efficiencies)) if efficiencies else None,
    }


class TwoRunnerTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: str | Path,
        device: str = "cpu",
        worker_env_factory: Callable[..., Any] = TwoRunnerArena2D,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.worker_env_factory = worker_env_factory
        seed = int(cfg.get("seed", 0))
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.workers = {
            aid: TwoRunnerWorker(aid, cfg, device=device, env_factory=worker_env_factory)
            for aid in TWO_RUNNER_IDS
        }
        joint_cfg = cfg.get("joint_collection", {})
        self.shared_collector: SharedTwoRunnerCollector | None = None
        if bool(joint_cfg.get("enabled", False)):
            self.shared_collector = SharedTwoRunnerCollector(
                cfg,
                device=device,
                env_factory=worker_env_factory,
            )
        initial = {aid: snapshot_model(self.workers[aid].model) for aid in TWO_RUNNER_IDS}
        self.exchange = MockPolicyExchange(initial, version=0, agent_ids=TWO_RUNNER_IDS)
        self.coordinator = SynchronousCoordinator(self.exchange)
        self.best_validation: dict[str, Any] | None = None
        self.metrics_path = self.output_dir / "metrics.jsonl"

    @property
    def current_round(self) -> int:
        return self.coordinator.current_round

    def policy_set(self):
        return self.exchange.get_committed_policy_set()

    def initialize_from_single_runner_checkpoint(self, path: str | Path) -> None:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(data.get("obs_dim", 15)) != 15 or int(data.get("action_dim", 2)) != 2:
            raise ValueError("single-runner checkpoint dimensions are incompatible")
        expanded = expand_observation_snapshot(data["model"], old_obs_dim=15, new_obs_dim=18)
        initial = {}
        for aid in TWO_RUNNER_IDS:
            load_model_snapshot(self.workers[aid].model, expanded)
            # Keep the fresh optimizer intentionally; Experiment 1 Adam state is not reused.
            initial[aid] = snapshot_model(self.workers[aid].model)
        self.exchange = MockPolicyExchange(initial, version=0, agent_ids=TWO_RUNNER_IDS)
        self.coordinator = SynchronousCoordinator(self.exchange)
        self.best_validation = None

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        checkpoint = Path(path) if path is not None else self.output_dir / "latest.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if self.shared_collector is None:
            worker_states = {
                aid: _worker_snapshot_training_state(self.workers[aid])
                for aid in TWO_RUNNER_IDS
            }
            shared_state = None
            training_state_version = 1
            collection_mode = "independent"
        else:
            worker_states = {
                aid: {
                    "optimizer": copy.deepcopy(
                        self.workers[aid].optimizer.state_dict()
                    )
                }
                for aid in TWO_RUNNER_IDS
            }
            shared_state = self.shared_collector.snapshot_state()
            training_state_version = 2
            collection_mode = "shared_joint"

        torch.save(
            {
                "version": self.current_round,
                "policy_set": self.policy_set(),
                "config": self.cfg,
                "obs_dim": 18,
                "action_dim": 2,
                "worker_states": worker_states,
                "shared_collector_state": shared_state,
                "collection_mode": collection_mode,
                "best_validation": copy.deepcopy(self.best_validation),
                "training_state_version": training_state_version,
            },
            checkpoint,
        )
        return checkpoint

    def resume_from_checkpoint(self, path: str | Path) -> str:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(data.get("obs_dim", 18)) != 18 or int(data.get("action_dim", 2)) != 2:
            raise ValueError("two-runner checkpoint dimensions do not match")
        policies = data["policy_set"]
        version = int(data["version"])
        self.exchange = MockPolicyExchange(policies, version=version, agent_ids=TWO_RUNNER_IDS)
        self.coordinator = SynchronousCoordinator(self.exchange)
        for aid in TWO_RUNNER_IDS:
            load_model_snapshot(self.workers[aid].model, policies[aid])
        if int(data.get("training_state_version", 0)) < 1 or "worker_states" not in data:
            return "legacy"

        state_version = int(data.get("training_state_version", 0))
        collection_mode = str(data.get("collection_mode", "independent"))
        if state_version >= 2 and collection_mode == "shared_joint":
            if self.shared_collector is None:
                raise ValueError(
                    "checkpoint uses shared_joint collection but current config does not"
                )
            for aid in TWO_RUNNER_IDS:
                self.workers[aid].optimizer.load_state_dict(
                    data["worker_states"][aid]["optimizer"]
                )
            shared_state = data.get("shared_collector_state")
            if shared_state is None:
                raise ValueError("shared_joint checkpoint missing shared_collector_state")
            self.shared_collector.restore_state(shared_state)
        else:
            if self.shared_collector is not None:
                raise ValueError(
                    "independent-collector checkpoint cannot resume into shared_joint mode"
                )
            for aid in TWO_RUNNER_IDS:
                _worker_restore_training_state(
                    self.workers[aid], data["worker_states"][aid]
                )
        self.best_validation = copy.deepcopy(data.get("best_validation"))
        return "full"

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, rounds: int | None = None, *, run_validation: bool = True) -> list[dict[str, Any]]:
        if rounds is None:
            rounds = int(self.cfg["training"]["rounds"])
        records: list[dict[str, Any]] = []
        checkpoint_every = max(1, int(self.cfg["training"].get("checkpoint_every", 1)))
        for _ in range(int(rounds)):
            current = self.current_round
            next_version = current + 1
            frozen = self.policy_set()
            round_metrics: dict[str, dict[str, Any]] = {}
            if self.shared_collector is None:
                for aid in TWO_RUNNER_IDS:
                    snapshot, metrics = self.workers[aid].train_round(
                        frozen, current
                    )
                    round_metrics[aid] = metrics
                    self.coordinator.submit_update(
                        aid, next_version, snapshot, metrics
                    )
            else:
                # One joint rollout is collected under exactly one frozen
                # policy version. Both agents then update from that same set of
                # team episodes, commit together, and the next round recollects
                # under the newly committed joint policy.
                batches, collection_metrics = self.shared_collector.collect_round(
                    frozen, current
                )
                for aid in TWO_RUNNER_IDS:
                    worker = self.workers[aid]
                    load_model_snapshot(worker.model, frozen[aid])
                    worker.model.train()
                    update_metrics = ppo_update(
                        worker.model,
                        worker.optimizer,
                        batches[aid],
                        self.cfg["ppo"],
                    )
                    metrics = {
                        **collection_metrics[aid],
                        **update_metrics,
                    }
                    snapshot = snapshot_model(worker.model)
                    round_metrics[aid] = metrics
                    self.coordinator.submit_update(
                        aid, next_version, snapshot, metrics
                    )
            if not self.coordinator.try_commit(next_version):
                raise RuntimeError(f"Two-runner synchronous commit failed: {self.coordinator.ready_status(next_version)}")
            record: dict[str, Any] = {"round": next_version, "agents": round_metrics}

            validation_cfg = self.cfg.get("validation") if run_validation else None
            if validation_cfg and next_version % int(validation_cfg["every"]) == 0:
                summary = evaluate_two_runner(
                    self.cfg,
                    self.policy_set(),
                    episodes=int(validation_cfg["episodes"]),
                    seed_start=int(validation_cfg["seed_start"]),
                    device=self.device,
                )
                record["validation"] = summary
                candidate = {
                    "round": next_version,
                    "team_success_rate": float(summary["team_success_rate"]),
                    "any_collision_rate": float(summary["any_collision_rate"]),
                    "both_dead_rate": float(summary["both_dead_rate"]),
                    "mean_team_episode_reward": float(summary["mean_team_episode_reward"]),
                }
                if two_runner_validation_is_better(candidate, self.best_validation):
                    self.best_validation = candidate
                    self.save_checkpoint(self.output_dir / "best.pt")

            records.append(record)
            self._append_metrics(record)
            self.save_checkpoint(self.output_dir / "latest.pt")
            if next_version % checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / f"round_{next_version:05d}.pt")
        return records

def load_two_runner_checkpoint(path: str | Path):
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return int(data["version"]), data["policy_set"], data["config"]
