from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .policy import ActorCritic


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: dict[str, Any],
) -> dict[str, float]:
    observations = batch.observations.float()
    actions = batch.actions.float()
    old_log_probs = batch.old_log_probs.float()
    returns = batch.returns.float()
    advantages = batch.advantages.float()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    clip_range = float(cfg["clip_range"])
    value_coef = float(cfg["value_coef"])
    entropy_coef = float(cfg["entropy_coef"])
    max_grad_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg["epochs"])
    minibatch_size = int(cfg["minibatch_size"])
    target_kl_raw = cfg.get("target_kl")
    target_kl = None if target_kl_raw is None else float(target_kl_raw)
    if target_kl is not None and target_kl <= 0.0:
        raise ValueError("ppo.target_kl must be > 0 when configured")
    n = observations.shape[0]

    metric_acc = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
    guard_kl_checks: list[float] = []
    optimizer_steps = 0
    epochs_completed = 0
    early_stopped = False

    for _ in range(epochs):
        order = torch.randperm(n, device=observations.device)
        completed_epoch = True
        for start in range(0, n, minibatch_size):
            idx = order[start : start + minibatch_size]
            new_log_prob, entropy, values = model.evaluate_actions(observations[idx], actions[idx])
            log_ratio = new_log_prob - old_log_probs[idx]
            ratio = log_ratio.exp()

            # Keep the historical signed first-order KL metric for continuity,
            # but use the non-negative second-order approximation for the
            # actual trust-region guard. The check happens BEFORE applying this
            # minibatch update, so the first minibatch that exceeds target_kl
            # is not optimized.
            with torch.no_grad():
                approx_kl = (old_log_probs[idx] - new_log_prob).mean()
                guard_kl = ((ratio - 1.0) - log_ratio).mean()
                guard_kl_value = float(guard_kl.detach().cpu())
            guard_kl_checks.append(guard_kl_value)
            if target_kl is not None and guard_kl_value > target_kl:
                early_stopped = True
                completed_epoch = False
                break

            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages[idx]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[idx] - values).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer_steps += 1

            metric_acc["policy_loss"].append(float(policy_loss.detach().cpu()))
            metric_acc["value_loss"].append(float(value_loss.detach().cpu()))
            metric_acc["entropy"].append(float(entropy_mean.detach().cpu()))
            metric_acc["approx_kl"].append(float(approx_kl.detach().cpu()))

        if completed_epoch:
            epochs_completed += 1
        if early_stopped:
            break

    result = {name: float(np.mean(values)) for name, values in metric_acc.items()}
    result.update(
        {
            "guard_kl": float(np.mean(guard_kl_checks)) if guard_kl_checks else 0.0,
            "max_guard_kl": float(max(guard_kl_checks, default=0.0)),
            "target_kl": float(target_kl) if target_kl is not None else 0.0,
            "optimizer_steps": float(optimizer_steps),
            "epochs_completed": float(epochs_completed),
            "early_stopped": float(early_stopped),
        }
    )
    return result
