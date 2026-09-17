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
    n = observations.shape[0]

    metric_acc = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(epochs):
        order = torch.randperm(n, device=observations.device)
        for start in range(0, n, minibatch_size):
            idx = order[start : start + minibatch_size]
            new_log_prob, entropy, values = model.evaluate_actions(observations[idx], actions[idx])
            log_ratio = new_log_prob - old_log_probs[idx]
            ratio = log_ratio.exp()
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

            with torch.no_grad():
                approx_kl = (old_log_probs[idx] - new_log_prob).mean()
            metric_acc["policy_loss"].append(float(policy_loss.detach().cpu()))
            metric_acc["value_loss"].append(float(value_loss.detach().cpu()))
            metric_acc["entropy"].append(float(entropy_mean.detach().cpu()))
            metric_acc["approx_kl"].append(float(approx_kl.detach().cpu()))

    return {name: float(np.mean(values)) for name, values in metric_acc.items()}
