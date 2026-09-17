from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        sizes = [int(obs_dim), *[int(x) for x in hidden_sizes]]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.Tanh()])
        self.backbone = nn.Sequential(*layers)
        last_dim = sizes[-1]
        self.actor_mean = nn.Linear(last_dim, int(action_dim))
        self.critic = nn.Linear(last_dim, 1)
        self.log_std = nn.Parameter(torch.full((int(action_dim),), -0.5))

    def _distribution_and_value(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), self.critic(features).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._distribution_and_value(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = self._squashed_log_prob(dist, raw, action)
        return action, log_prob, value

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        dist, _ = self._distribution_and_value(obs)
        return torch.tanh(dist.mean)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self._distribution_and_value(obs)
        return value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._distribution_and_value(obs)
        action = actions.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        log_prob = self._squashed_log_prob(dist, raw, action)
        # Exact tanh-policy entropy has no simple closed form; Normal entropy is a stable PPO bonus proxy.
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def snapshot_model(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())


def load_model_snapshot(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(snapshot, strict=True)


def expand_observation_snapshot(
    snapshot: dict[str, torch.Tensor],
    old_obs_dim: int = 15,
    new_obs_dim: int = 18,
) -> OrderedDict[str, torch.Tensor]:
    """Expand the first ActorCritic input layer without changing learned behavior."""
    old_obs_dim = int(old_obs_dim)
    new_obs_dim = int(new_obs_dim)
    if new_obs_dim < old_obs_dim:
        raise ValueError("new_obs_dim must be >= old_obs_dim")
    result = OrderedDict((key, value.detach().cpu().clone()) for key, value in snapshot.items())
    key = "backbone.0.weight"
    if key not in result:
        raise ValueError(f"snapshot missing {key}")
    weight = result[key]
    if weight.ndim != 2 or int(weight.shape[1]) != old_obs_dim:
        raise ValueError(
            f"expected {key} input dimension {old_obs_dim}, got shape {tuple(weight.shape)}"
        )
    expanded = torch.zeros(
        (int(weight.shape[0]), new_obs_dim),
        dtype=weight.dtype,
        device=weight.device,
    )
    expanded[:, :old_obs_dim] = weight
    result[key] = expanded
    return result
