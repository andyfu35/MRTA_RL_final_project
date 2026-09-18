from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

REWARD_API_VERSION = 1


@dataclass(frozen=True)
class RewardContext:
    """All reward-visible signals for one vectorized environment step.

    Shapes:
      - per-agent arrays: [num_envs, num_agents]
      - state arrays: [num_envs, num_agents, 3]
      - action arrays: [num_envs, num_agents, 2]
      - per-environment arrays: [num_envs]

    Reward plugins must treat this context as read-only.
    """

    old_state: np.ndarray
    new_state: np.ndarray
    actions: np.ndarray
    alive_before: np.ndarray
    alive_after: np.ndarray
    old_goal_distance: np.ndarray
    new_goal_distance: np.ndarray
    self_progress: np.ndarray
    team_progress: np.ndarray
    min_clearance: np.ndarray
    danger: np.ndarray
    collision: np.ndarray
    new_death: np.ndarray
    goal_reached: np.ndarray
    team_success: np.ndarray
    both_dead: np.ndarray
    timeout: np.ndarray
    steps: np.ndarray
    params: Mapping[str, Any]

    @property
    def num_envs(self) -> int:
        return int(self.alive_before.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.alive_before.shape[1])

    @property
    def agent_shape(self) -> tuple[int, int]:
        return self.num_envs, self.num_agents


@dataclass(frozen=True)
class RewardOverride:
    """Replace the accumulated reward wherever mask is true."""

    mask: np.ndarray
    values: np.ndarray | float


def zeros(ctx: RewardContext) -> np.ndarray:
    return np.zeros(ctx.agent_shape, dtype=np.float32)


def env_mask_to_agents(ctx: RewardContext, env_mask: np.ndarray) -> np.ndarray:
    env_mask = np.asarray(env_mask, dtype=bool)
    if env_mask.shape != (ctx.num_envs,):
        raise ValueError(
            f"expected environment mask shape {(ctx.num_envs,)}, got {env_mask.shape}"
        )
    return np.broadcast_to(env_mask[:, None], ctx.agent_shape)


def override(mask: np.ndarray, values: np.ndarray | float) -> RewardOverride:
    return RewardOverride(mask=np.asarray(mask, dtype=bool), values=values)


def _decorate(
    fn: Callable[..., Any],
    *,
    mode: str,
    name: str | None,
    order: int,
) -> Callable[..., Any]:
    setattr(fn, "__marl2d_reward_term__", True)
    setattr(fn, "__marl2d_reward_mode__", str(mode))
    setattr(fn, "__marl2d_reward_name__", str(name or fn.__name__))
    setattr(fn, "__marl2d_reward_order__", int(order))
    return fn


def reward_term(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    order: int = 0,
):
    """Register an additive reward/penalty term.

    Example:
        @reward_term
        def forward_progress(ctx):
            out = zeros(ctx)
            out[ctx.alive_before] = 2.0 * ctx.self_progress[ctx.alive_before]
            return out
    """

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate(func, mode="add", name=name, order=order)

    return wrap(fn) if fn is not None else wrap


def reward_override(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    order: int = 100,
):
    """Register a reward term that overwrites accumulated reward on a mask."""

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate(func, mode="override", name=name, order=order)

    return wrap(fn) if fn is not None else wrap
