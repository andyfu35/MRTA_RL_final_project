"""External reward plugin for MARL2D.

Edit this file without changing the training executable.

How to add a new reward/penalty:
1. Copy one @reward_term function.
2. Give it a new unique function name.
3. Return a [num_envs, num_agents] array (or a broadcastable scalar).
4. Restart all training nodes so every machine loads the same file.

All @reward_term functions are added together.
@reward_override functions replace the accumulated reward on terminal/event masks.
"""

from __future__ import annotations

import numpy as np

from marl2d.reward_api import (
    REWARD_API_VERSION,
    RewardContext,
    env_mask_to_agents,
    override,
    reward_override,
    reward_term,
    zeros,
)


@reward_term(order=10)
def own_goal_progress(ctx: RewardContext) -> np.ndarray:
    """Dense reward for reducing this Runner's distance to the goal."""
    out = zeros(ctx)
    scale = float(ctx.params["self_progress_scale"])
    out[ctx.alive_before] = (
        scale * ctx.self_progress[ctx.alive_before]
    )
    return out


@reward_term(order=20)
def team_goal_progress(ctx: RewardContext) -> np.ndarray:
    """Dense team reward when the closest alive Runner gets nearer the goal."""
    out = zeros(ctx)
    scale = float(ctx.params["team_progress_scale"])
    active = ctx.alive_before
    shared = np.broadcast_to(
        (scale * ctx.team_progress)[:, None],
        ctx.agent_shape,
    )
    out[active] = shared[active]
    return out


@reward_term(order=30)
def living_step_cost(ctx: RewardContext) -> np.ndarray:
    out = zeros(ctx)
    out[ctx.alive_before] = float(ctx.params["step_penalty"])
    return out


@reward_term(order=40)
def obstacle_safety_penalty(ctx: RewardContext) -> np.ndarray:
    out = zeros(ctx)
    scale = float(ctx.params["safety_scale"])
    out[ctx.alive_before] = (
        -scale * ctx.danger[ctx.alive_before] ** 2
    )
    return out


# ---------------------------------------------------------------------------
# TEMPLATE: add a new dense reward/penalty by copying this pattern.
#
# @reward_term(order=50)
# def my_new_reward(ctx: RewardContext) -> np.ndarray:
#     out = zeros(ctx)
#     # Example: -0.05 penalty while alive.
#     out[ctx.alive_before] = -0.05
#     return out
#
# Once decorated with @reward_term, it is automatically included.
# ---------------------------------------------------------------------------


@reward_override(order=100)
def collision_terminal(ctx: RewardContext):
    return override(
        ctx.new_death,
        float(ctx.params["collision_penalty"]),
    )


@reward_override(order=200)
def timeout_terminal(ctx: RewardContext):
    mask = env_mask_to_agents(ctx, ctx.timeout) & ctx.alive_after
    return override(
        mask,
        float(ctx.params["timeout_penalty"]),
    )


@reward_override(order=300)
def team_success_terminal(ctx: RewardContext):
    # Preserve the historical semantics: every Runner that was alive on the
    # successful step receives the team success bonus.
    mask = (
        env_mask_to_agents(ctx, ctx.team_success)
        & ctx.alive_before
    )
    return override(
        mask,
        float(ctx.params["team_success_bonus"]),
    )
