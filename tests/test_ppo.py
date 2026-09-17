import copy

import numpy as np
import torch

from marl2d.policy import ActorCritic, load_model_snapshot, snapshot_model
from marl2d.ppo import RolloutBatch, ppo_update


def make_ppo_cfg():
    return {
        'clip_range': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.005,
        'max_grad_norm': 0.5,
        'epochs': 2,
        'minibatch_size': 32,
    }


def test_sampled_actions_are_bounded_and_finite():
    torch.manual_seed(0)
    model = ActorCritic(obs_dim=21, action_dim=2, hidden_sizes=[32, 32])
    obs = torch.randn(64, 21)
    with torch.no_grad():
        action, log_prob, value = model.sample(obs)
    assert action.shape == (64, 2)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_snapshot_roundtrip_restores_parameters():
    torch.manual_seed(1)
    model = ActorCritic(obs_dim=21, action_dim=2, hidden_sizes=[16, 16])
    snapshot = snapshot_model(model)
    original = copy.deepcopy(snapshot)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    load_model_snapshot(model, snapshot)
    current = model.state_dict()
    for key, expected in original.items():
        torch.testing.assert_close(current[key], expected)


def test_ppo_update_returns_finite_metrics():
    torch.manual_seed(2)
    np.random.seed(2)
    model = ActorCritic(obs_dim=21, action_dim=2, hidden_sizes=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs = torch.randn(128, 21)
    with torch.no_grad():
        actions, old_log_prob, values = model.sample(obs)
    advantages = torch.randn(128)
    returns = values + advantages
    batch = RolloutBatch(
        observations=obs,
        actions=actions,
        old_log_probs=old_log_prob,
        returns=returns,
        advantages=advantages,
    )
    metrics = ppo_update(model, optimizer, batch, make_ppo_cfg())
    for name in ('policy_loss', 'value_loss', 'entropy', 'approx_kl'):
        assert name in metrics
        assert np.isfinite(metrics[name])
