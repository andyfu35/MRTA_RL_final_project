import copy

import torch

from marl2d import AGENT_IDS
from marl2d.policy import ActorCritic, snapshot_model
from marl2d.worker import AgentWorker


def make_cfg():
    return {
        'seed': 10,
        'environment': {
            'width': 10.0, 'height': 6.0, 'dt': 0.1,
            'wheel_radius': 0.1, 'wheel_base': 0.4, 'max_wheel_speed': 5.0,
            'robot_radius': 0.18, 'goal': [9.0, 3.0], 'goal_radius': 0.5,
            'max_steps': 50, 'lidar_rays': 8, 'lidar_range': 4.0,
            'reset_jitter': 0.05, 'obstacles': [],
        },
        'reward': {
            'runner': {'goal_bonus': 20.0, 'team_progress': 2.0, 'self_progress': 0.5, 'collision': -1.0, 'step': -0.01},
            'blocker': {'runner_progress': -2.0, 'timeout_bonus': 20.0, 'goal_failure': -20.0, 'collision': -1.0, 'step': -0.005, 'proximity': 0.01},
        },
        'training': {'samples_per_update': 64},
        'collection_profiles': {
            'runner_0': {'parallel_envs': 8, 'rollout_steps': 8, 'batches': 1},
            'runner_1': {'parallel_envs': 4, 'rollout_steps': 8, 'batches': 2},
            'blocker_0': {'parallel_envs': 2, 'rollout_steps': 8, 'batches': 4},
            'blocker_1': {'parallel_envs': 1, 'rollout_steps': 8, 'batches': 8},
        },
        'ppo': {
            'hidden_sizes': [16, 16], 'learning_rate': 3e-4,
            'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.0, 'max_grad_norm': 0.5,
            'epochs': 1, 'minibatch_size': 32,
        },
    }


def make_policy_set(cfg):
    torch.manual_seed(5)
    result = {}
    for agent_id in AGENT_IDS:
        model = ActorCritic(21, 2, cfg['ppo']['hidden_sizes'])
        result[agent_id] = snapshot_model(model)
    return result


def clone_policy_set(policy_set):
    return {aid: {k: v.clone() for k, v in state.items()} for aid, state in policy_set.items()}


def assert_policy_sets_equal(a, b):
    for aid in AGENT_IDS:
        for key in a[aid]:
            torch.testing.assert_close(a[aid][key], b[aid][key])


def test_heterogeneous_profiles_collect_same_sample_count():
    cfg = make_cfg()
    policy_set = make_policy_set(cfg)
    fast = AgentWorker('runner_0', cfg, device='cpu')
    slow = AgentWorker('blocker_1', cfg, device='cpu')
    fast_batch, fast_stats = fast.collect_round(policy_set, round_index=0)
    slow_batch, slow_stats = slow.collect_round(policy_set, round_index=0)
    assert fast_batch.observations.shape[0] == 64
    assert slow_batch.observations.shape[0] == 64
    assert fast_stats['samples'] == 64
    assert slow_stats['samples'] == 64


def test_collect_round_does_not_mutate_frozen_policy_set():
    cfg = make_cfg()
    policy_set = make_policy_set(cfg)
    before = clone_policy_set(policy_set)
    worker = AgentWorker('runner_1', cfg, device='cpu')
    worker.collect_round(policy_set, round_index=3)
    assert_policy_sets_equal(policy_set, before)
