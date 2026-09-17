from pathlib import Path

import numpy as np

from marl2d import AGENT_IDS
from marl2d.render import render_policy_set_gif
from marl2d.trainer import DistributedTrainer


def make_cfg():
    return {
        'seed': 123,
        'environment': {
            'width': 10.0, 'height': 6.0, 'dt': 0.1,
            'wheel_radius': 0.1, 'wheel_base': 0.4, 'max_wheel_linear_speed': 2.0,
            'robot_radius': 0.18, 'goal': [9.0, 3.0], 'goal_radius': 0.5,
            'max_steps': 20, 'lidar_rays': 8, 'lidar_range': 4.0,
            'reset_jitter': 0.02, 'obstacles': {'count': 0},
        },
        'reward': {
            'runner': {'goal_bonus': 20.0, 'team_progress': 2.0, 'self_progress': 0.5, 'collision': -1.0, 'step': -0.01},
            'blocker': {'runner_progress': -2.0, 'timeout_bonus': 20.0, 'goal_failure': -20.0, 'collision': -1.0, 'step': -0.005, 'proximity': 0.01},
        },
        'training': {'samples_per_update': 16, 'rounds': 1, 'checkpoint_every': 1},
        'collection_profiles': {
            'runner_0': {'parallel_envs': 4, 'rollout_steps': 4, 'batches': 1},
            'runner_1': {'parallel_envs': 2, 'rollout_steps': 4, 'batches': 2},
            'blocker_0': {'parallel_envs': 1, 'rollout_steps': 4, 'batches': 4},
            'blocker_1': {'parallel_envs': 1, 'rollout_steps': 2, 'batches': 8},
        },
        'ppo': {
            'hidden_sizes': [16, 16], 'learning_rate': 3e-4,
            'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.0, 'max_grad_norm': 0.5,
            'epochs': 1, 'minibatch_size': 8,
        },
        'network': {
            'mode': 'mock',
            'coordinator': {'ip': '127.0.0.1', 'port': 7400},
            'agents': {aid: {'ip': '127.0.0.1', 'port': 7401 + i} for i, aid in enumerate(AGENT_IDS)},
        },
    }


def test_one_synchronized_training_round_writes_checkpoint_and_metrics(tmp_path: Path):
    cfg = make_cfg()
    trainer = DistributedTrainer(cfg, output_dir=tmp_path, device='cpu')
    history = trainer.run(rounds=1)
    assert trainer.coordinator.current_round == 1
    assert len(history) == 1
    assert history[0]['round'] == 1
    for aid in AGENT_IDS:
        metrics = history[0]['agents'][aid]
        assert metrics['samples'] == 16
        assert np.isfinite(metrics['policy_loss'])
        assert np.isfinite(metrics['value_loss'])
    assert (tmp_path / 'latest.pt').exists()
    assert (tmp_path / 'metrics.jsonl').exists()


def test_renderer_creates_nonempty_gif(tmp_path: Path):
    cfg = make_cfg()
    trainer = DistributedTrainer(cfg, output_dir=tmp_path / 'train', device='cpu')
    trainer.run(rounds=1)
    policy_set = trainer.exchange.get_committed_policy_set()
    out = tmp_path / 'episode.gif'
    summary = render_policy_set_gif(cfg, policy_set, out, max_steps=4, device='cpu')
    assert out.exists()
    assert out.stat().st_size > 100
    assert summary['frames'] >= 1
