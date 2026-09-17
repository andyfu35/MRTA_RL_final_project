from pathlib import Path

import numpy as np
import torch

from marl2d.policy import ActorCritic, expand_observation_snapshot, snapshot_model
from marl2d.two_runner import TWO_RUNNER_IDS, TwoRunnerTrainer


def cfg(samples=2):
    return {
        'seed': 11,
        'environment': {
            'width': 20.0, 'height': 20.0, 'dt': 0.1,
            'wheel_radius': 0.1, 'wheel_base': 0.5,
            'max_wheel_linear_speed': 2.0, 'robot_radius': 0.3,
            'goal': [18.0, 10.0], 'goal_radius': 0.7, 'max_steps': 5,
            'lidar_rays': 8, 'lidar_range': 6.0, 'reset_jitter': 0.0,
            'obstacles': {'count': 0},
        },
        'two_runner_reward': {
            'team_success_bonus': 100.0, 'collision_penalty': -100.0,
            'timeout_penalty': -20.0, 'self_progress_scale': 3.0,
            'team_progress_scale': 2.0, 'step_penalty': -0.01,
            'safety_distance': 0.5, 'safety_scale': 3.0,
        },
        'training': {'samples_per_update': samples, 'rounds': 2, 'checkpoint_every': 1},
        'collection_profiles': {
            'runner_0': {'parallel_envs': 1},
            'runner_1': {'parallel_envs': 1},
        },
        'ppo': {
            'hidden_sizes': [8], 'learning_rate': 3e-4, 'gamma': 0.99,
            'gae_lambda': 0.95, 'clip_range': 0.2, 'value_coef': 0.5,
            'entropy_coef': 0.005, 'max_grad_norm': 0.5,
            'epochs': 1, 'minibatch_size': 2,
        },
        'validation': {'every': 1, 'episodes': 2, 'seed_start': 40000},
        'evaluation': {'episodes': 2, 'seed_start': 50000},
    }


class SerializableAlwaysSuccessEnv:
    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        self.num_envs = num_envs
        self.alive = np.ones((num_envs, 2), bool)
        self.map_seeds = np.arange(num_envs, dtype=np.int64) + seed
        self.step_count = np.zeros(num_envs, np.int32)

    def observe(self):
        return np.zeros((self.num_envs, 2, 18), np.float32)

    def reset_indices(self, mask):
        self.alive[mask] = True
        self.step_count[mask] = 0
        self.map_seeds[mask] += 1

    def step(self, actions):
        self.step_count += 1
        n = self.num_envs
        alive_before = self.alive.copy()
        rewards = np.full((n, 2), 100.0, np.float32)
        done = np.ones(n, bool)
        info = {
            'collision': np.zeros((n, 2), bool),
            'new_death': np.zeros((n, 2), bool),
            'goal_reached': np.ones((n, 2), bool),
            'team_success': np.ones(n, bool),
            'both_dead': np.zeros(n, bool),
            'timeout': np.zeros(n, bool),
            'alive_before': alive_before,
            'alive_after': self.alive.copy(),
            'min_clearance': np.ones((n, 2), np.float32),
            'map_seed': self.map_seeds.copy(),
        }
        return self.observe(), rewards, done, info

    def snapshot_state(self):
        return {
            'alive': self.alive.copy(),
            'map_seeds': self.map_seeds.copy(),
            'step_count': self.step_count.copy(),
        }

    def restore_state(self, state):
        self.alive[:] = state['alive']
        self.map_seeds[:] = state['map_seeds']
        self.step_count[:] = state['step_count']


def test_trainer_commits_only_after_both_runner_updates(tmp_path: Path):
    trainer = TwoRunnerTrainer(cfg(), tmp_path, device='cpu', worker_env_factory=SerializableAlwaysSuccessEnv)
    records = trainer.run(rounds=1, run_validation=False)
    assert trainer.current_round == 1
    assert records[0]['round'] == 1
    assert set(records[0]['agents']) == set(TWO_RUNNER_IDS)
    assert all(records[0]['agents'][aid]['samples'] == 2 for aid in TWO_RUNNER_IDS)


def test_warm_start_expands_single_runner_for_both_policies_with_fresh_optimizers(tmp_path: Path):
    c = cfg()
    torch.manual_seed(5)
    old = ActorCritic(15, 2, c['ppo']['hidden_sizes'])
    old_snapshot = snapshot_model(old)
    source = tmp_path / 'single.pt'
    torch.save({'version': 80, 'model': old_snapshot, 'config': {}, 'obs_dim': 15, 'action_dim': 2}, source)
    trainer = TwoRunnerTrainer(c, tmp_path / 'run', device='cpu', worker_env_factory=SerializableAlwaysSuccessEnv)
    trainer.initialize_from_single_runner_checkpoint(source)
    expected = expand_observation_snapshot(old_snapshot, 15, 18)
    assert trainer.current_round == 0
    for aid in TWO_RUNNER_IDS:
        for key, value in expected.items():
            torch.testing.assert_close(trainer.policy_set()[aid][key], value)
        assert trainer.workers[aid].optimizer.state_dict()['state'] == {}


def test_full_checkpoint_resume_reproduces_next_round(tmp_path: Path):
    c = cfg()
    first = TwoRunnerTrainer(c, tmp_path / 'first', device='cpu', worker_env_factory=SerializableAlwaysSuccessEnv)
    first.run(rounds=1, run_validation=False)
    checkpoint = first.save_checkpoint(tmp_path / 'resume.pt')
    expected_round = first.current_round
    original = first.run(rounds=1, run_validation=False)[0]
    original_policy = first.policy_set()

    resumed = TwoRunnerTrainer(c, tmp_path / 'resumed', device='cpu', worker_env_factory=SerializableAlwaysSuccessEnv)
    assert resumed.resume_from_checkpoint(checkpoint) == 'full'
    assert resumed.current_round == expected_round
    replay = resumed.run(rounds=1, run_validation=False)[0]
    assert replay['round'] == original['round']
    for aid in TWO_RUNNER_IDS:
        for key, value in original_policy[aid].items():
            torch.testing.assert_close(resumed.policy_set()[aid][key], value)
