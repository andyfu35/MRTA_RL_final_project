import numpy as np
import torch

from marl2d.policy import ActorCritic, snapshot_model
from marl2d.two_runner import (
    TWO_RUNNER_IDS,
    TargetTransition,
    TwoRunnerWorker,
    finalize_target_trajectory,
)


def tiny_cfg(samples=2, parallel_envs=1):
    return {
        'seed': 7,
        'environment': {
            'width': 20.0, 'height': 20.0, 'dt': 0.1,
            'wheel_radius': 0.1, 'wheel_base': 0.5,
            'max_wheel_linear_speed': 2.0, 'robot_radius': 0.3,
            'goal': [18.0, 10.0], 'goal_radius': 0.7, 'max_steps': 20,
            'lidar_rays': 8, 'lidar_range': 6.0, 'reset_jitter': 0.0,
            'obstacles': {'count': 0},
        },
        'two_runner_reward': {
            'team_success_bonus': 100.0, 'collision_penalty': -100.0,
            'timeout_penalty': -20.0, 'self_progress_scale': 3.0,
            'team_progress_scale': 2.0, 'step_penalty': -0.01,
            'safety_distance': 0.5, 'safety_scale': 3.0,
        },
        'training': {'samples_per_update': samples},
        'collection_profiles': {
            'runner_0': {'parallel_envs': parallel_envs},
            'runner_1': {'parallel_envs': parallel_envs},
        },
        'ppo': {
            'hidden_sizes': [8], 'learning_rate': 3e-4, 'gamma': 0.99,
            'gae_lambda': 0.95, 'clip_range': 0.2, 'value_coef': 0.5,
            'entropy_coef': 0.005, 'max_grad_norm': 0.5,
            'epochs': 1, 'minibatch_size': 2,
        },
    }


def policy_set(cfg):
    torch.manual_seed(3)
    return {
        aid: snapshot_model(ActorCritic(18, 2, cfg['ppo']['hidden_sizes']))
        for aid in TWO_RUNNER_IDS
    }


def tr(reward):
    return TargetTransition(
        observation=np.zeros(18, np.float32),
        action=np.zeros(2, np.float32),
        old_log_prob=0.0,
        reward=float(reward),
        value=0.0,
        next_value=0.0,
    )


def test_delayed_team_success_is_added_to_last_valid_transition_after_death():
    samples = finalize_target_trajectory(
        [tr(1.0), tr(-100.0)],
        delayed_team_success=True,
        team_success_bonus=100.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert len(samples) == 2
    assert samples[-1].reward == 0.0
    assert samples[-1].done is True


class DeathThenSuccessEnv:
    observation_dim = 18
    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        assert num_envs == 1
        self.num_envs = 1
        self.alive = np.ones((1, 2), dtype=bool)
        self.step_count = 0
        self.map_seeds = np.array([seed], dtype=np.int64)
    def observe(self):
        out = np.zeros((1, 2, 18), np.float32)
        out[:, :, 17] = self.alive[:, ::-1]
        return out
    def reset_indices(self, mask):
        if bool(mask[0]):
            self.alive[:] = True
            self.step_count = 0
    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros((1, 2), np.float32)
        collision = np.zeros((1, 2), bool)
        new_death = np.zeros((1, 2), bool)
        goal = np.zeros((1, 2), bool)
        success = np.zeros(1, bool)
        done = np.zeros(1, bool)
        alive_before = self.alive.copy()
        if self.step_count == 2:
            collision[0, 0] = new_death[0, 0] = True
            rewards[0, 0] = -100.0
            self.alive[0, 0] = False
        elif self.step_count == 4:
            goal[0, 1] = True
            success[0] = done[0] = True
            rewards[0, 1] = 100.0
        info = {
            'collision': collision, 'new_death': new_death,
            'goal_reached': goal, 'team_success': success,
            'both_dead': np.zeros(1, bool), 'timeout': np.zeros(1, bool),
            'alive_before': alive_before, 'alive_after': self.alive.copy(),
            'min_clearance': np.ones((1, 2), np.float32),
            'map_seed': self.map_seeds.copy(),
        }
        return self.observe(), rewards, done, info


def test_dead_period_steps_create_no_actor_samples_but_team_success_still_credits_dead_target():
    cfg = tiny_cfg(samples=2, parallel_envs=1)
    worker = TwoRunnerWorker('runner_0', cfg, device='cpu', env_factory=DeathThenSuccessEnv)
    batch, metrics = worker.collect_round(policy_set(cfg), round_index=0)
    assert batch.observations.shape[0] == 2
    assert metrics['samples'] == 2
    assert metrics['simulator_steps'] == 4
    assert metrics['target_deaths'] == 1
    assert metrics['delayed_team_credits'] == 1


class AlwaysSuccessEnv:
    observation_dim = 18
    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        self.num_envs = num_envs
        self.alive = np.ones((num_envs, 2), bool)
        self.map_seeds = np.arange(num_envs, dtype=np.int64) + seed
    def observe(self):
        return np.zeros((self.num_envs, 2, 18), np.float32)
    def reset_indices(self, mask):
        self.alive[mask] = True
    def step(self, actions):
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


def test_collect_round_consumes_exact_valid_sample_count_and_keeps_surplus():
    cfg = tiny_cfg(samples=3, parallel_envs=2)
    worker = TwoRunnerWorker('runner_0', cfg, device='cpu', env_factory=AlwaysSuccessEnv)
    batch, metrics = worker.collect_round(policy_set(cfg), round_index=0)
    assert batch.observations.shape[0] == 3
    assert metrics['samples'] == 3
    assert metrics['simulator_steps'] == 4
    assert len(worker.finalized_queue) == 1
