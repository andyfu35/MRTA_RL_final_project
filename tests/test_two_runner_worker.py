import numpy as np
import torch

from marl2d.policy import ActorCritic, snapshot_model
from marl2d.two_runner import (
    TWO_RUNNER_IDS,
    TargetTransition,
    TwoRunnerWorker,
    finalize_target_fragment,
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
        self.alive = np.ones((1,2), dtype=bool)
        self.step_count = 0
        self.map_seeds = np.array([seed], dtype=np.int64)
    def observe(self):
        out = np.zeros((1,2,18), np.float32)
        out[:,:,17] = self.alive[:,::-1]
        return out
    def reset_indices(self, mask):
        if bool(mask[0]):
            self.alive[:] = True
            self.step_count = 0
    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros((1,2), np.float32)
        collision = np.zeros((1,2), bool)
        new_death = np.zeros((1,2), bool)
        goal = np.zeros((1,2), bool)
        success = np.zeros(1, bool)
        done = np.zeros(1, bool)
        alive_before = self.alive.copy()
        if self.step_count == 2:
            collision[0,0] = new_death[0,0] = True
            rewards[0,0] = -100.0
            self.alive[0,0] = False
        elif self.step_count == 4:
            goal[0,1] = True
            success[0] = done[0] = True
            rewards[0,1] = 100.0
        info = {
            'collision': collision, 'new_death': new_death,
            'goal_reached': goal, 'team_success': success,
            'both_dead': np.zeros(1,bool), 'timeout': np.zeros(1,bool),
            'alive_before': alive_before, 'alive_after': self.alive.copy(),
            'min_clearance': np.ones((1,2),np.float32),
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
    # The collision transition receives delayed +100, so its final immediate reward is zero.
    assert metrics['delayed_team_credits'] == 1


class AlwaysSuccessEnv:
    observation_dim = 18
    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        self.num_envs = num_envs
        self.alive = np.ones((num_envs,2),bool)
        self.map_seeds = np.arange(num_envs,dtype=np.int64)+seed
    def observe(self): return np.zeros((self.num_envs,2,18),np.float32)
    def reset_indices(self, mask): self.alive[mask] = True
    def step(self, actions):
        n=self.num_envs; alive_before=self.alive.copy(); rewards=np.full((n,2),100.0,np.float32); done=np.ones(n,bool)
        info={'collision':np.zeros((n,2),bool),'new_death':np.zeros((n,2),bool),'goal_reached':np.ones((n,2),bool),'team_success':np.ones(n,bool),'both_dead':np.zeros(n,bool),'timeout':np.zeros(n,bool),'alive_before':alive_before,'alive_after':self.alive.copy(),'min_clearance':np.ones((n,2),np.float32),'map_seed':self.map_seeds.copy()}
        return self.observe(),rewards,done,info


def test_collect_round_consumes_exact_valid_sample_count_and_discards_same_round_surplus():
    cfg=tiny_cfg(samples=3,parallel_envs=2)
    worker=TwoRunnerWorker('runner_0',cfg,device='cpu',env_factory=AlwaysSuccessEnv)
    batch,metrics=worker.collect_round(policy_set(cfg),round_index=0)
    assert batch.observations.shape[0] == 3
    assert metrics['samples'] == 3
    assert metrics['simulator_steps'] == 4
    assert metrics['discarded_surplus_samples'] == 1
    assert len(worker.finalized_queue) == 0


def test_round_boundary_discards_surplus_instead_of_carrying_samples_across_policy_versions():
    cfg = tiny_cfg(samples=3, parallel_envs=2)
    worker = TwoRunnerWorker('runner_0', cfg, device='cpu', env_factory=AlwaysSuccessEnv)
    first_batch, first_metrics = worker.collect_round(policy_set(cfg), round_index=0)
    assert first_batch.observations.shape[0] == 3
    assert first_metrics['samples'] == 3
    assert first_metrics['discarded_surplus_samples'] == 1
    assert len(worker.finalized_queue) == 0
    assert all(len(items) == 0 for items in worker.pending_trajectories)

    second_batch, second_metrics = worker.collect_round(policy_set(cfg), round_index=1)
    assert second_batch.observations.shape[0] == 3
    assert second_metrics['samples'] == 3
    assert second_metrics['discarded_surplus_samples'] == 1
    assert len(worker.finalized_queue) == 0


class UnevenEpisodeEnv:
    observation_dim = 18

    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        assert num_envs == 2
        self.num_envs = 2
        self.alive = np.ones((2, 2), dtype=bool)
        self.map_seeds = np.arange(2, dtype=np.int64) + seed
        self.step_counts = np.zeros(2, dtype=np.int64)
        self.reset_calls = []

    def observe(self):
        return np.zeros((2, 2, 18), np.float32)

    def reset_indices(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.reset_calls.append(mask.copy())
        self.step_counts[mask] = 0
        self.alive[mask] = True

    def step(self, actions):
        self.step_counts += 1
        alive_before = self.alive.copy()
        # env 0 finishes after one step; env 1 needs three steps.
        done = np.array([self.step_counts[0] >= 1, self.step_counts[1] >= 3], dtype=bool)
        success = done.copy()
        rewards = np.zeros((2, 2), np.float32)
        rewards[done] = 100.0
        goal = np.zeros((2, 2), dtype=bool)
        goal[done, 0] = True
        info = {
            'collision': np.zeros((2, 2), bool),
            'new_death': np.zeros((2, 2), bool),
            'goal_reached': goal,
            'team_success': success,
            'both_dead': np.zeros(2, bool),
            'timeout': np.zeros(2, bool),
            'alive_before': alive_before,
            'alive_after': self.alive.copy(),
            'min_clearance': np.ones((2, 2), np.float32),
            'map_seed': self.map_seeds.copy(),
        }
        return self.observe(), rewards, done, info


def test_after_sample_target_is_reached_worker_drains_existing_episodes_without_launching_new_ones():
    cfg = tiny_cfg(samples=1, parallel_envs=2)
    worker = TwoRunnerWorker('runner_0', cfg, device='cpu', env_factory=UnevenEpisodeEnv)
    batch, metrics = worker.collect_round(policy_set(cfg), round_index=0)
    assert batch.observations.shape[0] == 1
    # The fast environment reaches success at step 1, but the slow environment is
    # allowed to finish its already-running episode at step 3 before the round ends.
    assert metrics['simulator_steps'] == 6
    assert metrics['completed_team_episodes'] == 2
    # No new episode is launched while draining; only one all-env reset occurs at
    # the round boundary after both old episodes are terminal.
    assert len(worker._collector_env.reset_calls) == 1
    np.testing.assert_array_equal(worker._collector_env.reset_calls[0], np.array([True, True]))
    assert len(worker.finalized_queue) == 0
    assert all(len(items) == 0 for items in worker.pending_trajectories)


class MarkerSuccessEnv:
    observation_dim = 18

    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        assert num_envs == 2
        self.num_envs = 2
        self.alive = np.ones((2, 2), dtype=bool)
        self.map_seeds = np.arange(2, dtype=np.int64) + seed
        self.generation = np.zeros(2, dtype=np.int64)

    def observe(self):
        obs = np.zeros((2, 2, 18), np.float32)
        markers = self.generation * 10 + np.arange(2)
        obs[:, :, 0] = markers[:, None]
        return obs

    def reset_indices(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.generation[mask] += 1
        self.alive[mask] = True

    def step(self, actions):
        alive_before = self.alive.copy()
        rewards = np.full((2, 2), 100.0, np.float32)
        done = np.ones(2, bool)
        info = {
            'collision': np.zeros((2, 2), bool),
            'new_death': np.zeros((2, 2), bool),
            'goal_reached': np.ones((2, 2), bool),
            'team_success': np.ones(2, bool),
            'both_dead': np.zeros(2, bool),
            'timeout': np.zeros(2, bool),
            'alive_before': alive_before,
            'alive_after': self.alive.copy(),
            'min_clearance': np.ones((2, 2), np.float32),
            'map_seed': self.map_seeds.copy(),
        }
        return self.observe(), rewards, done, info


def test_same_round_surplus_is_uniformly_subsampled_instead_of_always_dropping_tail_episodes():
    cfg = tiny_cfg(samples=3, parallel_envs=2)
    worker = TwoRunnerWorker('runner_0', cfg, device='cpu', env_factory=MarkerSuccessEnv)
    batch, metrics = worker.collect_round(policy_set(cfg), round_index=0)
    markers = set(batch.observations[:, 0].cpu().numpy().astype(int).tolist())
    # Round seed is 7 for runner_0/round0. The dedicated selection RNG uses
    # seed+1=8, whose 3-of-4 choice is indices [0, 1, 3]. The pool markers are
    # [0, 1, 10, 11], so the selected batch must include the later episode 11.
    assert markers == {0, 1, 11}
    assert metrics['discarded_surplus_samples'] == 1


class MixedOutcomeEnv:
    observation_dim = 18

    def __init__(self, num_envs, env_cfg, reward_cfg, seed=0):
        assert num_envs == 2
        self.num_envs = 2
        self.alive = np.ones((2, 2), dtype=bool)
        self.map_seeds = np.arange(2, dtype=np.int64) + seed

    def observe(self):
        return np.zeros((2, 2, 18), np.float32)

    def reset_indices(self, mask):
        self.alive[np.asarray(mask, dtype=bool)] = True

    def step(self, actions):
        alive_before = self.alive.copy()
        rewards = np.zeros((2, 2), np.float32)
        rewards[0] = 100.0
        rewards[1] = -20.0
        done = np.ones(2, dtype=bool)
        info = {
            'collision': np.zeros((2, 2), bool),
            'new_death': np.zeros((2, 2), bool),
            'goal_reached': np.array([[True, False], [False, False]], dtype=bool),
            'team_success': np.array([True, False], dtype=bool),
            'both_dead': np.array([False, False], dtype=bool),
            'timeout': np.array([False, True], dtype=bool),
            'alive_before': alive_before,
            'alive_after': self.alive.copy(),
            'min_clearance': np.ones((2, 2), np.float32),
            'map_seed': self.map_seeds.copy(),
        }
        return self.observe(), rewards, done, info


def test_collection_reports_success_failure_sample_composition_and_advantage_split():
    cfg = tiny_cfg(samples=2, parallel_envs=2)
    worker = TwoRunnerWorker('runner_0', cfg, device='cpu', env_factory=MixedOutcomeEnv)
    batch, metrics = worker.collect_round(policy_set(cfg), round_index=0)

    assert batch.observations.shape[0] == 2
    assert metrics['team_success_episodes'] == 1
    assert metrics['timeout_episodes'] == 1
    assert metrics['both_dead_episodes'] == 0
    assert metrics['sample_pool_success_transitions'] == 1
    assert metrics['sample_pool_failure_transitions'] == 1
    assert metrics['selected_success_transitions'] == 1
    assert metrics['selected_failure_transitions'] == 1
    assert metrics['selected_success_trajectories'] == 1
    assert metrics['selected_failure_trajectories'] == 1
    assert metrics['selected_success_fraction'] == 0.5
    assert 'selected_success_advantage_mean' in metrics
    assert 'selected_failure_advantage_mean' in metrics
    assert 'mean_success_terminal_credit_weight_at_start' in metrics


def test_nonterminal_fragment_bootstraps_value_at_fixed_horizon():
    transitions = [
        TargetTransition(
            observation=np.zeros(18, np.float32),
            action=np.zeros(2, np.float32),
            old_log_prob=0.0,
            reward=0.0,
            value=1.0,
            next_value=2.0,
        ),
        TargetTransition(
            observation=np.zeros(18, np.float32),
            action=np.zeros(2, np.float32),
            old_log_prob=0.0,
            reward=0.0,
            value=2.0,
            next_value=3.0,
        ),
    ]
    samples = finalize_target_fragment(
        transitions,
        terminal=False,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert len(samples) == 2
    assert samples[-1].done is False
    assert samples[-1].next_value == 3.0
    np.testing.assert_allclose(
        [sample.return_value for sample in samples],
        [3.0, 3.0],
        atol=1e-6,
    )


def test_terminal_fragment_does_not_bootstrap_past_terminal():
    transitions = [
        TargetTransition(
            observation=np.zeros(18, np.float32),
            action=np.zeros(2, np.float32),
            old_log_prob=0.0,
            reward=5.0,
            value=1.0,
            next_value=99.0,
        ),
    ]
    samples = finalize_target_fragment(
        transitions,
        terminal=True,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert samples[0].done is True
    assert samples[0].next_value == 0.0
    assert samples[0].return_value == 5.0
