from pathlib import Path

import numpy as np
import torch

from marl2d.single_runner import (
    SingleRunnerTrainer,
    compute_path_efficiency,
    evaluate_single_runner,
    load_single_runner_checkpoint,
    validation_is_better,
)


def make_cfg():
    return {
        'seed': 123,
        'environment': {
            'width': 20.0, 'height': 20.0, 'dt': 0.1,
            'wheel_radius': 0.1, 'wheel_base': 0.5,
            'max_wheel_linear_speed': 2.0,
            'robot_radius': 0.30,
            'body_length': 0.70, 'body_width': 0.44,
            'wheel_length': 0.34, 'wheel_width': 0.10,
            'goal': [18.0, 10.0], 'goal_radius': 0.6,
            'max_steps': 12, 'lidar_rays': 8, 'lidar_range': 6.0,
            'reset_jitter': 0.0,
            'obstacles': {'count': 0},
        },
        'single_runner_reward': {
            'mode': 'R2',
            'goal_bonus': 100.0,
            'collision_penalty': -100.0,
            'timeout_penalty': -20.0,
            'progress_scale': 5.0,
            'step_penalty': -0.01,
            'safety_distance': 0.5,
            'safety_scale': 0.5,
            'heading_scale': 0.02,
        },
        'training': {
            'samples_per_update': 16,
            'rounds': 1,
            'checkpoint_every': 1,
        },
        'collection': {
            'parallel_envs': 4,
            'rollout_steps': 4,
            'batches': 1,
        },
        'ppo': {
            'hidden_sizes': [16, 16], 'learning_rate': 3e-4,
            'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.0, 'max_grad_norm': 0.5,
            'epochs': 1, 'minibatch_size': 8,
        },
        'evaluation': {'episodes': 3, 'seed_start': 10000},
    }


def _optimizer_tensors(state_dict):
    tensors = []
    for state in state_dict['state'].values():
        for value in state.values():
            if torch.is_tensor(value):
                tensors.append(value.detach().cpu().clone())
    return tensors


def test_single_runner_one_update_writes_checkpoint_and_metrics(tmp_path: Path):
    cfg = make_cfg()
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path, device='cpu')
    records = trainer.run(rounds=1)
    assert len(records) == 1
    assert records[0]['round'] == 1
    assert records[0]['samples'] == 16
    assert np.isfinite(records[0]['policy_loss'])
    assert np.isfinite(records[0]['value_loss'])
    assert (tmp_path / 'latest.pt').exists()
    assert (tmp_path / 'metrics.jsonl').exists()

    version, snapshot, loaded_cfg = load_single_runner_checkpoint(tmp_path / 'latest.pt')
    assert version == 1
    assert snapshot
    assert loaded_cfg['seed'] == cfg['seed']


def test_single_runner_evaluation_uses_held_out_seed_range(tmp_path: Path):
    cfg = make_cfg()
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path, device='cpu')
    trainer.run(rounds=1)
    summary = evaluate_single_runner(
        cfg,
        trainer.snapshot(),
        episodes=3,
        seed_start=10000,
        device='cpu',
    )
    assert summary['episodes'] == 3
    assert summary['seeds'] == [10000, 10001, 10002]
    assert 0.0 <= summary['success_rate'] <= 1.0
    assert 0.0 <= summary['collision_rate'] <= 1.0
    assert 0.0 <= summary['timeout_rate'] <= 1.0
    assert summary['successes'] + summary['timeouts'] <= summary['episodes']
    assert summary['mean_collisions'] >= 0.0
    assert np.isfinite(summary['mean_episode_reward'])
    assert summary['mean_min_clearance_m'] is not None
    if summary['mean_path_efficiency'] is not None:
        assert 0.0 <= summary['mean_path_efficiency'] <= 1.0


def test_path_efficiency_is_direct_distance_over_actual_distance():
    assert compute_path_efficiency(10.0, 20.0, success=True) == 0.5
    assert compute_path_efficiency(10.0, 8.0, success=True) == 1.0
    assert compute_path_efficiency(10.0, 20.0, success=False) is None


def test_parallel_worlds_use_distinct_map_seeds_and_stochastic_rollout(tmp_path: Path):
    cfg = make_cfg()
    cfg['training']['samples_per_update'] = 16
    cfg['collection'] = {'parallel_envs': 4, 'rollout_steps': 4, 'batches': 1}
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path, device='cpu')
    batch, metrics = trainer._collect_round(round_index=0)
    assert batch.observations.shape[0] == 16
    assert metrics['parallel_envs'] == 4
    assert metrics['unique_map_seeds'] >= 4
    # PPO training must sample from the policy distribution, not use deterministic means.
    assert metrics['stochastic_action_std'] > 0.0


def test_parallel_worlds_persist_across_ppo_rounds_until_done(tmp_path: Path):
    cfg = make_cfg()
    cfg['environment']['max_steps'] = 6
    cfg['training']['samples_per_update'] = 8
    cfg['collection'] = {'parallel_envs': 2, 'rollout_steps': 4, 'batches': 1}
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path, device='cpu')

    _, first = trainer._collect_round(round_index=0)
    env = trainer._collector_envs[0]
    first_seeds = env.map_seeds.copy()
    assert env.steps.tolist() == [4, 4]
    assert env.episode_counts.tolist() == [0, 0]
    assert first['completed_episodes'] == 0

    _, second = trainer._collect_round(round_index=1)
    # Both worlds continue from step 4, timeout at step 6, reset independently,
    # then collect the remaining two steps of the new episode.
    assert trainer._collector_envs[0] is env
    assert env.episode_counts.tolist() == [1, 1]
    assert env.steps.tolist() == [2, 2]
    assert np.all(env.map_seeds == first_seeds + 1)
    assert second['completed_episodes'] == 2
    assert second['timeouts'] == 2
    assert second['goals'] == 0
    assert second['collisions'] == 0


def test_full_resume_restores_optimizer_round_and_persistent_world_state(tmp_path: Path):
    cfg = make_cfg()
    cfg['environment']['max_steps'] = 20
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'first', device='cpu')
    trainer.run(rounds=1)
    checkpoint = trainer.save_checkpoint(tmp_path / 'resume.pt')

    expected_round = trainer.current_round
    expected_steps = [env.steps.copy() for env in trainer._collector_envs]
    expected_episode_counts = [env.episode_counts.copy() for env in trainer._collector_envs]
    expected_optimizer = _optimizer_tensors(trainer.optimizer.state_dict())
    assert expected_optimizer

    resumed = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'resumed', device='cpu')
    mode = resumed.resume_from_checkpoint(checkpoint)

    assert mode == 'full'
    assert resumed.current_round == expected_round
    for env, steps, counts in zip(resumed._collector_envs, expected_steps, expected_episode_counts):
        np.testing.assert_array_equal(env.steps, steps)
        np.testing.assert_array_equal(env.episode_counts, counts)
    actual_optimizer = _optimizer_tensors(resumed.optimizer.state_dict())
    assert len(actual_optimizer) == len(expected_optimizer)
    for actual, expected in zip(actual_optimizer, expected_optimizer):
        torch.testing.assert_close(actual, expected)


def test_full_resume_reproduces_next_round_exactly(tmp_path: Path):
    cfg = make_cfg()
    cfg['environment']['max_steps'] = 20
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'first', device='cpu')
    trainer.run(rounds=1)
    checkpoint = trainer.save_checkpoint(tmp_path / 'resume.pt')

    original_record = trainer.run(rounds=1)[0]
    original_snapshot = trainer.snapshot()

    resumed = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'resumed', device='cpu')
    assert resumed.resume_from_checkpoint(checkpoint) == 'full'
    resumed_record = resumed.run(rounds=1)[0]

    for key in ('mean_step_reward', 'success_rate', 'stochastic_action_std', 'policy_loss', 'value_loss'):
        assert np.isclose(resumed_record[key], original_record[key])
    for key, expected in original_snapshot.items():
        torch.testing.assert_close(resumed.snapshot()[key], expected)


def test_legacy_resume_loads_model_and_round_with_fresh_training_state(tmp_path: Path):
    cfg = make_cfg()
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'source', device='cpu')
    legacy = tmp_path / 'legacy.pt'
    torch.save(
        {
            'version': 50,
            'model': trainer.snapshot(),
            'config': cfg,
            'obs_dim': trainer.obs_dim,
            'action_dim': trainer.action_dim,
        },
        legacy,
    )

    resumed = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'resumed', device='cpu')
    mode = resumed.resume_from_checkpoint(legacy)

    assert mode == 'legacy'
    assert resumed.current_round == 50
    assert resumed._collector_envs == []
    assert resumed._collector_obs == []


def test_validation_comparator_uses_success_then_collision_then_reward():
    best = {'success_rate': 0.20, 'collision_rate': 0.10, 'mean_episode_reward': 4.0}
    assert validation_is_better(
        {'success_rate': 0.21, 'collision_rate': 0.99, 'mean_episode_reward': -99.0}, best
    )
    assert validation_is_better(
        {'success_rate': 0.20, 'collision_rate': 0.09, 'mean_episode_reward': -99.0}, best
    )
    assert validation_is_better(
        {'success_rate': 0.20, 'collision_rate': 0.10, 'mean_episode_reward': 4.1}, best
    )
    assert not validation_is_better(
        {'success_rate': 0.19, 'collision_rate': 0.0, 'mean_episode_reward': 999.0}, best
    )


def test_validation_runs_on_schedule_and_writes_best_checkpoint(tmp_path: Path):
    cfg = make_cfg()
    cfg['validation'] = {'every': 1, 'episodes': 2, 'seed_start': 9000}
    trainer = SingleRunnerTrainer(cfg, output_dir=tmp_path, device='cpu')
    record = trainer.run(rounds=1)[0]

    assert record['val_episodes'] == 2
    assert 0.0 <= record['val_success_rate'] <= 1.0
    assert 0.0 <= record['val_collision_rate'] <= 1.0
    assert 0.0 <= record['val_timeout_rate'] <= 1.0
    assert np.isfinite(record['val_mean_episode_reward'])
    assert (tmp_path / 'best.pt').exists()
