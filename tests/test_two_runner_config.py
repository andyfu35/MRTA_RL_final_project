from pathlib import Path

import yaml

from marl2d.config import load_two_runner_config, validate_two_runner_config


def test_formal_two_runner_config_has_locked_experiment2_values():
    cfg = load_two_runner_config('config/two_runner.yaml')
    assert cfg['training']['samples_per_update'] == 8192
    assert set(cfg['collection_profiles']) == {'runner_0', 'runner_1'}
    assert cfg['two_runner_reward']['team_success_bonus'] == 100.0
    assert cfg['two_runner_reward']['collision_penalty'] == -100.0
    assert cfg['two_runner_reward']['self_progress_scale'] == 3.0
    assert cfg['two_runner_reward']['team_progress_scale'] == 2.0
    assert cfg['two_runner_reward']['safety_scale'] == 3.0
    assert cfg['validation'] == {'every': 5, 'episodes': 200, 'seed_start': 40000}
    assert cfg['evaluation']['episodes'] == 200
    assert cfg['evaluation']['seed_start'] == 50000
    assert cfg['final_test']['seed_start'] == 60000


def test_two_runner_config_rejects_wrong_agent_profiles():
    cfg = yaml.safe_load(Path('config/two_runner.yaml').read_text(encoding='utf-8'))
    cfg['collection_profiles'].pop('runner_1')
    try:
        validate_two_runner_config(cfg)
    except ValueError as exc:
        assert 'collection_profiles' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_large_batch_config_preserves_total_used_samples_and_changes_only_batch_schedule():
    baseline = load_two_runner_config('config/two_runner.yaml')
    large = load_two_runner_config('config/two_runner_large_batch.yaml')

    assert baseline['training']['samples_per_update'] == 8192
    assert baseline['training']['rounds'] == 100
    assert large['training']['samples_per_update'] == 32768
    assert large['training']['rounds'] == 25
    assert baseline['training']['samples_per_update'] * baseline['training']['rounds'] == (
        large['training']['samples_per_update'] * large['training']['rounds']
    )
    assert large['training']['checkpoint_every'] == 1
    assert large['validation'] == {'every': 1, 'episodes': 200, 'seed_start': 40000}
    assert large['environment'] == baseline['environment']
    assert large['two_runner_reward'] == baseline['two_runner_reward']
    assert large['collection_profiles'] == baseline['collection_profiles']
    assert large['ppo'] == baseline['ppo']


def test_kl_guard_config_changes_only_target_kl_from_large_batch():
    large = load_two_runner_config('config/two_runner_large_batch.yaml')
    guarded = load_two_runner_config('config/two_runner_large_batch_kl_guard.yaml')

    assert guarded['training'] == large['training']
    assert guarded['collection_profiles'] == large['collection_profiles']
    assert guarded['environment'] == large['environment']
    assert guarded['two_runner_reward'] == large['two_runner_reward']
    assert guarded['validation'] == large['validation']

    large_ppo = dict(large['ppo'])
    guarded_ppo = dict(guarded['ppo'])
    assert guarded_ppo.pop('target_kl') == 0.015
    assert guarded_ppo == large_ppo


def test_two_runner_config_rejects_nonpositive_target_kl():
    cfg = yaml.safe_load(Path('config/two_runner_large_batch.yaml').read_text(encoding='utf-8'))
    cfg['ppo']['target_kl'] = 0.0
    try:
        validate_two_runner_config(cfg)
    except ValueError as exc:
        assert 'target_kl' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_fresh_joint_rollout_config_uses_single_epoch_and_frequent_fresh_data():
    large = load_two_runner_config('config/two_runner_large_batch.yaml')
    fresh = load_two_runner_config('config/two_runner_fresh_joint.yaml')

    assert fresh['environment'] == large['environment']
    assert fresh['two_runner_reward'] == large['two_runner_reward']
    assert fresh['training']['samples_per_update'] == 8192
    assert fresh['training']['rounds'] == 100
    assert fresh['training']['samples_per_update'] * fresh['training']['rounds'] == 819200

    assert fresh['joint_collection'] == {
        'enabled': True,
        'parallel_envs': 64,
        'rollout_steps': 128,
    }
    assert fresh['ppo']['epochs'] == 1
    assert fresh['ppo']['minibatch_size'] == 256
    assert 'target_kl' not in fresh['ppo']

    # Every selected transition is consumed in exactly one PPO epoch before
    # the next policy version must collect a new rollout.
    assert fresh['training']['samples_per_update'] // fresh['ppo']['minibatch_size'] == 32

    for key in (
        'hidden_sizes', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range',
        'value_coef', 'entropy_coef', 'max_grad_norm', 'minibatch_size',
    ):
        assert fresh['ppo'][key] == large['ppo'][key]


def test_joint_collection_requires_positive_parallel_world_count():
    cfg = yaml.safe_load(Path('config/two_runner.yaml').read_text(encoding='utf-8'))
    cfg['joint_collection'] = {'enabled': True, 'parallel_envs': 0, 'rollout_steps': 128}
    try:
        validate_two_runner_config(cfg)
    except ValueError as exc:
        assert 'joint_collection.parallel_envs' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_joint_collection_requires_positive_rollout_steps():
    cfg = yaml.safe_load(Path('config/two_runner.yaml').read_text(encoding='utf-8'))
    cfg['joint_collection'] = {
        'enabled': True,
        'parallel_envs': 64,
        'rollout_steps': 0,
    }
    try:
        validate_two_runner_config(cfg)
    except ValueError as exc:
        assert 'joint_collection.rollout_steps' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_fresh_joint_dense_goal_progress_is_increased_without_changing_terminal_bonus():
    large = load_two_runner_config('config/two_runner_large_batch.yaml')
    fresh = load_two_runner_config('config/two_runner_fresh_joint.yaml')
    assert fresh['two_runner_reward']['team_success_bonus'] == large['two_runner_reward']['team_success_bonus'] == 100.0
    assert fresh['two_runner_reward']['self_progress_scale'] == 8.0
    assert fresh['two_runner_reward']['team_progress_scale'] == 4.0
    assert fresh['two_runner_reward']['collision_penalty'] == large['two_runner_reward']['collision_penalty']
    assert fresh['two_runner_reward']['safety_scale'] == large['two_runner_reward']['safety_scale']
