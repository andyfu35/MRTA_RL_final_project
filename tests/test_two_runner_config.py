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


def test_record_progress_config_changes_only_progress_shaping_from_large_batch():
    large = load_two_runner_config('config/two_runner_large_batch.yaml')
    record = load_two_runner_config('config/two_runner_record_progress.yaml')

    assert record['training'] == large['training']
    assert record['collection_profiles'] == large['collection_profiles']
    assert record['environment'] == large['environment']
    assert record['ppo'] == large['ppo']
    assert record['validation'] == large['validation']

    large_reward = dict(large['two_runner_reward'])
    record_reward = dict(record['two_runner_reward'])
    assert record_reward.pop('progress_mode') == 'record'
    assert record_reward.pop('record_progress_epsilon') == 0.01
    assert record_reward == large_reward


def test_two_runner_config_rejects_unknown_progress_mode_and_negative_record_epsilon():
    cfg = yaml.safe_load(Path('config/two_runner_large_batch.yaml').read_text(encoding='utf-8'))
    cfg['two_runner_reward']['progress_mode'] = 'not-a-mode'
    try:
        validate_two_runner_config(cfg)
    except ValueError as exc:
        assert 'progress_mode' in str(exc)
    else:
        raise AssertionError('expected ValueError')

    cfg['two_runner_reward']['progress_mode'] = 'record'
    cfg['two_runner_reward']['record_progress_epsilon'] = -0.01
    try:
        validate_two_runner_config(cfg)
    except ValueError as exc:
        assert 'record_progress_epsilon' in str(exc)
    else:
        raise AssertionError('expected ValueError')
