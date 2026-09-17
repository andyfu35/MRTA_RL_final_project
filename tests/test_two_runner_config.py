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
