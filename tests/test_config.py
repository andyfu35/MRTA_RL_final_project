from pathlib import Path

import pytest

from marl2d.config import load_config, validate_config


def test_default_profiles_collect_equal_samples():
    cfg = load_config(Path('config/default.yaml'))
    expected = cfg['training']['samples_per_update']
    for agent_id, profile in cfg['collection_profiles'].items():
        actual = profile['parallel_envs'] * profile['rollout_steps'] * profile['batches']
        assert actual == expected, (agent_id, actual, expected)


def test_default_map_uses_large_random_rectangular_arena():
    cfg = load_config(Path('config/default.yaml'))
    env = cfg['environment']
    assert env['width'] == 20.0
    assert env['height'] == 20.0
    assert env['obstacles']['count'] == 14
    assert env['body_length'] > env['wheel_width']
    assert env['max_wheel_linear_speed'] == 2.0
    assert 'max_wheel_speed' not in env


def test_obstacle_config_must_be_mapping():
    cfg = load_config(Path('config/default.yaml'))
    cfg['environment']['obstacles'] = []
    with pytest.raises(ValueError, match='obstacles must be a mapping'):
        validate_config(cfg)
