from pathlib import Path

from marl2d.config import load_single_runner_config


def test_experiment1_config_uses_r2_and_8192_samples_per_update():
    cfg = load_single_runner_config(Path('config/single_runner.yaml'))
    assert cfg['single_runner_reward']['mode'] == 'R2'
    assert cfg['training']['samples_per_update'] == 8192
    profile = cfg['collection']
    assert profile['parallel_envs'] == 32
    assert profile['rollout_steps'] == 256
    assert profile['batches'] == 1
    assert profile['parallel_envs'] * profile['rollout_steps'] * profile['batches'] == 8192
    assert cfg['environment']['max_wheel_linear_speed'] == 2.0
    assert cfg['validation']['every'] == 5
    assert cfg['validation']['episodes'] == 50
    assert cfg['validation']['seed_start'] == 9000
    assert cfg['evaluation']['seed_start'] == 10000
