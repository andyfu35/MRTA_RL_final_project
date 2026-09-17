from copy import deepcopy
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


def test_safety_ablation_configs_only_change_safety_scale_and_validation_size():
    baseline = load_single_runner_config(Path('config/single_runner.yaml'))

    for path, safety_scale in (
        ('config/single_runner_safe15.yaml', 1.5),
        ('config/single_runner_safe30.yaml', 3.0),
    ):
        cfg = load_single_runner_config(Path(path))
        assert cfg['single_runner_reward']['mode'] == 'R2'
        assert cfg['single_runner_reward']['safety_scale'] == safety_scale
        assert cfg['validation'] == {
            'every': 5,
            'episodes': 200,
            'seed_start': 9000,
        }
        assert cfg['evaluation']['seed_start'] == 10000

        normalized = deepcopy(cfg)
        normalized['single_runner_reward']['safety_scale'] = baseline['single_runner_reward']['safety_scale']
        normalized['validation']['episodes'] = baseline['validation']['episodes']
        assert normalized == baseline
