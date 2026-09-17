from pathlib import Path

from marl2d.config import load_config


def test_default_profiles_collect_equal_samples():
    cfg = load_config(Path('config/default.yaml'))
    expected = cfg['training']['samples_per_update']
    for agent_id, profile in cfg['collection_profiles'].items():
        actual = profile['parallel_envs'] * profile['rollout_steps'] * profile['batches']
        assert actual == expected, (agent_id, actual, expected)
