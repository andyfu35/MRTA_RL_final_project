from pathlib import Path

import torch

from marl2d.single_runner import SingleRunnerTrainer


def _minimal_cfg():
    return {
        'seed': 42,
        'single_runner_reward': {'mode': 'R2'},
        'ppo': {
            'hidden_sizes': [8],
            'learning_rate': 3e-4,
        },
    }


def test_resume_can_reset_inherited_best_validation(tmp_path: Path):
    cfg = _minimal_cfg()
    source = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'source', device='cpu')
    checkpoint = tmp_path / 'legacy_with_best.pt'
    inherited_best = {
        'round': 70,
        'success_rate': 0.24,
        'collision_rate': 0.06,
        'mean_episode_reward': 30.0,
    }
    torch.save(
        {
            'version': 70,
            'model': source.snapshot(),
            'config': cfg,
            'obs_dim': source.obs_dim,
            'action_dim': source.action_dim,
            'best_validation': inherited_best,
        },
        checkpoint,
    )

    resumed = SingleRunnerTrainer(cfg, output_dir=tmp_path / 'resumed', device='cpu')
    mode = resumed.resume_from_checkpoint(checkpoint, reset_best_validation=True)

    assert mode == 'legacy'
    assert resumed.current_round == 70
    assert resumed.best_validation is None
