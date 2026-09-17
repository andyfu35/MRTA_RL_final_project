import pytest

from marl2d.cli import build_parser


def test_existing_commands_remain_available_with_experiment2_cli():
    parser = build_parser()
    assert parser.parse_args(['train']).command == 'train'
    assert parser.parse_args(['single-train']).command == 'single-train'
    assert parser.parse_args(['single-eval', '--checkpoint', 'x.pt']).command == 'single-eval'
    assert parser.parse_args(['render', '--checkpoint', 'x.pt']).command == 'render'


def test_two_train_accepts_single_runner_initialization_checkpoint():
    args = build_parser().parse_args([
        'two-train',
        '--config', 'config/two_runner.yaml',
        '--init-single-runner-checkpoint', 'runs/exp1/round_00080.pt',
        '--rounds', '2',
    ])
    assert args.command == 'two-train'
    assert args.init_single_runner_checkpoint.endswith('round_00080.pt')
    assert args.resume is None


def test_two_train_init_and_resume_are_mutually_exclusive_and_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['two-train'])
    with pytest.raises(SystemExit):
        parser.parse_args([
            'two-train',
            '--init-single-runner-checkpoint', 'a.pt',
            '--resume', 'b.pt',
        ])


def test_two_eval_accepts_seed_controls():
    args = build_parser().parse_args([
        'two-eval',
        '--checkpoint', 'runs/exp2/best.pt',
        '--episodes', '200',
        '--seed-start', '40000',
    ])
    assert args.command == 'two-eval'
    assert args.episodes == 200
    assert args.seed_start == 40000
