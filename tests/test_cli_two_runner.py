import pytest

from marl2d.cli import build_parser, _format_two_runner_record


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


def test_two_runner_record_format_surfaces_success_batch_diagnostics():
    text = _format_two_runner_record({
        'round': 3,
        'agents': {
            'runner_0': {
                'samples': 32768, 'simulator_steps': 50000, 'target_deaths': 2,
                'completed_team_episodes': 120, 'team_success_episodes': 40,
                'selected_success_transitions': 9000, 'sample_pool_size': 36000,
                'discarded_surplus_samples': 3232, 'approx_kl': 0.0042,
            },
            'runner_1': {
                'samples': 32768, 'simulator_steps': 48000, 'target_deaths': 1,
                'completed_team_episodes': 118, 'team_success_episodes': 39,
                'selected_success_transitions': 8800, 'sample_pool_size': 35500,
                'discarded_surplus_samples': 2732, 'approx_kl': 0.0038,
            },
        },
    })
    assert 'round=   3' in text
    assert 'succ_ep=40' in text
    assert 'sel_succ=9000/32768' in text
    assert 'kl=0.00420' in text
