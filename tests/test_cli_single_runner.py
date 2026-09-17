from marl2d.cli import build_parser


def test_single_train_cli_accepts_reward_ablation_mode():
    args = build_parser().parse_args([
        'single-train',
        '--config', 'config/single_runner.yaml',
        '--reward-mode', 'R3',
        '--rounds', '2',
    ])
    assert args.command == 'single-train'
    assert args.reward_mode == 'R3'
    assert args.rounds == 2


def test_single_train_cli_accepts_resume_checkpoint():
    args = build_parser().parse_args([
        'single-train',
        '--config', 'config/single_runner.yaml',
        '--resume', 'runs/exp1/latest.pt',
        '--rounds', '50',
    ])
    assert args.command == 'single-train'
    assert args.resume == 'runs/exp1/latest.pt'
    assert args.rounds == 50


def test_single_eval_cli_accepts_held_out_seed_controls():
    args = build_parser().parse_args([
        'single-eval',
        '--checkpoint', 'runs/exp1/latest.pt',
        '--episodes', '25',
        '--seed-start', '20000',
    ])
    assert args.command == 'single-eval'
    assert args.episodes == 25
    assert args.seed_start == 20000
