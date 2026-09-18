from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import AGENT_IDS
from .config import (
    load_config,
    load_single_runner_config,
    load_two_runner_config,
    validate_single_runner_config,
)
from .render import find_two_runner_demo_seed, render_policy_set_gif, render_two_runner_gif
from .single_runner import SingleRunnerTrainer, evaluate_single_runner, load_single_runner_checkpoint
from .trainer import DistributedTrainer, load_checkpoint
from .two_runner import (
    TWO_RUNNER_IDS,
    TwoRunnerTrainer,
    evaluate_two_runner,
    load_two_runner_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic 2D synchronous multi-agent PPO prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run mock-distributed synchronous training")
    train.add_argument("--config", default="config/default.yaml")
    train.add_argument("--rounds", type=int, default=None)
    train.add_argument("--output", default="runs/default")
    train.add_argument("--device", default="cpu")

    render = sub.add_parser("render", help="Render one deterministic top-view episode from a checkpoint")
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--output", default="evaluation.gif")
    render.add_argument("--max-steps", type=int, default=None)
    render.add_argument("--device", default="cpu")
    render.add_argument("--config", default=None, help="Optional config override; normally checkpoint config is used")

    single_train = sub.add_parser("single-train", help="Train Experiment 1 single-runner PPO navigation")
    single_train.add_argument("--config", default="config/single_runner.yaml")
    single_train.add_argument("--rounds", type=int, default=None, help="Additional PPO rounds to run")
    single_train.add_argument("--output", default="runs/single_runner")
    single_train.add_argument("--device", default="cpu")
    single_train.add_argument("--reward-mode", choices=("R0", "R1", "R2", "R3"), default=None)
    single_train.add_argument(
        "--resume",
        default=None,
        help="Resume from a single-runner checkpoint. New-format checkpoints restore full training state.",
    )
    single_train.add_argument(
        "--reset-best-validation",
        action="store_true",
        help=(
            "Reset inherited best-checkpoint selection history after resume. "
            "Use this when forking a new experiment from an existing checkpoint."
        ),
    )

    single_eval = sub.add_parser("single-eval", help="Evaluate a single-runner checkpoint on held-out maps")
    single_eval.add_argument("--checkpoint", required=True)
    single_eval.add_argument("--episodes", type=int, default=None)
    single_eval.add_argument("--seed-start", type=int, default=None)
    single_eval.add_argument("--device", default="cpu")
    single_eval.add_argument("--output", default=None, help="Optional JSON file for evaluation summary")

    two_train = sub.add_parser("two-train", help="Train Experiment 2 cooperative two-runner PPO")
    two_train.add_argument("--config", default="config/two_runner.yaml")
    two_train.add_argument("--rounds", type=int, default=None, help="Additional PPO rounds to run")
    two_train.add_argument("--output", default="runs/exp2_two_runner")
    two_train.add_argument("--device", default="cpu")
    source = two_train.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--init-single-runner-checkpoint",
        default=None,
        help="Start Experiment 2 from the finalized 15-D single-runner checkpoint.",
    )
    source.add_argument(
        "--resume",
        default=None,
        help="Resume a full Experiment 2 checkpoint.",
    )
    two_train.add_argument(
        "--reset-best-validation",
        action="store_true",
        help="Reset inherited Experiment 2 best-validation history after resume.",
    )

    two_eval = sub.add_parser("two-eval", help="Evaluate an Experiment 2 two-runner checkpoint")
    two_eval.add_argument("--checkpoint", required=True)
    two_eval.add_argument("--episodes", type=int, default=None)
    two_eval.add_argument("--seed-start", type=int, default=None)
    two_eval.add_argument("--device", default="cpu")
    two_eval.add_argument("--output", default=None, help="Optional JSON file for evaluation summary")

    two_render = sub.add_parser(
        "two-render",
        help="Render one deterministic Experiment 2 episode as a top-view GIF",
    )
    two_render.add_argument("--checkpoint", required=True)
    two_render.add_argument("--output", default="two_runner_demo.gif")
    two_render.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Render this exact seed. If omitted, search validation seeds for a collision-free success.",
    )
    two_render.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Seed search start. Defaults to the checkpoint validation seed_start.",
    )
    two_render.add_argument(
        "--search-episodes",
        type=int,
        default=200,
        help="How many deterministic seeds to search when --seed is omitted.",
    )
    two_render.add_argument("--max-steps", type=int, default=None)
    two_render.add_argument("--device", default="cpu")

    return parser


def _print_round(record: dict) -> None:
    round_id = record["round"]
    print(f"\n=== COMMITTED ROUND {round_id} ===")
    for agent_id in AGENT_IDS:
        m = record["agents"][agent_id]
        print(
            f"{agent_id:9s} samples={int(m['samples']):5d} "
            f"reward={m['mean_step_reward']:+.4f} "
            f"policy_loss={m['policy_loss']:+.4f} "
            f"value_loss={m['value_loss']:.4f} "
            f"coll={m['collision_rate']:.3f}"
        )


def _format_two_runner_record(record: dict) -> str:
    pieces = [f"round={int(record['round']):4d}"]
    for agent_id in TWO_RUNNER_IDS:
        metrics = record["agents"][agent_id]
        freshness = ""
        if int(metrics.get("shared_joint_rollout", 0)):
            freshness = (
                f" joint=1"
                f" pv={int(metrics.get('rollout_policy_version', -1))}"
                f" dataep={int(metrics.get('ppo_data_epochs', 0))}"
            )
        pieces.append(
            f"{agent_id}:samples={int(metrics['samples'])} "
            f"sim={int(metrics['simulator_steps'])} "
            f"ep={int(metrics.get('completed_team_episodes', 0))} "
            f"succ_ep={int(metrics.get('team_success_episodes', 0))} "
            f"sel_succ={int(metrics.get('selected_success_transitions', 0))}/{int(metrics['samples'])} "
            f"pool={int(metrics.get('sample_pool_size', metrics['samples']))} "
            f"discard={int(metrics.get('discarded_surplus_samples', 0))} "
            f"kl={float(metrics.get('approx_kl', 0.0)):.5f} "
            f"guard_kl={float(metrics.get('max_guard_kl', 0.0)):.5f} "
            f"opt={int(metrics.get('optimizer_steps', 0))} "
            f"stop={int(metrics.get('early_stopped', 0))}"
            f"{freshness}"
        )
    if "validation" in record:
        validation = record["validation"]
        pieces.append(
            f"val_success={validation['team_success_rate']:.3f} "
            f"val_collision={validation['any_collision_rate']:.3f} "
            f"val_both_dead={validation['both_dead_rate']:.3f}"
        )
    return " ".join(pieces)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        cfg = load_config(args.config)
        trainer = DistributedTrainer(cfg, output_dir=args.output, device=args.device)
        records = trainer.run(rounds=args.rounds)
        for record in records:
            _print_round(record)
        print(f"\nCheckpoint: {Path(args.output) / 'latest.pt'}")
        print(f"Metrics:    {Path(args.output) / 'metrics.jsonl'}")
        return 0

    if args.command == "single-train":
        cfg = load_single_runner_config(args.config)
        if args.reward_mode is not None:
            cfg = copy.deepcopy(cfg)
            cfg["single_runner_reward"]["mode"] = args.reward_mode
            validate_single_runner_config(cfg)
        trainer = SingleRunnerTrainer(cfg, output_dir=args.output, device=args.device)
        if args.resume:
            resume_mode = trainer.resume_from_checkpoint(args.resume)
            if args.reset_best_validation:
                trainer.best_validation = None
                print("Reset inherited best-validation history for this experiment fork.")
            if resume_mode == "full":
                print(f"Resumed full training state from round {trainer.current_round}: {args.resume}")
            else:
                print(
                    f"Resumed legacy checkpoint from round {trainer.current_round}: {args.resume}\n"
                    "WARNING: this older checkpoint has no optimizer/world state; "
                    "optimizer and persistent worlds were reinitialized."
                )
        elif args.reset_best_validation:
            trainer.best_validation = None
        records = trainer.run(rounds=args.rounds)
        for record in records:
            validation_text = ""
            if "val_success_rate" in record:
                validation_text = (
                    f" val_success={record['val_success_rate']:.3f}"
                    f" val_collision={record['val_collision_rate']:.3f}"
                    f" val_timeout={record['val_timeout_rate']:.3f}"
                )
            print(
                f"round={int(record['round']):4d} "
                f"samples={int(record['samples']):5d} "
                f"reward={record['mean_step_reward']:+.4f} "
                f"episodes={int(record['episodes']):4d} "
                f"success={record['success_rate']:.3f} "
                f"maps={int(record['unique_map_seeds']):4d} "
                f"action_std={record['stochastic_action_std']:.3f}"
                f"{validation_text}"
            )
        print(f"\nReward mode: {cfg['single_runner_reward']['mode']}")
        print(f"Checkpoint:  {Path(args.output) / 'latest.pt'}")
        if (Path(args.output) / "best.pt").exists():
            print(f"Best:        {Path(args.output) / 'best.pt'}")
        print(f"Metrics:     {Path(args.output) / 'metrics.jsonl'}")
        return 0

    if args.command == "single-eval":
        version, snapshot, cfg = load_single_runner_checkpoint(args.checkpoint)
        evaluation_cfg = cfg["evaluation"]
        episodes = int(args.episodes if args.episodes is not None else evaluation_cfg["episodes"])
        seed_start = int(args.seed_start if args.seed_start is not None else evaluation_cfg["seed_start"])
        summary = evaluate_single_runner(
            cfg,
            snapshot,
            episodes=episodes,
            seed_start=seed_start,
            device=args.device,
        )
        summary["policy_version"] = version
        summary["reward_mode"] = cfg["single_runner_reward"]["mode"]
        text = json.dumps(summary, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
        return 0

    if args.command == "two-train":
        cfg = load_two_runner_config(args.config)
        trainer = TwoRunnerTrainer(cfg, output_dir=args.output, device=args.device)
        if args.resume:
            resume_mode = trainer.resume_from_checkpoint(args.resume)
            if args.reset_best_validation:
                trainer.best_validation = None
                print("Reset inherited Experiment 2 best-validation history.")
            print(f"Resumed {resume_mode} two-runner state from round {trainer.current_round}: {args.resume}")
        else:
            trainer.initialize_from_single_runner_checkpoint(args.init_single_runner_checkpoint)
            print(f"Initialized both runners from: {args.init_single_runner_checkpoint}")
        rounds_to_run = int(args.rounds if args.rounds is not None else cfg["training"]["rounds"])
        for _ in range(rounds_to_run):
            record = trainer.run(rounds=1)[0]
            print(_format_two_runner_record(record), flush=True)
        print(f"\nCheckpoint: {Path(args.output) / 'latest.pt'}")
        if (Path(args.output) / "best.pt").exists():
            print(f"Best:       {Path(args.output) / 'best.pt'}")
        print(f"Metrics:    {Path(args.output) / 'metrics.jsonl'}")
        return 0

    if args.command == "two-eval":
        version, policy_set, cfg = load_two_runner_checkpoint(args.checkpoint)
        evaluation_cfg = cfg["evaluation"]
        episodes = int(args.episodes if args.episodes is not None else evaluation_cfg["episodes"])
        seed_start = int(args.seed_start if args.seed_start is not None else evaluation_cfg["seed_start"])
        summary = evaluate_two_runner(
            cfg,
            policy_set,
            episodes=episodes,
            seed_start=seed_start,
            device=args.device,
        )
        summary["policy_version"] = version
        text = json.dumps(summary, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
        return 0

    if args.command == "two-render":
        version, policy_set, cfg = load_two_runner_checkpoint(args.checkpoint)
        if args.seed is None:
            validation_cfg = cfg.get("validation", {})
            seed_start = int(
                args.seed_start
                if args.seed_start is not None
                else validation_cfg.get("seed_start", 40000)
            )
            seed, found = find_two_runner_demo_seed(
                cfg,
                policy_set,
                seed_start=seed_start,
                search_episodes=int(args.search_episodes),
                device=args.device,
                max_steps=args.max_steps,
            )
            clean_text = "collision-free" if not found["any_collision"] else "successful"
            print(
                f"Found {clean_text} deterministic demo seed {seed}: "
                f"steps={found['steps']} collision={found['any_collision']}"
            )
        else:
            seed = int(args.seed)

        summary = render_two_runner_gif(
            cfg,
            policy_set,
            args.output,
            seed=seed,
            max_steps=args.max_steps,
            device=args.device,
        )
        summary["policy_version"] = version
        print(f"Rendered two-runner policy version {version} seed={seed} -> {args.output}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    version, policy_set, checkpoint_cfg = load_checkpoint(args.checkpoint)
    cfg = load_config(args.config) if args.config else checkpoint_cfg
    summary = render_policy_set_gif(
        cfg,
        policy_set,
        args.output,
        max_steps=args.max_steps,
        device=args.device,
    )
    print(f"Rendered policy version {version} -> {args.output}")
    print(f"Summary: {summary}")
    return 0
