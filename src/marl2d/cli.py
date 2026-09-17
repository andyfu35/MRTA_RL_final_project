from __future__ import annotations

import argparse
from pathlib import Path

from . import AGENT_IDS
from .config import load_config
from .render import render_policy_set_gif
from .trainer import DistributedTrainer, load_checkpoint


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
