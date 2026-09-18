from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reward_loader import RewardEngine
from .runtime_config import (
    load_runtime_config,
    resolve_local_node,
)
from .ros2_runtime import run_cluster_node


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MARL2D ROS2 distributed training node. "
            "By default it reads ./config.json and the reward.py "
            "referenced by that file."
        )
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Deployment JSON file (default: ./config.json)",
    )
    parser.add_argument(
        "--node-id",
        default=None,
        help=(
            "Optional explicit node id. Normally the executable "
            "auto-detects itself from the IP list in config.json."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for this computer (default: cpu)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config/reward/local identity and exit before ROS2.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = load_runtime_config(Path(args.config))
    local = resolve_local_node(
        runtime,
        explicit_node_id=args.node_id,
    )
    reward_engine = RewardEngine.from_file(
        runtime.reward_path,
        params=runtime.raw["reward"].get("params", {}),
    )

    print(
        json.dumps(
            {
                "node_id": local.node_id,
                "ip": local.ip,
                "agent_id": local.agent_id,
                "train": local.train,
                "config": str(runtime.source_path),
                "config_sha256": runtime.config_sha256,
                "reward": str(runtime.reward_path),
                "reward_sha256": runtime.reward_sha256,
                "reward_terms": [
                    {
                        "name": term.name,
                        "mode": term.mode,
                        "order": term.order,
                    }
                    for term in reward_engine.terms
                ],
                "ros_domain_id": runtime.network.get(
                    "ros_domain_id", 42
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.check:
        return 0

    run_cluster_node(
        runtime,
        local,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
