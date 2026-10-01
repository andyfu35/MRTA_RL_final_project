from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world

from .capability_oracle import (
    EXACT_AXES,
    solve_world_capability_ceilings,
)


def _build_split(
    *,
    config: EnvConfig,
    base_seed: int,
    count: int,
    time_limit: float,
    max_attempt_factor: int,
    label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attempts = 0
    max_attempts = max(count, count * max_attempt_factor)

    while len(rows) < count and attempts < max_attempts:
        seed = base_seed + attempts
        attempts += 1
        world = generate_world(config, seed)
        ceilings, results = solve_world_capability_ceilings(
            world,
            config,
            time_limit=time_limit,
        )

        status = ", ".join(
            f"{axis}={'ok' if result.optimal else 'timeout'}"
            for axis, result in results.items()
        )
        total_solve = sum(
            result.solve_seconds for result in results.values()
        )
        print(
            f"{label.upper()} {len(rows):03d}/{count:03d} "
            f"seed={seed} | {status} | total={total_solve:.2f}s"
        )

        if ceilings is None:
            continue

        rows.append(
            {
                "seed": seed,
                "ceilings": ceilings.to_dict(),
                "oracle_details": {
                    axis: {
                        "value": result.value,
                        "optimal": result.optimal,
                        "mip_gap": result.mip_gap,
                        "solve_seconds": result.solve_seconds,
                        "routes": [list(route) for route in result.routes],
                    }
                    for axis, result in results.items()
                },
            }
        )

    if len(rows) != count:
        raise RuntimeError(
            f"Only {len(rows)}/{count} {label} worlds obtained all exact "
            f"oracles within {max_attempts} attempts"
        )
    return rows


def build(args: argparse.Namespace) -> Path:
    config = EnvConfig()
    train = _build_split(
        config=config,
        base_seed=args.train_seed,
        count=args.train_count,
        time_limit=args.time_limit,
        max_attempt_factor=args.max_attempt_factor,
        label="train",
    )
    probe = _build_split(
        config=config,
        base_seed=args.probe_seed,
        count=args.probe_count,
        time_limit=args.time_limit,
        max_attempt_factor=args.max_attempt_factor,
        label="probe",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "gene_mrta_v17_multi_axis_capability_oracle",
        "created_at": datetime.now().isoformat(),
        "config": asdict(config),
        "exact_milp_axes": list(EXACT_AXES),
        "balance_reference": (
            "analytical upper bound completion_star; not claimed as exact "
            "Jain-balance optimum"
        ),
        "train_seed_base": args.train_seed,
        "probe_seed_base": args.probe_seed,
        "train": train,
        "probe": probe,
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"CAPABILITY_ORACLE_DATASET={output}")
    print(f"TRAIN_WORLDS={len(train)}")
    print(f"PROBE_WORLDS={len(probe)}")
    return output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train-count", type=int, default=32)
    p.add_argument("--probe-count", type=int, default=8)
    p.add_argument("--train-seed", type=int, default=210_000_000)
    p.add_argument("--probe-seed", type=int, default=220_000_000)
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--max-attempt-factor", type=int, default=4)
    p.add_argument(
        "--output",
        default="runs/gene_mrta_v17_oracle/capability_oracle.json",
    )
    return p


def main() -> None:
    build(parser().parse_args())


if __name__ == "__main__":
    main()
