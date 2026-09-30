from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from .env import EnvConfig, generate_world
from .global_optimal_core import solve_global_time_optimum


def _solve_split(
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
        result = solve_global_time_optimum(
            world,
            config,
            time_limit=time_limit,
        )

        print(
            f"{label.upper()} {len(rows):04d}/{count:04d} "
            f"seed={seed} optimal={result.optimal} "
            f"T*={result.time_optimality} gap={result.mip_gap} "
            f"solve={result.solve_seconds:.3f}s"
        )

        if not result.optimal or result.time_optimality is None:
            continue

        rows.append(
            {
                "seed": seed,
                "time_optimality_star": float(result.time_optimality),
                "completed_tasks_star": int(result.completed_tasks or 0),
                "solve_seconds": float(result.solve_seconds),
                "mip_gap": float(result.mip_gap or 0.0),
                "routes": [list(route) for route in result.routes],
            }
        )

    if len(rows) != count:
        raise RuntimeError(
            f"Could only prove {len(rows)}/{count} {label} worlds optimal "
            f"within {max_attempts} attempts."
        )
    return rows


def build(args: argparse.Namespace) -> Path:
    config = EnvConfig()
    train_rows = _solve_split(
        config=config,
        base_seed=args.train_seed,
        count=args.train_count,
        time_limit=args.time_limit,
        max_attempt_factor=args.max_attempt_factor,
        label="train",
    )
    probe_rows = _solve_split(
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
        "dataset": "gene_mrta_v16t_milp_oracle",
        "created_at": datetime.now().isoformat(),
        "objective": (
            "maximize sum(1 - finish_time / H) over completed tasks / N"
        ),
        "global_optimum_rule": (
            "Only HiGHS MILP status=optimal with requested mip_rel_gap=0 "
            "is retained."
        ),
        "config": asdict(config),
        "train_seed_base": args.train_seed,
        "probe_seed_base": args.probe_seed,
        "train": train_rows,
        "probe": probe_rows,
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"ORACLE_DATASET={output}")
    print(f"TRAIN_OPTIMAL_WORLDS={len(train_rows)}")
    print(f"PROBE_OPTIMAL_WORLDS={len(probe_rows)}")
    return output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train-count", type=int, default=256)
    p.add_argument("--probe-count", type=int, default=64)
    p.add_argument("--train-seed", type=int, default=110_000_000)
    p.add_argument("--probe-seed", type=int, default=120_000_000)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--max-attempt-factor", type=int, default=3)
    p.add_argument(
        "--output",
        default="runs/gene_mrta_v16to_oracle/oracle_dataset.json",
    )
    return p


def main() -> None:
    build(parser().parse_args())


if __name__ == "__main__":
    main()
