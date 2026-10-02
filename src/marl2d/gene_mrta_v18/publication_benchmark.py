from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from marl2d.gene_mrta_v16t.env import EnvConfig, generate_world
from marl2d.gene_mrta_v16t.global_optimal_core import solve_global_time_optimum
from marl2d.gene_mrta_v16t.hungarian_benchmark import rollout_world
from marl2d.gene_mrta_v17.rollout import rollout_direct_gene as rollout_v17

from .global_time_test import _load_v16to, _load_v17, _load_v18
from .rollout import rollout_direct_gene as rollout_v18


PROTOCOL_VERSION = "v18_publication_final_100_v1"
MODEL_CODE_FREEZE_COMMIT = "d58a621c4cf131e0c7a0a08bf64ea0b0a93d8476"
PUBLICATION_SEED_BASE = 98_000_000
PUBLICATION_WORLD_COUNT = 100
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_261_002


def _gene_hash(gene: Any) -> str:
    if hasattr(gene, "to_dict"):
        payload = gene.to_dict()
    elif hasattr(gene, "weights"):
        payload = {
            "type": gene.__class__.__name__,
            "weights": np.asarray(gene.weights, dtype=np.float64).tolist(),
        }
    else:
        payload = {
            "type": gene.__class__.__name__,
            "repr": repr(gene),
        }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(reps, dtype=np.float64)
    chunk = 1000
    offset = 0
    while offset < reps:
        count = min(chunk, reps - offset)
        ids = rng.integers(0, n, size=(count, n))
        means[offset : offset + count] = values[ids].mean(axis=1)
        offset += count

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _method_stats(
    values: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    low, high = _bootstrap_mean_ci(values, seed=seed)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "max": float(np.max(values)),
        "mean_ci95_low": low,
        "mean_ci95_high": high,
    }


def _paired_stats(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = a - b
    low, high = _bootstrap_mean_ci(delta, seed=seed)

    atol = 1e-12
    wins = int(np.sum(delta > atol))
    ties = int(np.sum(np.isclose(delta, 0.0, atol=atol)))
    losses = int(np.sum(delta < -atol))

    if delta.size > 1:
        delta_std = float(np.std(delta, ddof=1))
        cohen_dz = (
            float(np.mean(delta) / delta_std)
            if delta_std > 1e-15
            else 0.0
        )
    else:
        delta_std = 0.0
        cohen_dz = 0.0

    nonzero = delta[~np.isclose(delta, 0.0, atol=atol)]
    if nonzero.size == 0:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    else:
        result = wilcoxon(
            delta,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_stat = float(result.statistic)
        wilcoxon_p = float(result.pvalue)

    return {
        "mean_delta": float(np.mean(delta)),
        "std_delta": delta_std,
        "median_delta": float(np.median(delta)),
        "mean_delta_ci95_low": low,
        "mean_delta_ci95_high": high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cohen_dz": cohen_dz,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_two_sided": wilcoxon_p,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)


def _manifest_payload(
    *,
    args: argparse.Namespace,
    config: EnvConfig,
    hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "architecture_frozen": True,
        "model_code_freeze_commit": MODEL_CODE_FREEZE_COMMIT,
        "seed_base": int(args.world_seed),
        "world_count": int(args.worlds),
        "seed_end_inclusive": int(args.world_seed + args.worlds - 1),
        "reserved_before_evaluation": True,
        "primary_population": (
            "worlds with proven exact MILP optimum; report count explicitly"
        ),
        "oracle_initial_time_limit_seconds": float(args.time_limit),
        "oracle_retry_time_limit_seconds": float(args.retry_time_limit),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "environment": asdict(config),
        "runs": {
            "v18": args.v18_run,
            "v17": args.v17_run,
            "v16to": args.v16to_run,
        },
        "gene_sha256": hashes,
        "methods": [
            "v18_consequence_direct",
            "v17_direct",
            "v16to_bid_greedy",
            "hungarian_path_time",
            "global_milp_time_optimum",
        ],
        "primary_contrasts": [
            "v18_minus_v17",
            "v18_minus_v16to",
            "v18_minus_hungarian",
        ],
        "statistics": [
            "mean",
            "std",
            "median",
            "min",
            "p10",
            "p25",
            "max",
            "bootstrap_95pct_CI_of_mean",
            "paired_bootstrap_95pct_CI_of_mean_difference",
            "paired_wilcoxon_two_sided",
            "paired_cohen_dz",
            "win_tie_loss",
        ],
        "contamination_rule": (
            "These publication seeds must not be used to change or select "
            "the frozen V1.8 architecture. If the model is changed after "
            "inspection, this benchmark becomes analysis data and a new "
            "untouched seed range is required."
        ),
    }


def _validate_or_write_manifest(
    path: Path,
    payload: dict[str, object],
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "Publication freeze manifest already exists but does not "
                "match the requested protocol. Use the exact frozen settings "
                "or a new output directory."
            )
        return
    _atomic_json(path, payload)


def _evaluate_world(
    *,
    seed: int,
    config: EnvConfig,
    v18_gene,
    v17_gene,
    v16to_gene,
    time_limit: float,
    retry_time_limit: float,
) -> dict[str, object]:
    world = generate_world(config, seed)

    v18_eval = rollout_v18(v18_gene, world, config).evaluation
    v17_eval = rollout_v17(v17_gene, world, config).evaluation
    v16to_eval = rollout_world(
        world,
        config,
        score_mode="gene",
        matcher="greedy",
        gene=v16to_gene,
    ).evaluation
    hungarian_eval = rollout_world(
        world,
        config,
        score_mode="path_time",
        matcher="hungarian",
    ).evaluation

    oracle = solve_global_time_optimum(
        world,
        config,
        time_limit=time_limit,
    )
    oracle_attempts = 1
    if not oracle.optimal and retry_time_limit > time_limit:
        oracle = solve_global_time_optimum(
            world,
            config,
            time_limit=retry_time_limit,
        )
        oracle_attempts = 2

    row: dict[str, object] = {
        "world_seed": seed,
        "v18_time": float(v18_eval.time_optimality),
        "v17_time": float(v17_eval.time_optimality),
        "v16to_time": float(v16to_eval.time_optimality),
        "hungarian_time": float(hungarian_eval.time_optimality),
        "global_time": oracle.time_optimality,
        "optimal": bool(oracle.optimal),
        "mip_gap": oracle.mip_gap,
        "solve_seconds": float(oracle.solve_seconds),
        "oracle_attempts": oracle_attempts,
        "v18_completed_tasks": float(v18_eval.completed_tasks),
        "v17_completed_tasks": float(v17_eval.completed_tasks),
        "v16to_completed_tasks": float(v16to_eval.completed_tasks),
        "hungarian_completed_tasks": float(hungarian_eval.completed_tasks),
    }

    if oracle.optimal and oracle.time_optimality:
        star = float(oracle.time_optimality)
        row.update(
            {
                "v18_retention": float(v18_eval.time_optimality) / star,
                "v17_retention": float(v17_eval.time_optimality) / star,
                "v16to_retention": float(v16to_eval.time_optimality) / star,
                "hungarian_retention": (
                    float(hungarian_eval.time_optimality) / star
                ),
            }
        )

    return row


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    proven = [row for row in rows if row.get("optimal")]

    aggregate: dict[str, object] = {
        "worlds_requested": len(rows),
        "worlds_proven_optimal": len(proven),
        "worlds_not_proven_optimal": len(rows) - len(proven),
    }
    if not proven:
        return aggregate

    method_names = ("v18", "v17", "v16to", "hungarian")
    arrays = {
        name: np.asarray(
            [float(row[f"{name}_retention"]) for row in proven],
            dtype=np.float64,
        )
        for name in method_names
    }

    aggregate["methods"] = {
        name: _method_stats(
            values,
            seed=BOOTSTRAP_SEED + idx,
        )
        for idx, (name, values) in enumerate(arrays.items())
    }

    aggregate["paired"] = {
        "v18_minus_v17": _paired_stats(
            arrays["v18"],
            arrays["v17"],
            seed=BOOTSTRAP_SEED + 100,
        ),
        "v18_minus_v16to": _paired_stats(
            arrays["v18"],
            arrays["v16to"],
            seed=BOOTSTRAP_SEED + 101,
        ),
        "v18_minus_hungarian": _paired_stats(
            arrays["v18"],
            arrays["hungarian"],
            seed=BOOTSTRAP_SEED + 102,
        ),
    }

    aggregate["exact_optimum_matches"] = {
        name: int(np.sum(np.isclose(values, 1.0, atol=1e-12)))
        for name, values in arrays.items()
    }
    aggregate["oracle_solve_seconds"] = {
        "mean": float(
            np.mean([float(row["solve_seconds"]) for row in proven])
        ),
        "median": float(
            np.median([float(row["solve_seconds"]) for row in proven])
        ),
        "max": float(
            np.max([float(row["solve_seconds"]) for row in proven])
        ),
    }

    return aggregate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    fields = [
        "world_seed",
        "v18_time",
        "v17_time",
        "v16to_time",
        "hungarian_time",
        "global_time",
        "optimal",
        "mip_gap",
        "solve_seconds",
        "oracle_attempts",
        "v18_completed_tasks",
        "v17_completed_tasks",
        "v16to_completed_tasks",
        "hungarian_completed_tasks",
        "v18_retention",
        "v17_retention",
        "v16to_retention",
        "hungarian_retention",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_existing_world(path: Path, seed: int) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if int(row.get("world_seed", -1)) != seed:
        return None
    return row


def run(args: argparse.Namespace) -> Path:
    if args.worlds <= 0:
        raise ValueError("--worlds must be positive")
    if args.world_seed != PUBLICATION_SEED_BASE:
        raise ValueError(
            f"Publication seed base is frozen at {PUBLICATION_SEED_BASE}."
        )
    if args.worlds != PUBLICATION_WORLD_COUNT:
        raise ValueError(
            f"Publication world count is frozen at {PUBLICATION_WORLD_COUNT}."
        )

    config = EnvConfig()
    v18_gene = _load_v18(Path(args.v18_run))
    v17_gene = _load_v17(Path(args.v17_run))
    v16to_gene = _load_v16to(Path(args.v16to_run))

    hashes = {
        "v18": _gene_hash(v18_gene),
        "v17": _gene_hash(v17_gene),
        "v16to": _gene_hash(v16to_gene),
    }

    output_dir = Path(args.output_dir)
    worlds_dir = output_dir / "worlds"
    worlds_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest_payload(
        args=args,
        config=config,
        hashes=hashes,
    )
    _validate_or_write_manifest(
        output_dir / "freeze_manifest.json",
        manifest,
    )

    rows: list[dict[str, object]] = []
    for index in range(args.worlds):
        seed = args.world_seed + index
        world_path = worlds_dir / f"world_{seed}.json"
        row = _load_existing_world(world_path, seed)

        if row is None:
            row = _evaluate_world(
                seed=seed,
                config=config,
                v18_gene=v18_gene,
                v17_gene=v17_gene,
                v16to_gene=v16to_gene,
                time_limit=args.time_limit,
                retry_time_limit=args.retry_time_limit,
            )
            _atomic_json(world_path, row)
            source = "NEW"
        else:
            source = "CACHED"

        rows.append(row)

        retention_text = "NA"
        if row.get("optimal") and row.get("v18_retention") is not None:
            retention_text = (
                f"V18={100*float(row['v18_retention']):.3f}% "
                f"V17={100*float(row['v17_retention']):.3f}% "
                f"V16TO={100*float(row['v16to_retention']):.3f}% "
                f"H={100*float(row['hungarian_retention']):.3f}%"
            )
        print(
            f"[{index+1:03d}/{args.worlds}] {source} seed={seed} "
            f"optimal={row.get('optimal')} {retention_text}"
        )

    rows.sort(key=lambda row: int(row["world_seed"]))
    aggregate = _aggregate(rows)

    result = {
        "protocol": manifest,
        "rows": rows,
        "aggregate": aggregate,
    }
    json_path = output_dir / "publication_final_100.json"
    csv_path = output_dir / "publication_final_100.csv"
    _atomic_json(json_path, result)
    _write_csv(csv_path, rows)

    print("\nV1.8 PUBLICATION FINAL 100")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"FREEZE_MANIFEST={output_dir / 'freeze_manifest.json'}")
    print(f"RESULT_JSON={json_path}")
    print(f"RESULT_CSV={csv_path}")

    if int(aggregate["worlds_proven_optimal"]) != args.worlds:
        print(
            "WARNING: Primary statistics exclude worlds without a proven "
            "exact MILP optimum. Do not treat this run as final-complete "
            "until all 100 worlds are proven optimal."
        )

    return json_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--v18-run", required=True)
    p.add_argument("--v17-run", required=True)
    p.add_argument("--v16to-run", required=True)
    p.add_argument("--world-seed", type=int, default=PUBLICATION_SEED_BASE)
    p.add_argument("--worlds", type=int, default=PUBLICATION_WORLD_COUNT)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--retry-time-limit", type=float, default=900.0)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v18_publication_final_100",
    )
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
