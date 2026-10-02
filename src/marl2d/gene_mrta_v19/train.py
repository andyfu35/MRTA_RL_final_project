from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, World, generate_world
from marl2d.gene_mrta_v17.direct_time_scale import (
    _mutation_sigma,
    _parse_schedule,
    _schedule_value,
    load_v16to_time_oracle,
)
from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v18.global_time_test import _load_v18

from .robust_metrics import evaluate_robust_population


AXES = (
    "mean_time",
    "hard_world_time",
    "continuation_preservation",
    "fleet_option_reserve",
)


def _dedupe(
    genes: list[ConsequenceAwareDirectGene],
) -> list[ConsequenceAwareDirectGene]:
    out: list[ConsequenceAwareDirectGene] = []
    seen: set[tuple[float, ...]] = set()
    for gene in genes:
        key = gene.key()
        if key not in seen:
            seen.add(key)
            out.append(gene)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_hard_worlds(
    publication_result: Path,
    config: EnvConfig,
    bottom_count: int,
) -> tuple[list[World], np.ndarray, list[int]]:
    data = json.loads(publication_result.read_text(encoding="utf-8"))
    rows = [
        row
        for row in data["rows"]
        if bool(row.get("optimal"))
        and row.get("v18_retention") is not None
        and row.get("global_time") is not None
    ]
    rows.sort(key=lambda row: float(row["v18_retention"]))
    rows = rows[:bottom_count]
    if len(rows) != bottom_count:
        raise RuntimeError(
            f"Expected {bottom_count} proven hard worlds, got {len(rows)}"
        )

    seeds = [int(row["world_seed"]) for row in rows]
    stars = np.asarray(
        [float(row["global_time"]) for row in rows],
        dtype=np.float64,
    )
    worlds = [generate_world(config, seed) for seed in seeds]
    return worlds, stars, seeds


def _axis_scores(
    *,
    batch_tensor: np.ndarray,
    batch_star: np.ndarray,
    hard_tensor: np.ndarray,
    hard_star: np.ndarray,
) -> dict[str, np.ndarray]:
    batch_time = batch_tensor[:, :, 0]
    hard_time = hard_tensor[:, :, 0]

    mean_ratio = np.clip(
        batch_time / batch_star[None, :],
        0.0,
        1.0,
    )
    hard_ratio = np.clip(
        hard_time / hard_star[None, :],
        0.0,
        1.0,
    )

    continuation = np.concatenate(
        [
            batch_tensor[:, :, 1],
            hard_tensor[:, :, 1],
        ],
        axis=1,
    )
    reserve = np.concatenate(
        [
            batch_tensor[:, :, 2],
            hard_tensor[:, :, 2],
        ],
        axis=1,
    )

    return {
        "mean_time": np.mean(mean_ratio, axis=1),
        "hard_world_time": np.mean(hard_ratio, axis=1),
        "continuation_preservation": np.mean(continuation, axis=1),
        "fleet_option_reserve": np.mean(reserve, axis=1),
    }


def _fixed_axis_scores(
    *,
    genes: list[ConsequenceAwareDirectGene],
    probe_worlds: list[World],
    probe_star: np.ndarray,
    hard_worlds: list[World],
    hard_star: np.ndarray,
    config: EnvConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probe_tensor = evaluate_robust_population(
        genes,
        probe_worlds,
        config,
    )
    hard_tensor = evaluate_robust_population(
        genes,
        hard_worlds,
        config,
    )

    probe_ratio = np.clip(
        probe_tensor[:, :, 0] / probe_star[None, :],
        0.0,
        1.0,
    )
    hard_ratio = np.clip(
        hard_tensor[:, :, 0] / hard_star[None, :],
        0.0,
        1.0,
    )

    continuation = np.concatenate(
        [probe_tensor[:, :, 1], hard_tensor[:, :, 1]],
        axis=1,
    )
    reserve = np.concatenate(
        [probe_tensor[:, :, 2], hard_tensor[:, :, 2]],
        axis=1,
    )

    scores = {
        "mean_time": np.mean(probe_ratio, axis=1),
        "hard_world_time": np.mean(hard_ratio, axis=1),
        "continuation_preservation": np.mean(continuation, axis=1),
        "fleet_option_reserve": np.mean(reserve, axis=1),
    }
    details = {
        "probe_time_ratios": probe_ratio,
        "hard_time_ratios": hard_ratio,
        "continuation_values": continuation,
        "reserve_values": reserve,
    }
    return scores, details


def _select_axis_archives(
    candidates: list[ConsequenceAwareDirectGene],
    scores: dict[str, np.ndarray],
    size_per_axis: int,
) -> dict[str, list[ConsequenceAwareDirectGene]]:
    archives: dict[str, list[ConsequenceAwareDirectGene]] = {}
    for axis in AXES:
        order = np.argsort(-scores[axis])
        archives[axis] = [
            candidates[int(i)]
            for i in order[:size_per_axis]
        ]
    return archives


def _union_banks(
    banks: dict[str, list[ConsequenceAwareDirectGene]],
) -> list[ConsequenceAwareDirectGene]:
    genes: list[ConsequenceAwareDirectGene] = []
    for axis in AXES:
        genes.extend(banks.get(axis, []))
    return _dedupe(genes)


def _balanced_parent(
    banks: dict[str, list[ConsequenceAwareDirectGene]],
    rng: np.random.Generator,
) -> ConsequenceAwareDirectGene:
    available_axes = [
        axis for axis in AXES if banks.get(axis)
    ]
    if not available_axes:
        raise ValueError("No archive contains a parent")
    axis = available_axes[int(rng.integers(0, len(available_axes)))]
    bank = banks[axis]
    return bank[int(rng.integers(0, len(bank)))]


def _next_population(
    archives: dict[str, list[ConsequenceAwareDirectGene]],
    population_size: int,
    rng: np.random.Generator,
    sigma: float,
    mutation_rate: float,
    immigrant_fraction: float,
    hidden_dim: int,
) -> list[ConsequenceAwareDirectGene]:
    elites = _union_banks(archives)
    population = list(elites[:population_size])

    immigrant_count = max(
        1,
        int(round(population_size * immigrant_fraction)),
    )
    child_target = max(0, population_size - immigrant_count)

    while len(population) < child_target:
        parent_a = _balanced_parent(archives, rng)
        parent_b = _balanced_parent(archives, rng)
        population.append(
            parent_a.crossed(parent_b, rng).mutated(
                rng,
                sigma=sigma,
                mutation_rate=mutation_rate,
            )
        )

    while len(population) < population_size:
        population.append(
            ConsequenceAwareDirectGene.random(
                rng,
                hidden_dim=hidden_dim,
            )
        )

    return population[:population_size]


def _save_checkpoint(
    path: Path,
    *,
    next_generation: int,
    population: list[ConsequenceAwareDirectGene],
    archives: dict[str, list[ConsequenceAwareDirectGene]],
    hof: dict[str, list[ConsequenceAwareDirectGene]],
    rng: np.random.Generator,
) -> None:
    payload = {
        "next_generation": next_generation,
        "population": [gene.to_dict() for gene in population],
        "archives": {
            axis: [gene.to_dict() for gene in archives.get(axis, [])]
            for axis in AXES
        },
        "hof": {
            axis: [gene.to_dict() for gene in hof.get(axis, [])]
            for axis in AXES
        },
        "rng_state": _jsonable(rng.bit_generator.state),
    }
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _load_checkpoint(
    path: Path,
    rng: np.random.Generator,
) -> tuple[
    int,
    list[ConsequenceAwareDirectGene],
    dict[str, list[ConsequenceAwareDirectGene]],
    dict[str, list[ConsequenceAwareDirectGene]],
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rng.bit_generator.state = data["rng_state"]

    population = [
        ConsequenceAwareDirectGene.from_dict(item)
        for item in data["population"]
    ]
    archives = {
        axis: [
            ConsequenceAwareDirectGene.from_dict(item)
            for item in data["archives"].get(axis, [])
        ]
        for axis in AXES
    }
    hof = {
        axis: [
            ConsequenceAwareDirectGene.from_dict(item)
            for item in data["hof"].get(axis, [])
        ]
        for axis in AXES
    }
    return int(data["next_generation"]), population, archives, hof


def _refresh_hof(
    hof: dict[str, list[ConsequenceAwareDirectGene]],
    archives: dict[str, list[ConsequenceAwareDirectGene]],
    *,
    probe_worlds: list[World],
    probe_star: np.ndarray,
    hard_worlds: list[World],
    hard_star: np.ndarray,
    config: EnvConfig,
    limit_per_axis: int,
) -> tuple[
    dict[str, list[ConsequenceAwareDirectGene]],
    dict[str, float],
    dict[str, dict[str, float]],
]:
    candidates = _dedupe(
        _union_banks(hof) + _union_banks(archives)
    )
    scores, details = _fixed_axis_scores(
        genes=candidates,
        probe_worlds=probe_worlds,
        probe_star=probe_star,
        hard_worlds=hard_worlds,
        hard_star=hard_star,
        config=config,
    )

    refreshed: dict[str, list[ConsequenceAwareDirectGene]] = {}
    best_scores: dict[str, float] = {}
    best_details: dict[str, dict[str, float]] = {}

    for axis in AXES:
        order = np.argsort(-scores[axis])
        keep = order[:limit_per_axis]
        refreshed[axis] = [candidates[int(i)] for i in keep]
        best_idx = int(keep[0])
        best_scores[axis] = float(scores[axis][best_idx])

        best_details[axis] = {
            "probe_time_mean": float(
                np.mean(details["probe_time_ratios"][best_idx])
            ),
            "probe_time_min": float(
                np.min(details["probe_time_ratios"][best_idx])
            ),
            "hard_time_mean": float(
                np.mean(details["hard_time_ratios"][best_idx])
            ),
            "hard_time_min": float(
                np.min(details["hard_time_ratios"][best_idx])
            ),
            "continuation_mean": float(
                np.mean(details["continuation_values"][best_idx])
            ),
            "fleet_reserve_mean": float(
                np.mean(details["reserve_values"][best_idx])
            ),
        }

    return refreshed, best_scores, best_details


def train(args: argparse.Namespace) -> Path:
    config = EnvConfig()

    (
        train_worlds,
        train_star,
        probe_worlds,
        probe_star,
        oracle_meta,
    ) = load_v16to_time_oracle(
        Path(args.oracle_dataset),
        config,
    )
    hard_worlds, hard_star, hard_seeds = _load_hard_worlds(
        Path(args.publication_result),
        config,
        args.hard_world_count,
    )

    schedule = _parse_schedule(args.oracle_batch_schedule)
    rng = np.random.default_rng(args.seed)

    if args.resume:
        checkpoint_path = Path(args.resume)
        run_dir = checkpoint_path.parent
        (
            start_generation,
            population,
            archives,
            hof,
        ) = _load_checkpoint(checkpoint_path, rng)
        print(
            f"RESUME={checkpoint_path} "
            f"NEXT_GENERATION={start_generation}"
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            Path(args.output_dir)
            / f"gene_mrta_v19_robust_bank_{stamp}_seed{args.seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        bootstrap = _load_v18(Path(args.bootstrap_v18_run))
        population = [bootstrap]
        while len(population) < args.population:
            population.append(
                bootstrap.mutated(
                    rng,
                    sigma=args.bootstrap_sigma,
                    mutation_rate=args.mutation_rate,
                )
            )
        archives = {axis: [] for axis in AXES}
        hof = {axis: [bootstrap] for axis in AXES}
        start_generation = 0

    history_path = run_dir / "history.csv"
    history_exists = history_path.exists()

    for generation in range(start_generation, args.generations):
        batch_count = min(
            _schedule_value(schedule, generation),
            len(train_worlds),
        )
        ids = rng.choice(
            len(train_worlds),
            size=batch_count,
            replace=False,
        )
        batch_worlds = [train_worlds[int(i)] for i in ids]
        batch_star = train_star[ids]

        candidates = _dedupe(
            population + _union_banks(archives) + _union_banks(hof)
        )

        batch_tensor = evaluate_robust_population(
            candidates,
            batch_worlds,
            config,
        )
        hard_tensor = evaluate_robust_population(
            candidates,
            hard_worlds,
            config,
        )
        scores = _axis_scores(
            batch_tensor=batch_tensor,
            batch_star=batch_star,
            hard_tensor=hard_tensor,
            hard_star=hard_star,
        )
        archives = _select_axis_archives(
            candidates,
            scores,
            args.archive_size_per_axis,
        )

        probe_best = {axis: float("nan") for axis in AXES}
        probe_detail: dict[str, dict[str, float]] = {
            axis: {} for axis in AXES
        }

        if (
            generation % args.probe_every == 0
            or generation == args.generations - 1
        ):
            hof, probe_best, probe_detail = _refresh_hof(
                hof,
                archives,
                probe_worlds=probe_worlds,
                probe_star=probe_star,
                hard_worlds=hard_worlds,
                hard_star=hard_star,
                config=config,
                limit_per_axis=args.hof_limit_per_axis,
            )

        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        row: dict[str, object] = {
            "generation": generation,
            "oracle_batch_worlds": batch_count,
            "mutation_sigma": sigma,
        }
        for axis in AXES:
            row[f"train_{axis}"] = float(np.max(scores[axis]))
            row[f"probe_{axis}"] = probe_best[axis]
            detail = probe_detail.get(axis, {})
            row[f"{axis}_probe_time_mean"] = detail.get(
                "probe_time_mean", float("nan")
            )
            row[f"{axis}_probe_time_min"] = detail.get(
                "probe_time_min", float("nan")
            )
            row[f"{axis}_hard_time_mean"] = detail.get(
                "hard_time_mean", float("nan")
            )
            row[f"{axis}_hard_time_min"] = detail.get(
                "hard_time_min", float("nan")
            )

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            if not history_exists:
                writer.writeheader()
                history_exists = True
            writer.writerow(row)

        if (
            generation % args.log_every == 0
            or generation == args.generations - 1
        ):
            print(
                f"GEN {generation:04d} | oracle={batch_count} "
                f"mean={probe_best['mean_time']:.4f} "
                f"hard={probe_best['hard_world_time']:.4f} "
                f"cont={probe_best['continuation_preservation']:.4f} "
                f"reserve={probe_best['fleet_option_reserve']:.4f} "
                f"sigma={sigma:.4f}"
            )
            if probe_detail["mean_time"]:
                d = probe_detail["mean_time"]
                print(
                    "  MEAN_SPECIALIST "
                    f"probeT={d['probe_time_mean']:.4f} "
                    f"probeMin={d['probe_time_min']:.4f} "
                    f"hardT={d['hard_time_mean']:.4f} "
                    f"hardMin={d['hard_time_min']:.4f}"
                )
            if probe_detail["hard_world_time"]:
                d = probe_detail["hard_world_time"]
                print(
                    "  HARD_SPECIALIST "
                    f"probeT={d['probe_time_mean']:.4f} "
                    f"probeMin={d['probe_time_min']:.4f} "
                    f"hardT={d['hard_time_mean']:.4f} "
                    f"hardMin={d['hard_time_min']:.4f}"
                )

        population = _next_population(
            archives,
            args.population,
            rng,
            sigma,
            args.mutation_rate,
            args.immigrant_fraction,
            args.hidden_dim,
        )

        if (
            (generation + 1) % args.checkpoint_every == 0
            or generation == args.generations - 1
        ):
            _save_checkpoint(
                run_dir / "checkpoint.json",
                next_generation=generation + 1,
                population=population,
                archives=archives,
                hof=hof,
                rng=rng,
            )

    all_candidates = _dedupe(
        _union_banks(hof) + _union_banks(archives)
    )
    final_scores, final_details = _fixed_axis_scores(
        genes=all_candidates,
        probe_worlds=probe_worlds,
        probe_star=probe_star,
        hard_worlds=hard_worlds,
        hard_star=hard_star,
        config=config,
    )

    specialists: dict[str, object] = {}
    for axis in AXES:
        idx = int(np.argmax(final_scores[axis]))
        gene = all_candidates[idx]
        specialists[axis] = {
            "gene": gene.to_dict(),
            "axis_score": float(final_scores[axis][idx]),
            "probe_time_mean": float(
                np.mean(final_details["probe_time_ratios"][idx])
            ),
            "probe_time_std": float(
                np.std(final_details["probe_time_ratios"][idx], ddof=1)
            ),
            "probe_time_min": float(
                np.min(final_details["probe_time_ratios"][idx])
            ),
            "hard_time_mean": float(
                np.mean(final_details["hard_time_ratios"][idx])
            ),
            "hard_time_min": float(
                np.min(final_details["hard_time_ratios"][idx])
            ),
            "continuation_mean": float(
                np.mean(final_details["continuation_values"][idx])
            ),
            "fleet_reserve_mean": float(
                np.mean(final_details["reserve_values"][idx])
            ),
        }

    summary = {
        "experiment": "gene_mrta_v19_robust_gene_bank",
        "policy_architecture_changed_from_v18": False,
        "policy": {
            "observation_dim": 12,
            "hidden_dim": args.hidden_dim,
            "parameter_count": (
                ConsequenceAwareDirectGene.parameter_count(args.hidden_dim)
            ),
            "external_matcher": False,
            "milp_teaches_actions": False,
        },
        "gene_bank_axes": list(AXES),
        "weighted_scalarization": False,
        "hard_world_source": str(args.publication_result),
        "hard_world_seeds": hard_seeds,
        "hard_world_count": len(hard_worlds),
        "oracle_dataset": str(args.oracle_dataset),
        "oracle_train_worlds": len(train_worlds),
        "oracle_probe_worlds": len(probe_worlds),
        "oracle_train_seed_base": oracle_meta["train_seed_base"],
        "oracle_probe_seed_base": oracle_meta["probe_seed_base"],
        "bootstrap_v18_run": args.bootstrap_v18_run,
        "training": {
            "generations": args.generations,
            "population": args.population,
            "archive_size_per_axis": args.archive_size_per_axis,
            "hof_limit_per_axis": args.hof_limit_per_axis,
            "oracle_batch_schedule": args.oracle_batch_schedule,
            "mutation_rate": args.mutation_rate,
            "mutation_sigma_start": args.mutation_sigma_start,
            "mutation_sigma_end": args.mutation_sigma_end,
            "immigrant_fraction": args.immigrant_fraction,
            "balanced_parent_axis_sampling": True,
        },
        "final": {
            "axis_specialists": specialists,
            "candidate_count": len(all_candidates),
        },
        "v19_final_generalization_rule": (
            "98M is development data. Select/freeze V1.9 using development "
            "results, then evaluate once on untouched 99,000,000-99,000,099."
        ),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFINAL V1.9 ROBUST GENE BANK")
    for axis in AXES:
        item = specialists[axis]
        print(
            f"{axis}: axis={item['axis_score']:.4f} "
            f"probeT={item['probe_time_mean']:.4f} "
            f"probeMin={item['probe_time_min']:.4f} "
            f"hardT={item['hard_time_mean']:.4f} "
            f"hardMin={item['hard_time_min']:.4f}"
        )
    print(f"RUN_DIR={run_dir}")
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-dataset", required=True)
    p.add_argument("--publication-result", required=True)
    p.add_argument("--bootstrap-v18-run", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--hard-world-count", type=int, default=10)
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--archive-size-per-axis", type=int, default=16)
    p.add_argument("--hof-limit-per-axis", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument(
        "--oracle-batch-schedule",
        default="0:16,200:32,500:64",
    )
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--mutation-rate", type=float, default=0.20)
    p.add_argument("--bootstrap-sigma", type=float, default=0.06)
    p.add_argument("--mutation-sigma-start", type=float, default=0.12)
    p.add_argument("--mutation-sigma-end", type=float, default=0.015)
    p.add_argument("--immigrant-fraction", type=float, default=0.10)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v19_robust_bank",
    )
    return p


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
