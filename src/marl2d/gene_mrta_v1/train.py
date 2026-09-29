from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .env import EnvConfig, Evaluation, World, evaluate_gene_on_worlds, evaluate_genes_on_worlds, generate_world
from .gene import Gene


AXES = ("completion", "efficiency", "balance")


def _evaluate_candidates(
    genes: list[Gene],
    worlds: list[World],
    config: EnvConfig,
) -> list[Evaluation]:
    return evaluate_genes_on_worlds(genes, worlds, config)


def _dedupe(genes: list[Gene]) -> list[Gene]:
    out: list[Gene] = []
    seen: set[tuple[float, ...]] = set()
    for gene in genes:
        key = gene.key()
        if key not in seen:
            seen.add(key)
            out.append(gene)
    return out


def _axis_archives(
    genes: list[Gene],
    evaluations: list[Evaluation],
    archive_per_axis: int,
) -> dict[str, list[Gene]]:
    archives: dict[str, list[Gene]] = {}
    for axis in AXES:
        order = sorted(
            range(len(genes)),
            key=lambda idx: getattr(evaluations[idx], axis),
            reverse=True,
        )
        archives[axis] = [genes[idx] for idx in order[:archive_per_axis]]
    return archives


def _archive_union(archives: dict[str, list[Gene]]) -> list[Gene]:
    return _dedupe([gene for axis in AXES for gene in archives[axis]])


def _mutation_sigma(generation: int, generations: int, start: float, end: float) -> float:
    if generations <= 1:
        return end
    progress = generation / (generations - 1)
    return float(start * (1.0 - progress) + end * progress)


def _next_population(
    archives: dict[str, list[Gene]],
    population_size: int,
    rng: np.random.Generator,
    mutation_sigma: float,
    mutation_rate: float,
    immigrant_fraction: float,
) -> list[Gene]:
    elites = _archive_union(archives)
    population = list(elites[: min(len(elites), population_size)])
    immigrant_count = max(1, int(round(population_size * immigrant_fraction)))
    child_target = max(0, population_size - immigrant_count)

    while len(population) < child_target:
        axis_a = AXES[int(rng.integers(0, len(AXES)))]
        axis_b = AXES[int(rng.integers(0, len(AXES)))]
        parent_a = archives[axis_a][int(rng.integers(0, len(archives[axis_a])))]
        parent_b = archives[axis_b][int(rng.integers(0, len(archives[axis_b])))]
        child = parent_a.crossed(parent_b, rng).mutated(
            rng,
            sigma=mutation_sigma,
            mutation_rate=mutation_rate,
        )
        population.append(child)

    while len(population) < population_size:
        population.append(Gene.random(rng, scale=1.0))

    return population[:population_size]


def _make_worlds(config: EnvConfig, base_seed: int, count: int) -> list[World]:
    return [generate_world(config, base_seed + idx) for idx in range(count)]


def _baseline_metrics(worlds: list[World], config: EnvConfig) -> dict[str, dict[str, object]]:
    uninformed = Gene(np.zeros(4, dtype=np.float64), 0.0)
    nearest = Gene(np.array([-1.0, 0.0, 0.0, 0.0]), 0.0)
    return {
        "uninformed": evaluate_gene_on_worlds(uninformed, worlds, config).to_dict(),
        "nearest": evaluate_gene_on_worlds(nearest, worlds, config).to_dict(),
    }


def train(args: argparse.Namespace) -> Path:
    config = EnvConfig(
        world_size=args.world_size,
        num_robots=args.robots,
        num_tasks=args.tasks,
        robot_speed=args.robot_speed,
        service_time=args.service_time,
        episode_time=args.episode_time,
    )
    rng = np.random.default_rng(args.seed)
    population = [Gene.random(rng) for _ in range(args.population)]
    archives: dict[str, list[Gene]] | None = None
    history: list[dict[str, float | int]] = []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"gene_mrta_v1_{stamp}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for generation in range(args.generations):
        train_worlds = _make_worlds(
            config,
            base_seed=args.seed * 1_000_000 + generation * 10_000,
            count=args.worlds_per_generation,
        )
        candidates = population if archives is None else _dedupe(population + _archive_union(archives))
        evaluations = _evaluate_candidates(candidates, train_worlds, config)
        archives = _axis_archives(candidates, evaluations, args.archive_per_axis)

        best_values = {
            axis: max(getattr(ev, axis) for ev in evaluations)
            for axis in AXES
        }
        maximin_idx = max(range(len(candidates)), key=lambda idx: evaluations[idx].min_axis)
        maximin = evaluations[maximin_idx]
        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )
        row: dict[str, float | int] = {
            "generation": generation,
            "best_completion": best_values["completion"],
            "best_efficiency": best_values["efficiency"],
            "best_balance": best_values["balance"],
            "reference_min_axis": maximin.min_axis,
            "reference_completion": maximin.completion,
            "reference_efficiency": maximin.efficiency,
            "reference_balance": maximin.balance,
            "mutation_sigma": sigma,
            "bank_size": len(_archive_union(archives)),
        }
        history.append(row)

        if generation % args.log_every == 0 or generation == args.generations - 1:
            print(
                f"GEN {generation:04d} | "
                f"axis-best C={best_values['completion']:.3f} "
                f"E={best_values['efficiency']:.3f} B={best_values['balance']:.3f} | "
                f"reference min={maximin.min_axis:.3f} "
                f"[{maximin.completion:.3f}, {maximin.efficiency:.3f}, {maximin.balance:.3f}] | "
                f"bank={row['bank_size']} sigma={sigma:.3f}"
            )

        population = _next_population(
            archives,
            population_size=args.population,
            rng=rng,
            mutation_sigma=sigma,
            mutation_rate=args.mutation_rate,
            immigrant_fraction=args.immigrant_fraction,
        )

    if archives is None:
        raise RuntimeError("Training produced no archive")

    validation_worlds = _make_worlds(
        config,
        base_seed=args.seed * 1_000_000 + 900_000_000,
        count=args.validation_worlds,
    )
    final_candidates = _dedupe(population + _archive_union(archives))
    final_evaluations = _evaluate_candidates(final_candidates, validation_worlds, config)

    final_axis_best: dict[str, dict[str, object]] = {}
    for axis in AXES:
        idx = max(range(len(final_candidates)), key=lambda i: getattr(final_evaluations[i], axis))
        final_axis_best[axis] = {
            "gene": final_candidates[idx].to_dict(),
            "metrics": final_evaluations[idx].to_dict(),
        }

    reference_idx = max(range(len(final_candidates)), key=lambda i: final_evaluations[i].min_axis)
    reference_gene = final_candidates[reference_idx]
    reference_eval = final_evaluations[reference_idx]

    summary = {
        "experiment": "gene_homogeneous_mrta_v1",
        "seed": args.seed,
        "evolution_uses_scalar_reward": False,
        "selection": "three independent capability archives: completion, efficiency, balance",
        "reference_gene_selection": "maximin across the three validation axes; reporting only",
        "config": asdict(config),
        "training": {
            "generations": args.generations,
            "population": args.population,
            "worlds_per_generation": args.worlds_per_generation,
            "validation_worlds": args.validation_worlds,
            "archive_per_axis": args.archive_per_axis,
            "mutation_rate": args.mutation_rate,
            "mutation_sigma_start": args.mutation_sigma_start,
            "mutation_sigma_end": args.mutation_sigma_end,
            "immigrant_fraction": args.immigrant_fraction,
        },
        "validation": {
            "axis_best": final_axis_best,
            "reference_gene": {
                "gene": reference_gene.to_dict(),
                "metrics": reference_eval.to_dict(),
            },
            "baselines": _baseline_metrics(validation_worlds, config),
        },
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nVALIDATION")
    print(json.dumps(summary["validation"], indent=2, ensure_ascii=False))
    print(f"\nRUN_DIR={run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gene-based homogeneous MRTA v1")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--worlds-per-generation", type=int, default=8)
    parser.add_argument("--validation-worlds", type=int, default=64)
    parser.add_argument("--archive-per-axis", type=int, default=8)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--mutation-sigma-start", type=float, default=0.45)
    parser.add_argument("--mutation-sigma-end", type=float, default=0.06)
    parser.add_argument("--immigrant-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output-dir", default="runs/gene_mrta_v1")

    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--robot-speed", type=float, default=2.0)
    parser.add_argument("--service-time", type=float, default=3.0)
    parser.add_argument("--episode-time", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
