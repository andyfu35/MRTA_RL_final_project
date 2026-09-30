from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .env import (
    EnvConfig,
    Evaluation,
    World,
    evaluate_gene_on_worlds,
    evaluate_genes_on_worlds,
    evaluate_genes_time_optimality_matrix,
    generate_world,
)
from .gene import Gene
from .train import AXES as BASE_AXES
from .train import _baseline_metrics, _dedupe, _make_worlds, _mutation_sigma


ORACLE_AXIS = "global_optimality_retention"
ARCHIVE_AXES = BASE_AXES + (ORACLE_AXIS,)


def _parse_schedule(spec: str) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        generation, value = chunk.split(":", 1)
        rows.append((int(generation), int(value)))
    rows.sort()
    if not rows or rows[0][0] != 0:
        raise ValueError("schedule must start at generation 0")
    return rows


def _schedule_value(
    schedule: list[tuple[int, int]],
    generation: int,
) -> int:
    value = schedule[0][1]
    for start, candidate in schedule:
        if generation < start:
            break
        value = candidate
    return value


def _gene_from_dict(data: dict[str, object]) -> Gene:
    return Gene(np.asarray(data["weights"], dtype=np.float64))


def _load_bootstrap_genes(suite_dir: Path | None) -> list[Gene]:
    if suite_dir is None:
        return []
    genes: list[Gene] = []
    for summary_path in sorted((suite_dir / "runs").glob("*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        final = data["final"]
        for item in final["axis_best_selected_on_probe"].values():
            genes.append(_gene_from_dict(item["gene"]))
        genes.append(_gene_from_dict(final["reference_gene"]["gene"]))
    return _dedupe(genes)


def _load_oracle_dataset(
    path: Path,
    config: EnvConfig,
) -> tuple[list[World], np.ndarray, list[World], np.ndarray, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("dataset") != "gene_mrta_v16t_milp_oracle":
        raise ValueError("Not a V1.6-T MILP oracle dataset")
    if data.get("config") != asdict(config):
        raise ValueError(
            "Oracle dataset config does not exactly match training EnvConfig"
        )

    train_rows = data["train"]
    probe_rows = data["probe"]
    train_worlds = [
        generate_world(config, int(row["seed"])) for row in train_rows
    ]
    probe_worlds = [
        generate_world(config, int(row["seed"])) for row in probe_rows
    ]
    train_star = np.asarray(
        [float(row["time_optimality_star"]) for row in train_rows],
        dtype=np.float64,
    )
    probe_star = np.asarray(
        [float(row["time_optimality_star"]) for row in probe_rows],
        dtype=np.float64,
    )
    if np.any(train_star <= 0.0) or np.any(probe_star <= 0.0):
        raise ValueError("Oracle T* values must be positive")
    return train_worlds, train_star, probe_worlds, probe_star, data


def _oracle_scores(
    genes: list[Gene],
    worlds: list[World],
    stars: np.ndarray,
    config: EnvConfig,
) -> np.ndarray:
    matrix = evaluate_genes_time_optimality_matrix(
        genes,
        worlds,
        config,
    )
    if matrix.shape != (len(genes), len(worlds)):
        raise RuntimeError("Unexpected oracle evaluation matrix shape")
    return np.mean(matrix / stars[None, :], axis=1)


def _archives(
    genes: list[Gene],
    evaluations: list[Evaluation],
    oracle_scores: np.ndarray,
    archive_per_axis: int,
) -> dict[str, list[Gene]]:
    out: dict[str, list[Gene]] = {}
    for axis in BASE_AXES:
        order = sorted(
            range(len(genes)),
            key=lambda i: getattr(evaluations[i], axis),
            reverse=True,
        )
        out[axis] = [genes[i] for i in order[:archive_per_axis]]

    order = np.argsort(-oracle_scores)
    out[ORACLE_AXIS] = [
        genes[int(i)] for i in order[:archive_per_axis]
    ]
    return out


def _archive_union(archives: dict[str, list[Gene]]) -> list[Gene]:
    return _dedupe(
        [gene for axis in ARCHIVE_AXES for gene in archives[axis]]
    )


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
    immigrant_count = max(
        1,
        int(round(population_size * immigrant_fraction)),
    )
    child_target = max(0, population_size - immigrant_count)

    while len(population) < child_target:
        axis_a = ARCHIVE_AXES[int(rng.integers(0, len(ARCHIVE_AXES)))]
        axis_b = ARCHIVE_AXES[int(rng.integers(0, len(ARCHIVE_AXES)))]
        parent_a = archives[axis_a][
            int(rng.integers(0, len(archives[axis_a])))
        ]
        parent_b = archives[axis_b][
            int(rng.integers(0, len(archives[axis_b])))
        ]
        population.append(
            parent_a.crossed(parent_b, rng).mutated(
                rng,
                sigma=mutation_sigma,
                mutation_rate=mutation_rate,
            )
        )

    while len(population) < population_size:
        population.append(Gene.random(rng))
    return population[:population_size]


def _select_base_probe(
    genes: list[Gene],
    worlds: list[World],
    config: EnvConfig,
) -> tuple[dict[str, tuple[Gene, Evaluation]], Gene, Evaluation]:
    evaluations = evaluate_genes_on_worlds(genes, worlds, config)
    axis_best: dict[str, tuple[Gene, Evaluation]] = {}
    for axis in BASE_AXES:
        idx = max(
            range(len(genes)),
            key=lambda i: getattr(evaluations[i], axis),
        )
        axis_best[axis] = (genes[idx], evaluations[idx])
    ref_idx = max(
        range(len(genes)),
        key=lambda i: evaluations[i].min_axis,
    )
    return axis_best, genes[ref_idx], evaluations[ref_idx]


def train(args: argparse.Namespace) -> Path:
    config = EnvConfig()
    oracle_path = Path(args.oracle_dataset)
    (
        oracle_train_worlds,
        oracle_train_star,
        oracle_probe_worlds,
        oracle_probe_star,
        oracle_metadata,
    ) = _load_oracle_dataset(oracle_path, config)

    train_schedule = _parse_schedule(args.world_schedule)
    oracle_schedule = _parse_schedule(args.oracle_batch_schedule)

    rng = np.random.default_rng(args.seed)
    bootstrap = _load_bootstrap_genes(
        Path(args.bootstrap_suite) if args.bootstrap_suite else None
    )
    population = _dedupe(bootstrap)
    while len(population) < args.population:
        population.append(Gene.random(rng))
    population = population[: args.population]

    archives: dict[str, list[Gene]] | None = None
    base_hof: list[Gene] = []
    oracle_hof: list[Gene] = list(bootstrap)
    history: list[dict[str, float | int]] = []

    probe_worlds = _make_worlds(
        config,
        args.probe_seed,
        args.probe_worlds,
    )
    validation_worlds = _make_worlds(
        config,
        args.validation_seed,
        args.validation_worlds,
    )
    probe_baselines = _baseline_metrics(probe_worlds, config)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        / f"gene_mrta_v16to_{stamp}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    for generation in range(args.generations):
        train_world_count = _schedule_value(
            train_schedule,
            generation,
        )
        oracle_batch_count = min(
            _schedule_value(oracle_schedule, generation),
            len(oracle_train_worlds),
        )

        fresh_worlds = _make_worlds(
            config,
            base_seed=(
                args.training_seed_base
                + args.seed * 10_000_000
                + generation * 10_000
            ),
            count=train_world_count,
        )

        candidates = (
            population
            if archives is None
            else _dedupe(population + _archive_union(archives))
        )
        base_evals = evaluate_genes_on_worlds(
            candidates,
            fresh_worlds,
            config,
        )

        oracle_ids = rng.choice(
            len(oracle_train_worlds),
            size=oracle_batch_count,
            replace=False,
        )
        oracle_batch_worlds = [
            oracle_train_worlds[int(i)] for i in oracle_ids
        ]
        oracle_batch_star = oracle_train_star[oracle_ids]
        oracle_scores = _oracle_scores(
            candidates,
            oracle_batch_worlds,
            oracle_batch_star,
            config,
        )

        archives = _archives(
            candidates,
            base_evals,
            oracle_scores,
            args.archive_per_axis,
        )

        oracle_best_idx = int(np.argmax(oracle_scores))
        oracle_hof = _dedupe(
            oracle_hof
            + [candidates[oracle_best_idx]]
            + archives[ORACLE_AXIS]
        )
        if len(oracle_hof) > args.oracle_hof_limit:
            hof_scores = _oracle_scores(
                oracle_hof,
                oracle_probe_worlds,
                oracle_probe_star,
                config,
            )
            order = np.argsort(-hof_scores)[: args.oracle_hof_limit]
            oracle_hof = [oracle_hof[int(i)] for i in order]

        probe_candidates = _archive_union(archives)
        base_axis_best, reference_gene, reference_eval = _select_base_probe(
            _dedupe(base_hof + probe_candidates),
            probe_worlds,
            config,
        )
        base_hof = _dedupe(
            [base_axis_best[axis][0] for axis in BASE_AXES]
            + [reference_gene]
        )

        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        probe_oracle_best = float("nan")
        if (
            generation % args.oracle_probe_every == 0
            or generation == args.generations - 1
        ):
            probe_oracle_scores = _oracle_scores(
                oracle_hof,
                oracle_probe_worlds,
                oracle_probe_star,
                config,
            )
            probe_oracle_best = float(np.max(probe_oracle_scores))

        row: dict[str, float | int] = {
            "generation": generation,
            "train_worlds": train_world_count,
            "oracle_batch_worlds": oracle_batch_count,
            "train_best_completion": max(
                ev.completion for ev in base_evals
            ),
            "train_best_efficiency": max(
                ev.efficiency for ev in base_evals
            ),
            "train_best_priority_satisfaction": max(
                ev.priority_satisfaction for ev in base_evals
            ),
            "train_best_deadline_satisfaction": max(
                ev.deadline_satisfaction for ev in base_evals
            ),
            "train_best_balance": max(ev.balance for ev in base_evals),
            "train_best_time_optimality": max(
                ev.time_optimality for ev in base_evals
            ),
            "train_best_global_optimality_retention": float(
                np.max(oracle_scores)
            ),
            "probe_best_global_optimality_retention": probe_oracle_best,
            "probe_reference_min_axis": reference_eval.min_axis,
            "probe_reference_time_optimality": (
                reference_eval.time_optimality
            ),
            "mutation_sigma": sigma,
            "bank_size": len(probe_candidates),
            "oracle_hof_size": len(oracle_hof),
        }
        history.append(row)

        if (
            generation % args.log_every == 0
            or generation == args.generations - 1
        ):
            print(
                f"GEN {generation:04d} | "
                f"worlds={train_world_count} "
                f"oracle={oracle_batch_count} | "
                f"T={row['train_best_time_optimality']:.4f} "
                f"O={row['train_best_global_optimality_retention']:.4f} | "
                f"oracle-probe={probe_oracle_best:.4f} "
                f"ref-min={reference_eval.min_axis:.4f} | "
                f"bank={row['bank_size']} "
                f"hof={row['oracle_hof_size']} sigma={sigma:.3f}"
            )

        population = _next_population(
            archives,
            args.population,
            rng,
            sigma,
            args.mutation_rate,
            args.immigrant_fraction,
        )

    if archives is None:
        raise RuntimeError("No archives produced")

    final_oracle_candidates = _dedupe(
        oracle_hof
        + population
        + _archive_union(archives)
    )
    final_oracle_scores = _oracle_scores(
        final_oracle_candidates,
        oracle_probe_worlds,
        oracle_probe_star,
        config,
    )
    oracle_idx = int(np.argmax(final_oracle_scores))
    oracle_gene = final_oracle_candidates[oracle_idx]
    oracle_probe_retention = float(final_oracle_scores[oracle_idx])

    oracle_probe_matrix = evaluate_genes_time_optimality_matrix(
        [oracle_gene],
        oracle_probe_worlds,
        config,
    )[0]
    oracle_probe_per_world_retention = (
        oracle_probe_matrix / oracle_probe_star
    )

    validation_eval = evaluate_gene_on_worlds(
        oracle_gene,
        validation_worlds,
        config,
    )

    final_base_candidates = _dedupe(
        base_hof + population + _archive_union(archives)
    )
    base_axis_best, reference_gene, reference_probe = _select_base_probe(
        final_base_candidates,
        probe_worlds,
        config,
    )

    final_base = {}
    for axis, (gene, probe_eval) in base_axis_best.items():
        final_base[axis] = {
            "gene": gene.to_dict(),
            "probe_metrics": probe_eval.to_dict(),
            "validation_metrics": evaluate_gene_on_worlds(
                gene,
                validation_worlds,
                config,
            ).to_dict(),
        }

    summary = {
        "experiment": "gene_mrta_v16t_milp_oracle_guided",
        "seed": args.seed,
        "research_question": (
            "Can a seventh MILP-normalized capability archive push the "
            "unchanged 8D linear Gene plus greedy matcher closer to the "
            "fixed-environment global time optimum?"
        ),
        "deployment_uses_milp": False,
        "gene_observation_dimension": 8,
        "matching": "unchanged greedy matching",
        "archive_axes": list(ARCHIVE_AXES),
        "oracle_axis_definition": (
            "mean over cached oracle worlds of "
            "time_optimality_gene / time_optimality_global_MILP"
        ),
        "oracle_dataset": str(oracle_path),
        "oracle_dataset_train_worlds": len(oracle_train_worlds),
        "oracle_dataset_probe_worlds": len(oracle_probe_worlds),
        "oracle_dataset_metadata": {
            "train_seed_base": oracle_metadata["train_seed_base"],
            "probe_seed_base": oracle_metadata["probe_seed_base"],
        },
        "bootstrap_suite": args.bootstrap_suite,
        "config": asdict(config),
        "training": {
            "generations": args.generations,
            "population": args.population,
            "archive_per_axis": args.archive_per_axis,
            "world_schedule": args.world_schedule,
            "oracle_batch_schedule": args.oracle_batch_schedule,
            "mutation_rate": args.mutation_rate,
            "mutation_sigma_start": args.mutation_sigma_start,
            "mutation_sigma_end": args.mutation_sigma_end,
            "immigrant_fraction": args.immigrant_fraction,
        },
        "final": {
            "global_optimality_specialist": {
                "gene": oracle_gene.to_dict(),
                "oracle_probe_retention_mean": (
                    oracle_probe_retention
                ),
                "oracle_probe_retention_std": float(
                    np.std(
                        oracle_probe_per_world_retention,
                        ddof=1,
                    )
                )
                if len(oracle_probe_per_world_retention) > 1
                else 0.0,
                "oracle_probe_retention_min": float(
                    np.min(oracle_probe_per_world_retention)
                ),
                "oracle_probe_retention_max": float(
                    np.max(oracle_probe_per_world_retention)
                ),
                "validation_metrics": validation_eval.to_dict(),
            },
            "base_axis_specialists": final_base,
            "reference_gene": {
                "gene": reference_gene.to_dict(),
                "probe_metrics": reference_probe.to_dict(),
                "validation_metrics": evaluate_gene_on_worlds(
                    reference_gene,
                    validation_worlds,
                    config,
                ).to_dict(),
            },
            "probe_baselines": probe_baselines,
        },
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_dir / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nFINAL ORACLE-GUIDED RESULT")
    print(
        json.dumps(
            summary["final"]["global_optimality_specialist"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"RUN_DIR={run_dir}")
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-dataset", required=True)
    p.add_argument("--bootstrap-suite", default="")
    p.add_argument("--generations", type=int, default=2000)
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--archive-per-axis", type=int, default=16)
    p.add_argument(
        "--world-schedule",
        default="0:16,200:32,500:64,1000:128",
    )
    p.add_argument(
        "--oracle-batch-schedule",
        default="0:16,200:32,500:64,1000:128",
    )
    p.add_argument("--oracle-hof-limit", type=int, default=64)
    p.add_argument("--oracle-probe-every", type=int, default=10)
    p.add_argument("--probe-worlds", type=int, default=64)
    p.add_argument("--probe-seed", type=int, default=76_000_000)
    p.add_argument("--validation-worlds", type=int, default=128)
    p.add_argument("--validation-seed", type=int, default=96_000_000)
    p.add_argument("--training-seed-base", type=int, default=130_000_000)
    p.add_argument("--mutation-rate", type=float, default=0.35)
    p.add_argument("--mutation-sigma-start", type=float, default=0.45)
    p.add_argument("--mutation-sigma-end", type=float, default=0.02)
    p.add_argument("--immigrant-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v16to",
    )
    return p


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
