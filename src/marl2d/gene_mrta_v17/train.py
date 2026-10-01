from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig, Evaluation, generate_world

from .direct_gene import DirectAssignmentGene
from .rollout import AXES, evaluate_direct_population, rollout_direct_gene


def _dedupe(
    genes: list[DirectAssignmentGene],
) -> list[DirectAssignmentGene]:
    out: list[DirectAssignmentGene] = []
    seen: set[tuple[float, ...]] = set()
    for gene in genes:
        key = gene.key()
        if key not in seen:
            seen.add(key)
            out.append(gene)
    return out


def _parse_schedule(spec: str) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for part in spec.split(","):
        start, value = part.split(":", 1)
        rows.append((int(start), int(value)))
    rows.sort()
    if not rows or rows[0][0] != 0:
        raise ValueError("schedule must start at generation 0")
    return rows


def _schedule_value(
    schedule: list[tuple[int, int]],
    generation: int,
) -> int:
    result = schedule[0][1]
    for start, value in schedule:
        if generation < start:
            break
        result = value
    return result


def _mutation_sigma(
    generation: int,
    generations: int,
    start: float,
    end: float,
) -> float:
    if generations <= 1:
        return end
    p = generation / (generations - 1)
    return float(start * (1.0 - p) + end * p)


def _load_oracle_dataset(
    path: Path,
    config: EnvConfig,
) -> tuple[
    list,
    np.ndarray,
    list,
    np.ndarray,
    dict[str, object],
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("dataset") != "gene_mrta_v17_multi_axis_capability_oracle":
        raise ValueError("Not a V1.7 capability-oracle dataset")
    if data.get("config") != asdict(config):
        raise ValueError("Oracle dataset EnvConfig mismatch")

    def load_split(rows):
        worlds = [
            generate_world(config, int(row["seed"]))
            for row in rows
        ]
        ceilings = np.asarray(
            [
                [
                    float(row["ceilings"]["completion"]),
                    float(row["ceilings"]["efficiency"]),
                    float(row["ceilings"]["priority_satisfaction"]),
                    float(row["ceilings"]["deadline_satisfaction"]),
                    float(row["ceilings"]["balance"]),
                    float(row["ceilings"]["time_optimality"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        if np.any(ceilings <= 0.0):
            raise ValueError("All capability references must be positive")
        return worlds, ceilings

    train_worlds, train_ceilings = load_split(data["train"])
    probe_worlds, probe_ceilings = load_split(data["probe"])
    return (
        train_worlds,
        train_ceilings,
        probe_worlds,
        probe_ceilings,
        data,
    )


def normalized_capability_scores(
    raw_metric_tensor: np.ndarray,
    ceilings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    raw_metric_tensor: [gene, world, axis]
    ceilings: [world, axis]

    Returns:
      per_world_ratio [gene, world, axis]
      mean_score      [gene, axis]
    """
    if raw_metric_tensor.ndim != 3:
        raise ValueError("raw metric tensor must be rank 3")
    if ceilings.shape != raw_metric_tensor.shape[1:]:
        raise ValueError(
            f"ceiling shape {ceilings.shape} does not match "
            f"{raw_metric_tensor.shape[1:]}"
        )
    ratios = raw_metric_tensor / ceilings[None, :, :]
    if np.any(ratios > 1.00001):
        idx = np.argwhere(ratios > 1.00001)[0]
        raise RuntimeError(
            "Gene metric exceeded oracle reference beyond tolerance at "
            f"gene/world/axis={tuple(int(x) for x in idx)}, "
            f"ratio={ratios[tuple(idx)]}"
        )
    ratios = np.clip(ratios, 0.0, 1.0)
    return ratios, np.mean(ratios, axis=1)


def _archives(
    genes: list[DirectAssignmentGene],
    mean_scores: np.ndarray,
    archive_per_axis: int,
) -> dict[str, list[DirectAssignmentGene]]:
    out: dict[str, list[DirectAssignmentGene]] = {}
    for axis_idx, axis in enumerate(AXES):
        order = np.argsort(-mean_scores[:, axis_idx])
        out[axis] = [
            genes[int(i)]
            for i in order[:archive_per_axis]
        ]
    return out


def _archive_union(
    archives: dict[str, list[DirectAssignmentGene]],
) -> list[DirectAssignmentGene]:
    return _dedupe(
        [gene for axis in AXES for gene in archives[axis]]
    )


def _next_population(
    archives: dict[str, list[DirectAssignmentGene]],
    population_size: int,
    rng: np.random.Generator,
    sigma: float,
    mutation_rate: float,
    immigrant_fraction: float,
    hidden_dim: int,
) -> list[DirectAssignmentGene]:
    elites = _archive_union(archives)
    population = list(elites[: min(len(elites), population_size)])

    immigrant_count = max(
        1,
        int(round(population_size * immigrant_fraction)),
    )
    child_target = max(0, population_size - immigrant_count)

    while len(population) < child_target:
        axis_a = AXES[int(rng.integers(0, len(AXES)))]
        axis_b = AXES[int(rng.integers(0, len(AXES)))]
        parent_a = archives[axis_a][
            int(rng.integers(0, len(archives[axis_a])))
        ]
        parent_b = archives[axis_b][
            int(rng.integers(0, len(archives[axis_b])))
        ]
        population.append(
            parent_a.crossed(parent_b, rng).mutated(
                rng,
                sigma=sigma,
                mutation_rate=mutation_rate,
            )
        )

    while len(population) < population_size:
        population.append(
            DirectAssignmentGene.random(
                rng,
                hidden_dim=hidden_dim,
            )
        )
    return population[:population_size]


def _score_candidates(
    genes: list[DirectAssignmentGene],
    worlds: list,
    ceilings: np.ndarray,
    config: EnvConfig,
) -> tuple[list[Evaluation], np.ndarray, np.ndarray]:
    evaluations, raw = evaluate_direct_population(
        genes,
        worlds,
        config,
    )
    ratios, mean_scores = normalized_capability_scores(
        raw,
        ceilings,
    )
    return evaluations, ratios, mean_scores


def train(args: argparse.Namespace) -> Path:
    config = EnvConfig()
    (
        oracle_train_worlds,
        oracle_train_ceilings,
        oracle_probe_worlds,
        oracle_probe_ceilings,
        oracle_meta,
    ) = _load_oracle_dataset(
        Path(args.oracle_dataset),
        config,
    )

    rng = np.random.default_rng(args.seed)
    population = [
        DirectAssignmentGene.random(
            rng,
            hidden_dim=args.hidden_dim,
        )
        for _ in range(args.population)
    ]
    archives: dict[str, list[DirectAssignmentGene]] | None = None
    probe_hof: list[DirectAssignmentGene] = []
    batch_schedule = _parse_schedule(args.oracle_batch_schedule)
    history: list[dict[str, float | int]] = []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        / f"gene_mrta_v17_{stamp}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    for generation in range(args.generations):
        batch_count = min(
            _schedule_value(batch_schedule, generation),
            len(oracle_train_worlds),
        )
        ids = rng.choice(
            len(oracle_train_worlds),
            size=batch_count,
            replace=False,
        )
        worlds = [
            oracle_train_worlds[int(i)] for i in ids
        ]
        ceilings = oracle_train_ceilings[ids]

        candidates = (
            population
            if archives is None
            else _dedupe(population + _archive_union(archives))
        )
        _, _, train_scores = _score_candidates(
            candidates,
            worlds,
            ceilings,
            config,
        )
        archives = _archives(
            candidates,
            train_scores,
            args.archive_per_axis,
        )

        generation_best = [
            archives[axis][0] for axis in AXES
        ]
        probe_hof = _dedupe(probe_hof + generation_best)

        probe_best = np.full(len(AXES), np.nan, dtype=np.float64)
        if (
            generation % args.probe_every == 0
            or generation == args.generations - 1
        ):
            _, _, probe_scores = _score_candidates(
                probe_hof,
                oracle_probe_worlds,
                oracle_probe_ceilings,
                config,
            )

            keep: list[DirectAssignmentGene] = []
            for axis_idx in range(len(AXES)):
                order = np.argsort(-probe_scores[:, axis_idx])
                keep.extend(
                    probe_hof[int(i)]
                    for i in order[: args.probe_hof_per_axis]
                )
                probe_best[axis_idx] = float(
                    probe_scores[int(order[0]), axis_idx]
                )
            probe_hof = _dedupe(keep)

        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        row: dict[str, float | int] = {
            "generation": generation,
            "oracle_batch_worlds": batch_count,
            "mutation_sigma": sigma,
            "bank_size": len(_archive_union(archives)),
            "probe_hof_size": len(probe_hof),
        }
        for axis_idx, axis in enumerate(AXES):
            row[f"train_best_{axis}_capability"] = float(
                np.max(train_scores[:, axis_idx])
            )
            row[f"probe_best_{axis}_capability"] = float(
                probe_best[axis_idx]
            )
        history.append(row)

        if (
            generation % args.log_every == 0
            or generation == args.generations - 1
        ):
            train_text = " ".join(
                f"{axis[:1].upper()}={row[f'train_best_{axis}_capability']:.3f}"
                for axis in AXES
            )
            probe_text = " ".join(
                f"{axis[:1].upper()}*={row[f'probe_best_{axis}_capability']:.3f}"
                for axis in AXES
            )
            print(
                f"GEN {generation:04d} | worlds={batch_count} | "
                f"{train_text} | {probe_text} | "
                f"bank={row['bank_size']} hof={row['probe_hof_size']} "
                f"sigma={sigma:.3f}"
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

    if archives is None:
        raise RuntimeError("training produced no archives")

    final_candidates = _dedupe(
        probe_hof + population + _archive_union(archives)
    )
    final_evals, final_ratios, final_scores = _score_candidates(
        final_candidates,
        oracle_probe_worlds,
        oracle_probe_ceilings,
        config,
    )

    specialists: dict[str, object] = {}
    for axis_idx, axis in enumerate(AXES):
        idx = int(np.argmax(final_scores[:, axis_idx]))
        gene = final_candidates[idx]
        first_rollout = rollout_direct_gene(
            gene,
            oracle_probe_worlds[0],
            config,
        )
        specialists[axis] = {
            "gene": gene.to_dict(),
            "oracle_probe_capability_mean": float(
                final_scores[idx, axis_idx]
            ),
            "oracle_probe_capability_std": float(
                np.std(
                    final_ratios[idx, :, axis_idx],
                    ddof=1,
                )
            )
            if final_ratios.shape[1] > 1
            else 0.0,
            "oracle_probe_capability_min": float(
                np.min(final_ratios[idx, :, axis_idx])
            ),
            "oracle_probe_capability_max": float(
                np.max(final_ratios[idx, :, axis_idx])
            ),
            "raw_probe_metrics": final_evals[idx].to_dict(),
            "first_probe_assignment_events": [
                [list(pair) for pair in event]
                for event in first_rollout.assignment_events
            ],
        }

    min_scores = np.min(final_scores, axis=1)
    ref_idx = int(np.argmax(min_scores))
    reference_gene = final_candidates[ref_idx]

    summary = {
        "experiment": "gene_homogeneous_mrta_v1_7_direct_assignment",
        "seed": args.seed,
        "research_question": (
            "Can an evolved autoregressive Gene directly allocate robot-task "
            "pairs while independent capability archives are scored relative "
            "to per-world MILP capability references?"
        ),
        "environment_changed_from_v16t": False,
        "external_matching_optimizer": False,
        "policy": {
            "type": "autoregressive direct joint assignment",
            "pair_observation_dim": 8,
            "hidden_dim": args.hidden_dim,
            "parameter_count": (
                DirectAssignmentGene.parameter_count(args.hidden_dim)
            ),
            "decoding": (
                "policy emits one robot-task action, masks that robot/task, "
                "recomputes joint context, and emits the next action"
            ),
        },
        "capability_axes": list(AXES),
        "capability_score_definition": (
            "for each world and axis: Gene metric / oracle reference; "
            "then average normalized ratios across worlds"
        ),
        "oracle_semantics": {
            "exact_milp_axes": oracle_meta["exact_milp_axes"],
            "balance": oracle_meta["balance_reference"],
            "milp_teaches_actions": False,
            "milp_role": "external capability reference only",
        },
        "config": asdict(config),
        "training": {
            "generations": args.generations,
            "population": args.population,
            "archive_per_axis": args.archive_per_axis,
            "oracle_batch_schedule": args.oracle_batch_schedule,
            "mutation_rate": args.mutation_rate,
            "mutation_sigma_start": args.mutation_sigma_start,
            "mutation_sigma_end": args.mutation_sigma_end,
            "immigrant_fraction": args.immigrant_fraction,
            "oracle_train_worlds": len(oracle_train_worlds),
            "oracle_probe_worlds": len(oracle_probe_worlds),
        },
        "final": {
            "axis_specialists": specialists,
            "reference_gene": {
                "gene": reference_gene.to_dict(),
                "minimum_capability": float(min_scores[ref_idx]),
                "capability_vector": {
                    axis: float(final_scores[ref_idx, i])
                    for i, axis in enumerate(AXES)
                },
                "raw_probe_metrics": final_evals[ref_idx].to_dict(),
            },
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

    print("\nFINAL V1.7 DIRECT ASSIGNMENT")
    print(
        json.dumps(
            {
                axis: {
                    "capability": data["oracle_probe_capability_mean"],
                    "raw_probe_metrics": data["raw_probe_metrics"],
                }
                for axis, data in specialists.items()
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"RUN_DIR={run_dir}")
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-dataset", required=True)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=96)
    p.add_argument("--archive-per-axis", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument(
        "--oracle-batch-schedule",
        default="0:4,25:8,60:16",
    )
    p.add_argument("--probe-every", type=int, default=5)
    p.add_argument("--probe-hof-per-axis", type=int, default=4)
    p.add_argument("--mutation-rate", type=float, default=0.20)
    p.add_argument("--mutation-sigma-start", type=float, default=0.25)
    p.add_argument("--mutation-sigma-end", type=float, default=0.03)
    p.add_argument("--immigrant-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v17",
    )
    return p


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
