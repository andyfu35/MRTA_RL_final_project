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
    evaluate_baseline_on_worlds,
    evaluate_gene_on_worlds,
    evaluate_genes_on_worlds,
    generate_world,
)
from .gene import Gene


AXES = (
    "completion",
    "efficiency",
    "priority_satisfaction",
    "deadline_satisfaction",
    "balance",
)
BASELINES = (
    "uninformed",
    "nearest",
    "nearest_path",
    "shortest_service",
    "shortest_total_time",
    "shortest_path_time",
    "highest_priority",
    "priority_per_time",
    "priority_per_path_time",
    "earliest_deadline",
    "least_laxity",
    "max_battery_margin",
)


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


def _mutation_sigma(
    generation: int,
    generations: int,
    start: float,
    end: float,
) -> float:
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


def _make_worlds(
    config: EnvConfig,
    base_seed: int,
    count: int,
) -> list[World]:
    return [generate_world(config, base_seed + idx) for idx in range(count)]


def _baseline_metrics(
    worlds: list[World],
    config: EnvConfig,
) -> dict[str, dict[str, object]]:
    return {
        name: evaluate_baseline_on_worlds(name, worlds, config).to_dict()
        for name in BASELINES
    }


def _select_on_probe(
    candidates: list[Gene],
    probe_worlds: list[World],
    config: EnvConfig,
) -> tuple[dict[str, tuple[Gene, Evaluation]], Gene, Evaluation]:
    probe_evaluations = _evaluate_candidates(candidates, probe_worlds, config)

    axis_best: dict[str, tuple[Gene, Evaluation]] = {}
    for axis in AXES:
        idx = max(
            range(len(candidates)),
            key=lambda i: getattr(probe_evaluations[i], axis),
        )
        axis_best[axis] = (candidates[idx], probe_evaluations[idx])

    reference_idx = max(
        range(len(candidates)),
        key=lambda i: probe_evaluations[i].min_axis,
    )
    return (
        axis_best,
        candidates[reference_idx],
        probe_evaluations[reference_idx],
    )


def train(args: argparse.Namespace) -> Path:
    config = EnvConfig(
        world_size=args.world_size,
        num_robots=args.robots,
        num_tasks=args.tasks,
        robot_speed=args.robot_speed,
        service_time_min=args.service_time_min,
        service_time_max=args.service_time_max,
        priority_min=args.priority_min,
        priority_max=args.priority_max,
        deadline_min=args.deadline_min,
        deadline_max=args.deadline_max,
        episode_time=args.episode_time,
        obstacle_count=args.obstacle_count,
        obstacle_size_min=args.obstacle_size_min,
        obstacle_size_max=args.obstacle_size_max,
        obstacle_clearance=args.obstacle_clearance,
        grid_resolution=args.grid_resolution,
        battery_capacity=args.battery_capacity,
        initial_battery_min=args.initial_battery_min,
        initial_battery_max=args.initial_battery_max,
        energy_per_distance=args.energy_per_distance,
    )

    rng = np.random.default_rng(args.seed)
    population = [Gene.random(rng) for _ in range(args.population)]
    archives: dict[str, list[Gene]] | None = None
    probe_hof: list[Gene] = []
    history: list[dict[str, float | int]] = []

    probe_worlds = _make_worlds(config, args.probe_seed, args.probe_worlds)
    validation_worlds = _make_worlds(
        config,
        args.validation_seed,
        args.validation_worlds,
    )
    probe_baselines = _baseline_metrics(probe_worlds, config)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"gene_mrta_v16_{stamp}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for generation in range(args.generations):
        train_worlds = _make_worlds(
            config,
            base_seed=args.seed * 1_000_000 + generation * 10_000,
            count=args.worlds_per_generation,
        )

        candidates = (
            population
            if archives is None
            else _dedupe(population + _archive_union(archives))
        )
        train_evaluations = _evaluate_candidates(candidates, train_worlds, config)
        archives = _axis_archives(
            candidates,
            train_evaluations,
            args.archive_per_axis,
        )

        train_best = {
            axis: max(getattr(ev, axis) for ev in train_evaluations)
            for axis in AXES
        }

        probe_candidates = _archive_union(archives)
        probe_pool = _dedupe(probe_hof + probe_candidates)
        probe_axis_best, probe_reference_gene, probe_reference = _select_on_probe(
            probe_pool,
            probe_worlds,
            config,
        )
        probe_hof = _dedupe(
            [probe_axis_best[axis][0] for axis in AXES]
            + [probe_reference_gene]
        )

        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        row: dict[str, float | int] = {
            "generation": generation,
            "train_best_completion": train_best["completion"],
            "train_best_efficiency": train_best["efficiency"],
            "train_best_priority_satisfaction": train_best["priority_satisfaction"],
            "train_best_deadline_satisfaction": train_best["deadline_satisfaction"],
            "train_best_balance": train_best["balance"],
            "probe_best_completion": probe_axis_best["completion"][1].completion,
            "probe_best_efficiency": probe_axis_best["efficiency"][1].efficiency,
            "probe_best_priority_satisfaction": (
                probe_axis_best["priority_satisfaction"][1].priority_satisfaction
            ),
            "probe_best_deadline_satisfaction": (
                probe_axis_best["deadline_satisfaction"][1].deadline_satisfaction
            ),
            "probe_best_balance": probe_axis_best["balance"][1].balance,
            "probe_reference_min_axis": probe_reference.min_axis,
            "probe_reference_completion": probe_reference.completion,
            "probe_reference_efficiency": probe_reference.efficiency,
            "probe_reference_priority_satisfaction": (
                probe_reference.priority_satisfaction
            ),
            "probe_reference_deadline_satisfaction": (
                probe_reference.deadline_satisfaction
            ),
            "probe_reference_balance": probe_reference.balance,
            "probe_reference_detour_ratio": probe_reference.detour_ratio,
            "probe_reference_battery_remaining_fraction": (
                probe_reference.battery_remaining_fraction
            ),
            "probe_reference_battery_blocked_pair_events": (
                probe_reference.battery_blocked_pair_events
            ),
            "probe_shortest_path_time_completion": float(
                probe_baselines["shortest_path_time"]["completion"]
            ),
            "probe_shortest_path_time_efficiency": float(
                probe_baselines["shortest_path_time"]["efficiency"]
            ),
            "probe_priority_per_time_priority_satisfaction": float(
                probe_baselines["priority_per_time"]["priority_satisfaction"]
            ),
            "probe_priority_per_path_time_priority_satisfaction": float(
                probe_baselines["priority_per_path_time"]["priority_satisfaction"]
            ),
            "probe_least_laxity_deadline_satisfaction": float(
                probe_baselines["least_laxity"]["deadline_satisfaction"]
            ),
            "probe_max_battery_margin_completion": float(
                probe_baselines["max_battery_margin"]["completion"]
            ),
            "mutation_sigma": sigma,
            "bank_size": len(probe_candidates),
            "probe_hof_size": len(probe_hof),
        }
        history.append(row)

        if generation % args.log_every == 0 or generation == args.generations - 1:
            print(
                f"GEN {generation:04d} | "
                f"train-best C={train_best['completion']:.3f} "
                f"E={train_best['efficiency']:.3f} "
                f"P={train_best['priority_satisfaction']:.3f} "
                f"D={train_best['deadline_satisfaction']:.3f} "
                f"B={train_best['balance']:.3f} | "
                f"fixed-probe ref=[{probe_reference.completion:.3f}, "
                f"{probe_reference.efficiency:.3f}, "
                f"{probe_reference.priority_satisfaction:.3f}, "
                f"{probe_reference.deadline_satisfaction:.3f}, "
                f"{probe_reference.balance:.3f}] "
                f"detour={probe_reference.detour_ratio:.3f} "
                f"battery-rem={probe_reference.battery_remaining_fraction:.3f} "
                f"battery-blocked={probe_reference.battery_blocked_pair_events:.1f} | "
                f"shortest-path C="
                f"{float(probe_baselines['shortest_path_time']['completion']):.3f} | "
                f"least-laxity D="
                f"{float(probe_baselines['least_laxity']['deadline_satisfaction']):.3f} | "
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

    final_candidates = _dedupe(
        probe_hof + population + _archive_union(archives)
    )
    probe_axis_best, reference_gene, reference_probe_eval = _select_on_probe(
        final_candidates,
        probe_worlds,
        config,
    )

    final_axis_best: dict[str, dict[str, object]] = {}
    for axis, (gene, probe_eval) in probe_axis_best.items():
        validation_eval = evaluate_gene_on_worlds(
            gene,
            validation_worlds,
            config,
        )
        final_axis_best[axis] = {
            "gene": gene.to_dict(),
            "selection_probe_metrics": probe_eval.to_dict(),
            "validation_metrics": validation_eval.to_dict(),
        }

    reference_validation_eval = evaluate_gene_on_worlds(
        reference_gene,
        validation_worlds,
        config,
    )
    validation_baselines = _baseline_metrics(validation_worlds, config)

    deltas: dict[str, dict[str, float]] = {}
    for baseline_name, baseline_metrics in validation_baselines.items():
        if baseline_name == "uninformed":
            continue
        deltas[baseline_name] = {
            axis: (
                getattr(reference_validation_eval, axis)
                - float(baseline_metrics[axis])
            )
            for axis in AXES
        }

    summary = {
        "experiment": "gene_homogeneous_mrta_v1_6",
        "seed": args.seed,
        "research_question": (
            "Can gene evolution allocate tasks under heterogeneous remaining "
            "battery while retaining completion, efficiency, priority, "
            "deadline, and balance?"
        ),
        "evolution_uses_scalar_reward": False,
        "selection": (
            "five independent capability archives: completion, efficiency, "
            "priority_satisfaction, deadline_satisfaction, balance"
        ),
        "probe_is_used_for_parent_selection": False,
        "probe_hall_of_fame_is_reporting_only": True,
        "validation_is_used_for_selection": False,
        "observation": [
            "euclidean_distance_norm",
            "path_cost_norm",
            "service_time_norm",
            "priority_norm",
            "deadline_remaining_norm",
            "battery_remaining_norm",
            "robot_workload_norm",
            "competition_norm",
        ],
        "path_cost_semantics": (
            "static square obstacles; deterministic 8-connected grid A-star; "
            "travel time uses precomputed A-star path length"
        ),
        "battery_semantics": (
            "each robot starts with independently sampled battery; travel consumes "
            "path_length * energy_per_distance; robot-task pairs that exceed "
            "remaining battery are infeasible; service consumes no battery in v1.6"
        ),
        "efficiency_definition": (
            "sum(1 - clip(actual_path_length/world_diagonal, 0, 1)) "
            "over completed tasks / total tasks"
        ),
        "deadline_semantics": (
            "soft deadline: late tasks may still complete and count toward "
            "completion, but only on-time completions count toward "
            "deadline_satisfaction"
        ),
        "deadline_satisfaction_definition": (
            "number of tasks completed by their individual deadlines / total tasks"
        ),
        "priority_satisfaction_definition": (
            "sum(priority of completed tasks) / sum(priority of all tasks)"
        ),
        "balance_definition": (
            "completion multiplied by Jain fairness over accumulated "
            "travel+service workload"
        ),
        "config": asdict(config),
        "training": {
            "generations": args.generations,
            "population": args.population,
            "worlds_per_generation": args.worlds_per_generation,
            "probe_worlds": args.probe_worlds,
            "probe_seed": args.probe_seed,
            "validation_worlds": args.validation_worlds,
            "validation_seed": args.validation_seed,
            "archive_per_axis": args.archive_per_axis,
            "mutation_rate": args.mutation_rate,
            "mutation_sigma_start": args.mutation_sigma_start,
            "mutation_sigma_end": args.mutation_sigma_end,
            "immigrant_fraction": args.immigrant_fraction,
        },
        "final": {
            "axis_best_selected_on_probe": final_axis_best,
            "reference_gene": {
                "gene": reference_gene.to_dict(),
                "selection_probe_metrics": reference_probe_eval.to_dict(),
                "validation_metrics": reference_validation_eval.to_dict(),
            },
            "validation_baselines": validation_baselines,
            "validation_delta_gene_minus_baseline": deltas,
        },
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (run_dir / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nFINAL VALIDATION")
    print(json.dumps(summary["final"], indent=2, ensure_ascii=False))
    print(f"\nRUN_DIR={run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gene-based homogeneous MRTA v1.6 battery-aware experiment"
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--worlds-per-generation", type=int, default=8)
    parser.add_argument("--probe-worlds", type=int, default=64)
    parser.add_argument("--probe-seed", type=int, default=76_000_000)
    parser.add_argument("--validation-worlds", type=int, default=128)
    parser.add_argument("--validation-seed", type=int, default=96_000_000)
    parser.add_argument("--archive-per-axis", type=int, default=8)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--mutation-sigma-start", type=float, default=0.45)
    parser.add_argument("--mutation-sigma-end", type=float, default=0.05)
    parser.add_argument("--immigrant-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/gene_mrta_v16")

    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--robot-speed", type=float, default=4.0)
    parser.add_argument("--service-time-min", type=float, default=2.0)
    parser.add_argument("--service-time-max", type=float, default=35.0)
    parser.add_argument("--priority-min", type=float, default=0.1)
    parser.add_argument("--priority-max", type=float, default=1.0)
    parser.add_argument("--deadline-min", type=float, default=25.0)
    parser.add_argument("--deadline-max", type=float, default=50.0)
    parser.add_argument("--episode-time", type=float, default=50.0)
    parser.add_argument("--obstacle-count", type=int, default=10)
    parser.add_argument("--obstacle-size-min", type=float, default=12.0)
    parser.add_argument("--obstacle-size-max", type=float, default=20.0)
    parser.add_argument("--obstacle-clearance", type=float, default=4.0)
    parser.add_argument("--grid-resolution", type=float, default=5.0)
    parser.add_argument("--battery-capacity", type=float, default=70.0)
    parser.add_argument("--initial-battery-min", type=float, default=35.0)
    parser.add_argument("--initial-battery-max", type=float, default=70.0)
    parser.add_argument("--energy-per-distance", type=float, default=1.0)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
