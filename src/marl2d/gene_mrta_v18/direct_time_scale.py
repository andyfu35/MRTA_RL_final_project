from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import (
    EnvConfig,
    World,
)
from marl2d.gene_mrta_v17.direct_gene import (
    DirectAssignmentGene as V17DirectAssignmentGene,
)
from marl2d.gene_mrta_v17.direct_time_scale import (
    _mutation_sigma,
    _parse_schedule,
    _schedule_value,
    load_v16to_time_oracle,
)

from .direct_gene import (
    ConsequenceAwareDirectGene,
)
from .rollout import (
    AXES,
    evaluate_direct_population,
)


TIME_AXIS_INDEX = AXES.index(
    "time_optimality"
)


def _dedupe(
    genes: list[ConsequenceAwareDirectGene],
) -> list[ConsequenceAwareDirectGene]:
    out: list[
        ConsequenceAwareDirectGene
    ] = []
    seen: set[tuple[float, ...]] = set()

    for gene in genes:
        key = gene.key()
        if key not in seen:
            seen.add(key)
            out.append(gene)

    return out


def direct_time_scores(
    genes: list[ConsequenceAwareDirectGene],
    worlds: list[World],
    stars: np.ndarray,
    config: EnvConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not genes:
        return (
            np.zeros(
                (0, len(worlds)),
                dtype=np.float64,
            ),
            np.zeros(
                0,
                dtype=np.float64,
            ),
        )

    _, raw = evaluate_direct_population(
        genes,
        worlds,
        config,
    )
    time_values = raw[
        :,
        :,
        TIME_AXIS_INDEX,
    ]

    ratios = (
        time_values
        / stars[None, :]
    )

    if np.any(ratios > 1.00001):
        idx = np.argwhere(
            ratios > 1.00001
        )[0]
        raise RuntimeError(
            "V1.8 policy exceeded proven T* "
            "beyond tolerance at "
            f"gene/world="
            f"{tuple(int(x) for x in idx)}, "
            f"ratio={ratios[tuple(idx)]}"
        )

    ratios = np.clip(
        ratios,
        0.0,
        1.0,
    )

    return (
        ratios,
        np.mean(
            ratios,
            axis=1,
        ),
    )


def _load_v17_bootstrap(
    run_dir: Path | None,
) -> list[ConsequenceAwareDirectGene]:
    if run_dir is None:
        return []

    summary_path = (
        run_dir / "summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            summary_path
        )

    data = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )
    final = data["final"]

    if "time_specialist" in final:
        gene_data = final[
            "time_specialist"
        ]["gene"]
    else:
        gene_data = final[
            "axis_specialists"
        ][
            "time_optimality"
        ][
            "gene"
        ]

    v17_gene = (
        V17DirectAssignmentGene.from_dict(
            gene_data
        )
    )

    return [
        ConsequenceAwareDirectGene.from_v17(
            v17_gene
        )
    ]


def _next_population(
    archive: list[
        ConsequenceAwareDirectGene
    ],
    population_size: int,
    rng: np.random.Generator,
    sigma: float,
    mutation_rate: float,
    immigrant_fraction: float,
    hidden_dim: int,
) -> list[
    ConsequenceAwareDirectGene
]:
    if not archive:
        raise ValueError(
            "archive must not be empty"
        )

    population = list(
        archive[
            : min(
                len(archive),
                population_size,
            )
        ]
    )

    immigrant_count = max(
        1,
        int(
            round(
                population_size
                * immigrant_fraction
            )
        ),
    )
    child_target = max(
        0,
        population_size
        - immigrant_count,
    )

    while len(population) < child_target:
        parent_a = archive[
            int(
                rng.integers(
                    0,
                    len(archive),
                )
            )
        ]
        parent_b = archive[
            int(
                rng.integers(
                    0,
                    len(archive),
                )
            )
        ]
        population.append(
            parent_a.crossed(
                parent_b,
                rng,
            ).mutated(
                rng,
                sigma=sigma,
                mutation_rate=(
                    mutation_rate
                ),
            )
        )

    while len(population) < population_size:
        population.append(
            ConsequenceAwareDirectGene.random(
                rng,
                hidden_dim=hidden_dim,
            )
        )

    return population[
        :population_size
    ]


def _jsonable(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()
    if isinstance(
        value,
        np.integer,
    ):
        return int(value)
    if isinstance(
        value,
        np.floating,
    ):
        return float(value)
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _jsonable(item)
            for item in value
        ]
    return value


def _save_checkpoint(
    path: Path,
    *,
    next_generation: int,
    population: list[
        ConsequenceAwareDirectGene
    ],
    archive: list[
        ConsequenceAwareDirectGene
    ],
    hof: list[
        ConsequenceAwareDirectGene
    ],
    rng: np.random.Generator,
) -> None:
    payload = {
        "next_generation": (
            next_generation
        ),
        "population": [
            gene.to_dict()
            for gene in population
        ],
        "archive": [
            gene.to_dict()
            for gene in archive
        ],
        "hof": [
            gene.to_dict()
            for gene in hof
        ],
        "rng_state": _jsonable(
            rng.bit_generator.state
        ),
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_checkpoint(
    path: Path,
    rng: np.random.Generator,
) -> tuple[
    int,
    list[ConsequenceAwareDirectGene],
    list[ConsequenceAwareDirectGene],
    list[ConsequenceAwareDirectGene],
]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    rng.bit_generator.state = (
        data["rng_state"]
    )

    return (
        int(
            data[
                "next_generation"
            ]
        ),
        [
            ConsequenceAwareDirectGene.from_dict(
                item
            )
            for item in data[
                "population"
            ]
        ],
        [
            ConsequenceAwareDirectGene.from_dict(
                item
            )
            for item in data[
                "archive"
            ]
        ],
        [
            ConsequenceAwareDirectGene.from_dict(
                item
            )
            for item in data[
                "hof"
            ]
        ],
    )


def train(
    args: argparse.Namespace,
) -> Path:
    config = EnvConfig()

    oracle_path = Path(
        args.oracle_dataset
    )

    (
        train_worlds,
        train_star,
        probe_worlds,
        probe_star,
        oracle_meta,
    ) = load_v16to_time_oracle(
        oracle_path,
        config,
    )

    schedule = _parse_schedule(
        args.oracle_batch_schedule
    )
    rng = np.random.default_rng(
        args.seed
    )

    if args.resume:
        checkpoint_path = Path(
            args.resume
        )
        run_dir = (
            checkpoint_path.parent
        )

        (
            start_generation,
            population,
            archive,
            hof,
        ) = _load_checkpoint(
            checkpoint_path,
            rng,
        )

        bootstrap_count = 0
        print(
            f"RESUME={checkpoint_path} "
            f"NEXT_GENERATION="
            f"{start_generation}"
        )
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        run_dir = (
            Path(args.output_dir)
            / (
                "gene_mrta_v18t_scale_"
                f"{stamp}_seed{args.seed}"
            )
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        bootstrap = (
            _load_v17_bootstrap(
                Path(
                    args.bootstrap_v17_run
                )
                if args.bootstrap_v17_run
                else None
            )
        )
        bootstrap_count = len(
            bootstrap
        )

        population = list(
            bootstrap
        )
        while (
            len(population)
            < args.population
        ):
            population.append(
                ConsequenceAwareDirectGene.random(
                    rng,
                    hidden_dim=(
                        args.hidden_dim
                    ),
                )
            )

        population = population[
            : args.population
        ]
        archive: list[
            ConsequenceAwareDirectGene
        ] = []
        hof = list(bootstrap)
        start_generation = 0

    history_path = (
        run_dir / "history.csv"
    )
    history_exists = (
        history_path.exists()
    )

    for generation in range(
        start_generation,
        args.generations,
    ):
        batch_count = min(
            _schedule_value(
                schedule,
                generation,
            ),
            len(train_worlds),
        )

        ids = rng.choice(
            len(train_worlds),
            size=batch_count,
            replace=False,
        )
        batch_worlds = [
            train_worlds[int(i)]
            for i in ids
        ]
        batch_star = (
            train_star[ids]
        )

        candidates = _dedupe(
            population + archive
        )

        _, scores = (
            direct_time_scores(
                candidates,
                batch_worlds,
                batch_star,
                config,
            )
        )

        order = np.argsort(
            -scores
        )
        archive = [
            candidates[int(i)]
            for i in order[
                : args.archive_size
            ]
        ]

        hof = _dedupe(
            hof
            + archive[
                : min(
                    args.hof_add_per_generation,
                    len(archive),
                )
            ]
        )

        probe_best = float("nan")
        probe_min = float("nan")
        probe_std = float("nan")

        if (
            generation
            % args.probe_every
            == 0
            or generation
            == args.generations - 1
        ):
            (
                probe_ratios,
                probe_scores,
            ) = direct_time_scores(
                hof,
                probe_worlds,
                probe_star,
                config,
            )

            probe_order = np.argsort(
                -probe_scores
            )
            keep = probe_order[
                : args.hof_limit
            ]
            hof = [
                hof[int(i)]
                for i in keep
            ]

            best_idx = int(
                keep[0]
            )
            probe_best = float(
                probe_scores[
                    best_idx
                ]
            )
            probe_min = float(
                np.min(
                    probe_ratios[
                        best_idx
                    ]
                )
            )
            probe_std = float(
                np.std(
                    probe_ratios[
                        best_idx
                    ],
                    ddof=1,
                )
            )

        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        row = {
            "generation": generation,
            "oracle_batch_worlds": (
                batch_count
            ),
            "train_best_retention": float(
                scores[
                    int(order[0])
                ]
            ),
            "probe_best_retention": (
                probe_best
            ),
            "probe_best_retention_std": (
                probe_std
            ),
            "probe_best_retention_min": (
                probe_min
            ),
            "mutation_sigma": sigma,
            "archive_size": len(
                archive
            ),
            "hof_size": len(hof),
        }

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    row.keys()
                ),
            )
            if not history_exists:
                writer.writeheader()
                history_exists = True
            writer.writerow(row)

        if (
            generation
            % args.log_every
            == 0
            or generation
            == args.generations - 1
        ):
            print(
                f"GEN {generation:04d} | "
                f"oracle={batch_count} "
                f"train="
                f"{row['train_best_retention']:.4f} "
                f"probe={probe_best:.4f} "
                f"probe-min="
                f"{probe_min:.4f} "
                f"hof={len(hof)} "
                f"sigma={sigma:.4f}"
            )

        population = _next_population(
            archive,
            args.population,
            rng,
            sigma,
            args.mutation_rate,
            args.immigrant_fraction,
            args.hidden_dim,
        )

        if (
            (generation + 1)
            % args.checkpoint_every
            == 0
            or generation
            == args.generations - 1
        ):
            _save_checkpoint(
                run_dir
                / "checkpoint.json",
                next_generation=(
                    generation + 1
                ),
                population=population,
                archive=archive,
                hof=hof,
                rng=rng,
            )

    final_candidates = _dedupe(
        hof
        + archive
        + population
    )

    (
        final_ratios,
        final_scores,
    ) = direct_time_scores(
        final_candidates,
        probe_worlds,
        probe_star,
        config,
    )

    best_idx = int(
        np.argmax(
            final_scores
        )
    )
    best_gene = (
        final_candidates[
            best_idx
        ]
    )
    best_ratios = (
        final_ratios[
            best_idx
        ]
    )

    summary = {
        "experiment": (
            "gene_mrta_v18_"
            "consequence_direct_time_scale"
        ),
        "seed": args.seed,
        "research_question": (
            "Does adding deterministic "
            "candidate-consequence information "
            "to the direct assignment observation "
            "reduce V1.7 held-out failure modes "
            "without changing decoder capacity "
            "or using MILP action supervision?"
        ),
        "environment_changed_from_v17": (
            False
        ),
        "decoder_changed_from_v17": (
            False
        ),
        "stop_wait_changed_from_v17": (
            False
        ),
        "external_matching_optimizer": (
            False
        ),
        "milp_teaches_actions": False,
        "milp_role": (
            "external T* reference only"
        ),
        "oracle_dataset": str(
            oracle_path
        ),
        "oracle_train_worlds": len(
            train_worlds
        ),
        "oracle_probe_worlds": len(
            probe_worlds
        ),
        "oracle_train_seed_base": (
            oracle_meta[
                "train_seed_base"
            ]
        ),
        "oracle_probe_seed_base": (
            oracle_meta[
                "probe_seed_base"
            ]
        ),
        "bootstrap_v17_run": (
            args.bootstrap_v17_run
        ),
        "bootstrap_gene_count": (
            bootstrap_count
        ),
        "policy": {
            "type": (
                "V1.8 consequence-aware "
                "autoregressive direct assignment"
            ),
            "observation_dim": 12,
            "base_observation_dim": 8,
            "consequence_features": [
                "self_future_reachability",
                "self_future_best_time_utility",
                "other_robot_opportunity_cost",
                "residual_battery",
            ],
            "hidden_dim": (
                args.hidden_dim
            ),
            "parameter_count": (
                ConsequenceAwareDirectGene.parameter_count(
                    args.hidden_dim
                )
            ),
        },
        "training": {
            "generations": (
                args.generations
            ),
            "population": (
                args.population
            ),
            "archive_size": (
                args.archive_size
            ),
            "hof_limit": (
                args.hof_limit
            ),
            "oracle_batch_schedule": (
                args.oracle_batch_schedule
            ),
            "mutation_rate": (
                args.mutation_rate
            ),
            "mutation_sigma_start": (
                args.mutation_sigma_start
            ),
            "mutation_sigma_end": (
                args.mutation_sigma_end
            ),
            "immigrant_fraction": (
                args.immigrant_fraction
            ),
        },
        "final": {
            "time_specialist": {
                "gene": (
                    best_gene.to_dict()
                ),
                "oracle_probe_retention_mean": float(
                    final_scores[
                        best_idx
                    ]
                ),
                "oracle_probe_retention_std": float(
                    np.std(
                        best_ratios,
                        ddof=1,
                    )
                )
                if len(best_ratios) > 1
                else 0.0,
                "oracle_probe_retention_min": float(
                    np.min(
                        best_ratios
                    )
                ),
                "oracle_probe_retention_max": float(
                    np.max(
                        best_ratios
                    )
                ),
            }
        },
    }

    (
        run_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nFINAL V1.8-T "
        "CONSEQUENCE SCALE"
    )
    print(
        json.dumps(
            summary[
                "final"
            ][
                "time_specialist"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        f"RUN_DIR={run_dir}"
    )

    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--oracle-dataset",
        required=True,
    )
    p.add_argument(
        "--bootstrap-v17-run",
        default="",
    )
    p.add_argument(
        "--resume",
        default="",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--population",
        type=int,
        default=256,
    )
    p.add_argument(
        "--archive-size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--hof-limit",
        type=int,
        default=64,
    )
    p.add_argument(
        "--hof-add-per-generation",
        type=int,
        default=4,
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=8,
    )
    p.add_argument(
        "--oracle-batch-schedule",
        default=(
            "0:16,200:32,500:64"
        ),
    )
    p.add_argument(
        "--probe-every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--mutation-rate",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--mutation-sigma-start",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--mutation-sigma-end",
        type=float,
        default=0.1325,
    )
    p.add_argument(
        "--immigrant-fraction",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/gene_mrta_v18t_scale"
        ),
    )

    return p


def main() -> None:
    train(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
