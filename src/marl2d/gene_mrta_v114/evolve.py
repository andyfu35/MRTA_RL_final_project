from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig
from marl2d.gene_mrta_v17.direct_time_scale import _mutation_sigma
from marl2d.gene_mrta_v18.global_time_test import _load_v18
from marl2d.gene_mrta_v110.scenario_bank import load_scenario_bank
from marl2d.gene_mrta_v113.evolve import (
    AXES,
    GeneRecord,
    _active_parent_ids,
    _admit_normal_children,
    _best_by_axis,
    _capability_ceiling_retention,
    _certified_capabilities,
    _complementary_partner_ids,
    _evaluate_scores,
    _gene_id,
    _inheritance_retention,
    _normal_children,
    _parent_screen_scores,
    _passes_inheritance_gate,
    _prune_active_records,
    _quality_weight,
    _route_diagnostics,
    _sample_record_id,
    _score_dict_at,
    _screen_indices,
    _top_normal_full_indices,
)

from .recombination_gene import (
    RECOMBINATION_AXES,
    RecombinationRecord,
    initial_recombination_bank,
    pareto_front_ids,
    prune_recombination_bank,
    recombine_with_gene,
    sample_recombination_gene_id,
    spawn_recombination_mutants,
)


EPS = 1e-12


@dataclass(frozen=True)
class SelfMatingChild:
    gene: Any
    parent_a: str
    parent_b: str
    required_capabilities: tuple[str, ...]
    recombination_gene_id: str
    recombination_axis: str
    metadata: dict[str, Any]


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
            str(key): _jsonable(
                item
            )
            for key, item
            in value.items()
        }
    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _jsonable(
                item
            )
            for item in value
        ]
    return value


def _load_v113_policy_bank(
    path: Path,
) -> dict[str, GeneRecord]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    records = {
        str(item["record_id"]): (
            GeneRecord.from_dict(
                item
            )
        )
        for item in data.get(
            "records",
            []
        )
    }
    if not records:
        raise RuntimeError(
            "No V1.13 policy records found in checkpoint"
        )
    return records


def _self_mating_children(
    *,
    pair_count: int,
    children_per_pair: int,
    active_ids: list[str],
    policy_records: dict[str, GeneRecord],
    best: dict[str, float],
    recombination_records: dict[
        str,
        RecombinationRecord,
    ],
    anchor,
    rng: np.random.Generator,
    parent_q_power: float,
    parent_uniform_fraction: float,
    rule_selection_power: float,
    rule_uniform_fraction: float,
    generation: int,
) -> list[SelfMatingChild]:
    children: list[
        SelfMatingChild
    ] = []

    for _ in range(pair_count):
        parent_a = _sample_record_id(
            active_ids,
            policy_records,
            best,
            rng,
            power=parent_q_power,
            uniform_fraction=(
                parent_uniform_fraction
            ),
        )
        partner_ids = (
            _complementary_partner_ids(
                parent_a,
                active_ids,
                policy_records,
            )
        )
        if not partner_ids:
            continue
        parent_b = _sample_record_id(
            partner_ids,
            policy_records,
            best,
            rng,
            power=parent_q_power,
            uniform_fraction=(
                parent_uniform_fraction
            ),
        )

        record_a = policy_records[
            parent_a
        ]
        record_b = policy_records[
            parent_b
        ]
        required = tuple(
            sorted(
                set(
                    record_a.capabilities
                )
                | set(
                    record_b.capabilities
                )
            )
        )
        quality_a = _quality_weight(
            record_a,
            best,
        )
        quality_b = _quality_weight(
            record_b,
            best,
        )

        for _child_index in range(
            children_per_pair
        ):
            (
                rule_id,
                selection_axis,
            ) = (
                sample_recombination_gene_id(
                    recombination_records,
                    rng,
                    selection_power=(
                        rule_selection_power
                    ),
                    uniform_fraction=(
                        rule_uniform_fraction
                    ),
                )
            )
            rule_record = (
                recombination_records[
                    rule_id
                ]
            )
            child_gene, metadata = (
                recombine_with_gene(
                    rule_record.gene,
                    record_a.gene,
                    record_b.gene,
                    anchor,
                    quality_a=quality_a,
                    quality_b=quality_b,
                    capability_count_a=(
                        len(
                            record_a.capabilities
                        )
                    ),
                    capability_count_b=(
                        len(
                            record_b.capabilities
                        )
                    ),
                )
            )
            rule_record.generated += 1
            rule_record.last_generation_used = (
                generation
            )

            children.append(
                SelfMatingChild(
                    gene=child_gene,
                    parent_a=parent_a,
                    parent_b=parent_b,
                    required_capabilities=(
                        required
                    ),
                    recombination_gene_id=(
                        rule_id
                    ),
                    recombination_axis=(
                        selection_axis
                    ),
                    metadata={
                        **metadata,
                        "selection_axis": (
                            selection_axis
                        ),
                    },
                )
            )

    return children


def _screen_and_rank_mating(
    children: list[
        SelfMatingChild
    ],
    screen_scores: dict[
        str,
        np.ndarray,
    ],
    parent_screen: dict[
        str,
        dict[str, float],
    ],
    policy_records: dict[
        str,
        GeneRecord,
    ],
    recombination_records: dict[
        str,
        RecombinationRecord,
    ],
    limit: int,
    screen_log: Path,
    generation: int,
) -> list[int]:
    ranking: list[
        tuple[float, int]
    ] = []

    with screen_log.open(
        "a",
        encoding="utf-8",
    ) as file:
        for idx, child in enumerate(
            children
        ):
            child_score = (
                _score_dict_at(
                    screen_scores,
                    idx,
                )
            )
            retention = (
                _inheritance_retention(
                    child_score,
                    policy_records[
                        child.parent_a
                    ],
                    policy_records[
                        child.parent_b
                    ],
                    child.required_capabilities,
                    override_a=(
                        parent_screen[
                            child.parent_a
                        ]
                    ),
                    override_b=(
                        parent_screen[
                            child.parent_b
                        ]
                    ),
                )
            )
            min_retention = (
                min(
                    retention.values()
                )
                if retention
                else 0.0
            )
            recombination_records[
                child.recombination_gene_id
            ].screen_retention_sum += float(
                min(
                    max(
                        min_retention,
                        0.0,
                    ),
                    1.25,
                )
            )
            ranking.append(
                (
                    float(
                        min_retention
                    ),
                    idx,
                )
            )
            file.write(
                json.dumps(
                    _jsonable(
                        {
                            "generation": (
                                generation
                            ),
                            "child_index": (
                                idx
                            ),
                            "parent_a": (
                                child.parent_a
                            ),
                            "parent_b": (
                                child.parent_b
                            ),
                            "required_capabilities": list(
                                child.required_capabilities
                            ),
                            "recombination_gene_id": (
                                child.recombination_gene_id
                            ),
                            "selection_axis": (
                                child.recombination_axis
                            ),
                            "scores": (
                                child_score
                            ),
                            "retention": (
                                retention
                            ),
                            "min_retention": (
                                min_retention
                            ),
                        }
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    ranking.sort(
        reverse=True
    )
    selected = [
        idx
        for _score, idx
        in ranking[
            :limit
        ]
    ]
    for idx in selected:
        recombination_records[
            children[
                idx
            ].recombination_gene_id
        ].screen_selected += 1
    return selected


def _admit_self_mating(
    *,
    selected: list[
        SelfMatingChild
    ],
    scores: dict[
        str,
        np.ndarray,
    ],
    generation: int,
    policy_records: dict[
        str,
        GeneRecord,
    ],
    recombination_records: dict[
        str,
        RecombinationRecord,
    ],
    capability_ceiling: dict[
        str,
        float,
    ],
    threshold: float,
    accepted_log: Path,
) -> list[str]:
    admitted: list[str] = []

    with accepted_log.open(
        "a",
        encoding="utf-8",
    ) as file:
        for idx, child in enumerate(
            selected
        ):
            score = _score_dict_at(
                scores,
                idx,
            )
            parent_a = policy_records[
                child.parent_a
            ]
            parent_b = policy_records[
                child.parent_b
            ]
            parent_retention = (
                _inheritance_retention(
                    score,
                    parent_a,
                    parent_b,
                    child.required_capabilities,
                )
            )
            ceiling_retention = (
                _capability_ceiling_retention(
                    score,
                    capability_ceiling,
                    child.required_capabilities,
                )
            )
            success = (
                _passes_inheritance_gate(
                    parent_retention,
                    ceiling_retention,
                    threshold=threshold,
                )
            )

            rule_record = (
                recombination_records[
                    child.recombination_gene_id
                ]
            )
            if success:
                rule_record.accepted += 1
                if (
                    len(
                        child.required_capabilities
                    )
                    == len(AXES)
                ):
                    rule_record.four_capability_accepted += 1

                record_id = _gene_id(
                    child.gene
                )
                if (
                    record_id
                    not in policy_records
                ):
                    policy_records[
                        record_id
                    ] = GeneRecord(
                        record_id=record_id,
                        gene=child.gene,
                        capabilities=tuple(
                            sorted(
                                child.required_capabilities
                            )
                        ),
                        scores=score,
                        origin="mating",
                        generation=generation,
                        parents=(
                            child.parent_a,
                            child.parent_b,
                        ),
                        operator=(
                            "recombination_gene:"
                            + child.recombination_gene_id
                        ),
                        metadata={
                            "recombination_gene_id": (
                                child.recombination_gene_id
                            ),
                            "recombination_axis": (
                                child.recombination_axis
                            ),
                            "recombination_metadata": (
                                child.metadata
                            ),
                            "parent_retention": (
                                parent_retention
                            ),
                            "ceiling_retention": (
                                ceiling_retention
                            ),
                        },
                    )
                    admitted.append(
                        record_id
                    )
                else:
                    policy_records[
                        record_id
                    ].capabilities = tuple(
                        sorted(
                            set(
                                policy_records[
                                    record_id
                                ].capabilities
                            )
                            | set(
                                child.required_capabilities
                            )
                        )
                    )

            event = {
                "generation": generation,
                "parent_a": child.parent_a,
                "parent_b": child.parent_b,
                "parent_a_capabilities": list(
                    parent_a.capabilities
                ),
                "parent_b_capabilities": list(
                    parent_b.capabilities
                ),
                "required_capabilities": list(
                    child.required_capabilities
                ),
                "recombination_gene_id": (
                    child.recombination_gene_id
                ),
                "recombination_axis": (
                    child.recombination_axis
                ),
                "recombination_gene": (
                    rule_record.gene.to_dict()
                ),
                "scores": score,
                "parent_retention": (
                    parent_retention
                ),
                "ceiling_retention": (
                    ceiling_retention
                ),
                "threshold": threshold,
                "accepted": success,
            }
            file.write(
                json.dumps(
                    _jsonable(
                        event
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return admitted


def _save_checkpoint(
    path: Path,
    *,
    next_generation: int,
    policy_records: dict[
        str,
        GeneRecord,
    ],
    recombination_records: dict[
        str,
        RecombinationRecord,
    ],
    rng: np.random.Generator,
) -> None:
    payload = {
        "next_generation": int(
            next_generation
        ),
        "policy_records": [
            record.to_dict()
            for record
            in policy_records.values()
        ],
        "recombination_records": [
            record.to_dict()
            for record
            in recombination_records.values()
        ],
        "rng_state": _jsonable(
            rng.bit_generator.state
        ),
    }
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_checkpoint(
    path: Path,
    rng: np.random.Generator,
) -> tuple[
    int,
    dict[str, GeneRecord],
    dict[str, RecombinationRecord],
]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    rng.bit_generator.state = (
        data["rng_state"]
    )
    policy_records = {
        str(item["record_id"]): (
            GeneRecord.from_dict(
                item
            )
        )
        for item in data[
            "policy_records"
        ]
    }
    recombination_records: dict[
        str,
        RecombinationRecord
    ] = {}
    for item in data[
        "recombination_records"
    ]:
        record = (
            RecombinationRecord.from_dict(
                item
            )
        )
        recombination_records[
            record.gene_id
        ] = record
    return (
        int(
            data[
                "next_generation"
            ]
        ),
        policy_records,
        recombination_records,
    )


def _rule_specialists(
    records: dict[
        str,
        RecombinationRecord,
    ],
) -> dict[str, str]:
    return {
        axis: max(
            records,
            key=lambda gene_id: (
                records[
                    gene_id
                ].axis_scores()[
                    axis
                ],
                records[
                    gene_id
                ].generated,
            ),
        )
        for axis in RECOMBINATION_AXES
    }


def train(
    args: argparse.Namespace,
) -> Path:
    config = EnvConfig()
    worlds, stars, scenario_seeds = (
        load_scenario_bank(
            Path(
                args.scenario_bank
            ),
            config,
        )
    )
    if len(worlds) != 100:
        raise RuntimeError(
            "V1.14 requires the frozen 100-world development bank"
        )

    anchor = _load_v18(
        Path(
            args.anchor_v18_run
        )
    )
    rng = np.random.default_rng(
        args.seed
    )

    if args.resume:
        checkpoint = Path(
            args.resume
        )
        run_dir = checkpoint.parent
        (
            start_generation,
            policy_records,
            recombination_records,
        ) = _load_checkpoint(
            checkpoint,
            rng,
        )
        print(
            f"RESUME={checkpoint} "
            f"NEXT_GENERATION="
            f"{start_generation}"
        )
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        run_dir = (
            Path(
                args.output_dir
            )
            / (
                "gene_mrta_v114_self_recombination_"
                f"{stamp}_seed{args.seed}"
            )
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        policy_records = (
            _load_v113_policy_bank(
                Path(
                    args.bootstrap_v113_checkpoint
                )
            )
        )
        recombination_records = (
            initial_recombination_bank(
                count=(
                    args.initial_recombination_genes
                ),
                seed=(
                    args.seed
                    + 114_000
                ),
                initial_sigma=(
                    args.recombination_sigma_start
                ),
            )
        )
        start_generation = 0

    screen_log = (
        run_dir
        / "recombination_screen.jsonl"
    )
    accepted_log = (
        run_dir
        / "recombination_events.jsonl"
    )
    history_path = (
        run_dir
        / "history.csv"
    )
    history_exists = (
        history_path.exists()
    )

    for generation in range(
        start_generation,
        args.generations,
    ):
        active_ids, _archives = (
            _active_parent_ids(
                policy_records,
                args.archive_size_per_axis,
                args.hybrid_bank_limit,
            )
        )
        active_records = {
            record_id: policy_records[
                record_id
            ]
            for record_id
            in active_ids
        }
        best = _best_by_axis(
            active_records
        )
        generation_capability_ceiling = {
            axis: float(
                best[axis]
            )
            for axis in AXES
        }

        if (
            generation > start_generation
            or not args.resume
        ):
            mutants = (
                spawn_recombination_mutants(
                    recombination_records,
                    rng,
                    generation=generation,
                    count=(
                        args.recombination_mutants_per_generation
                    ),
                    selection_power=(
                        args.recombination_selection_power
                    ),
                    uniform_fraction=(
                        args.recombination_uniform_fraction
                    ),
                    gate_flip_rate=(
                        args.recombination_gate_flip_rate
                    ),
                    sigma_tau=(
                        args.recombination_sigma_tau
                    ),
                    sigma_min=(
                        args.recombination_sigma_min
                    ),
                    sigma_max=(
                        args.recombination_sigma_max
                    ),
                )
            )
            for gene in mutants:
                recombination_records[
                    gene.gene_id
                ] = RecombinationRecord(
                    gene=gene,
                    metadata={
                        "origin": "mutation"
                    },
                )

        policy_sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        normal = _normal_children(
            count=(
                args.normal_offspring
            ),
            active_ids=active_ids,
            records=policy_records,
            best=best,
            rng=rng,
            sigma=policy_sigma,
            mutation_rate=(
                args.mutation_rate
            ),
            uniform_fraction=(
                args.parent_uniform_fraction
            ),
        )

        mating = _self_mating_children(
            pair_count=(
                args.mating_pairs
            ),
            children_per_pair=(
                args.children_per_pair
            ),
            active_ids=active_ids,
            policy_records=(
                policy_records
            ),
            best=best,
            recombination_records=(
                recombination_records
            ),
            anchor=anchor,
            rng=rng,
            parent_q_power=(
                args.mating_q_power
            ),
            parent_uniform_fraction=(
                args.parent_uniform_fraction
            ),
            rule_selection_power=(
                args.recombination_selection_power
            ),
            rule_uniform_fraction=(
                args.recombination_uniform_fraction
            ),
            generation=generation,
        )
        if len(mating) != (
            args.mating_offspring
        ):
            raise RuntimeError(
                "V1.14 did not produce requested mating population"
            )

        screen_ids = _screen_indices(
            generation,
            len(worlds),
            args.screen_worlds,
        )
        screen_worlds = [
            worlds[
                int(idx)
            ]
            for idx in screen_ids
        ]
        screen_stars = stars[
            screen_ids
        ]

        normal_screen = _evaluate_scores(
            normal,
            screen_worlds,
            screen_stars,
            config,
        )
        normal_full_ids = (
            _top_normal_full_indices(
                normal_screen,
                args.normal_full_per_axis,
            )
        )

        mating_genes = [
            child.gene
            for child in mating
        ]
        mating_screen = _evaluate_scores(
            mating_genes,
            screen_worlds,
            screen_stars,
            config,
        )
        parent_screen = (
            _parent_screen_scores(
                [
                    parent_id
                    for child in mating
                    for parent_id
                    in (
                        child.parent_a,
                        child.parent_b,
                    )
                ],
                policy_records,
                screen_worlds,
                screen_stars,
                config,
            )
        )
        mating_full_ids = (
            _screen_and_rank_mating(
                mating,
                mating_screen,
                parent_screen,
                policy_records,
                recombination_records,
                args.mating_full_candidates,
                screen_log,
                generation,
            )
        )

        normal_full_genes = [
            normal[idx]
            for idx in normal_full_ids
        ]
        normal_full_scores = (
            _evaluate_scores(
                normal_full_genes,
                worlds,
                stars,
                config,
            )
        )
        admitted_normal = (
            _admit_normal_children(
                genes=normal_full_genes,
                scores=normal_full_scores,
                generation=generation,
                records=policy_records,
                archive_size=(
                    args.archive_size_per_axis
                ),
            )
        )

        selected_mating = [
            mating[idx]
            for idx
            in mating_full_ids
        ]
        selected_mating_scores = (
            _evaluate_scores(
                [
                    child.gene
                    for child
                    in selected_mating
                ],
                worlds,
                stars,
                config,
            )
        )
        admitted_mating = (
            _admit_self_mating(
                selected=selected_mating,
                scores=(
                    selected_mating_scores
                ),
                generation=generation,
                policy_records=(
                    policy_records
                ),
                recombination_records=(
                    recombination_records
                ),
                capability_ceiling=(
                    generation_capability_ceiling
                ),
                threshold=(
                    args.inheritance_threshold
                ),
                accepted_log=(
                    accepted_log
                ),
            )
        )

        policy_records = (
            _prune_active_records(
                policy_records,
                args.archive_size_per_axis,
                args.hybrid_bank_limit,
            )
        )
        recombination_records = (
            prune_recombination_bank(
                recombination_records,
                specialist_size=(
                    args.recombination_specialist_size
                ),
                pareto_limit=(
                    args.recombination_pareto_limit
                ),
                total_limit=(
                    args.recombination_bank_limit
                ),
            )
        )

        active_ids, _archives = (
            _active_parent_ids(
                policy_records,
                args.archive_size_per_axis,
                args.hybrid_bank_limit,
            )
        )
        best = _best_by_axis(
            policy_records,
            active_ids,
        )
        certified = {
            record_id: _certified_capabilities(
                policy_records[
                    record_id
                ],
                best,
                threshold=(
                    args.inheritance_threshold
                ),
            )
            for record_id in active_ids
        }
        best_certified_record_id = max(
            active_ids,
            key=lambda record_id: (
                len(
                    certified[
                        record_id
                    ]
                ),
                _quality_weight(
                    policy_records[
                        record_id
                    ],
                    best,
                ),
            ),
        )
        max_certified = max(
            (
                len(
                    certified[
                        record_id
                    ]
                )
                for record_id
                in active_ids
            ),
            default=0,
        )
        route_diag = _route_diagnostics(
            policy_records[
                best_certified_record_id
            ].gene,
            worlds,
            config,
            limit=10,
        )

        front = pareto_front_ids(
            recombination_records
        )
        specialists = (
            _rule_specialists(
                recombination_records
            )
        )
        most_used_rule_id = max(
            recombination_records,
            key=lambda gene_id: (
                recombination_records[
                    gene_id
                ].accepted,
                recombination_records[
                    gene_id
                ].screen_selected,
                recombination_records[
                    gene_id
                ].generated,
            ),
        )
        most_used_rule = (
            recombination_records[
                most_used_rule_id
            ]
        )

        row: dict[str, object] = {
            "generation": generation,
            "policy_bank_size": len(
                active_ids
            ),
            "normal_admitted": len(
                admitted_normal
            ),
            "mating_accepted": len(
                admitted_mating
            ),
            "max_certified_capability_count": (
                max_certified
            ),
            "best_certified_record_id": (
                best_certified_record_id
            ),
            "policy_mutation_sigma": (
                policy_sigma
            ),
            "mean_queue_depth": (
                route_diag[
                    "mean_queue_depth"
                ]
            ),
            "mean_max_queue_depth": (
                route_diag[
                    "mean_max_queue_depth"
                ]
            ),
            "multi_task_world_fraction": (
                route_diag[
                    "multi_task_world_fraction"
                ]
            ),
            "recombination_bank_size": (
                len(
                    recombination_records
                )
            ),
            "recombination_pareto_size": (
                len(front)
            ),
            "top_accept_rule_id": (
                most_used_rule_id
            ),
            "top_accept_rule_generated": (
                most_used_rule.generated
            ),
            "top_accept_rule_accepted": (
                most_used_rule.accepted
            ),
            "top_accept_rule_terms": (
                most_used_rule.gene.active_term_count
            ),
            "top_accept_rule_sigma": (
                most_used_rule.gene.mutation_sigma
            ),
        }
        for axis in AXES:
            row[
                f"best_{axis}"
            ] = best[
                axis
            ]
        for axis in RECOMBINATION_AXES:
            rule_id = specialists[
                axis
            ]
            row[
                f"rule_{axis}_id"
            ] = rule_id
            row[
                f"rule_{axis}_score"
            ] = (
                recombination_records[
                    rule_id
                ].axis_scores()[
                    axis
                ]
            )

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
            writer.writerow(
                row
            )

        if (
            generation
            % args.log_every
            == 0
            or generation
            == args.generations - 1
        ):
            print(
                f"GEN {generation:04d} | "
                f"Pbank={len(active_ids)} "
                f"Rbank={len(recombination_records)} "
                f"Rpareto={len(front)} "
                f"accepted={len(admitted_mating)} "
                f"certified_caps={max_certified} "
                f"mean={best['mean_time']:.4f} "
                f"tail10={best['tail10_time']:.4f} "
                f"cont={best['continuation_preservation']:.4f} "
                f"reserve={best['fleet_option_reserve']:.4f} "
                f"qmean={route_diag['mean_queue_depth']:.2f} "
                f"multi={route_diag['multi_task_world_fraction']:.2f} "
                f"Rtop={most_used_rule_id[:8]} "
                f"Racc={most_used_rule.accepted}/"
                f"{most_used_rule.generated} "
                f"Rterms={most_used_rule.gene.active_term_count} "
                f"Rsigma={most_used_rule.gene.mutation_sigma:.3f}"
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
                policy_records=(
                    policy_records
                ),
                recombination_records=(
                    recombination_records
                ),
                rng=rng,
            )

    active_ids, _archives = (
        _active_parent_ids(
            policy_records,
            args.archive_size_per_axis,
            args.hybrid_bank_limit,
        )
    )
    best = _best_by_axis(
        policy_records,
        active_ids,
    )
    specialists = (
        _rule_specialists(
            recombination_records
        )
    )
    front = pareto_front_ids(
        recombination_records
    )

    summary = {
        "experiment": (
            "gene_mrta_v114_self_evolving_recombination_bank"
        ),
        "policy_semantics": (
            "v113_route_tail_multi_task_append"
        ),
        "policy_parameter_count": 148,
        "bootstrap_v113_checkpoint": (
            args.bootstrap_v113_checkpoint
        ),
        "scenario_bank": (
            args.scenario_bank
        ),
        "scenario_seeds": (
            scenario_seeds
        ),
        "policy_axes": list(
            AXES
        ),
        "recombination_axes": list(
            RECOMBINATION_AXES
        ),
        "recombination_genotype": {
            "continuous_coefficients": 7,
            "binary_feature_gates": 7,
            "self_adaptive_mutation_sigma": True,
            "formula_family": (
                "parent-swap-symmetric adaptive ancestor-delta law"
            ),
        },
        "recombination_selection": {
            "axis_choice": (
                "uniform over independent recombination axes"
            ),
            "selection_power": (
                args.recombination_selection_power
            ),
            "uniform_exploration_fraction": (
                args.recombination_uniform_fraction
            ),
            "scalarized_reward": False,
        },
        "best_policy_axis_scores": (
            best
        ),
        "recombination_pareto_front": [
            recombination_records[
                gene_id
            ].to_dict()
            for gene_id in front
        ],
        "recombination_axis_specialists": {
            axis: recombination_records[
                specialists[
                    axis
                ]
            ].to_dict()
            for axis in RECOMBINATION_AXES
        },
        "protected_final_rule": (
            "95M is development data; 99M remains untouched "
            "until the V1.14 procedure and candidate are frozen."
        ),
    }
    (
        run_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            _jsonable(
                summary
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nFINAL V1.14 SELF-EVOLVING RECOMBINATION"
    )
    print(
        "POLICY_BEST="
        + json.dumps(
            best
        )
    )
    print(
        "RECOMBINATION_SPECIALISTS="
        + json.dumps(
            {
                axis: specialists[
                    axis
                ]
                for axis
                in RECOMBINATION_AXES
            }
        )
    )
    print(
        "RECOMBINATION_PARETO_SIZE="
        f"{len(front)}"
    )
    print(
        f"RUN_DIR={run_dir}"
    )
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenario-bank",
        required=True,
    )
    p.add_argument(
        "--bootstrap-v113-checkpoint",
        required=True,
    )
    p.add_argument(
        "--anchor-v18-run",
        required=True,
    )
    p.add_argument(
        "--resume",
        default="",
    )

    p.add_argument(
        "--generations",
        type=int,
        default=50,
    )
    p.add_argument(
        "--normal-offspring",
        type=int,
        default=128,
    )
    p.add_argument(
        "--mating-offspring",
        type=int,
        default=128,
    )
    p.add_argument(
        "--mating-pairs",
        type=int,
        default=32,
    )
    p.add_argument(
        "--children-per-pair",
        type=int,
        default=4,
    )
    p.add_argument(
        "--mating-q-power",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--parent-uniform-fraction",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--inheritance-threshold",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--screen-worlds",
        type=int,
        default=25,
    )
    p.add_argument(
        "--normal-full-per-axis",
        type=int,
        default=4,
    )
    p.add_argument(
        "--mating-full-candidates",
        type=int,
        default=16,
    )
    p.add_argument(
        "--archive-size-per-axis",
        type=int,
        default=16,
    )
    p.add_argument(
        "--hybrid-bank-limit",
        type=int,
        default=128,
    )
    p.add_argument(
        "--mutation-rate",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--mutation-sigma-start",
        type=float,
        default=0.04,
    )
    p.add_argument(
        "--mutation-sigma-end",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--initial-recombination-genes",
        type=int,
        default=24,
    )
    p.add_argument(
        "--recombination-mutants-per-generation",
        type=int,
        default=8,
    )
    p.add_argument(
        "--recombination-bank-limit",
        type=int,
        default=32,
    )
    p.add_argument(
        "--recombination-specialist-size",
        type=int,
        default=6,
    )
    p.add_argument(
        "--recombination-pareto-limit",
        type=int,
        default=12,
    )
    p.add_argument(
        "--recombination-selection-power",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--recombination-uniform-fraction",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--recombination-gate-flip-rate",
        type=float,
        default=0.08,
    )
    p.add_argument(
        "--recombination-sigma-start",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--recombination-sigma-tau",
        type=float,
        default=0.15,
    )
    p.add_argument(
        "--recombination-sigma-min",
        type=float,
        default=0.03,
    )
    p.add_argument(
        "--recombination-sigma-max",
        type=float,
        default=0.80,
    )

    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    p.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "gene_mrta_v114_self_recombination"
        ),
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if (
        args.mating_pairs
        * args.children_per_pair
        != args.mating_offspring
    ):
        raise ValueError(
            "mating_pairs * children_per_pair "
            "must equal mating_offspring"
        )
    if (
        args.normal_offspring
        != args.mating_offspring
    ):
        raise ValueError(
            "V1.14 keeps the policy population split at 50/50 "
            "normal mutation vs recombination"
        )
    if not (
        1 <= args.screen_worlds <= 100
    ):
        raise ValueError(
            "screen_worlds must be in 1..100"
        )
    train(
        args
    )


if __name__ == "__main__":
    main()
