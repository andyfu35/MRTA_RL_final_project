from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v110.recombination import (
    hidden_block_indices,
)
from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from marl2d.gene_mrta_v117.stage_a_train import (
    Record,
    _record_from_dict,
)
from marl2d.gene_mrta_v118.capabilities import (
    BASE_AXES,
    GeneAssessment,
    evaluate_gene,
)
from marl2d.gene_mrta_v118.oracle_bank import (
    load_oracle_bank,
)
from marl2d.gene_mrta_v118.pareto_bank import (
    analysis_best_by_axis,
    maximin_gene_id,
    rebuild_pareto_bank,
)


CHECKPOINT_VERSION = (
    "v118_feasibility_first_train_v1"
)


def _gene_id(
    gene: RouteTailDirectGene,
) -> str:
    return hashlib.sha256(
        np.asarray(
            gene.vector_data,
            dtype=np.float64,
        ).tobytes()
    ).hexdigest()[:20]


def _atomic_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    with temp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.flush()
        os.fsync(
            handle.fileno()
        )
    temp.replace(
        path
    )


def _assessment_to_dict(
    assessment: GeneAssessment,
) -> dict[str, Any]:
    return {
        "success": bool(
            assessment.success
        ),
        "worst_completion": float(
            assessment.worst_completion
        ),
        "mean_completion": float(
            assessment.mean_completion
        ),
        "success_worlds": int(
            assessment.success_worlds
        ),
        "world_count": int(
            assessment.world_count
        ),
    }


def _assessment_from_dict(
    data: dict[str, Any],
    scores: dict[str, float],
) -> GeneAssessment:
    return GeneAssessment(
        success=bool(
            data[
                "success"
            ]
        ),
        worst_completion=float(
            data[
                "worst_completion"
            ]
        ),
        mean_completion=float(
            data[
                "mean_completion"
            ]
        ),
        success_worlds=int(
            data[
                "success_worlds"
            ]
        ),
        world_count=int(
            data[
                "world_count"
            ]
        ),
        scores=scores,
    )


def _evaluate_one(
    gene: RouteTailDirectGene,
    worlds,
    config,
    time_optima,
    path_optima,
    priority_optima,
    deadline_optima,
) -> GeneAssessment:
    return evaluate_gene(
        gene,
        worlds,
        config,
        time_optima=(
            time_optima
        ),
        path_efficiency_optima=(
            path_optima
        ),
        priority_service_optima=(
            priority_optima
        ),
        deadline_optima=(
            deadline_optima
        ),
    )


def _evaluate_many(
    genes: list[
        RouteTailDirectGene
    ],
    worlds,
    config,
    time_optima,
    path_optima,
    priority_optima,
    deadline_optima,
) -> list[
    GeneAssessment
]:
    return [
        _evaluate_one(
            gene,
            worlds,
            config,
            time_optima,
            path_optima,
            priority_optima,
            deadline_optima,
        )
        for gene in genes
    ]


def _feasibility_key(
    assessment: GeneAssessment,
) -> tuple[
    float,
    float,
    int,
]:
    return (
        assessment.worst_completion,
        assessment.mean_completion,
        assessment.success_worlds,
    )


def _trim_bootstrap(
    records: dict[
        str,
        Record,
    ],
    assessments: dict[
        str,
        GeneAssessment,
    ],
    *,
    size: int,
) -> tuple[
    dict[str, Record],
    dict[
        str,
        GeneAssessment,
    ],
]:
    ordered = sorted(
        records,
        key=lambda rid: (
            _feasibility_key(
                assessments[
                    rid
                ]
            ),
            rid,
        ),
        reverse=True,
    )
    keep = ordered[
        :size
    ]
    return (
        {
            rid: records[
                rid
            ]
            for rid in keep
        },
        {
            rid: assessments[
                rid
            ]
            for rid in keep
        },
    )


def _mating_child(
    parent_a: RouteTailDirectGene,
    parent_b: RouteTailDirectGene,
    rng: np.random.Generator,
) -> tuple[
    RouteTailDirectGene,
    str,
]:
    blocks, globals_ = (
        hidden_block_indices(
            parent_a.hidden_dim
        )
    )
    recipient_is_a = bool(
        rng.integers(
            0,
            2,
        )
    )
    recipient = (
        parent_a
        if recipient_is_a
        else parent_b
    )
    donor = (
        parent_b
        if recipient_is_a
        else parent_a
    )
    vector = (
        recipient.vector_data.copy()
    )

    mode = int(
        rng.integers(
            0,
            3,
        )
    )
    if mode == 0:
        count = (
            1
            if rng.random()
            < 0.8
            else 2
        )
        selected = rng.choice(
            len(blocks),
            size=min(
                count,
                len(blocks),
            ),
            replace=False,
        )
        for index in selected:
            ids = blocks[
                int(index)
            ]
            vector[ids] = (
                donor.vector_data[
                    ids
                ]
            )
        operator = (
            "sparse_block_graft"
        )
    elif mode == 1:
        count = (
            1
            if rng.random()
            < 0.8
            else 2
        )
        selected = rng.choice(
            len(blocks),
            size=min(
                count,
                len(blocks),
            ),
            replace=False,
        )
        for index in selected:
            ids = blocks[
                int(index)
            ]
            weight = float(
                rng.uniform(
                    0.15,
                    0.40,
                )
            )
            vector[ids] = (
                (
                    1.0 - weight
                )
                * vector[ids]
                + weight
                * donor.vector_data[
                    ids
                ]
            )
        operator = (
            "sparse_block_blend"
        )
    else:
        weight = float(
            rng.uniform(
                0.05,
                0.20,
            )
        )
        vector = (
            (
                1.0 - weight
            )
            * recipient.vector_data
            + weight
            * donor.vector_data
        )
        operator = (
            "near_parent_blend"
        )

    if rng.random() < 0.10:
        vector[
            globals_
        ] = (
            donor.vector_data[
                globals_
            ]
        )

    return (
        RouteTailDirectGene(
            np.asarray(
                vector,
                dtype=np.float64,
            ),
            hidden_dim=(
                recipient.hidden_dim
            ),
        ),
        operator,
    )


def _maximin_summary(
    records: dict[
        str,
        Record,
    ],
) -> dict[str, Any] | None:
    rid = maximin_gene_id(
        records
    )
    if rid is None:
        return None
    record = records[
        rid
    ]
    return {
        "record_id": rid,
        "min_capability": float(
            min(
                record.scores[
                    axis
                ]
                for axis in BASE_AXES
            )
        ),
        "scores": record.scores,
    }


def run(
    args: argparse.Namespace,
) -> Path:
    (
        config,
        worlds,
        time_optima,
        path_optima,
        priority_optima,
        deadline_optima,
    ) = load_oracle_bank(
        Path(
            args.oracle_bank
        )
    )

    run_dir = Path(
        args.run_dir
    )
    checkpoint_path = (
        run_dir
        / "checkpoint.json"
    )

    if checkpoint_path.exists():
        data = json.loads(
            checkpoint_path.read_text(
                encoding="utf-8",
            )
        )
        if (
            data.get(
                "version"
            )
            != CHECKPOINT_VERSION
        ):
            raise ValueError(
                "Incompatible V1.18 checkpoint"
            )
        rng = np.random.default_rng()
        rng.bit_generator.state = (
            data[
                "rng_state"
            ]
        )
        bootstrap_records = {
            row["record_id"]:
                _record_from_dict(
                    row
                )
            for row in data[
                "bootstrap_records"
            ]
        }
        bootstrap_assessments = {
            rid: _assessment_from_dict(
                data[
                    "bootstrap_assessments"
                ][rid],
                bootstrap_records[
                    rid
                ].scores,
            )
            for rid in bootstrap_records
        }
        pareto_records = {
            row["record_id"]:
                _record_from_dict(
                    row
                )
            for row in data[
                "pareto_records"
            ]
        }
        history = list(
            data.get(
                "history",
                [],
            )
        )
        start_generation = (
            int(
                data[
                    "generation"
                ]
            )
            + 1
        )
    else:
        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )
        rng = np.random.default_rng(
            args.seed
        )
        genes = [
            RouteTailDirectGene.random(
                rng,
                hidden_dim=8,
                scale=(
                    args.initial_scale
                ),
            )
            for _ in range(
                args.population
            )
        ]
        assessments = _evaluate_many(
            genes,
            worlds,
            config,
            time_optima,
            path_optima,
            priority_optima,
            deadline_optima,
        )

        bootstrap_records = {}
        bootstrap_assessments = {}
        successful: dict[
            str,
            Record,
        ] = {}

        for gene, assessment in zip(
            genes,
            assessments,
            strict=True,
        ):
            rid = _gene_id(
                gene
            )
            record = Record(
                record_id=rid,
                gene=gene,
                scores=dict(
                    assessment.scores
                ),
                capabilities=(),
                origin=(
                    "random_initial"
                ),
                generation=-1,
            )
            bootstrap_records[
                rid
            ] = record
            bootstrap_assessments[
                rid
            ] = assessment
            if assessment.success:
                successful[
                    rid
                ] = record

        (
            bootstrap_records,
            bootstrap_assessments,
        ) = _trim_bootstrap(
            bootstrap_records,
            bootstrap_assessments,
            size=(
                args.bootstrap_size
            ),
        )

        pareto_records = dict(
            rebuild_pareto_bank(
                successful,
                max_size=(
                    args.pareto_max_size
                ),
                epsilon=(
                    args.pareto_epsilon
                ),
            ).records
        )
        history = []
        start_generation = 0

    for generation in range(
        start_generation,
        args.generations,
    ):
        parent_ids = list(
            pareto_records
        )
        if len(parent_ids) < 2:
            parent_ids = list(
                bootstrap_records
            )

        children: list[
            RouteTailDirectGene
        ] = []
        metadata: list[
            tuple[
                str,
                tuple[str, ...],
                str | None,
            ]
        ] = []

        for _ in range(
            args.mutation_children
        ):
            parent_id = parent_ids[
                int(
                    rng.integers(
                        0,
                        len(parent_ids),
                    )
                )
            ]
            child = (
                bootstrap_records[
                    parent_id
                ].gene
                if parent_id
                in bootstrap_records
                else pareto_records[
                    parent_id
                ].gene
            ).mutated(
                rng,
                sigma=(
                    args.mutation_sigma
                ),
                mutation_rate=(
                    args.mutation_rate
                ),
            )
            children.append(
                RouteTailDirectGene.from_v18(
                    child
                )
            )
            metadata.append(
                (
                    "mutation",
                    (
                        parent_id,
                    ),
                    None,
                )
            )

        if len(parent_ids) >= 2:
            for _ in range(
                args.mating_children
            ):
                selected = rng.choice(
                    len(parent_ids),
                    size=2,
                    replace=False,
                )
                a = parent_ids[
                    int(
                        selected[0]
                    )
                ]
                b = parent_ids[
                    int(
                        selected[1]
                    )
                ]
                gene_a = (
                    bootstrap_records[
                        a
                    ].gene
                    if a
                    in bootstrap_records
                    else pareto_records[
                        a
                    ].gene
                )
                gene_b = (
                    bootstrap_records[
                        b
                    ].gene
                    if b
                    in bootstrap_records
                    else pareto_records[
                        b
                    ].gene
                )
                child, operator = (
                    _mating_child(
                        gene_a,
                        gene_b,
                        rng,
                    )
                )
                children.append(
                    child
                )
                metadata.append(
                    (
                        "mating",
                        (
                            a,
                            b,
                        ),
                        operator,
                    )
                )

        assessments = _evaluate_many(
            children,
            worlds,
            config,
            time_optima,
            path_optima,
            priority_optima,
            deadline_optima,
        )

        merged_bootstrap = dict(
            bootstrap_records
        )
        merged_assessments = dict(
            bootstrap_assessments
        )
        merged_success = dict(
            pareto_records
        )
        newly_successful = 0

        for (
            gene,
            assessment,
            meta,
        ) in zip(
            children,
            assessments,
            metadata,
            strict=True,
        ):
            rid = _gene_id(
                gene
            )
            if rid in merged_bootstrap:
                continue
            origin, parents, operator = (
                meta
            )
            record = Record(
                record_id=rid,
                gene=gene,
                scores=dict(
                    assessment.scores
                ),
                capabilities=(),
                origin=origin,
                generation=(
                    generation
                ),
                parents=parents,
                operator=operator,
            )
            merged_bootstrap[
                rid
            ] = record
            merged_assessments[
                rid
            ] = assessment
            if assessment.success:
                merged_success[
                    rid
                ] = record
                if (
                    rid
                    not in pareto_records
                ):
                    newly_successful += 1

        (
            bootstrap_records,
            bootstrap_assessments,
        ) = _trim_bootstrap(
            merged_bootstrap,
            merged_assessments,
            size=(
                args.bootstrap_size
            ),
        )

        rebuilt = rebuild_pareto_bank(
            merged_success,
            max_size=(
                args.pareto_max_size
            ),
            epsilon=(
                args.pareto_epsilon
            ),
        )
        pareto_records = dict(
            rebuilt.records
        )

        best_bootstrap_id = max(
            bootstrap_assessments,
            key=lambda rid: (
                _feasibility_key(
                    bootstrap_assessments[
                        rid
                    ]
                )
            ),
        )
        best_bootstrap = (
            bootstrap_assessments[
                best_bootstrap_id
            ]
        )

        row = {
            "generation": generation,
            "success_gate": (
                "all_tasks_on_all_training_worlds"
            ),
            "bootstrap_size": len(
                bootstrap_records
            ),
            "best_worst_completion": float(
                best_bootstrap.worst_completion
            ),
            "best_mean_completion": float(
                best_bootstrap.mean_completion
            ),
            "best_success_worlds": int(
                best_bootstrap.success_worlds
            ),
            "world_count": int(
                best_bootstrap.world_count
            ),
            "newly_successful_genes": int(
                newly_successful
            ),
            "pareto_size": len(
                pareto_records
            ),
            "dominated_removed": len(
                rebuilt.dominated_ids
            ),
            "epsilon_removed": len(
                rebuilt.epsilon_duplicate_ids
            ),
            "crowding_removed": len(
                rebuilt.crowding_removed_ids
            ),
            "analysis_best_by_axis": (
                analysis_best_by_axis(
                    pareto_records
                )
            ),
            "maximin_gene": (
                _maximin_summary(
                    pareto_records
                )
            ),
        }
        history.append(
            row
        )
        print(
            "V118_TRAIN "
            + json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

        payload = {
            "version": (
                CHECKPOINT_VERSION
            ),
            "generation": generation,
            "oracle_bank": str(
                args.oracle_bank
            ),
            "axes": list(
                BASE_AXES
            ),
            "rng_state": (
                rng.bit_generator.state
            ),
            "bootstrap_records": [
                record.to_dict()
                for record
                in bootstrap_records.values()
            ],
            "bootstrap_assessments": {
                rid: _assessment_to_dict(
                    assessment
                )
                for rid, assessment
                in bootstrap_assessments.items()
            },
            "pareto_records": [
                record.to_dict()
                for record
                in pareto_records.values()
            ],
            "history": history,
        }
        _atomic_write(
            checkpoint_path,
            payload,
        )

    print(
        f"V118_RUN_DIR={run_dir}",
        flush=True,
    )
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--oracle-bank",
        required=True,
    )
    p.add_argument(
        "--run-dir",
        required=True,
    )
    p.add_argument(
        "--generations",
        type=int,
        default=50,
    )
    p.add_argument(
        "--population",
        type=int,
        default=256,
    )
    p.add_argument(
        "--bootstrap-size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--mutation-children",
        type=int,
        default=192,
    )
    p.add_argument(
        "--mating-children",
        type=int,
        default=64,
    )
    p.add_argument(
        "--initial-scale",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.12,
    )
    p.add_argument(
        "--mutation-rate",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--pareto-max-size",
        type=int,
        default=256,
    )
    p.add_argument(
        "--pareto-epsilon",
        type=float,
        default=0.005,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=118,
    )
    return p


def main() -> None:
    run(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
