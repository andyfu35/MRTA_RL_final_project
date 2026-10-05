from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
)
from .bank import (
    BankRecord,
    best_by_axis,
    global_best_id,
    rebuild_bank,
)
from .benchmark import (
    load_manifest,
    select_split,
)
from .capabilities import (
    GeneAssessment,
    InstanceAssessment,
    capability_axes,
    evaluate_gene,
    paired_delta,
)


CHECKPOINT_VERSION = (
    "v119_public_mtrpd_mutation_bank_v1"
)


def _gene_id(
    gene: RouteTailDirectGene,
) -> str:
    return hashlib.sha256(
        np.asarray(
            gene.vector_data,
            dtype=np.float64,
        ).tobytes()
    ).hexdigest()[
        :20
    ]


def _atomic_write(
    path: Path,
    payload: dict[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_suffix(
        path.suffix
        + ".tmp"
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
    value: GeneAssessment,
) -> dict[
    str,
    Any,
]:
    return {
        "success": value.success,
        "worst_completion": (
            value.worst_completion
        ),
        "mean_completion": (
            value.mean_completion
        ),
        "success_instances": (
            value.success_instances
        ),
        "instance_count": (
            value.instance_count
        ),
        "scores": dict(
            value.scores
        ),
        "overall_optimum_retention": (
            value.overall_optimum_retention
        ),
        "worst_optimum_retention": (
            value.worst_optimum_retention
        ),
        "exact_optimum_matches": (
            value.exact_optimum_matches
        ),
        "reference_violations": (
            value.reference_violations
        ),
        "instances": [
            row.to_dict()
            for row in (
                value.instances
            )
        ],
    }


def _assessment_from_dict(
    data: dict[
        str,
        Any,
    ],
) -> GeneAssessment:
    rows = tuple(
        InstanceAssessment(
            instance_id=str(
                row[
                    "instance_id"
                ]
            ),
            vertex_count=int(
                row[
                    "vertex_count"
                ]
            ),
            robot_count=int(
                row[
                    "robot_count"
                ]
            ),
            success=bool(
                row[
                    "success"
                ]
            ),
            completion=float(
                row[
                    "completion"
                ]
            ),
            total_latency=float(
                row[
                    "total_latency"
                ]
            ),
            optimum_total_latency=float(
                row[
                    "optimum_total_latency"
                ]
            ),
            optimum_retention=float(
                row[
                    "optimum_retention"
                ]
            ),
            optimality_gap=(
                None
                if row[
                    "optimality_gap"
                ] is None
                else float(
                    row[
                        "optimality_gap"
                    ]
                )
            ),
            reference_violation=bool(
                row[
                    "reference_violation"
                ]
            ),
        )
        for row in data[
            "instances"
        ]
    )
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
        success_instances=int(
            data[
                "success_instances"
            ]
        ),
        instance_count=int(
            data[
                "instance_count"
            ]
        ),
        scores={
            str(
                key
            ): float(
                value
            )
            for key, value
            in dict(
                data[
                    "scores"
                ]
            ).items()
        },
        overall_optimum_retention=float(
            data[
                "overall_optimum_retention"
            ]
        ),
        worst_optimum_retention=float(
            data[
                "worst_optimum_retention"
            ]
        ),
        exact_optimum_matches=int(
            data[
                "exact_optimum_matches"
            ]
        ),
        reference_violations=int(
            data[
                "reference_violations"
            ]
        ),
        instances=rows,
    )


def _record_from_gene(
    gene: RouteTailDirectGene,
    assessment: GeneAssessment,
    *,
    generation: int,
    origin: str,
    parents: tuple[
        str,
        ...,
    ] = (),
) -> BankRecord:
    return BankRecord(
        record_id=_gene_id(
            gene
        ),
        vector_data=tuple(
            float(
                value
            )
            for value in (
                gene.vector_data
            )
        ),
        hidden_dim=int(
            gene.hidden_dim
        ),
        scores=dict(
            assessment.scores
        ),
        overall_retention=float(
            assessment.overall_optimum_retention
        ),
        worst_retention=float(
            assessment.worst_optimum_retention
        ),
        generation=int(
            generation
        ),
        origin=origin,
        parents=parents,
    )


def _gene_from_record(
    record: BankRecord,
) -> RouteTailDirectGene:
    return RouteTailDirectGene(
        np.asarray(
            record.vector_data,
            dtype=np.float64,
        ),
        hidden_dim=(
            record.hidden_dim
        ),
    )


def _bootstrap_key(
    assessment: GeneAssessment,
) -> tuple[
    float,
    float,
    int,
    float,
]:
    return (
        assessment.worst_completion,
        assessment.mean_completion,
        assessment.success_instances,
        assessment.overall_optimum_retention,
    )


def _trim_bootstrap(
    records: dict[
        str,
        BankRecord,
    ],
    assessments: dict[
        str,
        GeneAssessment,
    ],
    *,
    size: int,
) -> tuple[
    dict[
        str,
        BankRecord,
    ],
    dict[
        str,
        GeneAssessment,
    ],
]:
    ordered = sorted(
        records,
        key=lambda rid: (
            _bootstrap_key(
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


def _parent_probabilities(
    parent_ids: list[
        str
    ],
    assessments: dict[
        str,
        GeneAssessment,
    ],
) -> np.ndarray:
    raw = []
    for rid in parent_ids:
        assessment = assessments[
            rid
        ]
        if assessment.success:
            score = max(
                assessment.overall_optimum_retention,
                1e-6,
            )
        else:
            score = max(
                0.5
                * assessment.worst_completion
                + 0.5
                * assessment.mean_completion,
                1e-6,
            )
        raw.append(
            score
            * score
        )
    weights = np.asarray(
        raw,
        dtype=np.float64,
    )
    return (
        weights
        / np.sum(
            weights
        )
    )


def _append_jsonl(
    path: Path,
    payload: dict[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )


def _checkpoint_payload(
    *,
    generation: int,
    manifest: str,
    manifest_sha256: str,
    instance_ids: tuple[str, ...],
    axes: tuple[
        str,
        ...,
    ],
    rng: np.random.Generator,
    bootstrap_records: dict[
        str,
        BankRecord,
    ],
    bank_records: dict[
        str,
        BankRecord,
    ],
    assessments: dict[
        str,
        GeneAssessment,
    ],
    history: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:
    keep_ids = sorted(
        set(
            bootstrap_records
        )
        | set(
            bank_records
        )
    )
    return {
        "version": (
            CHECKPOINT_VERSION
        ),
        "generation": int(
            generation
        ),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "instance_ids": list(
            instance_ids
        ),
        "split": "evolution",
        "axes": list(
            axes
        ),
        "rng_state": (
            rng.bit_generator.state
        ),
        "bootstrap_ids": list(
            bootstrap_records
        ),
        "bank_ids": list(
            bank_records
        ),
        "records": [
            (
                bootstrap_records.get(
                    rid
                )
                or bank_records[
                    rid
                ]
            ).to_dict()
            for rid in keep_ids
        ],
        "assessments": {
            rid: _assessment_to_dict(
                assessments[
                    rid
                ]
            )
            for rid in keep_ids
        },
        "history": history,
    }


def run(
    args: argparse.Namespace,
) -> Path:
    manifest_path = Path(
        args.manifest
    )
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    all_instances = (
        load_manifest(
            manifest_path
        )
    )
    instances = select_split(
        all_instances,
        "evolution",
        require_proven_optimum=True,
    )
    axes = capability_axes(
        instances
    )
    instance_ids = tuple(
        item.instance_id
        for item in instances
    )

    run_dir = Path(
        args.run_dir
    )
    checkpoint_path = (
        run_dir
        / "checkpoint.json"
    )
    paired_path = (
        run_dir
        / "paired_events.jsonl"
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
                "Incompatible V1.19 checkpoint"
            )
        if tuple(
            data.get(
                "axes",
                []
            )
        ) != axes:
            raise ValueError(
                "MTRPD capability axes changed since checkpoint"
            )
        if (
            data.get(
                "manifest_sha256"
            )
            != manifest_sha256
            or tuple(
                data.get(
                    "instance_ids",
                    []
                )
            )
            != instance_ids
        ):
            raise ValueError(
                "MTRPD manifest or evolution instance set changed since checkpoint"
            )

        records = {
            row[
                "record_id"
            ]: BankRecord.from_dict(
                row
            )
            for row in data[
                "records"
            ]
        }
        assessments = {
            rid: _assessment_from_dict(
                row
            )
            for rid, row in data[
                "assessments"
            ].items()
        }
        bootstrap_records = {
            rid: records[
                rid
            ]
            for rid in data[
                "bootstrap_ids"
            ]
        }
        bank_records = {
            rid: records[
                rid
            ]
            for rid in data[
                "bank_ids"
            ]
        }
        history = list(
            data.get(
                "history",
                [],
            )
        )
        rng = np.random.default_rng()
        rng.bit_generator.state = (
            data[
                "rng_state"
            ]
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
        records: dict[
            str,
            BankRecord,
        ] = {}
        assessments: dict[
            str,
            GeneAssessment,
        ] = {}

        for index in range(
            args.population
        ):
            gene = (
                RouteTailDirectGene.random(
                    rng,
                    hidden_dim=8,
                    scale=(
                        args.initial_scale
                    ),
                )
            )
            assessment = evaluate_gene(
                gene,
                instances,
                strict_reference=True,
            )
            record = _record_from_gene(
                gene,
                assessment,
                generation=-1,
                origin="random_initial",
            )
            records[
                record.record_id
            ] = record
            assessments[
                record.record_id
            ] = assessment
            if (
                args.progress_every
                > 0
                and (
                    index + 1
                )
                % args.progress_every
                == 0
            ):
                print(
                    "V119_INIT_EVAL "
                    f"{index + 1}/{args.population}",
                    flush=True,
                )

        (
            bootstrap_records,
            bootstrap_assessments,
        ) = _trim_bootstrap(
            records,
            assessments,
            size=(
                args.bootstrap_size
            ),
        )

        successful = {
            rid: record
            for rid, record
            in records.items()
            if assessments[
                rid
            ].success
        }
        rebuilt = rebuild_bank(
            successful,
            axes,
            epsilon=(
                args.pareto_epsilon
            ),
            max_size=(
                args.bank_max_size
            ),
        )
        bank_records = dict(
            rebuilt.records
        )
        assessments = {
            rid: assessments[
                rid
            ]
            for rid in (
                set(
                    bootstrap_records
                )
                | set(
                    bank_records
                )
            )
        }
        records = {
            rid: (
                bootstrap_records.get(
                    rid
                )
                or bank_records[
                    rid
                ]
            )
            for rid in assessments
        }
        bootstrap_records = {
            rid: records[
                rid
            ]
            for rid in (
                bootstrap_records
            )
        }
        history: list[
            dict[
                str,
                Any,
            ]
        ] = []
        start_generation = 0

    for generation in range(
        start_generation,
        args.generations,
    ):
        parent_ids = (
            list(
                bank_records
            )
            if bank_records
            else list(
                bootstrap_records
            )
        )
        probabilities = (
            _parent_probabilities(
                parent_ids,
                assessments,
            )
        )

        child_records: dict[
            str,
            BankRecord,
        ] = {}
        child_assessments: dict[
            str,
            GeneAssessment,
        ] = {}
        paired_events: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for index in range(
            args.mutation_children
        ):
            parent_id = str(
                rng.choice(
                    parent_ids,
                    p=probabilities,
                )
            )
            parent_record = (
                records[
                    parent_id
                ]
            )
            parent_gene = (
                _gene_from_record(
                    parent_record
                )
            )
            mutated = parent_gene.mutated(
                rng,
                sigma=(
                    args.mutation_sigma
                ),
                mutation_rate=(
                    args.mutation_rate
                ),
            )
            child_gene = (
                RouteTailDirectGene.from_v18(
                    mutated
                )
            )
            child_assessment = (
                evaluate_gene(
                    child_gene,
                    instances,
                    strict_reference=True,
                )
            )
            child_record = (
                _record_from_gene(
                    child_gene,
                    child_assessment,
                    generation=(
                        generation
                    ),
                    origin="mutation",
                    parents=(
                        parent_id,
                    ),
                )
            )
            child_id = (
                child_record.record_id
            )
            child_records[
                child_id
            ] = child_record
            child_assessments[
                child_id
            ] = child_assessment

            paired = paired_delta(
                assessments[
                    parent_id
                ],
                child_assessment,
            )
            event = {
                "generation": (
                    generation
                ),
                "child_index": index,
                "parent_id": (
                    parent_id
                ),
                "child_id": child_id,
                "parent_overall": (
                    assessments[
                        parent_id
                    ].overall_optimum_retention
                ),
                "child_overall": (
                    child_assessment.overall_optimum_retention
                ),
                "child_success": (
                    child_assessment.success
                ),
                **paired,
            }
            paired_events.append(
                event
            )
            _append_jsonl(
                paired_path,
                event,
            )

            if (
                args.progress_every
                > 0
                and (
                    index + 1
                )
                % args.progress_every
                == 0
            ):
                print(
                    "V119_GEN_EVAL "
                    f"generation={generation} "
                    f"{index + 1}/{args.mutation_children}",
                    flush=True,
                )

        merged_records = dict(
            records
        )
        merged_records.update(
            child_records
        )
        merged_assessments = dict(
            assessments
        )
        merged_assessments.update(
            child_assessments
        )

        (
            bootstrap_records,
            bootstrap_assessments,
        ) = _trim_bootstrap(
            merged_records,
            merged_assessments,
            size=(
                args.bootstrap_size
            ),
        )

        successful_records = {
            rid: record
            for rid, record
            in merged_records.items()
            if merged_assessments[
                rid
            ].success
        }
        rebuilt = rebuild_bank(
            successful_records,
            axes,
            epsilon=(
                args.pareto_epsilon
            ),
            max_size=(
                args.bank_max_size
            ),
        )
        bank_records = dict(
            rebuilt.records
        )

        keep_ids = (
            set(
                bootstrap_records
            )
            | set(
                bank_records
            )
        )
        records = {
            rid: merged_records[
                rid
            ]
            for rid in keep_ids
        }
        assessments = {
            rid: merged_assessments[
                rid
            ]
            for rid in keep_ids
        }
        bootstrap_records = {
            rid: records[
                rid
            ]
            for rid in (
                bootstrap_records
            )
        }

        best_bootstrap_id = max(
            bootstrap_records,
            key=lambda rid: (
                _bootstrap_key(
                    assessments[
                        rid
                    ]
                ),
                rid,
            ),
        )
        best_bootstrap = assessments[
            best_bootstrap_id
        ]

        global_id = global_best_id(
            bank_records
        )
        global_summary = (
            None
            if global_id
            is None
            else {
                "record_id": (
                    global_id
                ),
                "overall_optimum_retention": (
                    assessments[
                        global_id
                    ].overall_optimum_retention
                ),
                "worst_optimum_retention": (
                    assessments[
                        global_id
                    ].worst_optimum_retention
                ),
                "exact_optimum_matches": (
                    assessments[
                        global_id
                    ].exact_optimum_matches
                ),
                "scores": (
                    bank_records[
                        global_id
                    ].scores
                ),
            }
        )

        positive = sum(
            int(
                event[
                    "overall_delta"
                ]
                > 1e-12
            )
            for event in paired_events
        )
        negative = sum(
            int(
                event[
                    "overall_delta"
                ]
                < -1e-12
            )
            for event in paired_events
        )
        tied = (
            len(
                paired_events
            )
            - positive
            - negative
        )
        child_success_count = sum(
            int(
                assessment.success
            )
            for assessment
            in child_assessments.values()
        )

        row = {
            "generation": (
                generation
            ),
            "protocol": (
                "public_mtrpd_published_opt_paired_mutation"
            ),
            "instance_count": len(
                instances
            ),
            "axes": list(
                axes
            ),
            "best_worst_completion": (
                best_bootstrap.worst_completion
            ),
            "best_mean_completion": (
                best_bootstrap.mean_completion
            ),
            "best_success_instances": (
                best_bootstrap.success_instances
            ),
            "new_successful_children": (
                child_success_count
            ),
            "bank_size": len(
                bank_records
            ),
            "dominated_removed": len(
                rebuilt.dominated_ids
            ),
            "epsilon_removed": len(
                rebuilt.epsilon_removed_ids
            ),
            "crowding_removed": len(
                rebuilt.crowding_removed_ids
            ),
            "best_by_scale": (
                best_by_axis(
                    bank_records,
                    axes,
                )
            ),
            "global_best": (
                global_summary
            ),
            "paired_child_overall_wtl": {
                "win": positive,
                "tie": tied,
                "loss": negative,
            },
            "paired_mean_overall_delta": float(
                np.mean(
                    [
                        event[
                            "overall_delta"
                        ]
                        for event in (
                            paired_events
                        )
                    ]
                )
            ),
            "paired_instance_wins": int(
                sum(
                    event[
                        "wins"
                    ]
                    for event in (
                        paired_events
                    )
                )
            ),
            "paired_instance_ties": int(
                sum(
                    event[
                        "ties"
                    ]
                    for event in (
                        paired_events
                    )
                )
            ),
            "paired_instance_losses": int(
                sum(
                    event[
                        "losses"
                    ]
                    for event in (
                        paired_events
                    )
                )
            ),
        }
        history.append(
            row
        )
        print(
            "V119_TRAIN "
            + json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

        _atomic_write(
            checkpoint_path,
            _checkpoint_payload(
                generation=(
                    generation
                ),
                manifest=str(
                    args.manifest
                ),
                manifest_sha256=(
                    manifest_sha256
                ),
                instance_ids=(
                    instance_ids
                ),
                axes=axes,
                rng=rng,
                bootstrap_records=(
                    bootstrap_records
                ),
                bank_records=(
                    bank_records
                ),
                assessments=(
                    assessments
                ),
                history=history,
            ),
        )

    print(
        f"V119_RUN_DIR={run_dir}",
        flush=True,
    )
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
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
        default=128,
    )
    p.add_argument(
        "--bootstrap-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--mutation-children",
        type=int,
        default=128,
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
        "--bank-max-size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--pareto-epsilon",
        type=float,
        default=0.0025,
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=16,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=119,
    )
    return p


def main() -> None:
    run(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
