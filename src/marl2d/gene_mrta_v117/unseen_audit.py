from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.fusion1 import FUSION1_VERSION
from marl2d.gene_mrta_v117.pareto_bank import (
    crowding_trim_ids,
    pareto_front_ids,
)
from marl2d.gene_mrta_v117.stage_a_oracle_bank import (
    BANK_VERSION,
    load_stage_a_oracle_bank,
)
from marl2d.gene_mrta_v117.stage_a_train import (
    Record,
    _record_from_dict,
    evaluate_gene_base_axes,
)


AUDIT_VERSION = "v117_unseen_pareto_audit_v2"
HARD_BANK_VERSION = "v117_hard_world_bank_v2_pareto"


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
    temp.replace(path)


def _one_world_scores(
    record: Record,
    world,
    config,
    *,
    completion_star: float,
    time_star: float,
    path_star: float,
    priority_star: float,
    deadline_star: float,
) -> dict[str, float]:
    return evaluate_gene_base_axes(
        record.gene,
        [world],
        config,
        completion_optima=[
            completion_star
        ],
        time_optima=[
            time_star
        ],
        path_efficiency_optima=[
            path_star
        ],
        priority_optima=[
            priority_star
        ],
        deadline_optima=[
            deadline_star
        ],
    )


def _representative_ids(
    records: dict[str, Record],
    *,
    limit: int,
) -> list[str]:
    ids = list(
        records
    )
    if len(ids) <= limit:
        return ids

    kept, _removed = (
        crowding_trim_ids(
            records,
            ids,
            max_size=limit,
        )
    )
    return list(
        kept
    )


def _world_pareto_ids(
    records: dict[str, Record],
    scores_by_gene: dict[
        str,
        dict[str, float],
    ],
) -> list[str]:
    temp: dict[str, Record] = {}
    for rid, scores in (
        scores_by_gene.items()
    ):
        source = records[rid]
        temp[rid] = Record(
            record_id=rid,
            gene=source.gene,
            scores=scores,
            capabilities=(),
            origin=source.origin,
            generation=(
                source.generation
            ),
            parents=(
                source.parents
            ),
            operator=(
                source.operator
            ),
        )
    front, _dominated = (
        pareto_front_ids(
            temp
        )
    )
    return list(
        front
    )


def run(
    args: argparse.Namespace,
) -> Path:
    checkpoint_path = Path(
        args.fusion_checkpoint
    )
    checkpoint = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8",
        )
    )
    if (
        checkpoint.get("version")
        != FUSION1_VERSION
    ):
        raise ValueError(
            "Unseen audit requires a Pareto Fusion-1 checkpoint"
        )

    all_records = {
        row["record_id"]:
            _record_from_dict(
                row
            )
        for row in checkpoint[
            "records"
        ]
    }
    candidate_ids = (
        _representative_ids(
            all_records,
            limit=(
                args.pareto_candidates
            ),
        )
    )
    records = {
        rid: all_records[rid]
        for rid in candidate_ids
    }

    (
        config,
        worlds,
        completion_optima,
        time_optima,
        path_optima,
        priority_optima,
        deadline_optima,
    ) = load_stage_a_oracle_bank(
        Path(
            args.unseen_bank
        )
    )
    bank_raw = json.loads(
        Path(
            args.unseen_bank
        ).read_text(
            encoding="utf-8",
        )
    )
    if (
        bank_raw.get("version")
        != BANK_VERSION
    ):
        raise ValueError(
            "Unexpected unseen oracle bank version"
        )

    candidate_world_scores: dict[
        str,
        list[
            dict[str, float]
        ],
    ] = {
        rid: []
        for rid in candidate_ids
    }
    world_rows: list[
        dict[str, Any]
    ] = []

    for index, world in enumerate(
        worlds
    ):
        seed = int(
            bank_raw["rows"][
                index
            ]["seed"]
        )
        scores_by_gene: dict[
            str,
            dict[str, float],
        ] = {}

        for rid in candidate_ids:
            scores = _one_world_scores(
                records[rid],
                world,
                config,
                completion_star=(
                    completion_optima[
                        index
                    ]
                ),
                time_star=(
                    time_optima[
                        index
                    ]
                ),
                path_star=(
                    path_optima[
                        index
                    ]
                ),
                priority_star=(
                    priority_optima[
                        index
                    ]
                ),
                deadline_star=(
                    deadline_optima[
                        index
                    ]
                ),
            )
            scores_by_gene[
                rid
            ] = scores
            candidate_world_scores[
                rid
            ].append(
                scores
            )

        world_pareto = (
            _world_pareto_ids(
                records,
                scores_by_gene,
            )
        )

        frontier = {
            axis: max(
                scores_by_gene[
                    rid
                ][axis]
                for rid
                in candidate_ids
            )
            for axis in BASE_AXES
        }

        maximin_id = max(
            candidate_ids,
            key=lambda rid: (
                min(
                    scores_by_gene[
                        rid
                    ][axis]
                    for axis
                    in BASE_AXES
                )
            ),
        )
        maximin_scores = (
            scores_by_gene[
                maximin_id
            ]
        )
        worst_axis = min(
            BASE_AXES,
            key=lambda axis: (
                maximin_scores[
                    axis
                ]
            ),
        )
        maximin_value = float(
            maximin_scores[
                worst_axis
            ]
        )

        frontier_worst_axis = min(
            BASE_AXES,
            key=lambda axis: (
                frontier[axis]
            ),
        )
        frontier_min = float(
            frontier[
                frontier_worst_axis
            ]
        )

        row = {
            "world_index": index,
            "seed": seed,
            "hardness": float(
                1.0
                - maximin_value
            ),
            "bank_maximin_gene": (
                maximin_id
            ),
            "bank_maximin_value": (
                maximin_value
            ),
            "bank_maximin_worst_axis": (
                worst_axis
            ),
            "bank_maximin_scores": (
                maximin_scores
            ),
            "world_pareto_size": len(
                world_pareto
            ),
            "world_pareto_ids": (
                world_pareto
            ),
            "frontier_min": (
                frontier_min
            ),
            "frontier_worst_axis": (
                frontier_worst_axis
            ),
            "frontier": frontier,
            "candidate_scores": (
                scores_by_gene
            ),
        }
        world_rows.append(
            row
        )

        print(
            "V117_UNSEEN_PARETO_WORLD "
            + json.dumps(
                {
                    "index": index,
                    "seed": seed,
                    "bank_maximin_value": (
                        maximin_value
                    ),
                    "worst_axis": (
                        worst_axis
                    ),
                    "world_pareto_size": len(
                        world_pareto
                    ),
                    "frontier_min": (
                        frontier_min
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    candidate_summary: dict[
        str,
        dict[str, Any],
    ] = {}
    for rid in candidate_ids:
        rows = (
            candidate_world_scores[
                rid
            ]
        )
        axis_mean = {
            axis: float(
                np.mean(
                    [
                        row[axis]
                        for row in rows
                    ]
                )
            )
            for axis in BASE_AXES
        }
        axis_p10 = {
            axis: float(
                np.quantile(
                    [
                        row[axis]
                        for row in rows
                    ],
                    0.10,
                )
            )
            for axis in BASE_AXES
        }
        candidate_summary[rid] = {
            "mean": axis_mean,
            "p10": axis_p10,
            "min_mean": float(
                min(
                    axis_mean.values()
                )
            ),
            "origin": (
                records[rid].origin
            ),
            "parents": list(
                records[rid].parents
            ),
            "operator": (
                records[
                    rid
                ].operator
            ),
        }

    ranked_worlds = sorted(
        world_rows,
        key=lambda row: (
            row[
                "bank_maximin_value"
            ],
            row[
                "frontier_min"
            ],
        ),
    )
    hard_rows = ranked_worlds[
        : min(
            args.hard_count,
            len(ranked_worlds),
        )
    ]
    hard_seeds = [
        int(
            row["seed"]
        )
        for row in hard_rows
    ]

    bank_rows_by_seed = {
        int(row["seed"]): row
        for row in bank_raw[
            "rows"
        ]
    }
    hard_bank = {
        "version": (
            HARD_BANK_VERSION
        ),
        "source_unseen_bank": str(
            args.unseen_bank
        ),
        "source_pareto_checkpoint": str(
            checkpoint_path
        ),
        "selection": {
            "criterion": (
                "lowest_bank_maximin_then_frontier_min"
            ),
            "hard_count": len(
                hard_rows
            ),
        },
        "robots": bank_raw[
            "robots"
        ],
        "tasks": bank_raw[
            "tasks"
        ],
        "config": bank_raw[
            "config"
        ],
        "rows": [
            bank_rows_by_seed[
                seed
            ]
            for seed in hard_seeds
        ],
        "hardness_rows": [
            {
                key: value
                for key, value
                in row.items()
                if key
                != "candidate_scores"
            }
            for row in hard_rows
        ],
    }

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_payload = {
        "version": (
            AUDIT_VERSION
        ),
        "fusion_checkpoint": str(
            checkpoint_path
        ),
        "unseen_bank": str(
            args.unseen_bank
        ),
        "source_pareto_size": len(
            all_records
        ),
        "candidate_count": len(
            candidate_ids
        ),
        "candidate_ids": (
            candidate_ids
        ),
        "candidate_summary": (
            candidate_summary
        ),
        "worlds": world_rows,
        "hard_seeds": (
            hard_seeds
        ),
    }

    _atomic_write(
        output_dir
        / "unseen_audit.json",
        audit_payload,
    )
    _atomic_write(
        output_dir
        / "hard_world_bank.json",
        hard_bank,
    )

    print(
        "V117_UNSEEN_HARD_SEEDS="
        + ",".join(
            str(seed)
            for seed in hard_seeds
        ),
        flush=True,
    )
    print(
        f"V117_UNSEEN_AUDIT_DIR={output_dir}",
        flush=True,
    )
    return output_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fusion-checkpoint",
        required=True,
    )
    p.add_argument(
        "--unseen-bank",
        required=True,
    )
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument(
        "--pareto-candidates",
        type=int,
        default=256,
    )
    p.add_argument(
        "--hard-count",
        type=int,
        default=16,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
