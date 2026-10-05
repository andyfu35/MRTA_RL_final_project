from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.fusion1 import FUSION1_VERSION
from marl2d.gene_mrta_v117.stage_a_oracle_bank import (
    BANK_VERSION,
    load_stage_a_oracle_bank,
)
from marl2d.gene_mrta_v117.stage_a_train import (
    EPS,
    Record,
    _archives,
    _best,
    _record_from_dict,
    evaluate_gene_base_axes,
)


AUDIT_VERSION = "v117_unseen_audit_v1"
HARD_BANK_VERSION = "v117_hard_world_bank_v1"


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


def _candidate_ids(
    records: dict[str, Record],
    *,
    archive_size: int,
    hybrid_count: int,
) -> tuple[
    list[str],
    dict[str, str],
]:
    archives = _archives(
        records,
        archive_size,
    )
    selected: list[str] = []
    roles: dict[str, str] = {}

    for axis in BASE_AXES:
        ids = archives[axis]
        if not ids:
            continue
        rid = ids[0]
        if rid not in roles:
            selected.append(rid)
            roles[rid] = (
                f"specialist:{axis}"
            )
        else:
            roles[rid] += (
                f"|specialist:{axis}"
            )

    best = _best(records)
    hybrids = [
        record
        for record in records.values()
        if len(
            record.inherited_capabilities
        ) >= 2
    ]
    hybrids.sort(
        key=lambda record: (
            len(
                record.inherited_capabilities
            ),
            min(
                record.scores[axis]
                / max(best[axis], EPS)
                for axis in (
                    record.inherited_capabilities
                )
            ),
        ),
        reverse=True,
    )

    for record in hybrids[
        :hybrid_count
    ]:
        rid = record.record_id
        if rid not in roles:
            selected.append(rid)
            roles[rid] = "fusion"
        else:
            roles[rid] += "|fusion"

    return selected, roles


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


def run(args: argparse.Namespace) -> Path:
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
            "Unseen audit requires a Fusion-1 checkpoint"
        )

    records = {
        row["record_id"]:
            _record_from_dict(row)
        for row in checkpoint[
            "records"
        ]
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
        Path(args.unseen_bank)
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

    candidate_ids, roles = _candidate_ids(
        records,
        archive_size=(
            args.archive_size
        ),
        hybrid_count=(
            args.hybrid_count
        ),
    )
    if not candidate_ids:
        raise RuntimeError(
            "No candidates selected for unseen audit"
        )

    candidate_world_scores: dict[
        str,
        list[dict[str, float]],
    ] = {
        rid: []
        for rid in candidate_ids
    }

    world_rows: list[
        dict[str, Any]
    ] = []

    fusion_ids = [
        rid
        for rid in candidate_ids
        if "fusion" in roles[rid]
    ]
    if not fusion_ids:
        fusion_ids = list(
            candidate_ids
        )

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
            scores_by_gene[rid] = (
                scores
            )
            candidate_world_scores[
                rid
            ].append(scores)

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

        fusion_quality: list[
            tuple[
                float,
                str,
                str,
            ]
        ] = []
        for rid in fusion_ids:
            scores = scores_by_gene[
                rid
            ]
            worst_axis = min(
                BASE_AXES,
                key=lambda axis: (
                    scores[axis]
                ),
            )
            min_score = float(
                scores[worst_axis]
            )
            fusion_quality.append(
                (
                    min_score,
                    rid,
                    worst_axis,
                )
            )

        fusion_quality.sort(
            reverse=True
        )
        (
            best_fusion_min,
            best_fusion_id,
            best_fusion_worst_axis,
        ) = fusion_quality[0]

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

        # Main hard-world criterion:
        # even the best available hybrid has a weak worst capability.
        # Secondary criterion:
        # even the whole cohort frontier cannot cover one capability well.
        hardness = float(
            1.0
            - best_fusion_min
        )

        row = {
            "world_index": index,
            "seed": seed,
            "hardness": hardness,
            "best_fusion_min": float(
                best_fusion_min
            ),
            "best_fusion_id": (
                best_fusion_id
            ),
            "best_fusion_worst_axis": (
                best_fusion_worst_axis
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
        world_rows.append(row)
        print(
            "V117_UNSEEN_WORLD "
            + json.dumps(
                {
                    "index": index,
                    "seed": seed,
                    "best_fusion_min": (
                        best_fusion_min
                    ),
                    "worst_axis": (
                        best_fusion_worst_axis
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
        min_mean = float(
            min(
                axis_mean.values()
            )
        )
        candidate_summary[rid] = {
            "role": roles[rid],
            "inherited_capabilities": list(
                records[
                    rid
                ].inherited_capabilities
            ),
            "archive_capabilities": list(
                records[
                    rid
                ].archive_capabilities
            ),
            "mean": axis_mean,
            "p10": axis_p10,
            "min_mean": min_mean,
        }

    ranked_worlds = sorted(
        world_rows,
        key=lambda row: (
            row["best_fusion_min"],
            row["frontier_min"],
        ),
    )
    hard_rows = ranked_worlds[
        : min(
            args.hard_count,
            len(ranked_worlds),
        )
    ]
    hard_seeds = [
        int(row["seed"])
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
        "selection": {
            "criterion": (
                "lowest_best_hybrid_min_then_frontier_min"
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
        "version": AUDIT_VERSION,
        "fusion_checkpoint": str(
            checkpoint_path
        ),
        "unseen_bank": str(
            args.unseen_bank
        ),
        "candidate_ids": (
            candidate_ids
        ),
        "candidate_summary": (
            candidate_summary
        ),
        "worlds": world_rows,
        "hard_seeds": hard_seeds,
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
        "--archive-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--hybrid-count",
        type=int,
        default=12,
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
