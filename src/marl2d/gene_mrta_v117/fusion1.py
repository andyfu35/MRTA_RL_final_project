from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.stage_a_oracle_bank import load_stage_a_oracle_bank
from marl2d.gene_mrta_v117.stage_a_train import (
    EPS,
    CHECKPOINT_VERSION,
    Record,
    _archives,
    _best,
    _clean_offspring_family,
    _evaluate,
    _gene_id,
    _record_from_dict,
    _rebuild_active,
    _screen_indices,
    _subset,
)


FUSION1_VERSION = "v117_fusion1_checkpoint_v1"


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


def _record_quality(
    record: Record,
    best: dict[str, float],
) -> float:
    caps = (
        record.capabilities
        if record.capabilities
        else BASE_AXES
    )
    return float(
        min(
            record.scores[axis]
            / max(best[axis], EPS)
            for axis in caps
        )
    )


def _parent_pool(
    records: dict[str, Record],
    *,
    archive_size: int,
) -> list[str]:
    archives = _archives(
        records,
        archive_size,
    )
    ids: set[str] = set()
    for axis_ids in archives.values():
        ids.update(axis_ids)

    for rid, record in records.items():
        if len(
            record.inherited_capabilities
        ) >= 2:
            ids.add(rid)

    return sorted(ids)


def _choose_pairs(
    records: dict[str, Record],
    *,
    pair_count: int,
    archive_size: int,
    rng: np.random.Generator,
) -> list[tuple[str, str, tuple[str, ...]]]:
    ids = _parent_pool(
        records,
        archive_size=archive_size,
    )
    if len(ids) < 2:
        raise RuntimeError(
            "Fusion-1 requires at least two parent candidates"
        )

    best = _best(records)
    ranked: list[
        tuple[
            tuple[float, ...],
            str,
            str,
            tuple[str, ...],
        ]
    ] = []

    for i, a in enumerate(ids):
        caps_a = set(
            records[a].capabilities
        )
        if not caps_a:
            continue

        for b in ids[i + 1 :]:
            caps_b = set(
                records[b].capabilities
            )
            if not caps_b:
                continue

            union = tuple(
                sorted(
                    caps_a | caps_b
                )
            )
            if len(union) < 2:
                continue

            adds_a = len(
                caps_b - caps_a
            )
            adds_b = len(
                caps_a - caps_b
            )
            complementary = (
                adds_a + adds_b
            )
            if complementary <= 0:
                continue

            has_time_priority = int(
                (
                    "global_time_optimality"
                    in union
                )
                and (
                    "global_priority_optimality"
                    in union
                )
            )
            q = min(
                _record_quality(
                    records[a],
                    best,
                ),
                _record_quality(
                    records[b],
                    best,
                ),
            )

            # Primary goal: maximize capability union.
            # Secondary goal: explicitly expose the Time/Priority conflict.
            # Tiny random jitter prevents the same deterministic pair set
            # from dominating every Fusion-1 round.
            rank = (
                float(len(union)),
                float(has_time_priority),
                float(complementary),
                float(q),
                float(rng.random()),
            )
            ranked.append(
                (
                    rank,
                    a,
                    b,
                    union,
                )
            )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    selected = ranked[
        : min(
            pair_count,
            len(ranked),
        )
    ]
    return [
        (
            a,
            b,
            union,
        )
        for _rank, a, b, union
        in selected
    ]


def _inheritance_ratio(
    child_score: dict[str, float],
    union: tuple[str, ...],
    parent_a: dict[str, float],
    parent_b: dict[str, float],
) -> float:
    if not union:
        return 0.0

    ratios = []
    for axis in union:
        baseline = max(
            parent_a[axis],
            parent_b[axis],
            EPS,
        )
        ratios.append(
            child_score[axis]
            / baseline
        )
    return float(
        min(ratios)
    )


def _best_fusion_record(
    records: dict[str, Record],
) -> Record | None:
    best = _best(records)
    candidates = [
        record
        for record in records.values()
        if len(
            record.inherited_capabilities
        ) >= 2
    ]
    if not candidates:
        return None

    return max(
        candidates,
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
    )


def run(args: argparse.Namespace) -> Path:
    source_path = Path(
        args.source_checkpoint
    )
    source = json.loads(
        source_path.read_text(
            encoding="utf-8",
        )
    )
    if (
        source.get("version")
        != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "Fusion-1 source must be a final V1.17 Stage-A checkpoint"
        )
    if source.get("axes") != list(
        BASE_AXES
    ):
        raise ValueError(
            "Fusion-1 source capability axes do not match"
        )

    oracle_bank = Path(
        source["oracle_bank"]
    )
    (
        config,
        worlds,
        completion_optima,
        time_optima,
        path_optima,
        priority_optima,
        deadline_optima,
    ) = load_stage_a_oracle_bank(
        oracle_bank
    )

    output_dir = Path(
        args.output_dir
    )
    checkpoint_path = (
        output_dir
        / "checkpoint.json"
    )

    if checkpoint_path.exists():
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
                "Existing Fusion-1 checkpoint has incompatible version"
            )
        rng = np.random.default_rng()
        rng.bit_generator.state = (
            checkpoint["rng_state"]
        )
        records = {
            row["record_id"]:
                _record_from_dict(row)
            for row in checkpoint[
                "records"
            ]
        }
        history = list(
            checkpoint.get(
                "history",
                [],
            )
        )
        start_round = int(
            checkpoint["round"]
        ) + 1
        print(
            "V117_FUSION1_RESUME "
            + json.dumps(
                {
                    "next_round": (
                        start_round
                    ),
                    "target_rounds": (
                        args.rounds
                    ),
                    "active_genes": len(
                        records
                    ),
                }
            ),
            flush=True,
        )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        rng = np.random.default_rng(
            args.seed
        )
        records = {
            row["record_id"]:
                _record_from_dict(row)
            for row in source[
                "records"
            ]
        }
        history = []
        start_round = 0

    zero_anchor = RouteTailDirectGene(
        np.zeros(
            RouteTailDirectGene.parameter_count(
                8
            ),
            dtype=np.float64,
        ),
        hidden_dim=8,
    )

    for round_index in range(
        start_round,
        args.rounds,
    ):
        best_before = _best(
            records
        )
        pairs = _choose_pairs(
            records,
            pair_count=args.mating_pairs,
            archive_size=args.archive_size,
            rng=rng,
        )

        screen_idx = _screen_indices(
            10000 + round_index,
            len(worlds),
            min(
                args.screen_worlds,
                len(worlds),
            ),
        )
        screen_worlds = _subset(
            worlds,
            screen_idx,
        )
        screen_completion = _subset(
            completion_optima,
            screen_idx,
        )
        screen_time = _subset(
            time_optima,
            screen_idx,
        )
        screen_path = _subset(
            path_optima,
            screen_idx,
        )
        screen_priority = _subset(
            priority_optima,
            screen_idx,
        )
        screen_deadline = _subset(
            deadline_optima,
            screen_idx,
        )

        child_genes: list[
            RouteTailDirectGene
        ] = []
        child_pairs: list[
            tuple[str, str]
        ] = []
        child_unions: list[
            tuple[str, ...]
        ] = []
        child_operators: list[
            str
        ] = []

        for a, b, union in pairs:
            family = _clean_offspring_family(
                records[a].gene,
                records[b].gene,
                zero_anchor,
                rng,
                children=(
                    args.children_per_pair
                ),
                mutation_sigma=(
                    args.post_mating_sigma
                ),
                mutation_rate=(
                    args.post_mating_rate
                ),
            )
            for gene, operator in family:
                child_genes.append(gene)
                child_pairs.append(
                    (a, b)
                )
                child_unions.append(
                    union
                )
                child_operators.append(
                    operator
                )

        child_screen = _evaluate(
            child_genes,
            screen_worlds,
            config,
            screen_completion,
            screen_time,
            screen_path,
            screen_priority,
            screen_deadline,
        )

        parent_ids = sorted(
            {
                rid
                for pair in child_pairs
                for rid in pair
            }
        )
        parent_screen_values = _evaluate(
            [
                records[rid].gene
                for rid in parent_ids
            ],
            screen_worlds,
            config,
            screen_completion,
            screen_time,
            screen_path,
            screen_priority,
            screen_deadline,
        )
        parent_screen = {
            rid: score
            for rid, score in zip(
                parent_ids,
                parent_screen_values,
                strict=True,
            )
        }

        ranked = sorted(
            range(
                len(child_genes)
            ),
            key=lambda idx: (
                len(
                    child_unions[idx]
                ),
                _inheritance_ratio(
                    child_screen[idx],
                    child_unions[idx],
                    parent_screen[
                        child_pairs[idx][0]
                    ],
                    parent_screen[
                        child_pairs[idx][1]
                    ],
                ),
            ),
            reverse=True,
        )
        full_idx = ranked[
            : min(
                args.full_candidates,
                len(ranked),
            )
        ]

        full_genes = [
            child_genes[idx]
            for idx in full_idx
        ]
        full_scores = _evaluate(
            full_genes,
            worlds,
            config,
            completion_optima,
            time_optima,
            path_optima,
            priority_optima,
            deadline_optima,
        )

        accepted = 0
        accepted_sizes: list[int] = []
        for local_idx, (
            gene,
            score,
        ) in enumerate(
            zip(
                full_genes,
                full_scores,
                strict=True,
            )
        ):
            source_idx = full_idx[
                local_idx
            ]
            a, b = child_pairs[
                source_idx
            ]
            union = child_unions[
                source_idx
            ]

            passed = bool(
                union
            )
            for axis in union:
                parent_floor = max(
                    records[a].scores[
                        axis
                    ],
                    records[b].scores[
                        axis
                    ],
                )
                if (
                    score[axis]
                    / max(
                        parent_floor,
                        EPS,
                    )
                    < args.threshold
                    or score[axis]
                    / max(
                        best_before[axis],
                        EPS,
                    )
                    < args.threshold
                ):
                    passed = False
                    break

            rid = _gene_id(
                gene
            )
            if rid in records:
                continue

            records[rid] = Record(
                record_id=rid,
                gene=gene,
                scores=score,
                capabilities=(
                    union
                    if passed
                    else ()
                ),
                inherited_capabilities=(
                    union
                    if passed
                    else ()
                ),
                origin="mating",
                generation=(
                    1000
                    + round_index
                ),
                parents=(a, b),
                operator=(
                    child_operators[
                        source_idx
                    ]
                ),
            )
            if passed:
                accepted += 1
                accepted_sizes.append(
                    len(union)
                )

        records, archives = _rebuild_active(
            records,
            archive_size=(
                args.archive_size
            ),
            hybrid_limit=(
                args.hybrid_limit
            ),
            certification_threshold=(
                args.threshold
            ),
        )

        best_after = _best(
            records
        )
        fusion = _best_fusion_record(
            records
        )
        max_inherited = max(
            (
                len(
                    record.inherited_capabilities
                )
                for record
                in records.values()
            ),
            default=0,
        )
        row = {
            "round": round_index,
            "active_genes": len(
                records
            ),
            "mating_pairs": len(
                pairs
            ),
            "children_screened": len(
                child_genes
            ),
            "children_full_evaluated": (
                len(full_genes)
            ),
            "accepted_full_union_children": (
                accepted
            ),
            "accepted_union_sizes": (
                accepted_sizes
            ),
            "max_inherited_capabilities": (
                max_inherited
            ),
            "best": best_after,
            "specialists": {
                axis: (
                    archives[axis][0]
                    if archives[axis]
                    else None
                )
                for axis in BASE_AXES
            },
            "best_fusion_gene": (
                None
                if fusion is None
                else {
                    "record_id": (
                        fusion.record_id
                    ),
                    "operator": (
                        fusion.operator
                    ),
                    "parents": list(
                        fusion.parents
                    ),
                    "inherited_capabilities": list(
                        fusion.inherited_capabilities
                    ),
                    "archive_capabilities": list(
                        fusion.archive_capabilities
                    ),
                    "scores": (
                        fusion.scores
                    ),
                }
            ),
        }
        history.append(row)
        print(
            "V117_FUSION1 "
            + json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

        payload = {
            "version": FUSION1_VERSION,
            "round": round_index,
            "source_checkpoint": str(
                source_path
            ),
            "oracle_bank": str(
                oracle_bank
            ),
            "axes": list(
                BASE_AXES
            ),
            "rng_state": (
                rng.bit_generator.state
            ),
            "records": [
                record.to_dict()
                for record
                in records.values()
            ],
            "history": history,
        }
        _atomic_write(
            checkpoint_path,
            payload,
        )

        if (
            max_inherited
            >= len(BASE_AXES)
        ):
            print(
                "V117_FUSION1_ALL_CAPABILITIES=true",
                flush=True,
            )
            if args.stop_on_all:
                break

    print(
        f"V117_FUSION1_RUN_DIR={output_dir}",
        flush=True,
    )
    return output_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-checkpoint",
        required=True,
    )
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=20,
    )
    p.add_argument(
        "--mating-pairs",
        type=int,
        default=64,
    )
    p.add_argument(
        "--children-per-pair",
        type=int,
        default=4,
    )
    p.add_argument(
        "--screen-worlds",
        type=int,
        default=8,
    )
    p.add_argument(
        "--full-candidates",
        type=int,
        default=32,
    )
    p.add_argument(
        "--archive-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--hybrid-limit",
        type=int,
        default=192,
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--post-mating-sigma",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--post-mating-rate",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=11701,
    )
    p.add_argument(
        "--stop-on-all",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
