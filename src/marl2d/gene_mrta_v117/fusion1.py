from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v110.recombination import hidden_block_indices
from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v117.capabilities import BASE_AXES
from marl2d.gene_mrta_v117.pareto_bank import (
    analysis_best_by_axis,
    capability_distance,
    crowding_trim_ids,
    maximin_gene_id,
    pareto_front_ids,
    rebuild_pareto_bank,
)
from marl2d.gene_mrta_v117.stage_a_oracle_bank import load_stage_a_oracle_bank
from marl2d.gene_mrta_v117.stage_a_train import (
    CHECKPOINT_VERSION,
    Record,
    _evaluate,
    _gene_id,
    _record_from_dict,
    _screen_indices,
    _subset,
)


FUSION1_VERSION = "v117_fusion1_checkpoint_v3_pareto"


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


def _progressive_offspring_family(
    parent_a: RouteTailDirectGene,
    parent_b: RouteTailDirectGene,
    rng: np.random.Generator,
    *,
    children: int,
) -> list[tuple[RouteTailDirectGene, str]]:
    if children <= 0:
        raise ValueError(
            "children must be positive"
        )
    if (
        parent_a.hidden_dim
        != parent_b.hidden_dim
    ):
        raise ValueError(
            "Parent hidden dimensions must match"
        )

    blocks, globals_ = hidden_block_indices(
        parent_a.hidden_dim
    )
    result: list[
        tuple[
            RouteTailDirectGene,
            str,
        ]
    ] = []

    operators = (
        "sparse_block_graft",
        "sparse_block_blend",
        "near_parent_blend",
    )

    for child_index in range(
        children
    ):
        recipient_is_a = (
            child_index % 2 == 0
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
        direction = (
            "A<-B"
            if recipient_is_a
            else "B<-A"
        )

        operator = operators[
            child_index
            % len(operators)
        ]

        recipient_vec = (
            recipient.vector_data
        )
        donor_vec = (
            donor.vector_data
        )
        child = recipient_vec.copy()

        if operator in {
            "sparse_block_graft",
            "sparse_block_blend",
        }:
            graft_count = (
                1
                if rng.random() < 0.75
                else 2
            )
            chosen = rng.choice(
                len(blocks),
                size=min(
                    graft_count,
                    len(blocks),
                ),
                replace=False,
            )
            for block_index in chosen:
                ids = blocks[
                    int(block_index)
                ]
                if (
                    operator
                    == "sparse_block_graft"
                ):
                    child[ids] = (
                        donor_vec[ids]
                    )
                else:
                    donor_weight = float(
                        rng.uniform(
                            0.15,
                            0.40,
                        )
                    )
                    child[ids] = (
                        (
                            1.0
                            - donor_weight
                        )
                        * recipient_vec[ids]
                        + donor_weight
                        * donor_vec[ids]
                    )

            if rng.random() < 0.20:
                if (
                    operator
                    == "sparse_block_graft"
                ):
                    child[globals_] = (
                        donor_vec[
                            globals_
                        ]
                    )
                else:
                    donor_weight = float(
                        rng.uniform(
                            0.10,
                            0.30,
                        )
                    )
                    child[globals_] = (
                        (
                            1.0
                            - donor_weight
                        )
                        * recipient_vec[
                            globals_
                        ]
                        + donor_weight
                        * donor_vec[
                            globals_
                        ]
                    )

        elif (
            operator
            == "near_parent_blend"
        ):
            donor_weight = float(
                rng.uniform(
                    0.05,
                    0.20,
                )
            )
            child = (
                (
                    1.0
                    - donor_weight
                )
                * recipient_vec
                + donor_weight
                * donor_vec
            )
        else:
            raise AssertionError(
                operator
            )

        gene = RouteTailDirectGene(
            np.asarray(
                child,
                dtype=np.float64,
            ),
            hidden_dim=(
                recipient.hidden_dim
            ),
        )
        result.append(
            (
                gene,
                (
                    f"{operator}:"
                    f"{direction}"
                ),
            )
        )

    return result


def _choose_pairs(
    records: dict[str, Record],
    *,
    pair_count: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    ids = list(records)
    if len(ids) < 2:
        raise RuntimeError(
            "Pareto mating requires at least two Genes"
        )

    ranked: list[
        tuple[
            float,
            float,
            str,
            str,
        ]
    ] = []

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            distance = (
                capability_distance(
                    records[a].scores,
                    records[b].scores,
                )
            )
            ranked.append(
                (
                    distance,
                    float(
                        rng.random()
                    ),
                    a,
                    b,
                )
            )

    ranked.sort(
        reverse=True
    )
    return [
        (a, b)
        for _distance, _jitter, a, b
        in ranked[
            : min(
                pair_count,
                len(ranked),
            )
        ]
    ]


def _screen_pareto_indices(
    genes: list[RouteTailDirectGene],
    scores: list[dict[str, float]],
    *,
    limit: int,
) -> list[int]:
    temp: dict[str, Record] = {}
    index_by_id: dict[str, int] = {}

    for index, (
        gene,
        score,
    ) in enumerate(
        zip(
            genes,
            scores,
            strict=True,
        )
    ):
        rid = (
            f"screen_{index:06d}"
        )
        temp[rid] = Record(
            record_id=rid,
            gene=gene,
            scores=score,
            capabilities=(),
            origin="screen",
            generation=0,
        )
        index_by_id[rid] = (
            index
        )

    front, _dominated = (
        pareto_front_ids(
            temp
        )
    )
    kept, _removed = (
        crowding_trim_ids(
            temp,
            front,
            max_size=max(
                1,
                limit,
            ),
        )
    )
    return [
        index_by_id[rid]
        for rid in kept
    ]


def _bank_best_scores(
    records: dict[str, Record],
) -> dict[str, float]:
    if not records:
        return {
            axis: 0.0
            for axis in BASE_AXES
        }
    return {
        axis: max(
            float(
                record.scores[
                    axis
                ]
            )
            for record
            in records.values()
        )
        for axis in BASE_AXES
    }


def _maximin_summary(
    records: dict[str, Record],
) -> dict[str, Any] | None:
    rid = maximin_gene_id(
        records
    )
    if rid is None:
        return None
    record = records[rid]
    return {
        "record_id": rid,
        "scores": record.scores,
        "min_capability": float(
            min(
                record.scores[
                    axis
                ]
                for axis in BASE_AXES
            )
        ),
        "origin": record.origin,
        "parents": list(
            record.parents
        ),
        "operator": (
            record.operator
        ),
    }


def run(
    args: argparse.Namespace,
) -> Path:
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
            "Pareto Fusion-1 source must be the frozen Stage-A checkpoint"
        )
    if (
        source.get("axes")
        != list(BASE_AXES)
    ):
        raise ValueError(
            "Source capability axes do not match"
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
                "Existing Pareto Fusion-1 checkpoint has incompatible version"
            )
        if (
            float(
                checkpoint[
                    "pareto_epsilon"
                ]
            )
            != float(
                args.pareto_epsilon
            )
            or int(
                checkpoint[
                    "pareto_max_size"
                ]
            )
            != int(
                args.pareto_max_size
            )
        ):
            raise ValueError(
                "Pareto Bank settings do not match existing checkpoint"
            )

        rng = np.random.default_rng()
        rng.bit_generator.state = (
            checkpoint[
                "rng_state"
            ]
        )
        records = {
            row["record_id"]:
                _record_from_dict(
                    row
                )
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
        start_round = (
            int(
                checkpoint[
                    "round"
                ]
            )
            + 1
        )
        print(
            "V117_PARETO_FUSION_RESUME "
            + json.dumps(
                {
                    "next_round": (
                        start_round
                    ),
                    "target_rounds": (
                        args.rounds
                    ),
                    "pareto_size": len(
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
        source_records = {
            row["record_id"]:
                _record_from_dict(
                    row
                )
            for row in source[
                "records"
            ]
        }
        initial = rebuild_pareto_bank(
            source_records,
            max_size=(
                args.pareto_max_size
            ),
            epsilon=(
                args.pareto_epsilon
            ),
        )
        records = dict(
            initial.records
        )
        history = []
        start_round = 0

        print(
            "V117_PARETO_INIT "
            + json.dumps(
                {
                    "source_genes": len(
                        source_records
                    ),
                    "pareto_genes": len(
                        records
                    ),
                    "dominated_removed": len(
                        initial.dominated_ids
                    ),
                    "epsilon_removed": len(
                        initial.epsilon_duplicate_ids
                    ),
                    "crowding_removed": len(
                        initial.crowding_removed_ids
                    ),
                }
            ),
            flush=True,
        )

    for round_index in range(
        start_round,
        args.rounds,
    ):
        parent_pairs = _choose_pairs(
            records,
            pair_count=(
                args.mating_pairs
            ),
            rng=rng,
        )

        child_genes: list[
            RouteTailDirectGene
        ] = []
        child_parents: list[
            tuple[str, str]
        ] = []
        child_operators: list[
            str
        ] = []

        for a, b in parent_pairs:
            family = (
                _progressive_offspring_family(
                    records[a].gene,
                    records[b].gene,
                    rng,
                    children=(
                        args.children_per_pair
                    ),
                )
            )
            for gene, operator in family:
                child_genes.append(
                    gene
                )
                child_parents.append(
                    (a, b)
                )
                child_operators.append(
                    operator
                )

        screen_idx = _screen_indices(
            20000 + round_index,
            len(worlds),
            min(
                args.screen_worlds,
                len(worlds),
            ),
        )
        child_screen = _evaluate(
            child_genes,
            _subset(
                worlds,
                screen_idx,
            ),
            config,
            _subset(
                completion_optima,
                screen_idx,
            ),
            _subset(
                time_optima,
                screen_idx,
            ),
            _subset(
                path_optima,
                screen_idx,
            ),
            _subset(
                priority_optima,
                screen_idx,
            ),
            _subset(
                deadline_optima,
                screen_idx,
            ),
        )

        full_idx = (
            _screen_pareto_indices(
                child_genes,
                child_screen,
                limit=(
                    args.full_candidates
                ),
            )
        )
        full_genes = [
            child_genes[index]
            for index in full_idx
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

        before_ids = set(
            records
        )
        candidate_ids: list[
            str
        ] = []

        merged = dict(
            records
        )
        for local_index, (
            gene,
            scores,
        ) in enumerate(
            zip(
                full_genes,
                full_scores,
                strict=True,
            )
        ):
            source_index = (
                full_idx[
                    local_index
                ]
            )
            rid = _gene_id(
                gene
            )
            if rid in merged:
                continue

            candidate_ids.append(
                rid
            )
            merged[rid] = Record(
                record_id=rid,
                gene=gene,
                scores=scores,
                capabilities=(),
                archive_capabilities=(),
                inherited_capabilities=(),
                origin="pareto_mating",
                generation=(
                    2000
                    + round_index
                ),
                parents=(
                    child_parents[
                        source_index
                    ]
                ),
                operator=(
                    child_operators[
                        source_index
                    ]
                ),
            )

        rebuilt = rebuild_pareto_bank(
            merged,
            max_size=(
                args.pareto_max_size
            ),
            epsilon=(
                args.pareto_epsilon
            ),
        )
        records = dict(
            rebuilt.records
        )

        after_ids = set(
            records
        )
        inserted_ids = sorted(
            set(candidate_ids)
            & after_ids
        )
        old_removed_ids = sorted(
            before_ids
            - after_ids
        )

        best_view = (
            analysis_best_by_axis(
                records
            )
        )
        row = {
            "round": round_index,
            "pareto_size": len(
                records
            ),
            "mating_pairs": len(
                parent_pairs
            ),
            "children_screened": len(
                child_genes
            ),
            "screen_pareto_candidates": len(
                full_genes
            ),
            "new_children_full_evaluated": len(
                candidate_ids
            ),
            "pareto_inserted_children": len(
                inserted_ids
            ),
            "pareto_inserted_ids": (
                inserted_ids
            ),
            "old_pareto_removed": len(
                old_removed_ids
            ),
            "dominated_removed_total": len(
                rebuilt.dominated_ids
            ),
            "epsilon_removed_total": len(
                rebuilt.epsilon_duplicate_ids
            ),
            "crowding_removed_total": len(
                rebuilt.crowding_removed_ids
            ),
            "best_scores": (
                _bank_best_scores(
                    records
                )
            ),
            "analysis_best_by_axis": (
                best_view
            ),
            "maximin_gene": (
                _maximin_summary(
                    records
                )
            ),
        }
        history.append(
            row
        )

        print(
            "V117_PARETO_FUSION "
            + json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

        payload = {
            "version": (
                FUSION1_VERSION
            ),
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
            "pareto_epsilon": float(
                args.pareto_epsilon
            ),
            "pareto_max_size": int(
                args.pareto_max_size
            ),
            "rng_state": (
                rng.bit_generator.state
            ),
            "records": [
                record.to_dict()
                for record in (
                    records.values()
                )
            ],
            "history": history,
        }
        _atomic_write(
            checkpoint_path,
            payload,
        )

    print(
        f"V117_PARETO_FUSION_RUN_DIR={output_dir}",
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
        default=12,
    )
    p.add_argument(
        "--full-candidates",
        type=int,
        default=64,
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
        default=11702,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
