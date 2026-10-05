from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v110.recombination import recombine
from marl2d.gene_mrta_v113.direct_gene import RouteTailDirectGene
from marl2d.gene_mrta_v117.capabilities import BASE_AXES, evaluate_gene_base_axes
from marl2d.gene_mrta_v117.stage_a_oracle_bank import load_stage_a_oracle_bank


EPS = 1e-12


@dataclass
class Record:
    record_id: str
    gene: RouteTailDirectGene
    scores: dict[str, float]
    capabilities: tuple[str, ...]
    origin: str
    archive_capabilities: tuple[str, ...] = ()
    inherited_capabilities: tuple[str, ...] = ()
    generation: int
    parents: tuple[str, ...] = ()
    operator: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "gene": self.gene.to_dict(),
            "scores": self.scores,
            "capabilities": list(self.capabilities),
            "archive_capabilities": list(
                self.archive_capabilities
            ),
            "inherited_capabilities": list(
                self.inherited_capabilities
            ),
            "origin": self.origin,
            "generation": self.generation,
            "parents": list(self.parents),
            "operator": self.operator,
        }


def _gene_id(gene: RouteTailDirectGene) -> str:
    return hashlib.sha256(
        np.asarray(
            gene.vector_data,
            dtype=np.float64,
        ).tobytes()
    ).hexdigest()[:20]


def _evaluate(
    genes: list[RouteTailDirectGene],
    worlds,
    config,
    completion_optima: list[float],
    time_optima: list[float],
    path_efficiency_optima: list[float],
    priority_optima: list[float],
    deadline_optima: list[float],
) -> list[dict[str, float]]:
    return [
        evaluate_gene_base_axes(
            gene,
            worlds,
            config,
            completion_optima=(
                completion_optima
            ),
            time_optima=time_optima,
            path_efficiency_optima=(
                path_efficiency_optima
            ),
            priority_optima=priority_optima,
            deadline_optima=(
                deadline_optima
            ),
        )
        for gene in genes
    ]


def _archives(
    records: dict[str, Record],
    size: int,
) -> dict[str, list[str]]:
    return {
        axis: sorted(
            records,
            key=lambda rid: records[rid].scores[axis],
            reverse=True,
        )[:size]
        for axis in BASE_AXES
    }


def _best(
    records: dict[str, Record],
) -> dict[str, float]:
    return {
        axis: max(
            (
                record.scores[axis]
                for record in records.values()
            ),
            default=EPS,
        )
        for axis in BASE_AXES
    }


def _quality(
    record: Record,
    best: dict[str, float],
) -> float:
    caps = record.capabilities or BASE_AXES
    return float(
        min(
            record.scores[axis]
            / max(best[axis], EPS)
            for axis in caps
        )
    )


def _sample_parent(
    ids: list[str],
    records: dict[str, Record],
    best: dict[str, float],
    rng: np.random.Generator,
    *,
    power: float,
    uniform_fraction: float,
) -> str:
    q = np.asarray(
        [
            max(
                _quality(records[rid], best),
                1e-9,
            )
            for rid in ids
        ],
        dtype=np.float64,
    )
    weighted = q ** power
    weighted /= np.sum(weighted)
    uniform = np.full(
        len(ids),
        1.0 / len(ids),
        dtype=np.float64,
    )
    p = (
        (1.0 - uniform_fraction) * weighted
        + uniform_fraction * uniform
    )
    return ids[
        int(
            rng.choice(
                len(ids),
                p=p,
            )
        )
    ]


def _rebuild_active(
    records: dict[str, Record],
    *,
    archive_size: int,
    hybrid_limit: int,
    certification_threshold: float,
) -> tuple[
    dict[str, Record],
    dict[str, list[str]],
]:
    archives = _archives(
        records,
        archive_size,
    )
    archive_caps: dict[str, set[str]] = {}
    for axis, ids in archives.items():
        for rid in ids:
            archive_caps.setdefault(
                rid,
                set(),
            ).add(axis)

    best = _best(records)
    hybrid_ids = [
        rid
        for rid, record in records.items()
        if (
            record.origin == "mating"
            and len(
                record.inherited_capabilities
            ) >= 2
        )
    ]
    hybrid_ids.sort(
        key=lambda rid: (
            len(
                records[rid].inherited_capabilities
            ),
            min(
                records[rid].scores[axis]
                / max(best[axis], EPS)
                for axis in (
                    records[
                        rid
                    ].inherited_capabilities
                )
            ),
        ),
        reverse=True,
    )

    keep = set(archive_caps)
    keep.update(
        hybrid_ids[:hybrid_limit]
    )

    new_records: dict[str, Record] = {}
    for rid in keep:
        record = records[rid]
        current_archive = set(
            archive_caps.get(
                rid,
                set(),
            )
        )
        retained_inherited = {
            axis
            for axis in (
                record.inherited_capabilities
            )
            if (
                record.scores[axis]
                / max(best[axis], EPS)
                >= certification_threshold
            )
        }
        record.archive_capabilities = tuple(
            sorted(current_archive)
        )
        record.inherited_capabilities = tuple(
            sorted(retained_inherited)
        )
        record.capabilities = tuple(
            sorted(
                current_archive
                | retained_inherited
            )
        )
        new_records[rid] = record

    return (
        new_records,
        _archives(
            new_records,
            archive_size,
        ),
    )


def _screen_indices(
    generation: int,
    total: int,
    screen_count: int,
) -> np.ndarray:
    if screen_count >= total:
        return np.arange(
            total,
            dtype=np.int64,
        )
    rng = np.random.default_rng(
        117_900_000 + generation
    )
    return np.sort(
        rng.choice(
            total,
            size=screen_count,
            replace=False,
        )
    )


def _subset(
    values: list,
    indices: np.ndarray,
) -> list:
    return [
        values[int(i)]
        for i in indices
    ]


def _top_per_axis(
    scores: list[dict[str, float]],
    per_axis: int,
) -> list[int]:
    chosen: list[int] = []
    seen: set[int] = set()
    for axis in BASE_AXES:
        order = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx][axis],
            reverse=True,
        )
        for idx in order[:per_axis]:
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
    return chosen


CLEAN_MATING_OPERATORS = (
    "parameter_blend",
    "block_pick",
    "block_blend",
)


def _clean_offspring_family(
    parent_a: RouteTailDirectGene,
    parent_b: RouteTailDirectGene,
    zero_anchor: RouteTailDirectGene,
    rng: np.random.Generator,
    *,
    children: int,
    mutation_sigma: float,
    mutation_rate: float,
) -> list[tuple[RouteTailDirectGene, str]]:
    if children <= 0:
        raise ValueError(
            "children must be positive"
        )

    order = rng.permutation(
        len(CLEAN_MATING_OPERATORS)
    )
    result: list[
        tuple[RouteTailDirectGene, str]
    ] = []
    for child_idx in range(children):
        operator = CLEAN_MATING_OPERATORS[
            int(
                order[
                    child_idx
                    % len(order)
                ]
            )
        ]
        merged = recombine(
            parent_a,
            parent_b,
            zero_anchor,
            rng,
            operator=operator,
        )
        gene = RouteTailDirectGene.from_v18(
            merged.gene
        )
        if mutation_sigma > 0.0:
            gene = RouteTailDirectGene.from_v18(
                gene.mutated(
                    rng,
                    sigma=mutation_sigma,
                    mutation_rate=mutation_rate,
                )
            )
        result.append(
            (
                gene,
                operator,
            )
        )
    return result


def _mating_rank(
    child_scores: list[dict[str, float]],
    required_caps: list[tuple[str, ...]],
    parent_pairs: list[tuple[str, str]],
    parent_screen_scores: dict[str, dict[str, float]],
) -> list[int]:
    ranking: list[tuple[float, int]] = []
    for idx, score in enumerate(child_scores):
        a, b = parent_pairs[idx]
        ratios = []
        for axis in required_caps[idx]:
            baseline = max(
                parent_screen_scores[a][axis],
                parent_screen_scores[b][axis],
            )
            ratios.append(
                score[axis]
                / max(baseline, EPS)
            )
        ranking.append(
            (
                min(ratios) if ratios else 0.0,
                idx,
            )
        )
    ranking.sort(reverse=True)
    return [
        idx
        for _value, idx in ranking
    ]


def run(args: argparse.Namespace) -> Path:
    (
        config,
        worlds,
        completion_optima,
        time_optima,
        path_efficiency_optima,
        priority_optima,
        deadline_optima,
    ) = load_stage_a_oracle_bank(
        Path(args.oracle_bank)
    )

    rng = np.random.default_rng(
        args.seed
    )
    run_dir = (
        Path(args.output_dir)
        / (
            "gene_mrta_v117_stage_a_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + f"_seed{args.seed}"
        )
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    initial_genes = [
        RouteTailDirectGene.random(
            rng,
            hidden_dim=8,
            scale=args.initial_scale,
        )
        for _ in range(
            args.population
        )
    ]
    initial_scores = _evaluate(
        initial_genes,
        worlds,
        config,
        completion_optima,
        time_optima,
        path_efficiency_optima,
        priority_optima,
        deadline_optima,
    )

    records: dict[str, Record] = {}
    for gene, scores in zip(
        initial_genes,
        initial_scores,
        strict=True,
    ):
        rid = _gene_id(gene)
        records[rid] = Record(
            record_id=rid,
            gene=gene,
            scores=scores,
            capabilities=(),
            origin="random_initial",
            generation=-1,
        )

    records, archives = _rebuild_active(
        records,
        archive_size=args.archive_size,
        hybrid_limit=args.hybrid_limit,
        certification_threshold=(
            args.certification_threshold
        ),
    )

    zero_anchor = RouteTailDirectGene(
        np.zeros(
            RouteTailDirectGene.parameter_count(
                8
            ),
            dtype=np.float64,
        ),
        hidden_dim=8,
    )

    history: list[dict[str, Any]] = []

    for generation in range(
        args.generations
    ):
        best = _best(records)
        active_ids = list(records)

        screen_idx = _screen_indices(
            generation,
            len(worlds),
            args.screen_worlds,
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
            path_efficiency_optima,
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

        normal_genes: list[
            RouteTailDirectGene
        ] = []
        for _ in range(
            args.normal_children
        ):
            pid = _sample_parent(
                active_ids,
                records,
                best,
                rng,
                power=2.0,
                uniform_fraction=(
                    args.uniform_fraction
                ),
            )
            child = records[
                pid
            ].gene.mutated(
                rng,
                sigma=args.mutation_sigma,
                mutation_rate=(
                    args.mutation_rate
                ),
            )
            normal_genes.append(
                RouteTailDirectGene.from_v18(
                    child
                )
            )

        normal_screen = _evaluate(
            normal_genes,
            screen_worlds,
            config,
            screen_completion,
            screen_time,
            screen_path,
            screen_priority,
            screen_deadline,
        )
        normal_full_idx = _top_per_axis(
            normal_screen,
            args.normal_full_per_axis,
        )
        normal_full_genes = [
            normal_genes[i]
            for i in normal_full_idx
        ]
        normal_full_scores = _evaluate(
            normal_full_genes,
            worlds,
            config,
            completion_optima,
            time_optima,
            path_efficiency_optima,
            priority_optima,
            deadline_optima,
        )

        mating_genes: list[
            RouteTailDirectGene
        ] = []
        mating_operators: list[str] = []
        mating_pairs: list[
            tuple[str, str]
        ] = []
        required_caps: list[
            tuple[str, ...]
        ] = []

        for _ in range(
            args.mating_pairs
        ):
            a = _sample_parent(
                active_ids,
                records,
                best,
                rng,
                power=args.mating_power,
                uniform_fraction=(
                    args.uniform_fraction
                ),
            )
            complementary = [
                rid
                for rid in active_ids
                if (
                    rid != a
                    and (
                        set(
                            records[rid].capabilities
                        )
                        - set(
                            records[a].capabilities
                        )
                    )
                )
            ]
            if not complementary:
                complementary = [
                    rid
                    for rid in active_ids
                    if rid != a
                ]
            if not complementary:
                continue

            b = _sample_parent(
                complementary,
                records,
                best,
                rng,
                power=args.mating_power,
                uniform_fraction=(
                    args.uniform_fraction
                ),
            )
            union_caps = tuple(
                sorted(
                    set(
                        records[a].capabilities
                    )
                    | set(
                        records[b].capabilities
                    )
                )
            )
            family = _clean_offspring_family(
                records[a].gene,
                records[b].gene,
                zero_anchor,
                rng,
                children=args.children_per_pair,
                mutation_sigma=(
                    args.post_mating_sigma
                ),
                mutation_rate=(
                    args.post_mating_rate
                ),
            )
            for child_gene, operator in family:
                mating_genes.append(
                    child_gene
                )
                mating_operators.append(
                    operator
                )
                mating_pairs.append(
                    (a, b)
                )
                required_caps.append(
                    union_caps
                )

        mating_screen = _evaluate(
            mating_genes,
            screen_worlds,
            config,
            screen_completion,
            screen_time,
            screen_path,
            screen_priority,
            screen_deadline,
        ) if mating_genes else []

        parent_ids = sorted(
            set(
                rid
                for pair in mating_pairs
                for rid in pair
            )
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
        ) if parent_ids else []
        parent_screen = {
            rid: score
            for rid, score in zip(
                parent_ids,
                parent_screen_values,
                strict=True,
            )
        }

        mating_order = _mating_rank(
            mating_screen,
            required_caps,
            mating_pairs,
            parent_screen,
        ) if mating_genes else []
        mating_full_idx = mating_order[
            : args.mating_full_limit
        ]
        mating_full_genes = [
            mating_genes[i]
            for i in mating_full_idx
        ]
        mating_full_scores = _evaluate(
            mating_full_genes,
            worlds,
            config,
            completion_optima,
            time_optima,
            path_efficiency_optima,
            priority_optima,
            deadline_optima,
        ) if mating_full_genes else []

        pre_best = _best(records)

        for gene, score in zip(
            normal_full_genes,
            normal_full_scores,
            strict=True,
        ):
            rid = _gene_id(gene)
            if rid not in records:
                records[rid] = Record(
                    record_id=rid,
                    gene=gene,
                    scores=score,
                    capabilities=(),
                    origin="normal_mutation",
                    generation=generation,
                )

        for local_idx, (
            gene,
            score,
        ) in enumerate(
            zip(
                mating_full_genes,
                mating_full_scores,
                strict=True,
            )
        ):
            original_idx = mating_full_idx[
                local_idx
            ]
            a, b = mating_pairs[
                original_idx
            ]
            req = required_caps[
                original_idx
            ]
            passed = bool(req)
            for axis in req:
                parent_floor = max(
                    records[a].scores[axis],
                    records[b].scores[axis],
                )
                if (
                    score[axis]
                    / max(parent_floor, EPS)
                    < args.certification_threshold
                    or score[axis]
                    / max(pre_best[axis], EPS)
                    < args.certification_threshold
                ):
                    passed = False
                    break

            rid = _gene_id(gene)
            if rid not in records:
                records[rid] = Record(
                    record_id=rid,
                    gene=gene,
                    scores=score,
                    capabilities=(
                        req if passed else ()
                    ),
                    inherited_capabilities=(
                        req if passed else ()
                    ),
                    origin="mating",
                    generation=generation,
                    parents=(a, b),
                    operator=(
                        mating_operators[
                            original_idx
                        ]
                    ),
                )

        records, archives = _rebuild_active(
            records,
            archive_size=args.archive_size,
            hybrid_limit=args.hybrid_limit,
            certification_threshold=(
                args.certification_threshold
            ),
        )

        best = _best(records)
        max_caps = max(
            (
                len(record.capabilities)
                for record in records.values()
            ),
            default=0,
        )
        max_inherited_caps = max(
            (
                len(
                    record.inherited_capabilities
                )
                for record in records.values()
            ),
            default=0,
        )
        most_capable = max(
            records.values(),
            key=lambda record: (
                len(record.capabilities),
                _quality(record, best),
            ),
        )
        fusion_records = [
            record
            for record in records.values()
            if (
                record.origin == "mating"
                and len(
                    record.inherited_capabilities
                ) >= 2
            )
        ]
        best_fusion = (
            max(
                fusion_records,
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
            if fusion_records
            else None
        )
        time_id = (
            archives[
                "global_time_optimality"
            ][0]
            if archives[
                "global_time_optimality"
            ]
            else None
        )
        priority_id = (
            archives[
                "global_priority_optimality"
            ][0]
            if archives[
                "global_priority_optimality"
            ]
            else None
        )

        row = {
            "generation": generation,
            "active_genes": len(records),
            "max_capabilities": max_caps,
            "max_inherited_capabilities": (
                max_inherited_caps
            ),
            "time_priority_same_specialist": (
                time_id == priority_id
            ),
            "most_capable_gene": {
                "record_id": (
                    most_capable.record_id
                ),
                "origin": (
                    most_capable.origin
                ),
                "operator": (
                    most_capable.operator
                ),
                "capabilities": list(
                    most_capable.capabilities
                ),
                "archive_capabilities": list(
                    most_capable.archive_capabilities
                ),
                "inherited_capabilities": list(
                    most_capable.inherited_capabilities
                ),
                "scores": (
                    most_capable.scores
                ),
                "parents": list(
                    most_capable.parents
                ),
            },
            "best_fusion_gene": (
                {
                    "record_id": (
                        best_fusion.record_id
                    ),
                    "operator": (
                        best_fusion.operator
                    ),
                    "parents": list(
                        best_fusion.parents
                    ),
                    "archive_capabilities": list(
                        best_fusion.archive_capabilities
                    ),
                    "inherited_capabilities": list(
                        best_fusion.inherited_capabilities
                    ),
                    "scores": (
                        best_fusion.scores
                    ),
                }
                if best_fusion is not None
                else None
            ),
            "best": best,
            "specialists": {
                axis: (
                    archives[axis][0]
                    if archives[axis]
                    else None
                )
                for axis in BASE_AXES
            },
        }
        history.append(row)
        print(
            "V117_STAGE_A "
            + json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

        checkpoint = {
            "version": (
                "v117_stage_a_checkpoint_v4_capability_provenance"
            ),
            "generation": generation,
            "axes": list(BASE_AXES),
            "oracle_bank": str(
                args.oracle_bank
            ),
            "records": [
                record.to_dict()
                for record in records.values()
            ],
            "history": history,
        }
        (
            run_dir
            / "checkpoint.json"
        ).write_text(
            json.dumps(
                checkpoint,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(
        f"V117_STAGE_A_RUN_DIR={run_dir}",
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
        "--archive-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--hybrid-limit",
        type=int,
        default=128,
    )
    p.add_argument(
        "--normal-children",
        type=int,
        default=128,
    )
    p.add_argument(
        "--normal-full-per-axis",
        type=int,
        default=4,
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
        "--mating-full-limit",
        type=int,
        default=16,
    )
    p.add_argument(
        "--screen-worlds",
        type=int,
        default=8,
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
        "--post-mating-sigma",
        type=float,
        default=0.03,
    )
    p.add_argument(
        "--post-mating-rate",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--mating-power",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--uniform-fraction",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--certification-threshold",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=117,
    )
    p.add_argument(
        "--output-dir",
        default="runs/gene_mrta_v117_stage_a",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
