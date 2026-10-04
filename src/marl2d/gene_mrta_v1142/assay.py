from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig
from marl2d.gene_mrta_v18.global_time_test import _load_v18
from marl2d.gene_mrta_v110.scenario_bank import load_scenario_bank
from marl2d.gene_mrta_v113.evolve import (
    AXES,
    GeneRecord,
    _active_parent_ids,
    _best_by_axis,
    _capability_ceiling_retention,
    _gene_id,
    _inheritance_retention,
    _passes_inheritance_gate,
    _quality_weight,
    _sample_record_id,
    _complementary_partner_ids,
    _score_dict_at,
    _evaluate_scores,
)
from marl2d.gene_mrta_v1141.recombination_gene import (
    RecombinationGene,
    RecombinationRecord,
    recombine_with_gene,
    sample_recombination_gene_id,
)


EPS = 1e-12


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item)
            for item in value
        ]
    return value


def _load_policy_records(
    checkpoint: Path,
) -> dict[str, GeneRecord]:
    data = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )
    records = {
        str(item["record_id"]): (
            GeneRecord.from_dict(item)
        )
        for item in data.get(
            "records",
            []
        )
    }
    if not records:
        raise RuntimeError(
            "No V1.13 Policy records found"
        )
    return records


def _load_adaptive_rule_bank(
    checkpoint: Path,
    *,
    min_generations: int,
) -> dict[str, RecombinationRecord]:
    data = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )
    if data.get("version") != "v1141":
        raise RuntimeError(
            "V1.14.2 requires a V1.14.1 checkpoint"
        )
    if data.get(
        "recombination_mode"
    ) != "adaptive":
        raise RuntimeError(
            "V1.14.2 requires the adaptive V1.14.1 arm"
        )
    completed_generations = int(
        data.get(
            "next_generation",
            0,
        )
    )
    if completed_generations < min_generations:
        raise RuntimeError(
            "Adaptive V1.14.1 checkpoint is too early: "
            f"{completed_generations} < {min_generations}"
        )

    result: dict[
        str,
        RecombinationRecord
    ] = {}
    for item in data.get(
        "recombination_records",
        []
    ):
        record = (
            RecombinationRecord.from_dict(
                item
            )
        )
        result[
            record.gene_id
        ] = record

    if not result:
        raise RuntimeError(
            "No adaptive Recombination records found"
        )
    return result


def _center_rule() -> RecombinationGene:
    return RecombinationGene(
        coefficients=np.zeros(
            7,
            dtype=np.float64,
        ),
        gates=np.zeros(
            7,
            dtype=bool,
        ),
        mutation_sigma=0.35,
        generation=-1,
        parents=(),
    )


def _pair_key(
    a: str,
    b: str,
) -> tuple[str, str]:
    if a == b:
        raise ValueError(
            "A mating pair must contain two distinct parents"
        )
    return (
        (a, b)
        if a < b
        else (b, a)
    )


def build_pair_manifest(
    *,
    count: int,
    active_ids: list[str],
    records: dict[str, GeneRecord],
    best: dict[str, float],
    rng: np.random.Generator,
    parent_q_power: float,
    uniform_fraction: float,
) -> list[tuple[str, str]]:
    if count <= 0:
        raise ValueError(
            "count must be positive"
        )

    manifest: list[
        tuple[str, str]
    ] = []
    seen: set[
        tuple[str, str]
    ] = set()

    attempts = 0
    max_attempts = max(
        10_000,
        count * 500,
    )

    while (
        len(manifest) < count
        and attempts < max_attempts
    ):
        attempts += 1
        parent_a = _sample_record_id(
            active_ids,
            records,
            best,
            rng,
            power=parent_q_power,
            uniform_fraction=(
                uniform_fraction
            ),
        )
        partner_ids = (
            _complementary_partner_ids(
                parent_a,
                active_ids,
                records,
            )
        )
        if not partner_ids:
            continue
        parent_b = _sample_record_id(
            partner_ids,
            records,
            best,
            rng,
            power=parent_q_power,
            uniform_fraction=(
                uniform_fraction
            ),
        )

        key = _pair_key(
            parent_a,
            parent_b,
        )
        if key in seen:
            continue

        seen.add(key)
        manifest.append(key)

    if len(manifest) != count:
        raise RuntimeError(
            f"Could only sample {len(manifest)} unique pairs; "
            f"requested {count}"
        )

    return manifest


def _make_child(
    *,
    rule: RecombinationGene,
    parent_a: GeneRecord,
    parent_b: GeneRecord,
    anchor,
    best: dict[str, float],
) -> tuple[Any, dict[str, Any]]:
    return recombine_with_gene(
        rule,
        parent_a.gene,
        parent_b.gene,
        anchor,
        quality_a=_quality_weight(
            parent_a,
            best,
        ),
        quality_b=_quality_weight(
            parent_b,
            best,
        ),
        capability_count_a=len(
            parent_a.capabilities
        ),
        capability_count_b=len(
            parent_b.capabilities
        ),
    )


def _certified_capabilities(
    score: dict[str, float],
    best: dict[str, float],
    *,
    threshold: float,
) -> tuple[str, ...]:
    return tuple(
        axis
        for axis in AXES
        if (
            float(score[axis])
            / max(
                float(best[axis]),
                EPS,
            )
            >= threshold
        )
    )


def _minimum(
    values: dict[str, float],
) -> float:
    return (
        float(
            min(
                values.values()
            )
        )
        if values
        else 0.0
    )


def _axis_pair_stats(
    adaptive: list[float],
    center: list[float],
    *,
    tolerance: float,
) -> dict[str, Any]:
    a = np.asarray(
        adaptive,
        dtype=np.float64,
    )
    c = np.asarray(
        center,
        dtype=np.float64,
    )
    if a.shape != c.shape:
        raise ValueError(
            "Paired arrays must have the same shape"
        )
    d = a - c
    wins = int(
        np.sum(
            d > tolerance
        )
    )
    losses = int(
        np.sum(
            d < -tolerance
        )
    )
    ties = int(
        len(d)
        - wins
        - losses
    )
    return {
        "n": int(
            len(d)
        ),
        "adaptive_mean": float(
            np.mean(a)
        ) if len(a) else None,
        "center_mean": float(
            np.mean(c)
        ) if len(c) else None,
        "mean_delta": float(
            np.mean(d)
        ) if len(d) else None,
        "median_delta": float(
            np.median(d)
        ) if len(d) else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _dominance(
    adaptive_score: dict[str, float],
    center_score: dict[str, float],
    *,
    tolerance: float,
) -> str:
    delta = np.asarray(
        [
            adaptive_score[axis]
            - center_score[axis]
            for axis in AXES
        ],
        dtype=np.float64,
    )

    adaptive_ge = bool(
        np.all(
            delta >= -tolerance
        )
    )
    center_ge = bool(
        np.all(
            delta <= tolerance
        )
    )
    adaptive_gt = bool(
        np.any(
            delta > tolerance
        )
    )
    center_gt = bool(
        np.any(
            delta < -tolerance
        )
    )

    if adaptive_ge and adaptive_gt:
        return "adaptive"
    if center_ge and center_gt:
        return "center"
    if (
        not adaptive_gt
        and not center_gt
    ):
        return "tie"
    return "neither"


def run_assay(
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
            "V1.14.2 requires the frozen 100-world 95M bank"
        )

    policy_records = (
        _load_policy_records(
            Path(
                args.bootstrap_v113_checkpoint
            )
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

    adaptive_rules = (
        _load_adaptive_rule_bank(
            Path(
                args.adaptive_v1141_checkpoint
            ),
            min_generations=(
                args.adaptive_min_generations
            ),
        )
    )
    center_rule = _center_rule()
    anchor = _load_v18(
        Path(
            args.anchor_v18_run
        )
    )

    pair_rng = np.random.default_rng(
        args.seed
    )
    rule_rng = np.random.default_rng(
        args.seed + 114_200_003
    )

    manifest = build_pair_manifest(
        count=args.pairs,
        active_ids=active_ids,
        records=policy_records,
        best=best,
        rng=pair_rng,
        parent_q_power=(
            args.parent_q_power
        ),
        uniform_fraction=(
            args.parent_uniform_fraction
        ),
    )

    adaptive_genes = []
    center_genes = []
    pair_meta: list[
        dict[str, Any]
    ] = []
    rule_usage: Counter[str] = (
        Counter()
    )

    for pair_index, (
        parent_a_id,
        parent_b_id,
    ) in enumerate(manifest):
        parent_a = policy_records[
            parent_a_id
        ]
        parent_b = policy_records[
            parent_b_id
        ]

        (
            adaptive_rule_id,
            adaptive_rule_axis,
        ) = sample_recombination_gene_id(
            adaptive_rules,
            rule_rng,
            selection_power=(
                args.rule_selection_power
            ),
            uniform_fraction=(
                args.rule_uniform_fraction
            ),
            evidence_quantile=(
                args.rule_evidence_quantile
            ),
        )
        adaptive_rule = (
            adaptive_rules[
                adaptive_rule_id
            ].gene
        )
        rule_usage[
            adaptive_rule_id
        ] += 1

        adaptive_gene, adaptive_meta = (
            _make_child(
                rule=adaptive_rule,
                parent_a=parent_a,
                parent_b=parent_b,
                anchor=anchor,
                best=best,
            )
        )
        center_gene, center_meta = (
            _make_child(
                rule=center_rule,
                parent_a=parent_a,
                parent_b=parent_b,
                anchor=anchor,
                best=best,
            )
        )

        adaptive_genes.append(
            adaptive_gene
        )
        center_genes.append(
            center_gene
        )

        pair_meta.append(
            {
                "pair_index": pair_index,
                "parent_a": parent_a_id,
                "parent_b": parent_b_id,
                "parent_a_capabilities": list(
                    parent_a.capabilities
                ),
                "parent_b_capabilities": list(
                    parent_b.capabilities
                ),
                "required_capabilities": sorted(
                    set(
                        parent_a.capabilities
                    )
                    | set(
                        parent_b.capabilities
                    )
                ),
                "adaptive_rule_id": (
                    adaptive_rule_id
                ),
                "adaptive_rule_axis": (
                    adaptive_rule_axis
                ),
                "adaptive_rule_active_terms": (
                    adaptive_rule.active_term_count
                ),
                "adaptive_rule": (
                    adaptive_rule.to_dict()
                ),
                "adaptive_metadata": (
                    adaptive_meta
                ),
                "center_rule_id": (
                    center_rule.gene_id
                ),
                "center_metadata": (
                    center_meta
                ),
            }
        )

    adaptive_scores = (
        _evaluate_scores(
            adaptive_genes,
            worlds,
            stars,
            config,
        )
    )
    center_scores = (
        _evaluate_scores(
            center_genes,
            worlds,
            stars,
            config,
        )
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        Path(
            args.output_dir
        )
        / (
            "gene_mrta_v1142_matched_pair_"
            f"{stamp}_seed{args.seed}"
        )
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_payload = {
        "version": "v1142",
        "seed": int(
            args.seed
        ),
        "pair_count": int(
            len(manifest)
        ),
        "bootstrap_v113_checkpoint": (
            args.bootstrap_v113_checkpoint
        ),
        "adaptive_v1141_checkpoint": (
            args.adaptive_v1141_checkpoint
        ),
        "pairs": [
            {
                "pair_index": idx,
                "parent_a": pair[0],
                "parent_b": pair[1],
            }
            for idx, pair
            in enumerate(
                manifest
            )
        ],
    }
    (
        run_dir
        / "pair_manifest.json"
    ).write_text(
        json.dumps(
            manifest_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows: list[
        dict[str, Any]
    ] = []
    dominance_counter: Counter[
        str
    ] = Counter()
    adaptive_accepted = 0
    center_accepted = 0
    adaptive_fourcap = 0
    center_fourcap = 0
    exact_identical = 0

    for idx, meta in enumerate(
        pair_meta
    ):
        adaptive_score = (
            _score_dict_at(
                adaptive_scores,
                idx,
            )
        )
        center_score = (
            _score_dict_at(
                center_scores,
                idx,
            )
        )

        parent_a = policy_records[
            meta["parent_a"]
        ]
        parent_b = policy_records[
            meta["parent_b"]
        ]
        required = tuple(
            meta[
                "required_capabilities"
            ]
        )

        adaptive_parent_ret = (
            _inheritance_retention(
                adaptive_score,
                parent_a,
                parent_b,
                required,
            )
        )
        center_parent_ret = (
            _inheritance_retention(
                center_score,
                parent_a,
                parent_b,
                required,
            )
        )
        adaptive_ceiling_ret = (
            _capability_ceiling_retention(
                adaptive_score,
                best,
                required,
            )
        )
        center_ceiling_ret = (
            _capability_ceiling_retention(
                center_score,
                best,
                required,
            )
        )

        adaptive_gate = (
            _passes_inheritance_gate(
                adaptive_parent_ret,
                adaptive_ceiling_ret,
                threshold=(
                    args.inheritance_threshold
                ),
            )
        )
        center_gate = (
            _passes_inheritance_gate(
                center_parent_ret,
                center_ceiling_ret,
                threshold=(
                    args.inheritance_threshold
                ),
            )
        )
        adaptive_accepted += int(
            adaptive_gate
        )
        center_accepted += int(
            center_gate
        )

        adaptive_certified = (
            _certified_capabilities(
                adaptive_score,
                best,
                threshold=(
                    args.inheritance_threshold
                ),
            )
        )
        center_certified = (
            _certified_capabilities(
                center_score,
                best,
                threshold=(
                    args.inheritance_threshold
                ),
            )
        )
        adaptive_fourcap += int(
            len(
                adaptive_certified
            )
            == len(AXES)
        )
        center_fourcap += int(
            len(
                center_certified
            )
            == len(AXES)
        )

        dominance = _dominance(
            adaptive_score,
            center_score,
            tolerance=(
                args.tie_tolerance
            ),
        )
        dominance_counter[
            dominance
        ] += 1

        adaptive_gene_id = _gene_id(
            adaptive_genes[idx]
        )
        center_gene_id = _gene_id(
            center_genes[idx]
        )
        identical = (
            adaptive_gene_id
            == center_gene_id
        )
        exact_identical += int(
            identical
        )

        rows.append(
            {
                **meta,
                "adaptive_gene_id": (
                    adaptive_gene_id
                ),
                "center_gene_id": (
                    center_gene_id
                ),
                "exact_identical_child": (
                    identical
                ),
                "adaptive_scores": (
                    adaptive_score
                ),
                "center_scores": (
                    center_score
                ),
                "adaptive_minus_center": {
                    axis: float(
                        adaptive_score[axis]
                        - center_score[axis]
                    )
                    for axis in AXES
                },
                "dominance": dominance,
                "adaptive_parent_retention": (
                    adaptive_parent_ret
                ),
                "center_parent_retention": (
                    center_parent_ret
                ),
                "adaptive_ceiling_retention": (
                    adaptive_ceiling_ret
                ),
                "center_ceiling_retention": (
                    center_ceiling_ret
                ),
                "adaptive_min_parent_retention": (
                    _minimum(
                        adaptive_parent_ret
                    )
                ),
                "center_min_parent_retention": (
                    _minimum(
                        center_parent_ret
                    )
                ),
                "adaptive_min_ceiling_retention": (
                    _minimum(
                        adaptive_ceiling_ret
                    )
                ),
                "center_min_ceiling_retention": (
                    _minimum(
                        center_ceiling_ret
                    )
                ),
                "adaptive_dual_gate_pass": (
                    adaptive_gate
                ),
                "center_dual_gate_pass": (
                    center_gate
                ),
                "adaptive_certified_capabilities": list(
                    adaptive_certified
                ),
                "center_certified_capabilities": list(
                    center_certified
                ),
            }
        )

    with (
        run_dir
        / "pair_results.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=False,
                )
                + "\n"
            )

    axis_stats = {}
    for axis in AXES:
        axis_stats[axis] = (
            _axis_pair_stats(
                [
                    row[
                        "adaptive_scores"
                    ][axis]
                    for row in rows
                ],
                [
                    row[
                        "center_scores"
                    ][axis]
                    for row in rows
                ],
                tolerance=(
                    args.tie_tolerance
                ),
            )
        )

    noncenter_rows = [
        row
        for row in rows
        if row[
            "adaptive_rule_active_terms"
        ] > 0
    ]
    noncenter_axis_stats = {}
    for axis in AXES:
        noncenter_axis_stats[
            axis
        ] = _axis_pair_stats(
            [
                row[
                    "adaptive_scores"
                ][axis]
                for row
                in noncenter_rows
            ],
            [
                row[
                    "center_scores"
                ][axis]
                for row
                in noncenter_rows
            ],
            tolerance=(
                args.tie_tolerance
            ),
        )

    summary = {
        "experiment": (
            "gene_mrta_v1142_frozen_parent_matched_pair_assay"
        ),
        "status": "completed",
        "seed": int(
            args.seed
        ),
        "pair_count": int(
            len(rows)
        ),
        "world_count": int(
            len(worlds)
        ),
        "scenario_seeds": (
            scenario_seeds
        ),
        "bootstrap_v113_checkpoint": (
            args.bootstrap_v113_checkpoint
        ),
        "adaptive_v1141_checkpoint": (
            args.adaptive_v1141_checkpoint
        ),
        "anchor_v18_run": (
            args.anchor_v18_run
        ),
        "policy_axes": list(
            AXES
        ),
        "frozen_v113_ceiling": (
            best
        ),
        "design": {
            "same_parent_pair_manifest": True,
            "unique_unordered_parent_pairs": True,
            "one_child_per_pair_per_condition": True,
            "policy_mutation": False,
            "policy_bank_feedback": False,
            "primary_screening": False,
            "evaluation_worlds_per_child": 100,
            "scalarized_policy_score": False,
        },
        "center_rule": (
            center_rule.to_dict()
        ),
        "axis_stats": (
            axis_stats
        ),
        "dominance_counts": {
            key: int(value)
            for key, value
            in dominance_counter.items()
        },
        "adaptive_dual_gate_pass_count": int(
            adaptive_accepted
        ),
        "center_dual_gate_pass_count": int(
            center_accepted
        ),
        "adaptive_four_capability_count": int(
            adaptive_fourcap
        ),
        "center_four_capability_count": int(
            center_fourcap
        ),
        "exact_identical_child_count": int(
            exact_identical
        ),
        "adaptive_rule_usage": {
            rule_id: {
                "count": int(count),
                "active_term_count": (
                    adaptive_rules[
                        rule_id
                    ].gene.active_term_count
                ),
                "generated_in_v1141": (
                    adaptive_rules[
                        rule_id
                    ].generated
                ),
                "accepted_in_v1141": (
                    adaptive_rules[
                        rule_id
                    ].accepted
                ),
                "four_capability_accepted_in_v1141": (
                    adaptive_rules[
                        rule_id
                    ].four_capability_accepted
                ),
                "acceptance_evidence_q10": (
                    adaptive_rules[
                        rule_id
                    ].evidence_score(
                        "acceptance_yield",
                        quantile=(
                            args.rule_evidence_quantile
                        ),
                    )
                ),
            }
            for rule_id, count
            in rule_usage.most_common()
        },
        "adaptive_noncenter_pair_count": int(
            len(
                noncenter_rows
            )
        ),
        "noncenter_subset_axis_stats": (
            noncenter_axis_stats
        ),
        "protected_final_rule": (
            "95M is development data. 99M remains untouched."
        ),
    }
    (
        run_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            _jsonable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "V1142_MATCHED_PAIR_ASSAY"
    )
    print(
        f"PAIRS={len(rows)} "
        f"NONCENTER={len(noncenter_rows)} "
        f"IDENTICAL={exact_identical}"
    )
    for axis in AXES:
        stats = axis_stats[
            axis
        ]
        print(
            f"{axis}: "
            f"delta={stats['mean_delta']:.8f} "
            f"W/T/L="
            f"{stats['wins']}/"
            f"{stats['ties']}/"
            f"{stats['losses']}"
        )
    print(
        "DOMINANCE="
        + json.dumps(
            summary[
                "dominance_counts"
            ]
        )
    )
    print(
        f"DUAL_GATE adaptive="
        f"{adaptive_accepted}/{len(rows)} "
        f"center={center_accepted}/{len(rows)}"
    )
    print(
        f"FOUR_CAP adaptive="
        f"{adaptive_fourcap}/{len(rows)} "
        f"center={center_fourcap}/{len(rows)}"
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
        "--adaptive-v1141-checkpoint",
        required=True,
    )
    p.add_argument(
        "--anchor-v18-run",
        required=True,
    )
    p.add_argument(
        "--pairs",
        type=int,
        default=128,
    )
    p.add_argument(
        "--adaptive-min-generations",
        type=int,
        default=50,
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
        "--parent-q-power",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--parent-uniform-fraction",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--rule-selection-power",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--rule-uniform-fraction",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--rule-evidence-quantile",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--inheritance-threshold",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-12,
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
            "gene_mrta_v1142_matched_pair"
        ),
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.pairs <= 0:
        raise ValueError(
            "pairs must be positive"
        )
    run_assay(args)


if __name__ == "__main__":
    main()
