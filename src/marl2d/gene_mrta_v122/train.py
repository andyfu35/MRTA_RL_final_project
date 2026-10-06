from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from marl2d.gene_mrta_v120.bank import BankRecord, rebuild_bank
from marl2d.gene_mrta_v120.benchmark import (
    CERTIFICATE_ZIP,
    INSTANCE_ZIP,
    MinMaxMTSPInstance,
    describe_benchmark,
    load_benchmark,
    select_instances,
)
from marl2d.gene_mrta_v120.capabilities import (
    GeneAssessment,
    InstanceAssessment,
    _reference_comparison_tolerance,
    capability_axes,
)
from marl2d.gene_mrta_v120.gene import ScalableRouteTailGene
from marl2d.gene_mrta_v121.train import (
    _append_jsonl,
    _atomic_json,
    _gene,
    _record,
    _sha256,
    _world_event,
    build_round_summary,
    parent_probabilities,
)
from marl2d.gene_mrta_v122.mps_batch import (
    MPSInstanceCache,
    mps_available,
    prepare_instance,
    rollout_gene_batch_mps,
)


CHECKPOINT_VERSION = "v122_mps_resident_world_segb_round_v1"
PROTOCOL_NAME = "public_minmax_mtsp_resident_world_segb_mps_v1"
DEVICE = "mps"


def _instance_assessment(
    instance: MinMaxMTSPInstance,
    *,
    success: bool,
    objective: float | None,
    exact_tolerance: float,
) -> InstanceAssessment:
    if success:
        if objective is None:
            raise ValueError("Successful rollout must have an objective")
        objective = float(objective)
        comparison_tolerance = _reference_comparison_tolerance(
            instance.reference_value,
            exact_tolerance,
        )
        delta = objective - instance.reference_value

        if instance.is_exact_optimum and delta < -comparison_tolerance:
            raise RuntimeError(
                f"{instance.instance_id}: Gene objective {objective} is "
                f"better than published exact optimum "
                f"{instance.reference_value} by more than the published "
                f"precision tolerance {comparison_tolerance}; distance "
                "convention or benchmark parsing is inconsistent"
            )

        beat_bks = bool(
            instance.reference_kind == "best_known"
            and delta < -comparison_tolerance
        )
        if delta < 0.0 and not beat_bks:
            retention = 1.0
            gap = 0.0
        else:
            retention = float(instance.reference_value / objective)
            gap = float(delta / instance.reference_value)
    else:
        objective = None
        retention = 0.0
        gap = None
        beat_bks = False

    return InstanceAssessment(
        instance_id=instance.instance_id,
        benchmark_set=instance.benchmark_set,
        size_band=instance.size_band,
        vertex_count=instance.vertex_count,
        robot_count=instance.robot_count,
        reference_kind=instance.reference_kind,
        reference_value=float(instance.reference_value),
        success=bool(success),
        completion=1.0 if success else 0.0,
        objective=objective,
        reference_retention=float(retention),
        gap_to_reference=gap,
        beat_bks=bool(beat_bks),
    )


def _gene_assessment(
    rows: Sequence[InstanceAssessment],
) -> GeneAssessment:
    if not rows:
        raise ValueError("At least one instance assessment is required")

    completions = np.asarray(
        [row.completion for row in rows],
        dtype=np.float64,
    )
    retentions = np.asarray(
        [row.reference_retention for row in rows],
        dtype=np.float64,
    )

    scores: dict[str, float] = {}
    for band in ("small", "medium", "large"):
        values = [
            row.reference_retention
            for row in rows
            if row.size_band == band
        ]
        if values:
            scores[f"retention_{band}"] = float(np.mean(values))

    success_instances = sum(int(row.success) for row in rows)
    exact_matches = sum(
        int(
            row.success
            and row.reference_kind == "exact_optimum"
            and row.objective is not None
            and abs(row.objective - row.reference_value)
            <= _reference_comparison_tolerance(
                row.reference_value,
                1e-6,
            )
        )
        for row in rows
    )
    bks_improvements = sum(int(row.beat_bks) for row in rows)

    return GeneAssessment(
        success=bool(success_instances == len(rows)),
        worst_completion=float(np.min(completions)),
        mean_completion=float(np.mean(completions)),
        success_instances=int(success_instances),
        instance_count=len(rows),
        scores=scores,
        overall_reference_retention=float(np.mean(retentions)),
        worst_reference_retention=float(np.min(retentions)),
        exact_matches=int(exact_matches),
        bks_improvements=int(bks_improvements),
        instances=tuple(rows),
    )


def prepare_caches(
    instances: Sequence[MinMaxMTSPInstance],
) -> dict[str, MPSInstanceCache]:
    caches: dict[str, MPSInstanceCache] = {}
    for index, instance in enumerate(instances, start=1):
        caches[instance.instance_id] = prepare_instance(
            instance,
            device=DEVICE,
        )
        print(
            f"V122_MPS_CACHE {index}/{len(instances)} "
            f"instance={instance.instance_id} "
            f"vertices={instance.vertex_count} robots={instance.robot_count}",
            flush=True,
        )
    return caches


def evaluate_genes_mps(
    genes: Sequence[ScalableRouteTailGene],
    instances: Sequence[MinMaxMTSPInstance],
    caches: Mapping[str, MPSInstanceCache],
    *,
    candidate_k: int,
    batch_size: int,
    round_index: int,
    exact_tolerance: float = 1e-6,
) -> list[GeneAssessment]:
    if not genes:
        raise ValueError("At least one Gene is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rows_by_gene: list[list[InstanceAssessment]] = [
        [] for _ in genes
    ]

    for instance_index, instance in enumerate(instances, start=1):
        cache = caches[instance.instance_id]
        for start in range(0, len(genes), batch_size):
            stop = min(start + batch_size, len(genes))
            batch = rollout_gene_batch_mps(
                genes[start:stop],
                cache,
                candidate_k=candidate_k,
            )
            for local_index in range(stop - start):
                gene_index = start + local_index
                success = bool(batch.success[local_index])
                objective = (
                    float(batch.objectives[local_index])
                    if success
                    else None
                )
                rows_by_gene[gene_index].append(
                    _instance_assessment(
                        instance,
                        success=success,
                        objective=objective,
                        exact_tolerance=exact_tolerance,
                    )
                )

        print(
            f"V122_MPS_INSTANCE round={round_index} "
            f"{instance_index}/{len(instances)} "
            f"instance={instance.instance_id} genes={len(genes)} "
            f"batch_size={batch_size}",
            flush=True,
        )

    return [
        _gene_assessment(rows)
        for rows in rows_by_gene
    ]


def _checkpoint_payload(
    *,
    completed_round: int,
    args: argparse.Namespace,
    axes: Sequence[str],
    instance_ids: Sequence[str],
    instance_zip_sha: str,
    certificate_zip_sha: str,
    rng: np.random.Generator,
    bank_records: Mapping[str, BankRecord],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "protocol": PROTOCOL_NAME,
        "device": DEVICE,
        "completed_round": int(completed_round),
        "worlds_per_round": int(args.worlds_per_round),
        "rounds": int(args.rounds),
        "candidate_k": int(args.candidate_k),
        "mps_batch_size": int(args.mps_batch_size),
        "seed": int(args.seed),
        "initial_scale": float(args.initial_scale),
        "mutation_sigma": float(args.mutation_sigma),
        "mutation_rate": float(args.mutation_rate),
        "pareto_epsilon": float(args.pareto_epsilon),
        "bank_max_size": int(args.bank_max_size),
        "axes": list(axes),
        "instance_ids": list(instance_ids),
        "instance_zip_sha256": instance_zip_sha,
        "certificate_zip_sha256": certificate_zip_sha,
        "rng_state": rng.bit_generator.state,
        "bank_records": [
            record.to_dict()
            for record in bank_records.values()
        ],
        "history": list(history),
    }


def run(args: argparse.Namespace) -> Path:
    if not mps_available():
        raise RuntimeError(
            "V1.22 formal training requires Apple MPS GPU; "
            "CPU fallback is intentionally disabled"
        )

    all_instances = load_benchmark(
        Path(args.instance_zip),
        Path(args.certificate_zip),
    )
    instances = select_instances(all_instances, "evolution")
    if len(instances) != args.expected_instance_count:
        raise ValueError(
            f"V1.22 requires exactly {args.expected_instance_count} fixed "
            f"evolution instances, found {len(instances)}"
        )

    axes = capability_axes(instances)
    if tuple(axes) != ("retention_small", "retention_medium"):
        raise ValueError(f"Unexpected V1.22 capability axes: {axes}")

    instance_ids = tuple(item.instance_id for item in instances)
    instance_zip_sha = _sha256(Path(args.instance_zip))
    certificate_zip_sha = _sha256(Path(args.certificate_zip))

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoint.json"
    world_events_dir = run_dir / "world_events"
    snapshots_dir = run_dir / "bank_snapshots"

    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Incompatible V1.22 checkpoint")
        immutable = {
            "device": DEVICE,
            "worlds_per_round": args.worlds_per_round,
            "candidate_k": args.candidate_k,
            "seed": args.seed,
            "initial_scale": args.initial_scale,
            "mutation_sigma": args.mutation_sigma,
            "mutation_rate": args.mutation_rate,
            "pareto_epsilon": args.pareto_epsilon,
            "bank_max_size": args.bank_max_size,
            "instance_ids": list(instance_ids),
            "instance_zip_sha256": instance_zip_sha,
            "certificate_zip_sha256": certificate_zip_sha,
        }
        for key, expected in immutable.items():
            if data.get(key) != expected:
                raise ValueError(
                    f"Cannot resume: {key} changed from checkpoint"
                )
        if args.rounds < int(data["completed_round"]) + 1:
            raise ValueError(
                "Requested rounds end before completed checkpoint"
            )

        rng = np.random.default_rng()
        rng.bit_generator.state = data["rng_state"]
        bank_records = {
            row["record_id"]: BankRecord.from_dict(row)
            for row in data["bank_records"]
        }
        history = list(data.get("history", []))
        start_round = int(data["completed_round"]) + 1
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        bank_records: dict[str, BankRecord] = {}
        history: list[dict[str, Any]] = []
        start_round = 0

    print(
        "V122_PROTOCOL "
        + json.dumps(
            {
                "device": DEVICE,
                "torch": torch.__version__,
                "worlds_per_round": args.worlds_per_round,
                "rounds": args.rounds,
                "fixed_instances_per_world": len(instances),
                "rollouts_per_round": (
                    args.worlds_per_round * len(instances)
                ),
                "planned_total_rollouts": (
                    args.worlds_per_round
                    * len(instances)
                    * args.rounds
                ),
                "candidate_k": args.candidate_k,
                "mps_batch_size": args.mps_batch_size,
                "axes": list(axes),
                "instance_summary": describe_benchmark(instances),
                "gene_bank_admission": "external_axes_pareto_only",
                "parent_sampling": "equal_axis_total_score_squared",
                "parent_child_delta": "diagnostic_only",
                "cpu_fallback": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    caches = prepare_caches(instances)

    for round_index in range(start_round, args.rounds):
        bank_before_records = dict(bank_records)
        parent_ids: list[str] = []
        parent_probs: np.ndarray | None = None
        if round_index > 0:
            if not bank_before_records:
                raise RuntimeError(
                    "Gene Bank is empty after round 0; cannot inherit"
                )
            parent_ids, parent_probs = parent_probabilities(
                bank_before_records,
                axes,
            )

        candidate_genes: list[ScalableRouteTailGene] = []
        parent_records: list[BankRecord | None] = []
        origins: list[str] = []

        for _ in range(args.worlds_per_round):
            if round_index == 0:
                base = ScalableRouteTailGene.random(
                    rng,
                    hidden_dim=8,
                    scale=args.initial_scale,
                )
                gene = ScalableRouteTailGene.from_gene(base)
                parent = None
                origin = "round0_random"
            else:
                assert parent_probs is not None
                parent_id = str(rng.choice(parent_ids, p=parent_probs))
                parent = bank_before_records[parent_id]
                mutated = _gene(parent).mutated(
                    rng,
                    sigma=args.mutation_sigma,
                    mutation_rate=args.mutation_rate,
                )
                gene = ScalableRouteTailGene.from_gene(mutated)
                origin = "bank_inherit_mutation"

            candidate_genes.append(gene)
            parent_records.append(parent)
            origins.append(origin)

        print(
            f"V122_ROUND_START round={round_index} "
            f"genes={len(candidate_genes)} device={DEVICE} "
            f"batch_size={args.mps_batch_size}",
            flush=True,
        )

        candidate_assessments = evaluate_genes_mps(
            candidate_genes,
            instances,
            caches,
            candidate_k=args.candidate_k,
            batch_size=args.mps_batch_size,
            round_index=round_index,
        )

        candidate_records: list[BankRecord] = []
        for gene, assessment, parent, origin in zip(
            candidate_genes,
            candidate_assessments,
            parent_records,
            origins,
        ):
            candidate_records.append(
                _record(
                    gene,
                    assessment,
                    round_index=round_index,
                    origin=origin,
                    parent_id=(
                        None if parent is None else parent.record_id
                    ),
                )
            )

        world_events_dir.mkdir(parents=True, exist_ok=True)
        round_world_path = (
            world_events_dir / f"round_{round_index:03d}.jsonl"
        )
        staged_world_path = (
            world_events_dir / f"round_{round_index:03d}.jsonl.tmp"
        )
        if staged_world_path.exists():
            staged_world_path.unlink()

        for world_index, (record, assessment, parent) in enumerate(
            zip(
                candidate_records,
                candidate_assessments,
                parent_records,
            )
        ):
            _append_jsonl(
                staged_world_path,
                _world_event(
                    round_index=round_index,
                    world_index=world_index,
                    record=record,
                    assessment=assessment,
                    parent=parent,
                    instance_count=len(instances),
                ),
            )

        eligible: dict[str, BankRecord] = dict(bank_before_records)
        for record, assessment in zip(
            candidate_records,
            candidate_assessments,
        ):
            if assessment.success:
                eligible[record.record_id] = record

        rebuilt = rebuild_bank(
            eligible,
            axes,
            epsilon=args.pareto_epsilon,
            max_size=args.bank_max_size,
        )
        bank_records = dict(rebuilt.records)

        previous_summary = None if not history else history[-1]
        summary = build_round_summary(
            round_index=round_index,
            worlds_per_round=args.worlds_per_round,
            instance_count=len(instances),
            axes=axes,
            candidate_records=candidate_records,
            candidate_assessments=candidate_assessments,
            parent_records=parent_records,
            bank_records=bank_records,
            bank_before=len(bank_before_records),
            dominated_removed=len(rebuilt.dominated_ids),
            epsilon_removed=len(rebuilt.epsilon_removed_ids),
            crowding_removed=len(rebuilt.crowding_removed_ids),
            previous_summary=previous_summary,
        )
        history.append(summary)

        _atomic_json(
            snapshots_dir / f"round_{round_index:03d}.json",
            {
                "round": round_index,
                "summary": summary,
                "bank_records": [
                    record.to_dict()
                    for record in bank_records.values()
                ],
            },
        )
        staged_world_path.replace(round_world_path)

        _atomic_json(
            checkpoint_path,
            _checkpoint_payload(
                completed_round=round_index,
                args=args,
                axes=axes,
                instance_ids=instance_ids,
                instance_zip_sha=instance_zip_sha,
                certificate_zip_sha=certificate_zip_sha,
                rng=rng,
                bank_records=bank_records,
                history=history,
            ),
        )

        print(
            "V122_ROUND "
            + json.dumps(summary, ensure_ascii=False),
            flush=True,
        )

    print(f"V122_RUN_DIR={run_dir}", flush=True)
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-zip", default=str(INSTANCE_ZIP))
    p.add_argument(
        "--certificate-zip",
        default=str(CERTIFICATE_ZIP),
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worlds-per-round", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--expected-instance-count", type=int, default=34)
    p.add_argument("--candidate-k", type=int, default=32)
    p.add_argument("--mps-batch-size", type=int, default=32)
    p.add_argument("--initial-scale", type=float, default=0.35)
    p.add_argument("--mutation-sigma", type=float, default=0.12)
    p.add_argument("--mutation-rate", type=float, default=0.20)
    p.add_argument("--bank-max-size", type=int, default=128)
    p.add_argument("--pareto-epsilon", type=float, default=0.0025)
    p.add_argument("--seed", type=int, default=121)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.worlds_per_round <= 0:
        raise ValueError("worlds_per_round must be positive")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    if args.mps_batch_size <= 0:
        raise ValueError("mps_batch_size must be positive")
    run(args)


if __name__ == "__main__":
    main()
