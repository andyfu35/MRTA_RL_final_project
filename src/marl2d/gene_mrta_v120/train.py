from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bank import (
    BankRecord,
    best_by_axis,
    global_best_id,
    rebuild_bank,
)
from .benchmark import (
    CERTIFICATE_ZIP,
    INSTANCE_ZIP,
    describe_benchmark,
    load_benchmark,
    select_instances,
)
from .capabilities import (
    GeneAssessment,
    InstanceAssessment,
    capability_axes,
    evaluate_gene,
    paired_delta,
)
from .gene import (
    ScalableRouteTailGene,
)


CHECKPOINT_VERSION = "v120_public_minmax_mtsp_mutation_v1"


def _gene_id(gene: ScalableRouteTailGene) -> str:
    return hashlib.sha256(
        np.asarray(gene.vector_data, dtype=np.float64).tobytes()
    ).hexdigest()[:20]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _assessment_to_dict(value: GeneAssessment) -> dict[str, Any]:
    return {
        **value.to_summary_dict(),
        "instances": [row.to_dict() for row in value.instances],
    }


def _assessment_from_dict(data: dict[str, Any]) -> GeneAssessment:
    rows = tuple(
        InstanceAssessment(
            instance_id=str(row["instance_id"]),
            benchmark_set=str(row["benchmark_set"]),
            size_band=str(row["size_band"]),
            vertex_count=int(row["vertex_count"]),
            robot_count=int(row["robot_count"]),
            reference_kind=str(row["reference_kind"]),
            reference_value=float(row["reference_value"]),
            success=bool(row["success"]),
            completion=float(row["completion"]),
            objective=(
                None
                if row["objective"] is None
                else float(row["objective"])
            ),
            reference_retention=float(row["reference_retention"]),
            gap_to_reference=(
                None
                if row["gap_to_reference"] is None
                else float(row["gap_to_reference"])
            ),
            beat_bks=bool(row["beat_bks"]),
        )
        for row in data["instances"]
    )
    return GeneAssessment(
        success=bool(data["success"]),
        worst_completion=float(data["worst_completion"]),
        mean_completion=float(data["mean_completion"]),
        success_instances=int(data["success_instances"]),
        instance_count=int(data["instance_count"]),
        scores={
            str(key): float(value)
            for key, value in dict(data["scores"]).items()
        },
        overall_reference_retention=float(
            data["overall_reference_retention"]
        ),
        worst_reference_retention=float(
            data["worst_reference_retention"]
        ),
        exact_matches=int(data["exact_matches"]),
        bks_improvements=int(data["bks_improvements"]),
        instances=rows,
    )


def _record(
    gene: ScalableRouteTailGene,
    assessment: GeneAssessment,
    *,
    generation: int,
    origin: str,
    parents: tuple[str, ...] = (),
) -> BankRecord:
    return BankRecord(
        record_id=_gene_id(gene),
        vector_data=tuple(float(value) for value in gene.vector_data),
        hidden_dim=int(gene.hidden_dim),
        scores=dict(assessment.scores),
        overall_retention=float(assessment.overall_reference_retention),
        worst_retention=float(assessment.worst_reference_retention),
        generation=int(generation),
        origin=origin,
        parents=parents,
    )


def _gene(record: BankRecord) -> ScalableRouteTailGene:
    return ScalableRouteTailGene(
        np.asarray(record.vector_data, dtype=np.float64),
        hidden_dim=record.hidden_dim,
    )


def _bootstrap_key(value: GeneAssessment) -> tuple[float, float, int, float]:
    return (
        value.worst_completion,
        value.mean_completion,
        value.success_instances,
        value.overall_reference_retention,
    )


def _trim_bootstrap(
    records: dict[str, BankRecord],
    assessments: dict[str, GeneAssessment],
    *,
    size: int,
) -> tuple[dict[str, BankRecord], dict[str, GeneAssessment]]:
    ordered = sorted(
        records,
        key=lambda rid: (_bootstrap_key(assessments[rid]), rid),
        reverse=True,
    )
    keep = ordered[:size]
    return (
        {rid: records[rid] for rid in keep},
        {rid: assessments[rid] for rid in keep},
    )


def _parent_probabilities(
    parent_ids: list[str],
    assessments: dict[str, GeneAssessment],
) -> np.ndarray:
    raw: list[float] = []
    for rid in parent_ids:
        value = assessments[rid]
        if value.success:
            score = max(value.overall_reference_retention, 1e-6)
        else:
            score = max(
                0.5 * value.worst_completion
                + 0.5 * value.mean_completion,
                1e-6,
            )
        raw.append(score * score)
    weights = np.asarray(raw, dtype=np.float64)
    return weights / np.sum(weights)


def _checkpoint(
    *,
    generation: int,
    axes: tuple[str, ...],
    candidate_k: int,
    instance_ids: tuple[str, ...],
    instance_zip_sha256: str,
    certificate_zip_sha256: str,
    rng: np.random.Generator,
    bootstrap_records: dict[str, BankRecord],
    bank_records: dict[str, BankRecord],
    assessments: dict[str, GeneAssessment],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    keep_ids = sorted(set(bootstrap_records) | set(bank_records))
    return {
        "version": CHECKPOINT_VERSION,
        "generation": generation,
        "axes": list(axes),
        "candidate_k": candidate_k,
        "instance_ids": list(instance_ids),
        "instance_zip_sha256": instance_zip_sha256,
        "certificate_zip_sha256": certificate_zip_sha256,
        "rng_state": rng.bit_generator.state,
        "bootstrap_ids": list(bootstrap_records),
        "bank_ids": list(bank_records),
        "records": [
            (bootstrap_records.get(rid) or bank_records[rid]).to_dict()
            for rid in keep_ids
        ],
        "assessments": {
            rid: _assessment_to_dict(assessments[rid])
            for rid in keep_ids
        },
        "history": history,
    }


def run(args: argparse.Namespace) -> Path:
    all_instances = load_benchmark(
        Path(args.instance_zip),
        Path(args.certificate_zip),
    )
    instances = select_instances(all_instances, args.split)
    if args.max_vertices is not None:
        instances = [
            item
            for item in instances
            if item.vertex_count <= args.max_vertices
        ]
    if not instances:
        raise ValueError("No instances remain after filtering")

    axes = capability_axes(instances)
    instance_ids = tuple(item.instance_id for item in instances)
    instance_zip_sha = _sha256(Path(args.instance_zip))
    cert_zip_sha = _sha256(Path(args.certificate_zip))

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoint.json"
    paired_path = run_dir / "paired_events.jsonl"

    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Incompatible V1.20 checkpoint")
        if tuple(data.get("axes", [])) != axes:
            raise ValueError("Capability axes changed since checkpoint")
        if tuple(data.get("instance_ids", [])) != instance_ids:
            raise ValueError("Evolution instance set changed since checkpoint")
        if int(data.get("candidate_k")) != int(args.candidate_k):
            raise ValueError("candidate_k changed since checkpoint")
        if (
            data.get("instance_zip_sha256") != instance_zip_sha
            or data.get("certificate_zip_sha256") != cert_zip_sha
        ):
            raise ValueError("Mirrored public benchmark bytes changed")

        records = {
            row["record_id"]: BankRecord.from_dict(row)
            for row in data["records"]
        }
        assessments = {
            rid: _assessment_from_dict(row)
            for rid, row in data["assessments"].items()
        }
        bootstrap_records = {
            rid: records[rid]
            for rid in data["bootstrap_ids"]
        }
        bank_records = {
            rid: records[rid]
            for rid in data["bank_ids"]
        }
        history = list(data.get("history", []))
        rng = np.random.default_rng()
        rng.bit_generator.state = data["rng_state"]
        start_generation = int(data["generation"]) + 1
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        rng = np.random.default_rng(args.seed)
        records: dict[str, BankRecord] = {}
        assessments: dict[str, GeneAssessment] = {}

        for index in range(args.population):
            base = ScalableRouteTailGene.random(
                rng,
                hidden_dim=8,
                scale=args.initial_scale,
            )
            gene = ScalableRouteTailGene.from_gene(base)
            value = evaluate_gene(
                gene,
                instances,
                candidate_k=args.candidate_k,
            )
            rec = _record(
                gene,
                value,
                generation=-1,
                origin="random_initial",
            )
            records[rec.record_id] = rec
            assessments[rec.record_id] = value
            if (
                args.progress_every > 0
                and (index + 1) % args.progress_every == 0
            ):
                print(
                    f"V120_INIT_EVAL {index + 1}/{args.population}",
                    flush=True,
                )

        bootstrap_records, _ = _trim_bootstrap(
            records,
            assessments,
            size=args.bootstrap_size,
        )
        successful = {
            rid: record
            for rid, record in records.items()
            if assessments[rid].success
        }
        rebuilt = rebuild_bank(
            successful,
            axes,
            epsilon=args.pareto_epsilon,
            max_size=args.bank_max_size,
        )
        bank_records = dict(rebuilt.records)

        keep = set(bootstrap_records) | set(bank_records)
        records = {rid: records[rid] for rid in keep}
        assessments = {rid: assessments[rid] for rid in keep}
        bootstrap_records = {
            rid: records[rid]
            for rid in bootstrap_records
        }
        history: list[dict[str, Any]] = []
        start_generation = 0

    print(
        "V120_PROTOCOL "
        + json.dumps(
            {
                "split": args.split,
                "candidate_k": args.candidate_k,
                "instance_summary": describe_benchmark(instances),
                "axes": list(axes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for generation in range(start_generation, args.generations):
        parent_ids = (
            list(bank_records)
            if bank_records
            else list(bootstrap_records)
        )
        probs = _parent_probabilities(parent_ids, assessments)

        child_records: dict[str, BankRecord] = {}
        child_assessments: dict[str, GeneAssessment] = {}
        events: list[dict[str, Any]] = []

        for index in range(args.mutation_children):
            parent_id = str(rng.choice(parent_ids, p=probs))
            parent = _gene(records[parent_id])
            mutated = parent.mutated(
                rng,
                sigma=args.mutation_sigma,
                mutation_rate=args.mutation_rate,
            )
            child = ScalableRouteTailGene.from_gene(mutated)
            value = evaluate_gene(
                child,
                instances,
                candidate_k=args.candidate_k,
            )
            rec = _record(
                child,
                value,
                generation=generation,
                origin="mutation",
                parents=(parent_id,),
            )
            child_records[rec.record_id] = rec
            child_assessments[rec.record_id] = value

            delta = paired_delta(assessments[parent_id], value)
            event = {
                "generation": generation,
                "child_index": index,
                "parent_id": parent_id,
                "child_id": rec.record_id,
                "parent_overall": (
                    assessments[parent_id].overall_reference_retention
                ),
                "child_overall": value.overall_reference_retention,
                "child_success": value.success,
                **delta,
            }
            events.append(event)
            _append_jsonl(paired_path, event)

            if (
                args.progress_every > 0
                and (index + 1) % args.progress_every == 0
            ):
                print(
                    f"V120_GEN_EVAL generation={generation} "
                    f"{index + 1}/{args.mutation_children}",
                    flush=True,
                )

        merged_records = dict(records)
        merged_records.update(child_records)
        merged_assessments = dict(assessments)
        merged_assessments.update(child_assessments)

        bootstrap_records, _ = _trim_bootstrap(
            merged_records,
            merged_assessments,
            size=args.bootstrap_size,
        )

        successful = {
            rid: record
            for rid, record in merged_records.items()
            if merged_assessments[rid].success
        }
        rebuilt = rebuild_bank(
            successful,
            axes,
            epsilon=args.pareto_epsilon,
            max_size=args.bank_max_size,
        )
        bank_records = dict(rebuilt.records)

        keep = set(bootstrap_records) | set(bank_records)
        records = {rid: merged_records[rid] for rid in keep}
        assessments = {
            rid: merged_assessments[rid]
            for rid in keep
        }
        bootstrap_records = {
            rid: records[rid]
            for rid in bootstrap_records
        }

        best_bootstrap_id = max(
            bootstrap_records,
            key=lambda rid: (_bootstrap_key(assessments[rid]), rid),
        )
        best_bootstrap = assessments[best_bootstrap_id]

        best_id = global_best_id(bank_records)
        global_best = (
            None
            if best_id is None
            else {
                "record_id": best_id,
                "overall_reference_retention": (
                    assessments[best_id].overall_reference_retention
                ),
                "worst_reference_retention": (
                    assessments[best_id].worst_reference_retention
                ),
                "exact_matches": assessments[best_id].exact_matches,
                "bks_improvements": assessments[best_id].bks_improvements,
                "scores": bank_records[best_id].scores,
            }
        )

        positive = sum(
            int(event["overall_delta"] > 1e-12)
            for event in events
        )
        negative = sum(
            int(event["overall_delta"] < -1e-12)
            for event in events
        )
        tied = len(events) - positive - negative

        row = {
            "generation": generation,
            "protocol": "public_minmax_mtsp_paired_mutation_v1",
            "instance_count": len(instances),
            "candidate_k": args.candidate_k,
            "best_worst_completion": best_bootstrap.worst_completion,
            "best_mean_completion": best_bootstrap.mean_completion,
            "best_success_instances": best_bootstrap.success_instances,
            "new_successful_children": sum(
                int(value.success)
                for value in child_assessments.values()
            ),
            "bank_size": len(bank_records),
            "dominated_removed": len(rebuilt.dominated_ids),
            "epsilon_removed": len(rebuilt.epsilon_removed_ids),
            "crowding_removed": len(rebuilt.crowding_removed_ids),
            "best_by_size": best_by_axis(bank_records, axes),
            "global_best": global_best,
            "paired_child_overall_wtl": {
                "win": positive,
                "tie": tied,
                "loss": negative,
            },
            "paired_mean_overall_delta": float(
                np.mean([event["overall_delta"] for event in events])
            ),
            "paired_instance_wins": int(
                sum(event["wins"] for event in events)
            ),
            "paired_instance_ties": int(
                sum(event["ties"] for event in events)
            ),
            "paired_instance_losses": int(
                sum(event["losses"] for event in events)
            ),
        }
        history.append(row)
        print(
            "V120_TRAIN "
            + json.dumps(row, ensure_ascii=False),
            flush=True,
        )

        _atomic_json(
            checkpoint_path,
            _checkpoint(
                generation=generation,
                axes=axes,
                candidate_k=args.candidate_k,
                instance_ids=instance_ids,
                instance_zip_sha256=instance_zip_sha,
                certificate_zip_sha256=cert_zip_sha,
                rng=rng,
                bootstrap_records=bootstrap_records,
                bank_records=bank_records,
                assessments=assessments,
                history=history,
            ),
        )

    print(f"V120_RUN_DIR={run_dir}", flush=True)
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-zip", default=str(INSTANCE_ZIP))
    p.add_argument("--certificate-zip", default=str(CERTIFICATE_ZIP))
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--split",
        choices=("evolution", "validation", "protected_test", "all"),
        default="evolution",
    )
    p.add_argument("--max-vertices", type=int, default=None)
    p.add_argument("--candidate-k", type=int, default=32)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--bootstrap-size", type=int, default=32)
    p.add_argument("--mutation-children", type=int, default=64)
    p.add_argument("--initial-scale", type=float, default=0.35)
    p.add_argument("--mutation-sigma", type=float, default=0.12)
    p.add_argument("--mutation-rate", type=float, default=0.20)
    p.add_argument("--bank-max-size", type=int, default=64)
    p.add_argument("--pareto-epsilon", type=float, default=0.0025)
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--seed", type=int, default=120)
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
