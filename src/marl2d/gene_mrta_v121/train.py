from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marl2d.gene_mrta_v120.bank import (
    BankRecord,
    best_by_axis,
    global_best_id,
    rebuild_bank,
)
from marl2d.gene_mrta_v120.benchmark import (
    CERTIFICATE_ZIP,
    INSTANCE_ZIP,
    describe_benchmark,
    load_benchmark,
    select_instances,
)
from marl2d.gene_mrta_v120.capabilities import (
    GeneAssessment,
    capability_axes,
    evaluate_gene,
)
from marl2d.gene_mrta_v120.gene import ScalableRouteTailGene


CHECKPOINT_VERSION = "v121_resident_world_segb_round_v1"
PROTOCOL_NAME = "public_minmax_mtsp_resident_world_segb_v1"


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _record(
    gene: ScalableRouteTailGene,
    assessment: GeneAssessment,
    *,
    round_index: int,
    origin: str,
    parent_id: str | None,
) -> BankRecord:
    return BankRecord(
        record_id=_gene_id(gene),
        vector_data=tuple(float(value) for value in gene.vector_data),
        hidden_dim=int(gene.hidden_dim),
        scores=dict(assessment.scores),
        overall_retention=float(assessment.overall_reference_retention),
        worst_retention=float(assessment.worst_reference_retention),
        generation=int(round_index),
        origin=origin,
        parents=(() if parent_id is None else (parent_id,)),
    )


def _gene(record: BankRecord) -> ScalableRouteTailGene:
    return ScalableRouteTailGene(
        np.asarray(record.vector_data, dtype=np.float64),
        hidden_dim=record.hidden_dim,
    )


def _equal_axis_total_score(
    record: BankRecord,
    axes: Sequence[str],
) -> float:
    values = np.asarray(
        [float(record.scores[axis]) for axis in axes],
        dtype=np.float64,
    )
    return float(np.mean(values))


def parent_probabilities(
    records: Mapping[str, BankRecord],
    axes: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    ids = sorted(records)
    if not ids:
        raise ValueError("Cannot sample a parent from an empty Gene Bank")

    raw = np.asarray(
        [
            max(_equal_axis_total_score(records[rid], axes), 1e-9) ** 2
            for rid in ids
        ],
        dtype=np.float64,
    )
    return ids, raw / np.sum(raw)


def _population_axis_stats(
    assessments: Sequence[GeneAssessment],
    axes: Sequence[str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for axis in axes:
        values = np.asarray(
            [float(item.scores[axis]) for item in assessments],
            dtype=np.float64,
        )
        result[axis] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }
    return result


def build_round_summary(
    *,
    round_index: int,
    worlds_per_round: int,
    instance_count: int,
    axes: Sequence[str],
    candidate_records: Sequence[BankRecord],
    candidate_assessments: Sequence[GeneAssessment],
    parent_records: Sequence[BankRecord | None],
    bank_records: Mapping[str, BankRecord],
    bank_before: int,
    dominated_removed: int,
    epsilon_removed: int,
    crowding_removed: int,
    previous_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if len(candidate_records) != worlds_per_round:
        raise ValueError("Round did not evaluate the requested number of worlds")
    if len(candidate_assessments) != worlds_per_round:
        raise ValueError("Assessment count does not equal worlds_per_round")
    if len(parent_records) != worlds_per_round:
        raise ValueError("Parent metadata count does not equal worlds_per_round")

    axis_stats = _population_axis_stats(candidate_assessments, axes)
    previous_axis_stats = (
        None
        if previous_summary is None
        else previous_summary.get("population_axis_stats")
    )
    axis_mean_delta = {
        axis: (
            None
            if previous_axis_stats is None
            else float(
                axis_stats[axis]["mean"]
                - float(previous_axis_stats[axis]["mean"])
            )
        )
        for axis in axes
    }

    overall = np.asarray(
        [item.overall_reference_retention for item in candidate_assessments],
        dtype=np.float64,
    )
    worst = np.asarray(
        [item.worst_reference_retention for item in candidate_assessments],
        dtype=np.float64,
    )

    population_best_by_axis: dict[str, str] = {}
    for axis in axes:
        best_index = max(
            range(worlds_per_round),
            key=lambda index: (
                candidate_records[index].scores[axis],
                candidate_records[index].overall_retention,
                candidate_records[index].record_id,
            ),
        )
        population_best_by_axis[axis] = candidate_records[best_index].record_id

    child_axis_deltas: dict[str, list[float]] = {axis: [] for axis in axes}
    child_overall_deltas: list[float] = []
    child_wins = child_ties = child_losses = 0
    for record, parent in zip(candidate_records, parent_records):
        if parent is None:
            continue
        delta = float(record.overall_retention - parent.overall_retention)
        child_overall_deltas.append(delta)
        if delta > 1e-12:
            child_wins += 1
        elif delta < -1e-12:
            child_losses += 1
        else:
            child_ties += 1
        for axis in axes:
            child_axis_deltas[axis].append(
                float(record.scores[axis] - parent.scores[axis])
            )

    best_id = global_best_id(bank_records)
    global_best = (
        None
        if best_id is None
        else {
            "record_id": best_id,
            "overall_reference_retention": float(
                bank_records[best_id].overall_retention
            ),
            "worst_reference_retention": float(
                bank_records[best_id].worst_retention
            ),
            "scores": dict(bank_records[best_id].scores),
        }
    )

    return {
        "round": int(round_index),
        "protocol": PROTOCOL_NAME,
        "worlds_per_round": int(worlds_per_round),
        "fixed_instances_per_world": int(instance_count),
        "rollouts_this_round": int(worlds_per_round * instance_count),
        "cumulative_rollouts": int(
            (round_index + 1) * worlds_per_round * instance_count
        ),
        "successful_worlds": int(
            sum(int(item.success) for item in candidate_assessments)
        ),
        "population_axis_stats": axis_stats,
        "population_axis_mean_delta_from_previous_round": axis_mean_delta,
        "population_overall": {
            "mean": float(np.mean(overall)),
            "median": float(np.median(overall)),
            "min": float(np.min(overall)),
            "max": float(np.max(overall)),
            "std": float(np.std(overall)),
        },
        "population_worst_instance_retention": {
            "mean": float(np.mean(worst)),
            "median": float(np.median(worst)),
            "min": float(np.min(worst)),
            "max": float(np.max(worst)),
        },
        "population_best_by_axis": population_best_by_axis,
        "bank_size_before": int(bank_before),
        "bank_size_after": int(len(bank_records)),
        "bank_best_by_axis": best_by_axis(bank_records, axes),
        "bank_global_best": global_best,
        "bank_dominated_removed": int(dominated_removed),
        "bank_epsilon_removed": int(epsilon_removed),
        "bank_crowding_removed": int(crowding_removed),
        "parent_child_diagnostic": {
            "available": bool(child_overall_deltas),
            "overall_win_tie_loss": {
                "win": int(child_wins),
                "tie": int(child_ties),
                "loss": int(child_losses),
            },
            "mean_overall_delta": (
                None
                if not child_overall_deltas
                else float(np.mean(child_overall_deltas))
            ),
            "mean_axis_delta": {
                axis: (
                    None
                    if not child_axis_deltas[axis]
                    else float(np.mean(child_axis_deltas[axis]))
                )
                for axis in axes
            },
        },
    }


def _world_event(
    *,
    round_index: int,
    world_index: int,
    record: BankRecord,
    assessment: GeneAssessment,
    parent: BankRecord | None,
    instance_count: int,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "round": int(round_index),
        "world": int(world_index),
        "record_id": record.record_id,
        "parent_id": None if parent is None else parent.record_id,
        "origin": record.origin,
        "fixed_instances": int(instance_count),
        "successful_all_instances": bool(assessment.success),
        "scores": dict(record.scores),
        "overall_reference_retention": float(record.overall_retention),
        "worst_reference_retention": float(record.worst_retention),
        "exact_matches": int(assessment.exact_matches),
        "bks_improvements": int(assessment.bks_improvements),
    }
    if parent is not None:
        event["parent_child_delta"] = {
            "overall": float(
                record.overall_retention - parent.overall_retention
            ),
            "axes": {
                axis: float(record.scores[axis] - parent.scores[axis])
                for axis in record.scores
            },
        }
    return event


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
        "completed_round": int(completed_round),
        "worlds_per_round": int(args.worlds_per_round),
        "rounds": int(args.rounds),
        "candidate_k": int(args.candidate_k),
        "seed": int(args.seed),
        "mutation_sigma": float(args.mutation_sigma),
        "mutation_rate": float(args.mutation_rate),
        "pareto_epsilon": float(args.pareto_epsilon),
        "bank_max_size": int(args.bank_max_size),
        "axes": list(axes),
        "instance_ids": list(instance_ids),
        "instance_zip_sha256": instance_zip_sha,
        "certificate_zip_sha256": certificate_zip_sha,
        "rng_state": rng.bit_generator.state,
        "bank_records": [record.to_dict() for record in bank_records.values()],
        "history": list(history),
    }


def run(args: argparse.Namespace) -> Path:
    all_instances = load_benchmark(
        Path(args.instance_zip),
        Path(args.certificate_zip),
    )
    instances = select_instances(all_instances, "evolution")
    if len(instances) != args.expected_instance_count:
        raise ValueError(
            f"V1.21 requires exactly {args.expected_instance_count} fixed "
            f"evolution instances, found {len(instances)}"
        )

    axes = capability_axes(instances)
    if tuple(axes) != ("retention_small", "retention_medium"):
        raise ValueError(f"Unexpected V1.21 capability axes: {axes}")

    instance_ids = tuple(item.instance_id for item in instances)
    instance_zip_sha = _sha256(Path(args.instance_zip))
    certificate_zip_sha = _sha256(Path(args.certificate_zip))

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoint.json"
    world_events_path = run_dir / "world_events.jsonl"
    round_history_path = run_dir / "round_history.jsonl"
    snapshots_dir = run_dir / "bank_snapshots"

    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Incompatible V1.21 checkpoint")
        immutable = {
            "worlds_per_round": args.worlds_per_round,
            "candidate_k": args.candidate_k,
            "seed": args.seed,
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
            raise ValueError("Requested rounds end before completed checkpoint")

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
        "V121_PROTOCOL "
        + json.dumps(
            {
                "worlds_per_round": args.worlds_per_round,
                "rounds": args.rounds,
                "fixed_instances_per_world": len(instances),
                "rollouts_per_round": args.worlds_per_round * len(instances),
                "planned_total_rollouts": (
                    args.worlds_per_round * len(instances) * args.rounds
                ),
                "candidate_k": args.candidate_k,
                "axes": list(axes),
                "instance_summary": describe_benchmark(instances),
                "gene_bank_admission": "external_axes_pareto_only",
                "parent_sampling": "equal_axis_total_score_squared",
                "parent_child_delta": "diagnostic_only",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

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

        candidate_records: list[BankRecord] = []
        candidate_assessments: list[GeneAssessment] = []
        parent_records: list[BankRecord | None] = []

        for world_index in range(args.worlds_per_round):
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

            assessment = evaluate_gene(
                gene,
                instances,
                candidate_k=args.candidate_k,
            )
            record = _record(
                gene,
                assessment,
                round_index=round_index,
                origin=origin,
                parent_id=None if parent is None else parent.record_id,
            )

            candidate_records.append(record)
            candidate_assessments.append(assessment)
            parent_records.append(parent)
            _append_jsonl(
                world_events_path,
                _world_event(
                    round_index=round_index,
                    world_index=world_index,
                    record=record,
                    assessment=assessment,
                    parent=parent,
                    instance_count=len(instances),
                ),
            )

            if (
                args.progress_every > 0
                and (world_index + 1) % args.progress_every == 0
            ):
                print(
                    f"V121_WORLD round={round_index} "
                    f"{world_index + 1}/{args.worlds_per_round} "
                    f"rollouts={(world_index + 1) * len(instances)}",
                    flush=True,
                )

        # The Bank is updated only after all 1000 worlds of the round have
        # completed all 34 fixed benchmark instances.
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

        _append_jsonl(round_history_path, summary)
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
            "V121_ROUND "
            + json.dumps(summary, ensure_ascii=False),
            flush=True,
        )

    print(f"V121_RUN_DIR={run_dir}", flush=True)
    return run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-zip", default=str(INSTANCE_ZIP))
    p.add_argument("--certificate-zip", default=str(CERTIFICATE_ZIP))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worlds-per-round", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--expected-instance-count", type=int, default=34)
    p.add_argument("--candidate-k", type=int, default=32)
    p.add_argument("--initial-scale", type=float, default=0.35)
    p.add_argument("--mutation-sigma", type=float, default=0.12)
    p.add_argument("--mutation-rate", type=float, default=0.20)
    p.add_argument("--bank-max-size", type=int, default=128)
    p.add_argument("--pareto-epsilon", type=float, default=0.0025)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=121)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.worlds_per_round <= 0:
        raise ValueError("worlds_per_round must be positive")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    run(args)


if __name__ == "__main__":
    main()
