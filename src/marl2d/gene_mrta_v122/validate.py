from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from marl2d.gene_mrta_v120.benchmark import load_benchmark
from marl2d.gene_mrta_v120.gene import ScalableRouteTailGene
from marl2d.gene_mrta_v120.rollout import rollout_gene
from marl2d.gene_mrta_v122.mps_batch import (
    mps_available,
    prepare_instance,
    rollout_gene_batch_mps,
)


DEFAULT_INSTANCE_IDS = (
    "mtsp51_3",
    "mtsp100_10",
    "kroa200_20",
    "att532_20",
)


def _genes(count: int, seed: int) -> list[ScalableRouteTailGene]:
    rng = np.random.default_rng(seed)
    result: list[ScalableRouteTailGene] = []
    for _ in range(count):
        base = ScalableRouteTailGene.random(
            rng,
            hidden_dim=8,
            scale=0.35,
        )
        result.append(ScalableRouteTailGene.from_gene(base))
    return result


def _by_id():
    return {
        item.instance_id: item
        for item in load_benchmark()
    }


def _cpu_rollouts(genes, instance, candidate_k):
    return [
        rollout_gene(
            gene,
            instance,
            candidate_k=candidate_k,
        )
        for gene in genes
    ]


def _compare(cpu, batch, instance):
    mismatches = []
    objective_abs_error = []
    for index, cpu_result in enumerate(cpu):
        batch_routes = batch.routes_for_world(
            index,
            instance.robot_count,
        )
        route_match = batch_routes == cpu_result.routes
        obj_error = abs(
            float(batch.objectives[index])
            - float(cpu_result.objective)
        )
        objective_abs_error.append(obj_error)
        if not route_match or obj_error > 1e-8:
            mismatches.append(
                {
                    "world": index,
                    "route_match": route_match,
                    "cpu_objective": float(cpu_result.objective),
                    "batch_objective": float(batch.objectives[index]),
                    "objective_abs_error": obj_error,
                }
            )
    return mismatches, objective_abs_error


def run_equivalence(args) -> int:
    if args.device == "mps" and not mps_available():
        raise RuntimeError("Apple MPS is not available")

    rows = _by_id()
    genes = _genes(args.genes, args.seed)
    total_mismatches = 0
    summaries = []

    for instance_id in args.instances:
        instance = rows[instance_id]
        cpu = _cpu_rollouts(
            genes,
            instance,
            args.candidate_k,
        )
        cache = prepare_instance(
            instance,
            device=args.device,
        )
        batch = rollout_gene_batch_mps(
            genes,
            cache,
            candidate_k=args.candidate_k,
        )
        mismatches, errors = _compare(
            cpu,
            batch,
            instance,
        )
        total_mismatches += len(mismatches)
        row = {
            "instance_id": instance_id,
            "edge_type": instance.edge_weight_type,
            "vertices": instance.vertex_count,
            "robots": instance.robot_count,
            "genes": len(genes),
            "route_mismatches": len(mismatches),
            "max_objective_abs_error": max(errors) if errors else 0.0,
            "examples": mismatches[:3],
        }
        summaries.append(row)
        print(
            "V122_EQUIV "
            + json.dumps(row, ensure_ascii=False),
            flush=True,
        )

    final = {
        "device": args.device,
        "instances": len(args.instances),
        "genes_per_instance": len(genes),
        "total_cases": len(args.instances) * len(genes),
        "route_mismatches": total_mismatches,
        "status": "PASS" if total_mismatches == 0 else "FAIL",
    }
    print(
        "V122_EQUIV_SUMMARY "
        + json.dumps(final, ensure_ascii=False),
        flush=True,
    )
    return 0 if total_mismatches == 0 else 3


def run_benchmark(args) -> int:
    if args.device == "mps" and not mps_available():
        raise RuntimeError("Apple MPS is not available")

    rows = _by_id()
    genes = _genes(args.genes, args.seed)

    for instance_id in args.instances:
        instance = rows[instance_id]
        cache = prepare_instance(
            instance,
            device=args.device,
        )

        start = time.perf_counter()
        cpu = _cpu_rollouts(
            genes,
            instance,
            args.candidate_k,
        )
        cpu_seconds = time.perf_counter() - start

        if args.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        batch = rollout_gene_batch_mps(
            genes,
            cache,
            candidate_k=args.candidate_k,
        )
        if args.device == "mps":
            torch.mps.synchronize()
        batch_seconds = time.perf_counter() - start

        mismatches, _ = _compare(
            cpu,
            batch,
            instance,
        )
        summary = {
            "instance_id": instance_id,
            "vertices": instance.vertex_count,
            "robots": instance.robot_count,
            "genes": len(genes),
            "cpu_seconds": cpu_seconds,
            "batch_seconds": batch_seconds,
            "speedup": (
                cpu_seconds / batch_seconds
                if batch_seconds > 0.0
                else None
            ),
            "route_mismatches": len(mismatches),
        }
        print(
            "V122_BENCH "
            + json.dumps(summary, ensure_ascii=False),
            flush=True,
        )
    return 0


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=("equivalence", "benchmark"),
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--genes", type=int, default=8)
    p.add_argument("--candidate-k", type=int, default=32)
    p.add_argument("--seed", type=int, default=122)
    p.add_argument(
        "--instances",
        nargs="+",
        default=list(DEFAULT_INSTANCE_IDS),
    )
    return p


def main():
    args = parser().parse_args()
    if args.mode == "equivalence":
        raise SystemExit(run_equivalence(args))
    raise SystemExit(run_benchmark(args))


if __name__ == "__main__":
    main()
