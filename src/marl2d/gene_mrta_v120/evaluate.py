from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bank import BankRecord
from .benchmark import (
    CERTIFICATE_ZIP,
    INSTANCE_ZIP,
    describe_benchmark,
    load_benchmark,
    select_instances,
)
from .capabilities import evaluate_gene
from .gene import ScalableRouteTailGene


def _gene(record: BankRecord) -> ScalableRouteTailGene:
    return ScalableRouteTailGene(
        np.asarray(record.vector_data, dtype=np.float64),
        hidden_dim=record.hidden_dim,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = Path(args.checkpoint)
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    records = {
        row["record_id"]: BankRecord.from_dict(row)
        for row in data["records"]
    }
    history = data.get("history", [])
    if not history:
        raise ValueError("Checkpoint has no training history")

    last = history[-1]
    selected: list[str] = []

    global_best = last.get("global_best")
    if global_best is not None:
        selected.append(str(global_best["record_id"]))

    for rid in (last.get("best_by_size") or {}).values():
        if rid is not None and str(rid) not in selected:
            selected.append(str(rid))

    if not selected:
        raise ValueError("Checkpoint has no formal Bank genes to evaluate")

    all_instances = load_benchmark(
        Path(args.instance_zip),
        Path(args.certificate_zip),
    )
    instances = select_instances(all_instances, args.split)
    candidate_k = int(data["candidate_k"])

    reports: dict[str, object] = {}
    for rid in selected:
        if rid not in records:
            raise ValueError(f"Gene {rid} is missing from checkpoint records")
        assessment = evaluate_gene(
            _gene(records[rid]),
            instances,
            candidate_k=candidate_k,
        )
        reports[rid] = {
            "training_record": records[rid].to_dict(),
            "evaluation": assessment.to_summary_dict(),
            "instances": [row.to_dict() for row in assessment.instances],
        }
        print(
            "V120_EVAL "
            + json.dumps(
                {
                    "record_id": rid,
                    "split": args.split,
                    **assessment.to_summary_dict(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "protocol": "v120_public_minmax_mtsp_protected_eval_v1",
        "checkpoint": str(checkpoint_path),
        "training_generation": int(data["generation"]),
        "candidate_k": candidate_k,
        "split": args.split,
        "benchmark": describe_benchmark(instances),
        "genes": reports,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"V120_EVAL_REPORT={output}", flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--split",
        choices=("validation", "protected_test", "all"),
        default="protected_test",
    )
    p.add_argument("--instance-zip", default=str(INSTANCE_ZIP))
    p.add_argument("--certificate-zip", default=str(CERTIFICATE_ZIP))
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
