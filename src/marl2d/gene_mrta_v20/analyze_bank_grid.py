from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bank import BankRecord, rebuild_bank
from .gene import SetAssignmentGene


CANDIDATES = {
    "fine": {
        "total_time": 0.25,
        "priority": 0.05,
        "on_time_completed_tasks": 0.25,
    },
    "default": {
        "total_time": 0.50,
        "priority": 0.10,
        "on_time_completed_tasks": 0.50,
    },
    "coarse": {
        "total_time": 1.00,
        "priority": 0.20,
        "on_time_completed_tasks": 1.00,
    },
}


def _record(row: dict[str, object]) -> BankRecord:
    raw_scores = dict(row["scores"])
    if (
        "on_time_completed_tasks"
        not in raw_scores
        and "completed_tasks"
        in raw_scores
    ):
        raw_scores[
            "on_time_completed_tasks"
        ] = raw_scores.pop(
            "completed_tasks"
        )
    return BankRecord(
        record_id=str(row["record_id"]),
        gene=SetAssignmentGene.from_dict(
            row["gene"]
        ),
        scores={
            str(k): float(v)
            for k, v in raw_scores.items()
        },
        round_index=int(
            row["round_index"]
        ),
        parent_id=(
            None
            if row.get("parent_id") is None
            else str(row["parent_id"])
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/gene_mrta_v20/"
            "fixed100_50r_seed200/"
            "checkpoint.json"
        ),
    )
    args = parser.parse_args()

    path = Path(args.checkpoint)
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    records = [
        _record(row)
        for row in data["bank_records"]
    ]
    print(
        f"V20_GRID_SOURCE={path}"
    )
    print(
        f"V20_GRID_ORIGINAL_BANK={len(records)}"
    )
    for name, resolutions in CANDIDATES.items():
        reduced = rebuild_bank(
            records,
            resolutions=resolutions,
        )
        print(
            "V20_GRID "
            + json.dumps(
                {
                    "name": name,
                    "resolutions": resolutions,
                    "bank_size": len(reduced),
                    "reduction_fraction": (
                        1.0
                        - len(reduced)
                        / max(len(records), 1)
                    ),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
