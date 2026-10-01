from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig
from marl2d.gene_mrta_v17.direct_time_scale import (
    load_v16to_time_oracle,
)

from .direct_gene import ConsequenceAwareDirectGene
from .direct_time_scale import (
    _dedupe,
    direct_time_scores,
)


def _genes_from_checkpoint(
    checkpoint_path: Path,
) -> tuple[
    int,
    list[ConsequenceAwareDirectGene],
]:
    data = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8"
        )
    )

    next_generation = int(
        data["next_generation"]
    )

    hof = [
        ConsequenceAwareDirectGene.from_dict(
            item
        )
        for item in data["hof"]
    ]
    archive = [
        ConsequenceAwareDirectGene.from_dict(
            item
        )
        for item in data["archive"]
    ]

    candidates = _dedupe(
        hof + archive
    )
    if not candidates:
        raise RuntimeError(
            "Checkpoint contains no "
            "selected V1.8 candidates"
        )

    return (
        next_generation,
        candidates,
    )


def export_checkpoint(
    args: argparse.Namespace,
) -> Path:
    checkpoint_path = Path(
        args.checkpoint
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    config = EnvConfig()
    oracle_path = Path(
        args.oracle_dataset
    )

    (
        _train_worlds,
        _train_star,
        probe_worlds,
        probe_star,
        oracle_meta,
    ) = load_v16to_time_oracle(
        oracle_path,
        config,
    )

    (
        next_generation,
        candidates,
    ) = _genes_from_checkpoint(
        checkpoint_path
    )

    (
        ratios,
        scores,
    ) = direct_time_scores(
        candidates,
        probe_worlds,
        probe_star,
        config,
    )

    best_idx = int(
        np.argmax(scores)
    )
    best_gene = candidates[
        best_idx
    ]
    best_ratios = ratios[
        best_idx
    ]

    run_dir = (
        checkpoint_path.parent
    )
    output_path = (
        Path(args.output)
        if args.output
        else run_dir / "summary.json"
    )

    summary = {
        "experiment": (
            "gene_mrta_v18_"
            "consequence_direct_checkpoint_finalized"
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "completed_generations": (
            next_generation
        ),
        "selection_candidates": (
            "deduplicated HOF + archive only"
        ),
        "excluded_from_selection": (
            "checkpoint population because "
            "it is the unevaluated next generation"
        ),
        "environment_changed_from_v17": (
            False
        ),
        "decoder_changed_from_v17": (
            False
        ),
        "external_matching_optimizer": (
            False
        ),
        "milp_teaches_actions": False,
        "milp_role": (
            "external T* reference only"
        ),
        "oracle_dataset": str(
            oracle_path
        ),
        "oracle_train_worlds": len(
            oracle_meta["train"]
        ),
        "oracle_probe_worlds": len(
            oracle_meta["probe"]
        ),
        "policy": {
            "type": (
                "V1.8 consequence-aware "
                "autoregressive direct assignment"
            ),
            "observation_dim": 12,
            "hidden_dim": (
                best_gene.hidden_dim
            ),
            "parameter_count": (
                ConsequenceAwareDirectGene.parameter_count(
                    best_gene.hidden_dim
                )
            ),
        },
        "final": {
            "time_specialist": {
                "gene": (
                    best_gene.to_dict()
                ),
                "oracle_probe_retention_mean": float(
                    scores[
                        best_idx
                    ]
                ),
                "oracle_probe_retention_std": float(
                    np.std(
                        best_ratios,
                        ddof=1,
                    )
                )
                if len(best_ratios) > 1
                else 0.0,
                "oracle_probe_retention_min": float(
                    np.min(
                        best_ratios
                    )
                ),
                "oracle_probe_retention_max": float(
                    np.max(
                        best_ratios
                    )
                ),
                "candidate_count": len(
                    candidates
                ),
            }
        },
    }

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "V1.8-T CHECKPOINT FINALIZED"
    )
    print(
        f"CHECKPOINT={checkpoint_path}"
    )
    print(
        "COMPLETED_GENERATIONS="
        f"{next_generation}"
    )
    print(
        f"CANDIDATES={len(candidates)}"
    )
    print(
        "PROBE_RETENTION_MEAN="
        f"{100.0 * float(scores[best_idx]):.4f}%"
    )
    print(
        "PROBE_RETENTION_MIN="
        f"{100.0 * float(np.min(best_ratios)):.4f}%"
    )
    print(
        "PROBE_RETENTION_MAX="
        f"{100.0 * float(np.max(best_ratios)):.4f}%"
    )
    print(
        f"SUMMARY={output_path}"
    )

    return output_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        required=True,
    )
    p.add_argument(
        "--oracle-dataset",
        required=True,
    )
    p.add_argument(
        "--output",
        default="",
    )
    return p


def main() -> None:
    export_checkpoint(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
