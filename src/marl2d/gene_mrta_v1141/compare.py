from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from marl2d.gene_mrta_v113.evolve import AXES
from marl2d.gene_mrta_v1141.recombination_gene import (
    RECOMBINATION_AXES,
)


def _load(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def compare(
    adaptive_path: Path,
    center_path: Path,
) -> dict[str, Any]:
    adaptive = _load(
        adaptive_path
    )
    center = _load(
        center_path
    )

    if adaptive.get(
        "recombination_mode"
    ) != "adaptive":
        raise ValueError(
            "Adaptive summary is not adaptive mode"
        )
    if center.get(
        "recombination_mode"
    ) != "center":
        raise ValueError(
            "Center summary is not center mode"
        )

    for key in (
        "bootstrap_v113_checkpoint",
        "scenario_bank",
        "seed",
    ):
        if adaptive.get(
            key
        ) != center.get(
            key
        ):
            raise ValueError(
                f"Paired-control mismatch for {key}"
            )

    adaptive_best = adaptive[
        "best_policy_axis_scores"
    ]
    center_best = center[
        "best_policy_axis_scores"
    ]
    policy_delta = {
        axis: float(
            adaptive_best[axis]
            - center_best[axis]
        )
        for axis in AXES
    }

    specialists: dict[
        str,
        dict[str, Any],
    ] = {}
    for axis in RECOMBINATION_AXES:
        record = adaptive[
            "recombination_axis_specialists"
        ][axis]
        gene = record[
            "gene"
        ]
        specialists[axis] = {
            "phenotype_id": gene[
                "phenotype_id"
            ],
            "active_term_count": gene[
                "active_term_count"
            ],
            "gates": gene[
                "gates"
            ],
            "masked_coefficients": gene[
                "masked_coefficients"
            ],
            "mutation_sigma": gene[
                "mutation_sigma"
            ],
            "equation": gene[
                "equation"
            ],
            "generated": record[
                "generated"
            ],
            "screen_selected": record[
                "screen_selected"
            ],
            "accepted": record[
                "accepted"
            ],
            "four_capability_accepted": record[
                "four_capability_accepted"
            ],
            "selected_evidence_score": record[
                "selected_evidence_score"
            ],
        }

    return {
        "comparison": (
            "v1141_adaptive_minus_center"
        ),
        "seed": adaptive[
            "seed"
        ],
        "bootstrap_v113_checkpoint": (
            adaptive[
                "bootstrap_v113_checkpoint"
            ]
        ),
        "scenario_bank": (
            adaptive[
                "scenario_bank"
            ]
        ),
        "policy_axes": list(
            AXES
        ),
        "adaptive_policy_best": (
            adaptive_best
        ),
        "center_policy_best": (
            center_best
        ),
        "adaptive_minus_center": (
            policy_delta
        ),
        "adaptive_specialists_are_mature": (
            adaptive[
                "recombination_specialists_are_mature"
            ]
        ),
        "adaptive_center_phenotype_count": (
            adaptive[
                "recombination_center_phenotype_count"
            ]
        ),
        "adaptive_pareto_size": len(
            adaptive[
                "recombination_pareto_front"
            ]
        ),
        "adaptive_recombination_specialists": (
            specialists
        ),
        "interpretation_guardrail": (
            "Single-seed paired development comparison only. "
            "Do not treat positive deltas as a generalization claim."
        ),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adaptive",
        required=True,
    )
    p.add_argument(
        "--center",
        required=True,
    )
    p.add_argument(
        "--output",
        default="",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = compare(
        Path(
            args.adaptive
        ),
        Path(
            args.center
        ),
    )

    if args.output:
        output = Path(
            args.output
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(
        "V1141_PAIRED_COMPARISON="
        + json.dumps(
            result,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
