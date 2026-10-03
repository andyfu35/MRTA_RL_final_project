from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v16t.env import EnvConfig
from marl2d.gene_mrta_v18.global_time_test import _load_v18
from marl2d.gene_mrta_v110.recombination import recombine
from marl2d.gene_mrta_v110.scenario_bank import load_scenario_bank
from marl2d.gene_mrta_v110.train import (
    AXES,
    GeneRecord,
    _active_parent_ids,
    _best_by_axis,
    _certified_capabilities,
    _evaluate_scores,
    _score_dict_at,
)

from .law import (
    LawCoefficients,
    adaptive_delta_child,
    sample_law_candidates,
)


EPS = 1e-12
BASELINE_METHODS = (
    "block_blend",
    "parameter_blend",
    "ties_delta",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def _atomic_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    temp.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temp.replace(path)


def _write_jsonl(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
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


def _load_checkpoint_records(
    path: Path,
) -> tuple[
    int,
    dict[str, GeneRecord],
]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    records = {
        str(item["record_id"]): (
            GeneRecord.from_dict(
                item
            )
        )
        for item in data["records"]
    }
    if not records:
        raise RuntimeError(
            "V1.10 checkpoint contains no Gene records"
        )
    return (
        int(
            data.get(
                "next_generation",
                -1,
            )
        ),
        records,
    )


def _quality_from_caps(
    record: GeneRecord,
    ceiling: dict[str, float],
    caps: tuple[str, ...],
) -> float:
    if not caps:
        return 0.0
    return float(
        np.clip(
            min(
                float(record.scores[axis])
                / max(
                    float(ceiling[axis]),
                    EPS,
                )
                for axis in caps
            ),
            0.0,
            1.0,
        )
    )


def _certified_map(
    records: dict[str, GeneRecord],
    active_ids: list[str],
    ceiling: dict[str, float],
    threshold: float,
) -> dict[str, tuple[str, ...]]:
    return {
        record_id: (
            _certified_capabilities(
                records[record_id],
                ceiling,
                threshold=threshold,
            )
        )
        for record_id in active_ids
    }


def _sample_complementary_pairs(
    *,
    records: dict[str, GeneRecord],
    active_ids: list[str],
    certified: dict[str, tuple[str, ...]],
    ceiling: dict[str, float],
    count: int,
    rng: np.random.Generator,
    q_power: float,
    uniform_fraction: float,
) -> list[tuple[str, str]]:
    candidates: list[
        tuple[str, str]
    ] = []
    weights: list[float] = []

    ordered = sorted(
        record_id
        for record_id in active_ids
        if certified.get(
            record_id,
            (),
        )
    )

    for i, first in enumerate(
        ordered
    ):
        caps_a = set(
            certified[first]
        )
        for second in ordered[
            i + 1 :
        ]:
            caps_b = set(
                certified[second]
            )

            # True complementary mating: each parent contributes at least
            # one currently certified capability not carried by the other.
            if not (
                caps_a - caps_b
                and caps_b - caps_a
            ):
                continue

            q_a = _quality_from_caps(
                records[first],
                ceiling,
                certified[first],
            )
            q_b = _quality_from_caps(
                records[second],
                ceiling,
                certified[second],
            )
            pair_q = min(
                q_a,
                q_b,
            )
            candidates.append(
                (
                    first,
                    second,
                )
            )
            weights.append(
                max(
                    pair_q,
                    1e-9,
                )
                ** q_power
            )

    if len(candidates) < count:
        raise RuntimeError(
            "Not enough distinct complementary parent pairs: "
            f"need {count}, found {len(candidates)}"
        )

    weighted = np.asarray(
        weights,
        dtype=np.float64,
    )
    weighted /= np.sum(
        weighted
    )
    uniform = np.full(
        len(candidates),
        1.0 / len(candidates),
        dtype=np.float64,
    )
    p = (
        (1.0 - uniform_fraction)
        * weighted
        + uniform_fraction
        * uniform
    )

    selected = rng.choice(
        len(candidates),
        size=count,
        replace=False,
        p=p,
    )
    selected = np.asarray(
        selected,
        dtype=np.int64,
    )
    selected = selected[
        rng.permutation(
            selected.size
        )
    ]
    return [
        candidates[int(idx)]
        for idx in selected
    ]


def _score_map(
    ids: list[str],
    records: dict[str, GeneRecord],
    worlds,
    stars: np.ndarray,
    config: EnvConfig,
) -> dict[
    str,
    dict[str, float],
]:
    unique = list(
        dict.fromkeys(
            ids
        )
    )
    genes = [
        records[
            record_id
        ].gene
        for record_id in unique
    ]
    scores = _evaluate_scores(
        genes,
        worlds,
        stars,
        config,
    )
    return {
        record_id: (
            _score_dict_at(
                scores,
                idx,
            )
        )
        for idx, record_id
        in enumerate(unique)
    }


def _ceiling_from_score_map(
    score_map: dict[
        str,
        dict[str, float],
    ],
) -> dict[str, float]:
    return {
        axis: max(
            float(scores[axis])
            for scores
            in score_map.values()
        )
        for axis in AXES
    }


def _dual_retention(
    *,
    child_scores: dict[str, float],
    parent_a_scores: dict[str, float],
    parent_b_scores: dict[str, float],
    caps_a: tuple[str, ...],
    caps_b: tuple[str, ...],
    ceiling: dict[str, float],
    threshold: float,
) -> dict[str, object]:
    set_a = set(caps_a)
    set_b = set(caps_b)
    required = tuple(
        sorted(
            set_a
            | set_b
        )
    )

    parent_retention: dict[
        str,
        float
    ] = {}
    ceiling_retention: dict[
        str,
        float
    ] = {}
    dual_retention: dict[
        str,
        float
    ] = {}

    for axis in required:
        baselines: list[float] = []
        if axis in set_a:
            baselines.append(
                float(
                    parent_a_scores[
                        axis
                    ]
                )
            )
        if axis in set_b:
            baselines.append(
                float(
                    parent_b_scores[
                        axis
                    ]
                )
            )
        parent_base = max(
            baselines
        )
        parent_ratio = (
            float(
                child_scores[axis]
            )
            / max(
                parent_base,
                EPS,
            )
        )
        ceiling_ratio = (
            float(
                child_scores[axis]
            )
            / max(
                float(
                    ceiling[axis]
                ),
                EPS,
            )
        )
        parent_retention[
            axis
        ] = parent_ratio
        ceiling_retention[
            axis
        ] = ceiling_ratio
        dual_retention[
            axis
        ] = min(
            parent_ratio,
            ceiling_ratio,
        )

    minimum = (
        min(
            dual_retention.values()
        )
        if dual_retention
        else 0.0
    )
    accepted = bool(
        dual_retention
    ) and all(
        value >= threshold
        for value
        in dual_retention.values()
    )

    return {
        "required_capabilities": list(
            required
        ),
        "parent_retention": (
            parent_retention
        ),
        "ceiling_retention": (
            ceiling_retention
        ),
        "dual_retention": (
            dual_retention
        ),
        "min_dual_retention": float(
            minimum
        ),
        "margin": float(
            minimum - threshold
        ),
        "accepted": accepted,
    }


def _evaluate_in_batches(
    genes,
    *,
    worlds,
    stars: np.ndarray,
    config: EnvConfig,
    batch_size: int,
    label: str,
) -> list[dict[str, float]]:
    result: list[
        dict[str, float]
    ] = []
    total = len(genes)
    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            total,
            start + batch_size,
        )
        scores = _evaluate_scores(
            genes[start:end],
            worlds,
            stars,
            config,
        )
        for local_idx in range(
            end - start
        ):
            result.append(
                _score_dict_at(
                    scores,
                    local_idx,
                )
            )
        print(
            f"{label} {end}/{total}",
            flush=True,
        )
    return result


def _law_statistics(
    law: LawCoefficients,
    rows: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["law_id"]
        == law.law_id
    ]
    values = np.asarray(
        [
            float(
                row[
                    "min_dual_retention"
                ]
            )
            for row in selected
        ],
        dtype=np.float64,
    )
    accepted = np.asarray(
        [
            bool(
                row["accepted"]
            )
            for row in selected
        ],
        dtype=bool,
    )
    four = [
        row
        for row in selected
        if len(
            row[
                "required_capabilities"
            ]
        )
        == 4
    ]
    four_acceptance = (
        float(
            np.mean(
                [
                    bool(
                        row["accepted"]
                    )
                    for row in four
                ]
            )
        )
        if four
        else None
    )
    return {
        "law_id": law.law_id,
        "coefficients": (
            law.to_dict()
        ),
        "equation": (
            law.equation()
        ),
        "n": int(
            len(selected)
        ),
        "success_rate": float(
            np.mean(
                accepted
            )
        ),
        "mean_min_dual_retention": float(
            np.mean(
                values
            )
        ),
        "p10_min_dual_retention": float(
            np.quantile(
                values,
                0.10,
            )
        ),
        "median_min_dual_retention": float(
            np.median(
                values
            )
        ),
        "four_capability_n": int(
            len(four)
        ),
        "four_capability_success_rate": (
            four_acceptance
        ),
    }


def _law_rank_key(
    row: dict[str, object],
) -> tuple[
    float,
    float,
    float,
    float,
    str,
]:
    four_rate = row[
        "four_capability_success_rate"
    ]
    primary = (
        float(four_rate)
        if four_rate is not None
        else float(
            row[
                "success_rate"
            ]
        )
    )
    return (
        primary,
        float(
            row[
                "success_rate"
            ]
        ),
        float(
            row[
                "p10_min_dual_retention"
            ]
        ),
        float(
            row[
                "mean_min_dual_retention"
            ]
        ),
        str(
            row["law_id"]
        ),
    )


def _stable_seed(
    base_seed: int,
    pair_id: str,
    method: str,
) -> int:
    digest = hashlib.sha256(
        (
            f"{base_seed}|"
            f"{pair_id}|"
            f"{method}"
        ).encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    )


def _summarize_validation(
    rows: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    methods = sorted(
        set(
            str(row["method"])
            for row in rows
        )
    )
    result: dict[
        str,
        object
    ] = {}
    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"]
            == method
        ]
        values = np.asarray(
            [
                float(
                    row[
                        "min_dual_retention"
                    ]
                )
                for row in subset
            ],
            dtype=np.float64,
        )
        accepted = np.asarray(
            [
                bool(
                    row["accepted"]
                )
                for row in subset
            ],
            dtype=bool,
        )
        four = [
            row
            for row in subset
            if len(
                row[
                    "required_capabilities"
                ]
            )
            == 4
        ]
        axis_means = {
            axis: float(
                np.mean(
                    [
                        float(
                            row[
                                "scores"
                            ][axis]
                        )
                        for row
                        in subset
                    ]
                )
            )
            for axis in AXES
        }
        result[method] = {
            "n": len(subset),
            "acceptance_rate": float(
                np.mean(
                    accepted
                )
            ),
            "mean_min_dual_retention": float(
                np.mean(
                    values
                )
            ),
            "p10_min_dual_retention": float(
                np.quantile(
                    values,
                    0.10,
                )
            ),
            "median_min_dual_retention": float(
                np.median(
                    values
                )
            ),
            "four_capability_n": len(
                four
            ),
            "four_capability_acceptance_rate": (
                float(
                    np.mean(
                        [
                            bool(
                                row[
                                    "accepted"
                                ]
                            )
                            for row in four
                        ]
                    )
                )
                if four
                else None
            ),
            "axis_mean_scores": (
                axis_means
            ),
        }
    return result


def _paired_validation(
    rows: list[
        dict[str, object]
    ],
    selected_method: str,
) -> dict[str, object]:
    by_method: dict[
        str,
        dict[
            str,
            dict[str, object],
        ],
    ] = {}
    for row in rows:
        by_method.setdefault(
            str(
                row["method"]
            ),
            {},
        )[
            str(
                row["pair_id"]
            )
        ] = row

    target = by_method[
        selected_method
    ]
    result: dict[
        str,
        object
    ] = {}

    for baseline in BASELINE_METHODS:
        other = by_method[
            baseline
        ]
        common = sorted(
            set(target)
            & set(other)
        )
        deltas = np.asarray(
            [
                float(
                    target[pair_id][
                        "min_dual_retention"
                    ]
                )
                - float(
                    other[pair_id][
                        "min_dual_retention"
                    ]
                )
                for pair_id in common
            ],
            dtype=np.float64,
        )
        wins = int(
            np.sum(
                deltas > 1e-12
            )
        )
        losses = int(
            np.sum(
                deltas < -1e-12
            )
        )
        ties = int(
            deltas.size
            - wins
            - losses
        )
        result[baseline] = {
            "n": int(
                deltas.size
            ),
            "mean_delta_min_dual_retention": float(
                np.mean(
                    deltas
                )
            ),
            "median_delta_min_dual_retention": float(
                np.median(
                    deltas
                )
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    return result


def run_discovery(
    args: argparse.Namespace,
) -> Path:
    if not (
        0 < args.train_pairs
        < args.pairs
    ):
        raise ValueError(
            "train-pairs must be between 1 and pairs-1"
        )
    if not (
        1 <= args.screen_worlds
        <= 100
    ):
        raise ValueError(
            "screen-worlds must be in [1,100]"
        )

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
            "V1.11 requires the frozen 100-world V1.10 development bank"
        )

    checkpoint_generation, records = (
        _load_checkpoint_records(
            Path(
                args.v110_checkpoint
            )
        )
    )
    anchor = _load_v18(
        Path(
            args.anchor_v18_run
        )
    )

    active_ids, _archives = (
        _active_parent_ids(
            records,
            args.archive_size_per_axis,
            args.hybrid_bank_limit,
        )
    )
    full_ceiling = _best_by_axis(
        records,
        active_ids,
    )
    certified = _certified_map(
        records,
        active_ids,
        full_ceiling,
        args.inheritance_threshold,
    )

    rng = np.random.default_rng(
        args.seed
    )
    pairs = _sample_complementary_pairs(
        records=records,
        active_ids=active_ids,
        certified=certified,
        ceiling=full_ceiling,
        count=args.pairs,
        rng=rng,
        q_power=args.parent_q_power,
        uniform_fraction=(
            args.parent_uniform_fraction
        ),
    )

    train_pairs = pairs[
        : args.train_pairs
    ]
    validation_pairs = pairs[
        args.train_pairs :
    ]

    screen_rng = np.random.default_rng(
        args.seed + 111
    )
    screen_indices = np.sort(
        screen_rng.choice(
            len(worlds),
            size=args.screen_worlds,
            replace=False,
        )
    )
    screen_worlds = [
        worlds[
            int(idx)
        ]
        for idx in screen_indices
    ]
    screen_stars = stars[
        screen_indices
    ]

    # Screen ceilings are recomputed from the fixed active parent pool.
    active_screen_scores = (
        _score_map(
            active_ids,
            records,
            screen_worlds,
            screen_stars,
            config,
        )
    )
    screen_ceiling = (
        _ceiling_from_score_map(
            active_screen_scores
        )
    )

    laws = sample_law_candidates(
        args.laws,
        args.seed + 211,
    )

    train_genes = []
    train_manifest: list[
        dict[str, object]
    ] = []

    for pair_index, (
        parent_a_id,
        parent_b_id,
    ) in enumerate(
        train_pairs
    ):
        parent_a = records[
            parent_a_id
        ]
        parent_b = records[
            parent_b_id
        ]
        caps_a = certified[
            parent_a_id
        ]
        caps_b = certified[
            parent_b_id
        ]
        q_a = _quality_from_caps(
            parent_a,
            full_ceiling,
            caps_a,
        )
        q_b = _quality_from_caps(
            parent_b,
            full_ceiling,
            caps_b,
        )
        pair_id = (
            f"train_{pair_index:03d}_"
            f"{parent_a_id[:8]}_"
            f"{parent_b_id[:8]}"
        )

        for law in laws:
            gene, metadata = (
                adaptive_delta_child(
                    parent_a.gene,
                    parent_b.gene,
                    anchor,
                    law,
                    quality_a=q_a,
                    quality_b=q_b,
                    capability_count_a=(
                        len(caps_a)
                    ),
                    capability_count_b=(
                        len(caps_b)
                    ),
                )
            )
            train_genes.append(
                gene
            )
            train_manifest.append(
                {
                    "split": "discovery",
                    "pair_id": pair_id,
                    "parent_a": (
                        parent_a_id
                    ),
                    "parent_b": (
                        parent_b_id
                    ),
                    "parent_a_capabilities": list(
                        caps_a
                    ),
                    "parent_b_capabilities": list(
                        caps_b
                    ),
                    "quality_a": q_a,
                    "quality_b": q_b,
                    "law_id": (
                        law.law_id
                    ),
                    "law_metadata": (
                        metadata
                    ),
                }
            )

    train_scores = (
        _evaluate_in_batches(
            train_genes,
            worlds=screen_worlds,
            stars=screen_stars,
            config=config,
            batch_size=args.batch_size,
            label="DISCOVERY_SCREEN",
        )
    )

    screen_rows: list[
        dict[str, object]
    ] = []
    for manifest, child_scores in zip(
        train_manifest,
        train_scores,
    ):
        parent_a_id = str(
            manifest[
                "parent_a"
            ]
        )
        parent_b_id = str(
            manifest[
                "parent_b"
            ]
        )
        retention = _dual_retention(
            child_scores=(
                child_scores
            ),
            parent_a_scores=(
                active_screen_scores[
                    parent_a_id
                ]
            ),
            parent_b_scores=(
                active_screen_scores[
                    parent_b_id
                ]
            ),
            caps_a=certified[
                parent_a_id
            ],
            caps_b=certified[
                parent_b_id
            ],
            ceiling=screen_ceiling,
            threshold=(
                args.inheritance_threshold
            ),
        )
        screen_rows.append(
            {
                **manifest,
                "scores": (
                    child_scores
                ),
                **retention,
            }
        )

    law_summaries = [
        _law_statistics(
            law,
            screen_rows,
        )
        for law in laws
    ]
    law_summaries.sort(
        key=_law_rank_key,
        reverse=True,
    )
    selected_law_id = str(
        law_summaries[0][
            "law_id"
        ]
    )
    law_map = {
        law.law_id: law
        for law in laws
    }
    selected_law = law_map[
        selected_law_id
    ]

    print(
        "DISCOVERY_SELECTED="
        f"{selected_law_id} "
        f"success="
        f"{law_summaries[0]['success_rate']:.4f} "
        f"p10="
        f"{law_summaries[0]['p10_min_dual_retention']:.4f}",
        flush=True,
    )

    # Freeze the discovered law before evaluating held-out parent pairs.
    validation_genes = []
    validation_manifest: list[
        dict[str, object]
    ] = []

    for pair_index, (
        parent_a_id,
        parent_b_id,
    ) in enumerate(
        validation_pairs
    ):
        parent_a = records[
            parent_a_id
        ]
        parent_b = records[
            parent_b_id
        ]
        caps_a = certified[
            parent_a_id
        ]
        caps_b = certified[
            parent_b_id
        ]
        q_a = _quality_from_caps(
            parent_a,
            full_ceiling,
            caps_a,
        )
        q_b = _quality_from_caps(
            parent_b,
            full_ceiling,
            caps_b,
        )
        pair_id = (
            f"validation_{pair_index:03d}_"
            f"{parent_a_id[:8]}_"
            f"{parent_b_id[:8]}"
        )

        discovered_gene, discovered_metadata = (
            adaptive_delta_child(
                parent_a.gene,
                parent_b.gene,
                anchor,
                selected_law,
                quality_a=q_a,
                quality_b=q_b,
                capability_count_a=(
                    len(caps_a)
                ),
                capability_count_b=(
                    len(caps_b)
                ),
            )
        )
        validation_genes.append(
            discovered_gene
        )
        validation_manifest.append(
            {
                "split": "validation",
                "pair_id": pair_id,
                "parent_a": parent_a_id,
                "parent_b": parent_b_id,
                "parent_a_capabilities": list(
                    caps_a
                ),
                "parent_b_capabilities": list(
                    caps_b
                ),
                "method": "discovered_law",
                "law_id": (
                    selected_law.law_id
                ),
                "operator_metadata": (
                    discovered_metadata
                ),
            }
        )

        for method in BASELINE_METHODS:
            baseline_rng = np.random.default_rng(
                _stable_seed(
                    args.seed,
                    pair_id,
                    method,
                )
            )
            baseline = recombine(
                parent_a.gene,
                parent_b.gene,
                anchor,
                baseline_rng,
                operator=method,
            )
            validation_genes.append(
                baseline.gene
            )
            validation_manifest.append(
                {
                    "split": "validation",
                    "pair_id": pair_id,
                    "parent_a": (
                        parent_a_id
                    ),
                    "parent_b": (
                        parent_b_id
                    ),
                    "parent_a_capabilities": list(
                        caps_a
                    ),
                    "parent_b_capabilities": list(
                        caps_b
                    ),
                    "method": method,
                    "law_id": None,
                    "operator_metadata": (
                        baseline.metadata
                    ),
                }
            )

    validation_scores = (
        _evaluate_in_batches(
            validation_genes,
            worlds=worlds,
            stars=stars,
            config=config,
            batch_size=args.batch_size,
            label="VALIDATION_FULL100",
        )
    )

    validation_rows: list[
        dict[str, object]
    ] = []
    for manifest, child_scores in zip(
        validation_manifest,
        validation_scores,
    ):
        parent_a_id = str(
            manifest[
                "parent_a"
            ]
        )
        parent_b_id = str(
            manifest[
                "parent_b"
            ]
        )
        retention = _dual_retention(
            child_scores=(
                child_scores
            ),
            parent_a_scores=(
                records[
                    parent_a_id
                ].scores
            ),
            parent_b_scores=(
                records[
                    parent_b_id
                ].scores
            ),
            caps_a=certified[
                parent_a_id
            ],
            caps_b=certified[
                parent_b_id
            ],
            ceiling=full_ceiling,
            threshold=(
                args.inheritance_threshold
            ),
        )
        validation_rows.append(
            {
                **manifest,
                "scores": (
                    child_scores
                ),
                **retention,
            }
        )

    validation_summary = (
        _summarize_validation(
            validation_rows
        )
    )
    paired = _paired_validation(
        validation_rows,
        "discovered_law",
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        Path(
            args.output_dir
        )
        / (
            "gene_mrta_v111_law_"
            f"{stamp}_seed{args.seed}"
        )
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_manifest = {
        "checkpoint": (
            args.v110_checkpoint
        ),
        "checkpoint_next_generation": (
            checkpoint_generation
        ),
        "active_parent_count": len(
            active_ids
        ),
        "pairs": [
            {
                "split": (
                    "discovery"
                    if idx
                    < args.train_pairs
                    else "validation"
                ),
                "parent_a": pair[0],
                "parent_b": pair[1],
                "parent_a_capabilities": list(
                    certified[
                        pair[0]
                    ]
                ),
                "parent_b_capabilities": list(
                    certified[
                        pair[1]
                    ]
                ),
            }
            for idx, pair
            in enumerate(pairs)
        ],
    }

    selected_payload = {
        "law": (
            selected_law.to_dict()
        ),
        "equation": (
            selected_law.equation()
        ),
        "selection_basis": (
            "discovery parent pairs on frozen "
            f"{args.screen_worlds}/100 development worlds only"
        ),
        "discovery_statistics": (
            law_summaries[0]
        ),
    }

    summary = {
        "experiment": (
            "gene_mrta_v111_empirical_recombination_law_discovery"
        ),
        "policy_architecture_changed": False,
        "oracle_action_teacher": False,
        "scenario_bank": (
            args.scenario_bank
        ),
        "scenario_seeds": (
            scenario_seeds
        ),
        "source_v110_checkpoint": (
            args.v110_checkpoint
        ),
        "source_checkpoint_next_generation": (
            checkpoint_generation
        ),
        "parent_pair_protocol": {
            "total_pairs": args.pairs,
            "discovery_pairs": (
                args.train_pairs
            ),
            "validation_pairs": (
                args.pairs
                - args.train_pairs
            ),
            "currently_certified_only": True,
            "true_complementarity_required": True,
            "q_power": (
                args.parent_q_power
            ),
            "uniform_fraction": (
                args.parent_uniform_fraction
            ),
        },
        "discovery": {
            "law_candidates": (
                args.laws
            ),
            "screen_world_count": (
                args.screen_worlds
            ),
            "screen_world_indices": (
                screen_indices.tolist()
            ),
            "selected_law_id": (
                selected_law_id
            ),
            "selected_law": (
                selected_law.to_dict()
            ),
            "selected_equation": (
                selected_law.equation()
            ),
            "top_laws": (
                law_summaries[:10]
            ),
        },
        "validation": {
            "held_out_from_formula_selection": True,
            "world_count": 100,
            "methods": (
                validation_summary
            ),
            "paired_against_discovered_law": (
                paired
            ),
        },
        "interpretation_guardrail": (
            "This pilot discovers coefficients on 95M development worlds "
            "and held-out parent pairs. It does not touch the protected "
            "99M final benchmark and does not teach actions from the MILP."
        ),
    }

    _atomic_json(
        run_dir
        / "pair_manifest.json",
        pair_manifest,
    )
    _atomic_json(
        run_dir
        / "formula_candidates.json",
        {
            "laws": [
                law.to_dict()
                for law in laws
            ],
            "ranked_discovery_statistics": (
                law_summaries
            ),
        },
    )
    _atomic_json(
        run_dir
        / "selected_formula.json",
        selected_payload,
    )
    _write_jsonl(
        run_dir
        / "recombination_dataset.jsonl",
        screen_rows,
    )
    _write_jsonl(
        run_dir
        / "validation_results.jsonl",
        validation_rows,
    )
    _atomic_json(
        run_dir
        / "summary.json",
        summary,
    )

    print(
        "\nFINAL V1.11 LAW DISCOVERY"
    )
    print(
        "SELECTED_FORMULA="
        + json.dumps(
            _jsonable(
                selected_payload
            ),
            ensure_ascii=False,
        )
    )
    print(
        "VALIDATION="
        + json.dumps(
            _jsonable(
                validation_summary
            ),
            ensure_ascii=False,
        )
    )
    print(
        "PAIRED="
        + json.dumps(
            _jsonable(
                paired
            ),
            ensure_ascii=False,
        )
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
        "--v110-checkpoint",
        required=True,
    )
    p.add_argument(
        "--anchor-v18-run",
        required=True,
    )
    p.add_argument(
        "--pairs",
        type=int,
        default=64,
    )
    p.add_argument(
        "--train-pairs",
        type=int,
        default=48,
    )
    p.add_argument(
        "--laws",
        type=int,
        default=32,
    )
    p.add_argument(
        "--screen-worlds",
        type=int,
        default=25,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
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
        "--inheritance-threshold",
        type=float,
        default=0.95,
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
            "gene_mrta_v111_law_discovery"
        ),
    )
    return p


def main() -> None:
    run_discovery(
        parser().parse_args()
    )


if __name__ == "__main__":
    main()
