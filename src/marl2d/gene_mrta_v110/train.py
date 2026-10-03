from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from marl2d.gene_mrta_v17.direct_time_scale import _mutation_sigma
from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v18.global_time_test import _load_v18
from marl2d.gene_mrta_v19.robust_metrics import evaluate_robust_population

from .recombination import OPERATORS, RecombinationResult, offspring_family
from .scenario_bank import load_scenario_bank


AXES = (
    "mean_time",
    "tail10_time",
    "continuation_preservation",
    "fleet_option_reserve",
)

V19_AXIS_MAP = {
    "mean_time": "mean_time",
    "hard_world_time": "tail10_time",
    "continuation_preservation": "continuation_preservation",
    "fleet_option_reserve": "fleet_option_reserve",
}


@dataclass
class GeneRecord:
    record_id: str
    gene: ConsequenceAwareDirectGene
    capabilities: tuple[str, ...]
    scores: dict[str, float]
    origin: str
    generation: int
    parents: tuple[str, ...] = ()
    operator: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "gene": self.gene.to_dict(),
            "capabilities": list(self.capabilities),
            "scores": {
                key: float(value)
                for key, value in self.scores.items()
            },
            "origin": self.origin,
            "generation": self.generation,
            "parents": list(self.parents),
            "operator": self.operator,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "GeneRecord":
        return cls(
            record_id=str(data["record_id"]),
            gene=ConsequenceAwareDirectGene.from_dict(
                data["gene"]
            ),
            capabilities=tuple(
                str(x)
                for x in data["capabilities"]
            ),
            scores={
                str(k): float(v)
                for k, v in data["scores"].items()
            },
            origin=str(data["origin"]),
            generation=int(data["generation"]),
            parents=tuple(
                str(x)
                for x in data.get("parents", [])
            ),
            operator=(
                None
                if data.get("operator") is None
                else str(data["operator"])
            ),
            metadata=dict(
                data.get("metadata", {})
            ),
        )


@dataclass(frozen=True)
class MatingChild:
    result: RecombinationResult
    parent_a: str
    parent_b: str
    required_capabilities: tuple[str, ...]


def _gene_id(
    gene: ConsequenceAwareDirectGene,
) -> str:
    digest = hashlib.sha256(
        np.asarray(
            gene.vector_data,
            dtype=np.float64,
        ).tobytes()
    ).hexdigest()
    return digest[:20]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]
    return value


def _axis_scores(
    tensor: np.ndarray,
    stars: np.ndarray,
) -> dict[str, np.ndarray]:
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(
            "Expected metric tensor [gene, world, 3]"
        )
    if tensor.shape[1] != stars.size:
        raise ValueError(
            "World count and T* count do not match"
        )

    ratios = np.clip(
        tensor[:, :, 0]
        / stars[None, :],
        0.0,
        1.0,
    )
    tail_count = max(
        1,
        int(
            np.ceil(
                0.10 * stars.size
            )
        ),
    )
    sorted_ratios = np.sort(
        ratios,
        axis=1,
    )

    return {
        "mean_time": np.mean(
            ratios,
            axis=1,
        ),
        "tail10_time": np.mean(
            sorted_ratios[
                :, :tail_count
            ],
            axis=1,
        ),
        "continuation_preservation": np.mean(
            tensor[:, :, 1],
            axis=1,
        ),
        "fleet_option_reserve": np.mean(
            tensor[:, :, 2],
            axis=1,
        ),
    }


def _evaluate_scores(
    genes: list[ConsequenceAwareDirectGene],
    worlds,
    stars: np.ndarray,
    config,
) -> dict[str, np.ndarray]:
    if not genes:
        return {
            axis: np.zeros(
                0,
                dtype=np.float64,
            )
            for axis in AXES
        }
    tensor = evaluate_robust_population(
        genes,
        worlds,
        config,
    )
    return _axis_scores(
        tensor,
        stars,
    )


def _score_dict_at(
    scores: dict[str, np.ndarray],
    idx: int,
) -> dict[str, float]:
    return {
        axis: float(
            scores[axis][idx]
        )
        for axis in AXES
    }


def _best_by_axis(
    records: dict[str, GeneRecord],
    active_ids: list[str] | None = None,
) -> dict[str, float]:
    ids = (
        active_ids
        if active_ids is not None
        else list(records)
    )
    best = {
        axis: 1e-12
        for axis in AXES
    }
    for record_id in ids:
        record = records[record_id]
        for axis in AXES:
            best[axis] = max(
                best[axis],
                float(
                    record.scores[axis]
                ),
            )
    return best


def _quality_weight(
    record: GeneRecord,
    best: dict[str, float],
) -> float:
    if not record.capabilities:
        return 0.0
    retention = [
        float(
            record.scores[axis]
        )
        / max(
            float(best[axis]),
            1e-12,
        )
        for axis in record.capabilities
    ]
    return float(
        np.clip(
            min(retention),
            0.0,
            1.0,
        )
    )


def _sample_record_id(
    candidate_ids: list[str],
    records: dict[str, GeneRecord],
    best: dict[str, float],
    rng: np.random.Generator,
    *,
    power: float,
    uniform_fraction: float,
) -> str:
    if not candidate_ids:
        raise ValueError(
            "No parent candidates"
        )

    q = np.asarray(
        [
            _quality_weight(
                records[record_id],
                best,
            )
            for record_id
            in candidate_ids
        ],
        dtype=np.float64,
    )
    weighted = np.power(
        np.maximum(
            q,
            1e-9,
        ),
        power,
    )
    weighted /= np.sum(weighted)
    uniform = np.full(
        len(candidate_ids),
        1.0 / len(candidate_ids),
        dtype=np.float64,
    )
    p = (
        (1.0 - uniform_fraction)
        * weighted
        + uniform_fraction
        * uniform
    )
    idx = int(
        rng.choice(
            len(candidate_ids),
            p=p,
        )
    )
    return candidate_ids[idx]


def _complementary_partner_ids(
    first_id: str,
    active_ids: list[str],
    records: dict[str, GeneRecord],
) -> list[str]:
    first_caps = set(
        records[first_id].capabilities
    )
    distinct = [
        record_id
        for record_id in active_ids
        if record_id != first_id
    ]
    complementary = [
        record_id
        for record_id in distinct
        if (
            set(
                records[
                    record_id
                ].capabilities
            )
            - first_caps
        )
    ]
    return (
        complementary
        if complementary
        else distinct
    )


def _specialist_archives(
    records: dict[str, GeneRecord],
    size_per_axis: int,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for axis in AXES:
        ordered = sorted(
            records,
            key=lambda record_id: (
                records[
                    record_id
                ].scores[axis]
            ),
            reverse=True,
        )
        result[axis] = ordered[
            :size_per_axis
        ]
    return result


def _hybrid_ids(
    records: dict[str, GeneRecord],
    best: dict[str, float],
    limit: int,
) -> list[str]:
    ids = [
        record_id
        for record_id, record
        in records.items()
        if record.origin == "mating"
        and len(
            record.capabilities
        ) >= 2
    ]
    ids.sort(
        key=lambda record_id: (
            len(
                records[
                    record_id
                ].capabilities
            ),
            _quality_weight(
                records[record_id],
                best,
            ),
        ),
        reverse=True,
    )
    return ids[:limit]


def _active_parent_ids(
    records: dict[str, GeneRecord],
    specialist_size: int,
    hybrid_limit: int,
) -> tuple[
    list[str],
    dict[str, list[str]],
]:
    archives = _specialist_archives(
        records,
        specialist_size,
    )
    base_ids: list[str] = []
    seen: set[str] = set()
    for axis in AXES:
        for record_id in archives[axis]:
            if record_id not in seen:
                seen.add(record_id)
                base_ids.append(record_id)

    best = _best_by_axis(
        records,
        base_ids,
    )
    for record_id in _hybrid_ids(
        records,
        best,
        hybrid_limit,
    ):
        if record_id not in seen:
            seen.add(record_id)
            base_ids.append(record_id)

    return base_ids, archives


def _bootstrap_from_v19_checkpoint(
    checkpoint: Path,
) -> tuple[
    dict[str, ConsequenceAwareDirectGene],
    dict[str, set[str]],
]:
    data = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )
    genes: dict[
        str,
        ConsequenceAwareDirectGene
    ] = {}
    caps: dict[
        str,
        set[str]
    ] = {}

    for bank_name in (
        "archives",
        "hof",
    ):
        banks = data.get(
            bank_name,
            {},
        )
        for old_axis, items in banks.items():
            if old_axis not in V19_AXIS_MAP:
                continue
            axis = V19_AXIS_MAP[
                old_axis
            ]
            for item in items:
                gene = (
                    ConsequenceAwareDirectGene.from_dict(
                        item
                    )
                )
                record_id = _gene_id(
                    gene
                )
                genes[record_id] = gene
                caps.setdefault(
                    record_id,
                    set(),
                ).add(axis)

    if not genes:
        raise RuntimeError(
            "No V1.9 genes found in checkpoint"
        )
    return genes, caps


def _make_initial_records(
    checkpoint: Path,
    worlds,
    stars: np.ndarray,
    config,
) -> dict[str, GeneRecord]:
    genes, caps = (
        _bootstrap_from_v19_checkpoint(
            checkpoint
        )
    )
    ids = list(genes)
    gene_list = [
        genes[record_id]
        for record_id in ids
    ]
    scores = _evaluate_scores(
        gene_list,
        worlds,
        stars,
        config,
    )

    records: dict[
        str,
        GeneRecord
    ] = {}
    for idx, record_id in enumerate(ids):
        records[record_id] = GeneRecord(
            record_id=record_id,
            gene=genes[record_id],
            capabilities=tuple(
                sorted(
                    caps[record_id]
                )
            ),
            scores=_score_dict_at(
                scores,
                idx,
            ),
            origin="v19_bootstrap",
            generation=-1,
            metadata={
                "source_checkpoint": str(
                    checkpoint
                ),
            },
        )
    return records


def _screen_indices(
    generation: int,
    total_worlds: int,
    screen_worlds: int,
) -> np.ndarray:
    if total_worlds % screen_worlds == 0:
        groups = (
            total_worlds
            // screen_worlds
        )
        group = generation % groups
        start = group * screen_worlds
        return np.arange(
            start,
            start + screen_worlds,
            dtype=np.int64,
        )

    rng = np.random.default_rng(
        100_000 + generation
    )
    return np.sort(
        rng.choice(
            total_worlds,
            size=screen_worlds,
            replace=False,
        )
    )


def _normal_children(
    *,
    count: int,
    active_ids: list[str],
    records: dict[str, GeneRecord],
    best: dict[str, float],
    rng: np.random.Generator,
    sigma: float,
    mutation_rate: float,
    uniform_fraction: float,
) -> list[
    ConsequenceAwareDirectGene
]:
    children: list[
        ConsequenceAwareDirectGene
    ] = []
    for _ in range(count):
        parent_id = _sample_record_id(
            active_ids,
            records,
            best,
            rng,
            power=2.0,
            uniform_fraction=(
                uniform_fraction
            ),
        )
        children.append(
            records[
                parent_id
            ].gene.mutated(
                rng,
                sigma=sigma,
                mutation_rate=(
                    mutation_rate
                ),
            )
        )
    return children


def _mating_children(
    *,
    pair_count: int,
    children_per_pair: int,
    active_ids: list[str],
    records: dict[str, GeneRecord],
    best: dict[str, float],
    anchor: ConsequenceAwareDirectGene,
    rng: np.random.Generator,
    q_power: float,
    uniform_fraction: float,
    mutation_sigma: float,
    mutation_rate: float,
) -> list[MatingChild]:
    children: list[
        MatingChild
    ] = []

    for _ in range(pair_count):
        parent_a = _sample_record_id(
            active_ids,
            records,
            best,
            rng,
            power=q_power,
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
            power=q_power,
            uniform_fraction=(
                uniform_fraction
            ),
        )

        required = tuple(
            sorted(
                set(
                    records[
                        parent_a
                    ].capabilities
                )
                | set(
                    records[
                        parent_b
                    ].capabilities
                )
            )
        )

        family = offspring_family(
            records[
                parent_a
            ].gene,
            records[
                parent_b
            ].gene,
            anchor,
            rng,
            children=(
                children_per_pair
            ),
            mutation_sigma=(
                mutation_sigma
            ),
            mutation_rate=(
                mutation_rate
            ),
        )
        for result in family:
            children.append(
                MatingChild(
                    result=result,
                    parent_a=parent_a,
                    parent_b=parent_b,
                    required_capabilities=(
                        required
                    ),
                )
            )

    return children


def _top_normal_full_indices(
    screen_scores: dict[
        str,
        np.ndarray,
    ],
    per_axis: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()

    for axis in AXES:
        order = np.argsort(
            -screen_scores[axis]
        )
        for idx in order[
            :per_axis
        ]:
            item = int(idx)
            if item not in seen:
                seen.add(item)
                selected.append(item)

    return selected


def _parent_screen_scores(
    parent_ids: list[str],
    records: dict[str, GeneRecord],
    worlds,
    stars: np.ndarray,
    config,
) -> dict[
    str,
    dict[str, float],
]:
    unique_ids = list(
        dict.fromkeys(
            parent_ids
        )
    )
    genes = [
        records[
            record_id
        ].gene
        for record_id in unique_ids
    ]
    scores = _evaluate_scores(
        genes,
        worlds,
        stars,
        config,
    )
    return {
        record_id: _score_dict_at(
            scores,
            idx,
        )
        for idx, record_id
        in enumerate(
            unique_ids
        )
    }


def _inheritance_retention(
    child_scores: dict[str, float],
    parent_a: GeneRecord,
    parent_b: GeneRecord,
    capabilities: tuple[str, ...],
    *,
    override_a: dict[str, float] | None = None,
    override_b: dict[str, float] | None = None,
) -> dict[str, float]:
    score_a = (
        override_a
        if override_a is not None
        else parent_a.scores
    )
    score_b = (
        override_b
        if override_b is not None
        else parent_b.scores
    )

    result: dict[str, float] = {}
    caps_a = set(
        parent_a.capabilities
    )
    caps_b = set(
        parent_b.capabilities
    )

    for axis in capabilities:
        baselines: list[float] = []
        if axis in caps_a:
            baselines.append(
                float(
                    score_a[axis]
                )
            )
        if axis in caps_b:
            baselines.append(
                float(
                    score_b[axis]
                )
            )
        if not baselines:
            continue

        baseline = max(
            baselines
        )
        result[axis] = (
            float(
                child_scores[axis]
            )
            / max(
                baseline,
                1e-12,
            )
        )

    return result


def _top_mating_full_indices(
    children: list[MatingChild],
    screen_scores: dict[str, np.ndarray],
    parent_screen: dict[
        str,
        dict[str, float],
    ],
    records: dict[str, GeneRecord],
    limit: int,
) -> list[int]:
    ranking: list[
        tuple[float, int]
    ] = []

    for idx, child in enumerate(
        children
    ):
        child_score = (
            _score_dict_at(
                screen_scores,
                idx,
            )
        )
        retention = (
            _inheritance_retention(
                child_score,
                records[
                    child.parent_a
                ],
                records[
                    child.parent_b
                ],
                child.required_capabilities,
                override_a=(
                    parent_screen[
                        child.parent_a
                    ]
                ),
                override_b=(
                    parent_screen[
                        child.parent_b
                    ]
                ),
            )
        )
        min_retention = (
            min(
                retention.values()
            )
            if retention
            else 0.0
        )
        ranking.append(
            (
                float(
                    min_retention
                ),
                idx,
            )
        )

    ranking.sort(
        reverse=True
    )
    return [
        idx
        for _score, idx
        in ranking[:limit]
    ]


def _admit_normal_children(
    *,
    genes: list[ConsequenceAwareDirectGene],
    scores: dict[str, np.ndarray],
    generation: int,
    records: dict[str, GeneRecord],
    archive_size: int,
) -> list[str]:
    admitted: list[str] = []

    current_archives = (
        _specialist_archives(
            records,
            archive_size,
        )
    )
    thresholds = {
        axis: (
            records[
                ids[-1]
            ].scores[axis]
            if len(ids)
            >= archive_size
            else -np.inf
        )
        for axis, ids
        in current_archives.items()
    }

    for idx, gene in enumerate(
        genes
    ):
        score = _score_dict_at(
            scores,
            idx,
        )
        caps = tuple(
            axis
            for axis in AXES
            if score[axis]
            > float(
                thresholds[axis]
            )
            + 1e-12
        )
        if not caps:
            continue

        record_id = _gene_id(
            gene
        )
        if record_id in records:
            merged = tuple(
                sorted(
                    set(
                        records[
                            record_id
                        ].capabilities
                    )
                    | set(caps)
                )
            )
            records[
                record_id
            ].capabilities = merged
            continue

        records[record_id] = (
            GeneRecord(
                record_id=record_id,
                gene=gene,
                capabilities=tuple(
                    sorted(caps)
                ),
                scores=score,
                origin="normal_mutation",
                generation=generation,
            )
        )
        admitted.append(
            record_id
        )

    return admitted


def _admit_mating_children(
    *,
    selected: list[MatingChild],
    scores: dict[str, np.ndarray],
    generation: int,
    records: dict[str, GeneRecord],
    threshold: float,
    accepted_log: Path,
) -> tuple[
    list[str],
    dict[str, int],
]:
    admitted: list[str] = []
    operator_success = {
        operator: 0
        for operator in OPERATORS
    }

    with accepted_log.open(
        "a",
        encoding="utf-8",
    ) as file:
        for idx, child in enumerate(
            selected
        ):
            score = _score_dict_at(
                scores,
                idx,
            )
            parent_a = records[
                child.parent_a
            ]
            parent_b = records[
                child.parent_b
            ]
            retention = (
                _inheritance_retention(
                    score,
                    parent_a,
                    parent_b,
                    child.required_capabilities,
                )
            )
            success = bool(
                retention
            ) and all(
                value
                >= threshold
                for value
                in retention.values()
            )

            event = {
                "generation": (
                    generation
                ),
                "parent_a": (
                    child.parent_a
                ),
                "parent_b": (
                    child.parent_b
                ),
                "parent_a_capabilities": list(
                    parent_a.capabilities
                ),
                "parent_b_capabilities": list(
                    parent_b.capabilities
                ),
                "required_capabilities": list(
                    child.required_capabilities
                ),
                "operator": (
                    child.result.operator
                ),
                "operator_metadata": (
                    child.result.metadata
                ),
                "scores": score,
                "retention": retention,
                "threshold": threshold,
                "accepted": success,
            }

            if success:
                gene = child.result.gene
                record_id = _gene_id(
                    gene
                )
                event[
                    "record_id"
                ] = record_id
                event[
                    "gene"
                ] = gene.to_dict()

                if record_id not in records:
                    records[
                        record_id
                    ] = GeneRecord(
                        record_id=record_id,
                        gene=gene,
                        capabilities=tuple(
                            sorted(
                                child.required_capabilities
                            )
                        ),
                        scores=score,
                        origin="mating",
                        generation=generation,
                        parents=(
                            child.parent_a,
                            child.parent_b,
                        ),
                        operator=(
                            child.result.operator
                        ),
                        metadata={
                            "retention": (
                                retention
                            ),
                            "operator_metadata": (
                                child.result.metadata
                            ),
                        },
                    )
                    admitted.append(
                        record_id
                    )
                else:
                    records[
                        record_id
                    ].capabilities = tuple(
                        sorted(
                            set(
                                records[
                                    record_id
                                ].capabilities
                            )
                            | set(
                                child.required_capabilities
                            )
                        )
                    )

                operator_success[
                    child.result.operator
                ] += 1

            file.write(
                json.dumps(
                    _jsonable(event),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return admitted, operator_success


def _prune_active_records(
    records: dict[str, GeneRecord],
    specialist_size: int,
    hybrid_limit: int,
) -> dict[str, GeneRecord]:
    active_ids, _ = (
        _active_parent_ids(
            records,
            specialist_size,
            hybrid_limit,
        )
    )
    return {
        record_id: records[
            record_id
        ]
        for record_id in active_ids
    }


def _save_checkpoint(
    path: Path,
    *,
    next_generation: int,
    records: dict[str, GeneRecord],
    rng: np.random.Generator,
    operator_attempts: dict[str, int],
    operator_successes: dict[str, int],
) -> None:
    payload = {
        "next_generation": (
            next_generation
        ),
        "records": [
            record.to_dict()
            for record
            in records.values()
        ],
        "rng_state": _jsonable(
            rng.bit_generator.state
        ),
        "operator_attempts": (
            operator_attempts
        ),
        "operator_successes": (
            operator_successes
        ),
    }
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_checkpoint(
    path: Path,
    rng: np.random.Generator,
) -> tuple[
    int,
    dict[str, GeneRecord],
    dict[str, int],
    dict[str, int],
]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    rng.bit_generator.state = (
        data["rng_state"]
    )
    records = {
        str(item["record_id"]): (
            GeneRecord.from_dict(
                item
            )
        )
        for item in data["records"]
    }
    attempts = {
        operator: int(
            data.get(
                "operator_attempts",
                {},
            ).get(
                operator,
                0,
            )
        )
        for operator in OPERATORS
    }
    successes = {
        operator: int(
            data.get(
                "operator_successes",
                {},
            ).get(
                operator,
                0,
            )
        )
        for operator in OPERATORS
    }
    return (
        int(
            data[
                "next_generation"
            ]
        ),
        records,
        attempts,
        successes,
    )


def train(
    args: argparse.Namespace,
) -> Path:
    from marl2d.gene_mrta_v16t.env import EnvConfig

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
            "V1.10 requires 100 scenarios"
        )

    anchor = _load_v18(
        Path(
            args.anchor_v18_run
        )
    )
    rng = np.random.default_rng(
        args.seed
    )

    if args.resume:
        checkpoint = Path(
            args.resume
        )
        run_dir = (
            checkpoint.parent
        )
        (
            start_generation,
            records,
            operator_attempts,
            operator_successes,
        ) = _load_checkpoint(
            checkpoint,
            rng,
        )
        print(
            f"RESUME={checkpoint} "
            f"NEXT_GENERATION="
            f"{start_generation}"
        )
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        run_dir = (
            Path(
                args.output_dir
            )
            / (
                "gene_mrta_v110_mating_"
                f"{stamp}_seed{args.seed}"
            )
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        records = _make_initial_records(
            Path(
                args.bootstrap_v19_checkpoint
            ),
            worlds,
            stars,
            config,
        )
        start_generation = 0
        operator_attempts = {
            operator: 0
            for operator in OPERATORS
        }
        operator_successes = {
            operator: 0
            for operator in OPERATORS
        }

    accepted_log = (
        run_dir
        / "mating_events.jsonl"
    )
    history_path = (
        run_dir / "history.csv"
    )
    history_exists = (
        history_path.exists()
    )

    for generation in range(
        start_generation,
        args.generations,
    ):
        active_ids, archives = (
            _active_parent_ids(
                records,
                args.archive_size_per_axis,
                args.hybrid_bank_limit,
            )
        )
        active_records = {
            record_id: records[
                record_id
            ]
            for record_id
            in active_ids
        }
        best = _best_by_axis(
            active_records
        )

        sigma = _mutation_sigma(
            generation,
            args.generations,
            args.mutation_sigma_start,
            args.mutation_sigma_end,
        )

        normal = _normal_children(
            count=(
                args.normal_offspring
            ),
            active_ids=active_ids,
            records=records,
            best=best,
            rng=rng,
            sigma=sigma,
            mutation_rate=(
                args.mutation_rate
            ),
            uniform_fraction=(
                args.parent_uniform_fraction
            ),
        )

        mating = _mating_children(
            pair_count=(
                args.mating_pairs
            ),
            children_per_pair=(
                args.children_per_pair
            ),
            active_ids=active_ids,
            records=records,
            best=best,
            anchor=anchor,
            rng=rng,
            q_power=(
                args.mating_q_power
            ),
            uniform_fraction=(
                args.parent_uniform_fraction
            ),
            mutation_sigma=(
                args.mating_mutation_sigma
            ),
            mutation_rate=(
                args.mating_mutation_rate
            ),
        )

        if len(mating) != args.mating_offspring:
            raise RuntimeError(
                "Mating configuration did not "
                "produce requested offspring: "
                f"{len(mating)} vs "
                f"{args.mating_offspring}"
            )

        screen_ids = _screen_indices(
            generation,
            len(worlds),
            args.screen_worlds,
        )
        screen_worlds = [
            worlds[int(i)]
            for i in screen_ids
        ]
        screen_stars = stars[
            screen_ids
        ]

        normal_screen = (
            _evaluate_scores(
                normal,
                screen_worlds,
                screen_stars,
                config,
            )
        )
        normal_full_ids = (
            _top_normal_full_indices(
                normal_screen,
                args.normal_full_per_axis,
            )
        )

        mating_genes = [
            child.result.gene
            for child in mating
        ]
        mating_screen = (
            _evaluate_scores(
                mating_genes,
                screen_worlds,
                screen_stars,
                config,
            )
        )
        parent_screen = (
            _parent_screen_scores(
                [
                    parent_id
                    for child in mating
                    for parent_id in (
                        child.parent_a,
                        child.parent_b,
                    )
                ],
                records,
                screen_worlds,
                screen_stars,
                config,
            )
        )
        mating_full_ids = (
            _top_mating_full_indices(
                mating,
                mating_screen,
                parent_screen,
                records,
                args.mating_full_candidates,
            )
        )

        normal_full_genes = [
            normal[idx]
            for idx in normal_full_ids
        ]
        normal_full_scores = (
            _evaluate_scores(
                normal_full_genes,
                worlds,
                stars,
                config,
            )
        )
        admitted_normal = (
            _admit_normal_children(
                genes=normal_full_genes,
                scores=normal_full_scores,
                generation=generation,
                records=records,
                archive_size=(
                    args.archive_size_per_axis
                ),
            )
        )

        selected_mating = [
            mating[idx]
            for idx in mating_full_ids
        ]
        selected_mating_genes = [
            child.result.gene
            for child
            in selected_mating
        ]
        mating_full_scores = (
            _evaluate_scores(
                selected_mating_genes,
                worlds,
                stars,
                config,
            )
        )

        for child in selected_mating:
            operator_attempts[
                child.result.operator
            ] += 1

        (
            admitted_mating,
            success_now,
        ) = _admit_mating_children(
            selected=selected_mating,
            scores=mating_full_scores,
            generation=generation,
            records=records,
            threshold=(
                args.inheritance_threshold
            ),
            accepted_log=accepted_log,
        )
        for operator, count in (
            success_now.items()
        ):
            operator_successes[
                operator
            ] += count

        active_ids, archives = (
            _active_parent_ids(
                records,
                args.archive_size_per_axis,
                args.hybrid_bank_limit,
            )
        )
        best = _best_by_axis(
            records,
            active_ids,
        )

        hybrid_active = [
            record_id
            for record_id
            in active_ids
            if records[
                record_id
            ].origin
            == "mating"
        ]
        max_capability_count = max(
            (
                len(
                    records[
                        record_id
                    ].capabilities
                )
                for record_id
                in active_ids
            ),
            default=1,
        )

        row: dict[str, object] = {
            "generation": generation,
            "active_bank_size": (
                len(active_ids)
            ),
            "hybrid_active": (
                len(hybrid_active)
            ),
            "normal_admitted": (
                len(admitted_normal)
            ),
            "mating_accepted": (
                len(admitted_mating)
            ),
            "max_capability_count": (
                max_capability_count
            ),
            "mutation_sigma": sigma,
        }
        for axis in AXES:
            row[
                f"best_{axis}"
            ] = best[axis]
        for operator in OPERATORS:
            attempts = (
                operator_attempts[
                    operator
                ]
            )
            successes = (
                operator_successes[
                    operator
                ]
            )
            row[
                f"{operator}_success_rate"
            ] = (
                successes
                / attempts
                if attempts
                else 0.0
            )

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    row.keys()
                ),
            )
            if not history_exists:
                writer.writeheader()
                history_exists = True
            writer.writerow(row)

        if (
            generation
            % args.log_every
            == 0
            or admitted_mating
            or generation
            == args.generations - 1
        ):
            print(
                f"GEN {generation:04d} | "
                f"bank={len(active_ids)} "
                f"hybrid={len(hybrid_active)} "
                f"accepted={len(admitted_mating)} "
                f"caps={max_capability_count} "
                f"mean={best['mean_time']:.4f} "
                f"tail10={best['tail10_time']:.4f} "
                f"cont={best['continuation_preservation']:.4f} "
                f"reserve={best['fleet_option_reserve']:.4f} "
                f"sigma={sigma:.4f}"
            )
            if admitted_mating:
                for record_id in (
                    admitted_mating[:5]
                ):
                    record = records[
                        record_id
                    ]
                    print(
                        "  ACCEPT "
                        f"{record_id} "
                        f"caps="
                        f"{','.join(record.capabilities)} "
                        f"operator={record.operator}"
                    )

        if (
            (generation + 1)
            % args.checkpoint_every
            == 0
            or generation
            == args.generations - 1
        ):
            active_records = (
                _prune_active_records(
                    records,
                    args.archive_size_per_axis,
                    args.hybrid_bank_limit,
                )
            )
            _save_checkpoint(
                run_dir
                / "checkpoint.json",
                next_generation=(
                    generation + 1
                ),
                records=active_records,
                rng=rng,
                operator_attempts=(
                    operator_attempts
                ),
                operator_successes=(
                    operator_successes
                ),
            )

    active_ids, archives = (
        _active_parent_ids(
            records,
            args.archive_size_per_axis,
            args.hybrid_bank_limit,
        )
    )
    best = _best_by_axis(
        records,
        active_ids,
    )
    best_global = {
        axis: records[
            max(
                active_ids,
                key=lambda record_id: (
                    records[
                        record_id
                    ].scores[axis]
                ),
            )
        ].to_dict()
        for axis in AXES
    }

    best_hybrids = sorted(
        [
            records[
                record_id
            ]
            for record_id
            in active_ids
            if records[
                record_id
            ].origin
            == "mating"
        ],
        key=lambda record: (
            len(
                record.capabilities
            ),
            _quality_weight(
                record,
                best,
            ),
        ),
        reverse=True,
    )

    summary = {
        "experiment": (
            "gene_mrta_v110_mating_evolution"
        ),
        "policy_architecture_changed": False,
        "policy": {
            "observation_dim": 12,
            "hidden_dim": (
                anchor.hidden_dim
            ),
            "parameter_count": (
                ConsequenceAwareDirectGene.parameter_count(
                    anchor.hidden_dim
                )
            ),
        },
        "scenario_bank": (
            args.scenario_bank
        ),
        "scenario_seeds": (
            scenario_seeds
        ),
        "axes": list(AXES),
        "parent_selection": {
            "normal_power": 2.0,
            "mating_power": (
                args.mating_q_power
            ),
            "uniform_fraction": (
                args.parent_uniform_fraction
            ),
            "quality_definition": (
                "minimum score retention relative "
                "to active best across the Gene's "
                "declared capability set"
            ),
        },
        "offspring": {
            "population": (
                args.normal_offspring
                + args.mating_offspring
            ),
            "normal": (
                args.normal_offspring
            ),
            "mating": (
                args.mating_offspring
            ),
            "mating_pairs": (
                args.mating_pairs
            ),
            "children_per_pair": (
                args.children_per_pair
            ),
        },
        "evaluation": {
            "screen_worlds": (
                args.screen_worlds
            ),
            "normal_full_per_axis": (
                args.normal_full_per_axis
            ),
            "mating_full_candidates": (
                args.mating_full_candidates
            ),
            "inheritance_threshold": (
                args.inheritance_threshold
            ),
        },
        "best_axis_scores": best,
        "best_axis_records": (
            best_global
        ),
        "best_hybrids": [
            record.to_dict()
            for record
            in best_hybrids[:20]
        ],
        "operator_attempts": (
            operator_attempts
        ),
        "operator_successes": (
            operator_successes
        ),
        "final_generalization_rule": (
            "The 95M mating scenario bank is development data. "
            "The 98M set remains historical V1.8/V1.9 development evidence. "
            "Do not inspect 99,000,000-99,000,099 until the V1.10 "
            "candidate and procedure are frozen."
        ),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(
            _jsonable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nFINAL V1.10 MATING EVOLUTION")
    print(
        json.dumps(
            best,
            indent=2,
        )
    )
    print(
        "OPERATOR_SUCCESS="
        + json.dumps(
            {
                operator: {
                    "attempts": (
                        operator_attempts[
                            operator
                        ]
                    ),
                    "successes": (
                        operator_successes[
                            operator
                        ]
                    ),
                }
                for operator in OPERATORS
            }
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
        "--bootstrap-v19-checkpoint",
        required=True,
    )
    p.add_argument(
        "--anchor-v18-run",
        required=True,
    )
    p.add_argument(
        "--resume",
        default="",
    )

    p.add_argument(
        "--generations",
        type=int,
        default=500,
    )
    p.add_argument(
        "--normal-offspring",
        type=int,
        default=128,
    )
    p.add_argument(
        "--mating-offspring",
        type=int,
        default=128,
    )
    p.add_argument(
        "--mating-pairs",
        type=int,
        default=32,
    )
    p.add_argument(
        "--children-per-pair",
        type=int,
        default=4,
    )

    p.add_argument(
        "--mating-q-power",
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
        "--screen-worlds",
        type=int,
        default=25,
    )
    p.add_argument(
        "--normal-full-per-axis",
        type=int,
        default=4,
    )
    p.add_argument(
        "--mating-full-candidates",
        type=int,
        default=16,
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
        "--mutation-rate",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--mutation-sigma-start",
        type=float,
        default=0.08,
    )
    p.add_argument(
        "--mutation-sigma-end",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--mating-mutation-sigma",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--mating-mutation-rate",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=5,
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
            "gene_mrta_v110_mating"
        ),
    )

    return p


def main() -> None:
    args = parser().parse_args()

    if (
        args.mating_pairs
        * args.children_per_pair
        != args.mating_offspring
    ):
        raise ValueError(
            "mating_pairs * children_per_pair "
            "must equal mating_offspring"
        )
    if (
        args.normal_offspring
        != args.mating_offspring
    ):
        raise ValueError(
            "V1.10 freezes the population split at 50/50 "
            "normal vs mating offspring"
        )
    if args.screen_worlds <= 0 or args.screen_worlds > 100:
        raise ValueError(
            "screen_worlds must be in 1..100"
        )

    train(args)


if __name__ == "__main__":
    main()
