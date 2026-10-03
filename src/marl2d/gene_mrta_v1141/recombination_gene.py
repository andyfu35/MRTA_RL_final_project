from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v111.law import (
    LawCoefficients,
    adaptive_delta_child,
)


COEFFICIENT_NAMES = (
    "beta_norm_ratio",
    "beta_quality_diff",
    "beta_capability_diff",
    "gamma_bias",
    "gamma_cosine",
    "gamma_sign_agreement",
    "gamma_magnitude",
)

RECOMBINATION_AXES = (
    "screen_yield",
    "acceptance_yield",
    "retention_quality",
    "four_capability_yield",
)

LOWER_BOUNDS = np.asarray(
    [
        -4.0,
        -4.0,
        -4.0,
        -2.5,
        -4.0,
        -4.0,
        -2.5,
    ],
    dtype=np.float64,
)
UPPER_BOUNDS = np.asarray(
    [
        4.0,
        4.0,
        4.0,
        2.5,
        4.0,
        4.0,
        2.5,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RecombinationGene:
    """
    Interpretable genotype controlling policy mating.

    coefficients:
      Seven continuous parameters of the V1.11 adaptive-delta law.

    gates:
      Seven binary structural switches. A disabled term contributes exactly
      zero, allowing evolution to remove inputs from the mating equation.

    mutation_sigma:
      Self-adaptive mutation step size. It is part of the genotype so the
      recombination population can evolve its own exploration scale.
    """

    coefficients: np.ndarray
    gates: np.ndarray
    mutation_sigma: float
    generation: int
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        coefficients = np.asarray(
            self.coefficients,
            dtype=np.float64,
        )
        gates = np.asarray(
            self.gates,
            dtype=bool,
        )
        if coefficients.shape != (7,):
            raise ValueError(
                "Recombination coefficients must have shape (7,)"
            )
        if gates.shape != (7,):
            raise ValueError(
                "Recombination gates must have shape (7,)"
            )
        if not np.all(
            np.isfinite(
                coefficients
            )
        ):
            raise ValueError(
                "Recombination coefficients must be finite"
            )
        if not np.isfinite(
            self.mutation_sigma
        ) or self.mutation_sigma <= 0.0:
            raise ValueError(
                "mutation_sigma must be positive and finite"
            )
        # V1.14.1 canonical phenotype: disabled coefficients are exactly
        # zero. A disabled term therefore has no hidden latent value that
        # can create a different ID while producing the same formula.
        canonical = np.where(
            gates,
            coefficients,
            0.0,
        )
        object.__setattr__(
            self,
            "coefficients",
            canonical.copy(),
        )
        object.__setattr__(
            self,
            "gates",
            gates.copy(),
        )

    @property
    def phenotype_id(self) -> str:
        """
        Canonical mating-law identity.

        Only the effective equation participates in the ID. mutation_sigma
        controls future exploration and is deliberately excluded. Disabled
        coefficients were already canonicalized to zero in __post_init__.
        """
        payload = (
            self.coefficients.tobytes()
            + self.gates.astype(
                np.uint8
            ).tobytes()
        )
        return hashlib.sha256(
            payload
        ).hexdigest()[:20]

    @property
    def gene_id(self) -> str:
        # Backwards-compatible name used by the bank: one slot per phenotype.
        return self.phenotype_id

    @property
    def genotype_fingerprint(self) -> str:
        payload = (
            self.coefficients.tobytes()
            + self.gates.astype(
                np.uint8
            ).tobytes()
            + np.asarray(
                [self.mutation_sigma],
                dtype=np.float64,
            ).tobytes()
        )
        return hashlib.sha256(
            payload
        ).hexdigest()[:20]

    @property
    def active_term_count(self) -> int:
        return int(
            np.sum(
                self.gates
            )
        )

    def masked_coefficients(
        self,
    ) -> np.ndarray:
        return (
            self.coefficients
            * self.gates.astype(
                np.float64
            )
        )

    def to_law(
        self,
    ) -> LawCoefficients:
        values = (
            self.masked_coefficients()
        )
        return LawCoefficients(
            law_id=self.gene_id,
            beta_norm_ratio=float(
                values[0]
            ),
            beta_quality_diff=float(
                values[1]
            ),
            beta_capability_diff=float(
                values[2]
            ),
            gamma_bias=float(
                values[3]
            ),
            gamma_cosine=float(
                values[4]
            ),
            gamma_sign_agreement=float(
                values[5]
            ),
            gamma_magnitude=float(
                values[6]
            ),
        )

    def equation(
        self,
    ) -> dict[str, str]:
        equation = dict(
            self.to_law().equation()
        )
        equation["alpha"] = equation[
            "alpha"
        ].replace(
            "*(H_A-H_B)",
            "*((H_A-H_B)/4)",
        )
        return equation

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "gene_id": self.gene_id,
            "phenotype_id": self.phenotype_id,
            "genotype_fingerprint": self.genotype_fingerprint,
            "coefficients": {
                name: float(value)
                for name, value
                in zip(
                    COEFFICIENT_NAMES,
                    self.coefficients,
                )
            },
            "gates": {
                name: bool(value)
                for name, value
                in zip(
                    COEFFICIENT_NAMES,
                    self.gates,
                )
            },
            "masked_coefficients": {
                name: float(value)
                for name, value
                in zip(
                    COEFFICIENT_NAMES,
                    self.masked_coefficients(),
                )
            },
            "mutation_sigma": float(
                self.mutation_sigma
            ),
            "generation": int(
                self.generation
            ),
            "parents": list(
                self.parents
            ),
            "active_term_count": (
                self.active_term_count
            ),
            "equation": self.equation(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "RecombinationGene":
        coefficients_raw = data[
            "coefficients"
        ]
        gates_raw = data[
            "gates"
        ]
        if isinstance(
            coefficients_raw,
            dict,
        ):
            coefficients = np.asarray(
                [
                    coefficients_raw[name]
                    for name
                    in COEFFICIENT_NAMES
                ],
                dtype=np.float64,
            )
        else:
            coefficients = np.asarray(
                coefficients_raw,
                dtype=np.float64,
            )
        if isinstance(
            gates_raw,
            dict,
        ):
            gates = np.asarray(
                [
                    gates_raw[name]
                    for name
                    in COEFFICIENT_NAMES
                ],
                dtype=bool,
            )
        else:
            gates = np.asarray(
                gates_raw,
                dtype=bool,
            )
        return cls(
            coefficients=coefficients,
            gates=gates,
            mutation_sigma=float(
                data[
                    "mutation_sigma"
                ]
            ),
            generation=int(
                data.get(
                    "generation",
                    0,
                )
            ),
            parents=tuple(
                str(x)
                for x
                in data.get(
                    "parents",
                    []
                )
            ),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        *,
        generation: int,
        gate_flip_rate: float,
        sigma_tau: float,
        sigma_min: float,
        sigma_max: float,
    ) -> "RecombinationGene":
        sigma = float(
            np.clip(
                self.mutation_sigma
                * np.exp(
                    rng.normal(
                        0.0,
                        sigma_tau,
                    )
                ),
                sigma_min,
                sigma_max,
            )
        )
        coefficients = np.clip(
            self.coefficients
            + rng.normal(
                0.0,
                sigma,
                size=7,
            ),
            LOWER_BOUNDS,
            UPPER_BOUNDS,
        )
        gates = self.gates.copy()
        flips = (
            rng.random(7)
            < gate_flip_rate
        )
        gates ^= flips

        return RecombinationGene(
            coefficients=coefficients,
            gates=gates,
            mutation_sigma=sigma,
            generation=generation,
            parents=(
                self.gene_id,
            ),
        )


@dataclass
class RecombinationRecord:
    gene: RecombinationGene
    generated: int = 0
    screen_selected: int = 0
    accepted: int = 0
    four_capability_accepted: int = 0
    screen_retention_sum: float = 0.0
    last_generation_used: int = -1
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def gene_id(
        self,
    ) -> str:
        return self.gene.gene_id

    def axis_scores(
        self,
    ) -> dict[str, float]:
        """
        Bayesian-smoothed, nonnegative rule-quality axes.

        Untested rules receive neutral priors rather than zero, which keeps
        newly mutated rules explorable without hand-authored bonuses.
        """
        generated = float(
            self.generated
        )
        # Conservative priors prevent an untested newborn rule from
        # looking artificially better than a mature, empirically good rule.
        # Exploration is supplied separately by uniform sampling.
        screen_yield = (
            self.screen_selected + 0.25
        ) / (
            generated + 2.0
        )
        acceptance_yield = (
            self.accepted + 0.20
        ) / (
            generated + 2.0
        )
        four_yield = (
            self.four_capability_accepted
            + 0.10
        ) / (
            generated + 2.0
        )
        retention = (
            (
                self.screen_retention_sum
                + 0.95
            )
            / (
                generated + 1.0
            )
        )
        return {
            "screen_yield": float(
                np.clip(
                    screen_yield,
                    0.0,
                    1.0,
                )
            ),
            "acceptance_yield": float(
                np.clip(
                    acceptance_yield,
                    0.0,
                    1.0,
                )
            ),
            "retention_quality": float(
                np.clip(
                    retention,
                    0.0,
                    1.25,
                )
            ),
            "four_capability_yield": float(
                np.clip(
                    four_yield,
                    0.0,
                    1.0,
                )
            ),
        }

    def evidence_count(
        self,
        axis: str,
    ) -> int:
        if axis not in RECOMBINATION_AXES:
            raise ValueError(
                f"Unknown recombination axis: {axis}"
            )
        return int(
            self.generated
        )

    def evidence_score(
        self,
        axis: str,
        *,
        quantile: float = 0.10,
    ) -> float:
        """
        Lower posterior quantile used for mature-rule selection.

        Yield axes use Beta posteriors. retention_quality is approximated as
        a bounded [0,1] fractional-success process after clipping cumulative
        retention to the number of generated trials.
        """
        if axis not in RECOMBINATION_AXES:
            raise ValueError(
                f"Unknown recombination axis: {axis}"
            )
        if not (
            0.0 < quantile < 0.5
        ):
            raise ValueError(
                "quantile must be in (0,0.5)"
            )

        n = float(
            self.generated
        )
        if axis == "screen_yield":
            success = float(
                self.screen_selected
            )
            alpha0, beta0 = 1.0, 3.0
        elif axis == "acceptance_yield":
            success = float(
                self.accepted
            )
            alpha0, beta0 = 1.0, 9.0
        elif axis == "four_capability_yield":
            success = float(
                self.four_capability_accepted
            )
            alpha0, beta0 = 0.5, 9.5
        else:
            success = float(
                np.clip(
                    self.screen_retention_sum,
                    0.0,
                    n,
                )
            )
            alpha0, beta0 = 1.0, 1.0

        failure = max(
            n - success,
            0.0,
        )
        return float(
            beta_distribution.ppf(
                quantile,
                alpha0 + success,
                beta0 + failure,
            )
        )

    def is_evidence_mature(
        self,
        min_evidence: int,
    ) -> bool:
        return (
            self.generated
            >= min_evidence
        )


    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "gene": self.gene.to_dict(),
            "generated": int(
                self.generated
            ),
            "screen_selected": int(
                self.screen_selected
            ),
            "accepted": int(
                self.accepted
            ),
            "four_capability_accepted": int(
                self.four_capability_accepted
            ),
            "screen_retention_sum": float(
                self.screen_retention_sum
            ),
            "last_generation_used": int(
                self.last_generation_used
            ),
            "axis_scores": (
                self.axis_scores()
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "RecombinationRecord":
        return cls(
            gene=RecombinationGene.from_dict(
                data["gene"]
            ),
            generated=int(
                data.get(
                    "generated",
                    0,
                )
            ),
            screen_selected=int(
                data.get(
                    "screen_selected",
                    0,
                )
            ),
            accepted=int(
                data.get(
                    "accepted",
                    0,
                )
            ),
            four_capability_accepted=int(
                data.get(
                    "four_capability_accepted",
                    0,
                )
            ),
            screen_retention_sum=float(
                data.get(
                    "screen_retention_sum",
                    0.0,
                )
            ),
            last_generation_used=int(
                data.get(
                    "last_generation_used",
                    -1,
                )
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {}
                )
            ),
        )


def initial_recombination_bank(
    *,
    count: int,
    seed: int,
    initial_sigma: float,
) -> dict[str, RecombinationRecord]:
    if count <= 0:
        raise ValueError(
            "count must be positive"
        )
    rng = np.random.default_rng(
        seed
    )
    result: dict[
        str,
        RecombinationRecord
    ] = {}

    center = RecombinationGene(
        coefficients=np.zeros(
            7,
            dtype=np.float64,
        ),
        gates=np.zeros(
            7,
            dtype=bool,
        ),
        mutation_sigma=initial_sigma,
        generation=-1,
        parents=(),
    )
    result[center.gene_id] = (
        RecombinationRecord(
            gene=center,
            metadata={
                "origin": "center_law"
            },
        )
    )

    while len(result) < count:
        coefficients = rng.uniform(
            LOWER_BOUNDS,
            UPPER_BOUNDS,
        )
        gates = (
            rng.random(7)
            < 0.70
        )
        gene = RecombinationGene(
            coefficients=coefficients,
            gates=gates,
            mutation_sigma=initial_sigma,
            generation=-1,
            parents=(),
        )
        result.setdefault(
            gene.gene_id,
            RecombinationRecord(
                gene=gene,
                metadata={
                    "origin": "random_seed"
                },
            ),
        )

    return result


def _sampling_score(
    record: RecombinationRecord,
    axis: str,
    *,
    evidence_quantile: float,
) -> float:
    if axis not in RECOMBINATION_AXES:
        raise ValueError(
            f"Unknown recombination axis: {axis}"
        )
    value = record.evidence_score(
        axis,
        quantile=evidence_quantile,
    )
    return max(
        float(value),
        1e-6,
    )


def sample_recombination_gene_id(
    records: dict[
        str,
        RecombinationRecord,
    ],
    rng: np.random.Generator,
    *,
    selection_power: float,
    uniform_fraction: float,
    evidence_quantile: float = 0.10,
) -> tuple[str, str]:
    if not records:
        raise RuntimeError(
            "Empty recombination Gene Bank"
        )
    ids = sorted(
        records
    )
    axis = RECOMBINATION_AXES[
        int(
            rng.integers(
                0,
                len(
                    RECOMBINATION_AXES
                ),
            )
        )
    ]

    scores = np.asarray(
        [
            _sampling_score(
                records[gene_id],
                axis,
                evidence_quantile=(
                    evidence_quantile
                ),
            )
            ** selection_power
            for gene_id in ids
        ],
        dtype=np.float64,
    )
    scores /= np.sum(
        scores
    )
    uniform = np.full(
        len(ids),
        1.0 / len(ids),
        dtype=np.float64,
    )
    p = (
        (
            1.0
            - uniform_fraction
        )
        * scores
        + uniform_fraction
        * uniform
    )
    index = int(
        rng.choice(
            len(ids),
            p=p,
        )
    )
    return ids[index], axis


def recombine_with_gene(
    recombination_gene: RecombinationGene,
    parent_a: ConsequenceAwareDirectGene,
    parent_b: ConsequenceAwareDirectGene,
    anchor: ConsequenceAwareDirectGene,
    *,
    quality_a: float,
    quality_b: float,
    capability_count_a: int,
    capability_count_b: int,
) -> tuple[
    ConsequenceAwareDirectGene,
    dict[str, Any],
]:
    child, metadata = (
        adaptive_delta_child(
            parent_a,
            parent_b,
            anchor,
            recombination_gene.to_law(),
            quality_a=quality_a,
            quality_b=quality_b,
            capability_count_a=(
                capability_count_a
            ),
            capability_count_b=(
                capability_count_b
            ),
        )
    )
    return child, {
        **metadata,
        "recombination_gene": (
            recombination_gene.to_dict()
        ),
    }


def spawn_recombination_mutants(
    records: dict[
        str,
        RecombinationRecord,
    ],
    rng: np.random.Generator,
    *,
    generation: int,
    count: int,
    selection_power: float,
    uniform_fraction: float,
    gate_flip_rate: float,
    sigma_tau: float,
    sigma_min: float,
    sigma_max: float,
    evidence_quantile: float = 0.10,
) -> list[RecombinationGene]:
    children: list[
        RecombinationGene
    ] = []

    attempts = 0
    max_attempts = max(
        100,
        count * 20,
    )
    while (
        len(children) < count
        and attempts < max_attempts
    ):
        attempts += 1
        parent_id, _axis = (
            sample_recombination_gene_id(
                records,
                rng,
                selection_power=(
                    selection_power
                ),
                uniform_fraction=(
                    uniform_fraction
                ),
            )
        )
        child = records[
            parent_id
        ].gene.mutate(
            rng,
            generation=generation,
            gate_flip_rate=(
                gate_flip_rate
            ),
            sigma_tau=sigma_tau,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        if (
            child.gene_id
            not in records
            and all(
                child.gene_id
                != existing.gene_id
                for existing in children
            )
        ):
            children.append(
                child
            )

    if len(children) != count:
        raise RuntimeError(
            "Could not generate requested unique recombination mutants"
        )
    return children


def pareto_front_ids(
    records: dict[
        str,
        RecombinationRecord,
    ],
    *,
    min_evidence: int = 0,
    evidence_quantile: float = 0.10,
) -> list[str]:
    ids = [
        gene_id
        for gene_id, record
        in records.items()
        if record.generated
        >= min_evidence
    ]
    if not ids:
        return []

    values = {
        gene_id: {
            axis: records[
                gene_id
            ].evidence_score(
                axis,
                quantile=(
                    evidence_quantile
                ),
            )
            for axis
            in RECOMBINATION_AXES
        }
        for gene_id in ids
    }
    front: list[str] = []

    for candidate in sorted(ids):
        dominated = False
        for other in ids:
            if other == candidate:
                continue
            better_or_equal = all(
                values[other][axis]
                >= values[candidate][axis]
                - 1e-12
                for axis
                in RECOMBINATION_AXES
            )
            strictly_better = any(
                values[other][axis]
                > values[candidate][axis]
                + 1e-12
                for axis
                in RECOMBINATION_AXES
            )
            if (
                better_or_equal
                and strictly_better
            ):
                dominated = True
                break
        if not dominated:
            front.append(
                candidate
            )
    return front


def prune_recombination_bank(
    records: dict[
        str,
        RecombinationRecord,
    ],
    *,
    specialist_size: int,
    pareto_limit: int,
    total_limit: int,
    min_evidence: int = 32,
    evidence_quantile: float = 0.10,
    exploration_slots: int = 8,
) -> dict[str, RecombinationRecord]:
    if len(records) <= total_limit:
        return dict(records)

    mature_ids = [
        gene_id
        for gene_id, record
        in records.items()
        if record.generated
        >= min_evidence
    ]
    keep: set[str] = set()

    # Mature specialists are ranked by lower posterior quantile, never by a
    # small-sample point estimate.
    if mature_ids:
        for axis in RECOMBINATION_AXES:
            ranked = sorted(
                mature_ids,
                key=lambda gene_id: (
                    records[
                        gene_id
                    ].evidence_score(
                        axis,
                        quantile=(
                            evidence_quantile
                        ),
                    ),
                    records[
                        gene_id
                    ].generated,
                    -records[
                        gene_id
                    ].gene.active_term_count,
                ),
                reverse=True,
            )
            keep.update(
                ranked[
                    :specialist_size
                ]
            )

        front = pareto_front_ids(
            records,
            min_evidence=min_evidence,
            evidence_quantile=(
                evidence_quantile
            ),
        )
        front = sorted(
            front,
            key=lambda gene_id: (
                records[
                    gene_id
                ].accepted,
                records[
                    gene_id
                ].generated,
            ),
            reverse=True,
        )
        keep.update(
            front[
                :pareto_limit
            ]
        )

    # Explicit exploration reserve for newborn/provisional rules. Prefer the
    # newest, least-tested distinct phenotypes.
    provisional = [
        gene_id
        for gene_id, record
        in records.items()
        if gene_id not in keep
        and record.generated
        < min_evidence
    ]
    provisional.sort(
        key=lambda gene_id: (
            records[
                gene_id
            ].gene.generation,
            -records[
                gene_id
            ].generated,
        ),
        reverse=True,
    )
    keep.update(
        provisional[
            : min(
                exploration_slots,
                total_limit,
            )
        ]
    )

    # Fill remaining slots round-robin by mature evidence axes. If no mature
    # rules exist yet, retain the most-tested provisional rules.
    remaining_capacity = (
        total_limit
        - len(keep)
    )
    if remaining_capacity > 0:
        pool = [
            gene_id
            for gene_id in records
            if gene_id not in keep
        ]
        if mature_ids:
            rankings = {
                axis: sorted(
                    pool,
                    key=lambda gene_id: (
                        records[
                            gene_id
                        ].evidence_score(
                            axis,
                            quantile=(
                                evidence_quantile
                            ),
                        ),
                        records[
                            gene_id
                        ].generated,
                    ),
                    reverse=True,
                )
                for axis in RECOMBINATION_AXES
            }
            cursor = 0
            while (
                len(keep) < total_limit
                and pool
            ):
                axis = RECOMBINATION_AXES[
                    cursor
                    % len(
                        RECOMBINATION_AXES
                    )
                ]
                added = False
                for gene_id in rankings[
                    axis
                ]:
                    if gene_id not in keep:
                        keep.add(
                            gene_id
                        )
                        added = True
                        break
                if not added:
                    break
                cursor += 1
        else:
            pool.sort(
                key=lambda gene_id: (
                    records[
                        gene_id
                    ].generated,
                    records[
                        gene_id
                    ].gene.generation,
                ),
                reverse=True,
            )
            keep.update(
                pool[
                    :remaining_capacity
                ]
            )

    # Hard capacity guard.
    if len(keep) > total_limit:
        ordered = sorted(
            keep,
            key=lambda gene_id: (
                records[
                    gene_id
                ].generated
                >= min_evidence,
                max(
                    records[
                        gene_id
                    ].evidence_score(
                        axis,
                        quantile=(
                            evidence_quantile
                        ),
                    )
                    for axis
                    in RECOMBINATION_AXES
                ),
                records[
                    gene_id
                ].generated,
            ),
            reverse=True,
        )
        keep = set(
            ordered[
                :total_limit
            ]
        )

    return {
        gene_id: records[
            gene_id
        ]
        for gene_id in sorted(
            keep
        )
    }
