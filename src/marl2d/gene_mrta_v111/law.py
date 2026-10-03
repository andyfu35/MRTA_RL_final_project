from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import qmc

from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene
from marl2d.gene_mrta_v110.recombination import hidden_block_indices


EPS = 1e-12


@dataclass(frozen=True)
class LawCoefficients:
    """
    Seven-parameter, parent-swap-symmetric adaptive delta mixing law.

    alpha controls which parent's delta dominates a functional group.
    eta controls how strongly the mixed delta is applied relative to the
    common V1.8 ancestor.
    """

    law_id: str
    beta_norm_ratio: float
    beta_quality_diff: float
    beta_capability_diff: float
    gamma_bias: float
    gamma_cosine: float
    gamma_sign_agreement: float
    gamma_magnitude: float

    def to_dict(self) -> dict[str, object]:
        return {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in asdict(self).items()
        }

    def equation(self) -> dict[str, str]:
        return {
            "alpha": (
                "sigmoid("
                f"{self.beta_norm_ratio:.8g}*r_k + "
                f"{self.beta_quality_diff:.8g}*(Q_A-Q_B) + "
                f"{self.beta_capability_diff:.8g}*(H_A-H_B)"
                ")"
            ),
            "eta": (
                "0.5 + sigmoid("
                f"{self.gamma_bias:.8g} + "
                f"{self.gamma_cosine:.8g}*c_k + "
                f"{self.gamma_sign_agreement:.8g}*s_k + "
                f"{self.gamma_magnitude:.8g}*m_k"
                ")"
            ),
            "child": (
                "theta_C^(k) = theta_0^(k) + eta_k * "
                "[alpha_k*Delta_A^(k) + (1-alpha_k)*Delta_B^(k)]"
            ),
        }


def sample_law_candidates(
    count: int,
    seed: int,
) -> list[LawCoefficients]:
    if count <= 0:
        raise ValueError("count must be positive")

    # Include the exact center law as a deterministic reference:
    # alpha=0.5 and eta=1.0 for every group.
    rows: list[np.ndarray] = [
        np.zeros(7, dtype=np.float64)
    ]
    if count > 1:
        sampler = qmc.LatinHypercube(
            d=7,
            seed=seed,
        )
        unit = sampler.random(
            n=count - 1,
        )
        low = np.asarray(
            [
                -2.5,
                -3.0,
                -3.0,
                -1.5,
                -2.5,
                -2.5,
                -1.5,
            ],
            dtype=np.float64,
        )
        high = np.asarray(
            [
                2.5,
                3.0,
                3.0,
                1.5,
                2.5,
                2.5,
                1.5,
            ],
            dtype=np.float64,
        )
        scaled = qmc.scale(
            unit,
            low,
            high,
        )
        rows.extend(
            np.asarray(row, dtype=np.float64)
            for row in scaled
        )

    result: list[LawCoefficients] = []
    for idx, row in enumerate(rows):
        result.append(
            LawCoefficients(
                law_id=f"law_{idx:03d}",
                beta_norm_ratio=float(row[0]),
                beta_quality_diff=float(row[1]),
                beta_capability_diff=float(row[2]),
                gamma_bias=float(row[3]),
                gamma_cosine=float(row[4]),
                gamma_sign_agreement=float(row[5]),
                gamma_magnitude=float(row[6]),
            )
        )
    return result


def _group_indices(
    hidden_dim: int,
) -> list[np.ndarray]:
    blocks, global_ids = hidden_block_indices(
        hidden_dim
    )
    return [
        *blocks,
        global_ids,
    ]


def _cosine(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    norm_a = float(
        np.linalg.norm(a)
    )
    norm_b = float(
        np.linalg.norm(b)
    )
    if norm_a <= EPS or norm_b <= EPS:
        return 0.0
    return float(
        np.clip(
            np.dot(a, b)
            / (norm_a * norm_b),
            -1.0,
            1.0,
        )
    )


def _sign_agreement(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    active = (
        (np.abs(a) > EPS)
        | (np.abs(b) > EPS)
    )
    if not np.any(active):
        return 1.0
    agree = (
        np.sign(a[active])
        == np.sign(b[active])
    )
    # Map [0,1] to [-1,1] so zero means neutral agreement.
    return float(
        2.0 * np.mean(agree) - 1.0
    )


def group_features(
    parent_a: ConsequenceAwareDirectGene,
    parent_b: ConsequenceAwareDirectGene,
    anchor: ConsequenceAwareDirectGene,
) -> list[dict[str, float]]:
    if parent_a.hidden_dim != parent_b.hidden_dim:
        raise ValueError(
            "Parent hidden dimensions must match"
        )
    if parent_a.hidden_dim != anchor.hidden_dim:
        raise ValueError(
            "Anchor hidden dimension must match"
        )

    delta_a = (
        parent_a.vector_data
        - anchor.vector_data
    )
    delta_b = (
        parent_b.vector_data
        - anchor.vector_data
    )
    groups = _group_indices(
        parent_a.hidden_dim
    )

    norms_a: list[float] = []
    norms_b: list[float] = []
    for ids in groups:
        norms_a.append(
            float(
                np.linalg.norm(
                    delta_a[ids]
                )
            )
        )
        norms_b.append(
            float(
                np.linalg.norm(
                    delta_b[ids]
                )
            )
        )

    magnitudes = np.asarray(
        [
            0.5 * (a + b)
            for a, b
            in zip(norms_a, norms_b)
        ],
        dtype=np.float64,
    )
    magnitude_reference = max(
        float(
            np.median(
                magnitudes
            )
        ),
        EPS,
    )

    features: list[
        dict[str, float]
    ] = []
    for group_idx, ids in enumerate(groups):
        da = delta_a[ids]
        db = delta_b[ids]
        norm_a = norms_a[group_idx]
        norm_b = norms_b[group_idx]
        magnitude = magnitudes[
            group_idx
        ]
        features.append(
            {
                "group": float(group_idx),
                "norm_a": norm_a,
                "norm_b": norm_b,
                "norm_log_ratio": float(
                    np.log(
                        (norm_a + EPS)
                        / (norm_b + EPS)
                    )
                ),
                "cosine": _cosine(
                    da,
                    db,
                ),
                "sign_agreement": (
                    _sign_agreement(
                        da,
                        db,
                    )
                ),
                "magnitude_log_relative": float(
                    np.log(
                        (magnitude + EPS)
                        / magnitude_reference
                    )
                ),
            }
        )
    return features


def adaptive_delta_child(
    parent_a: ConsequenceAwareDirectGene,
    parent_b: ConsequenceAwareDirectGene,
    anchor: ConsequenceAwareDirectGene,
    law: LawCoefficients,
    *,
    quality_a: float,
    quality_b: float,
    capability_count_a: int,
    capability_count_b: int,
) -> tuple[
    ConsequenceAwareDirectGene,
    dict[str, Any],
]:
    if parent_a.hidden_dim != parent_b.hidden_dim:
        raise ValueError(
            "Parent hidden dimensions must match"
        )
    if parent_a.hidden_dim != anchor.hidden_dim:
        raise ValueError(
            "Anchor hidden dimension must match"
        )

    a = parent_a.vector_data
    b = parent_b.vector_data
    base = anchor.vector_data
    delta_a = a - base
    delta_b = b - base
    groups = _group_indices(
        parent_a.hidden_dim
    )
    features = group_features(
        parent_a,
        parent_b,
        anchor,
    )

    q_diff = float(
        quality_a - quality_b
    )
    cap_diff = float(
        (
            int(capability_count_a)
            - int(capability_count_b)
        )
        / 4.0
    )

    child = base.copy()
    group_metadata: list[
        dict[str, float]
    ] = []

    for ids, feature in zip(
        groups,
        features,
    ):
        alpha_logit = (
            law.beta_norm_ratio
            * feature[
                "norm_log_ratio"
            ]
            + law.beta_quality_diff
            * q_diff
            + law.beta_capability_diff
            * cap_diff
        )
        alpha = float(
            expit(
                alpha_logit
            )
        )

        eta_logit = (
            law.gamma_bias
            + law.gamma_cosine
            * feature["cosine"]
            + law.gamma_sign_agreement
            * feature[
                "sign_agreement"
            ]
            + law.gamma_magnitude
            * feature[
                "magnitude_log_relative"
            ]
        )
        # eta is bounded in [0.5, 1.5].
        eta = float(
            0.5
            + expit(
                eta_logit
            )
        )

        mixed_delta = (
            alpha * delta_a[ids]
            + (1.0 - alpha)
            * delta_b[ids]
        )
        child[ids] = (
            base[ids]
            + eta * mixed_delta
        )

        group_metadata.append(
            {
                **feature,
                "alpha": alpha,
                "eta": eta,
            }
        )

    gene = ConsequenceAwareDirectGene(
        np.asarray(
            child,
            dtype=np.float64,
        ),
        hidden_dim=parent_a.hidden_dim,
    )
    metadata: dict[str, Any] = {
        "law_id": law.law_id,
        "law": law.to_dict(),
        "quality_a": float(
            quality_a
        ),
        "quality_b": float(
            quality_b
        ),
        "capability_count_a": int(
            capability_count_a
        ),
        "capability_count_b": int(
            capability_count_b
        ),
        "groups": group_metadata,
    }
    return gene, metadata
