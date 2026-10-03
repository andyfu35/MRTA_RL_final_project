from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl2d.gene_mrta_v18.direct_gene import ConsequenceAwareDirectGene


OPERATORS = (
    "parameter_blend",
    "block_pick",
    "block_blend",
    "ancestor_delta",
    "ties_delta",
    "dare_delta",
)


@dataclass(frozen=True)
class RecombinationResult:
    gene: ConsequenceAwareDirectGene
    operator: str
    metadata: dict[str, Any]


def hidden_block_indices(
    hidden_dim: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Partition the V1.8/V1.9 policy vector into hidden-neuron functional blocks.

    One hidden block includes:
      - all 12 input weights feeding that hidden unit,
      - its pair-encoder bias,
      - the four decoder weights using that hidden coordinate
        (pair/global/robot/task),
      - the STOP weight using that hidden coordinate.

    Four scalar parameters are global rather than hidden-specific:
      decoder step weight, decoder bias, STOP step weight, STOP bias.
    """
    h = int(hidden_dim)
    if h <= 0:
        raise ValueError("hidden_dim must be positive")

    w_pair_start = 0
    b_pair_start = 12 * h
    w_decode_start = 13 * h
    decode_step = w_decode_start + 4 * h
    b_decode = decode_step + 1
    w_stop_start = b_decode + 1
    stop_step = w_stop_start + h
    b_stop = stop_step + 1

    expected = 18 * h + 4
    if b_stop + 1 != expected:
        raise AssertionError("Unexpected policy vector layout")

    blocks: list[np.ndarray] = []
    for k in range(h):
        ids: list[int] = []
        ids.extend(
            row * h + k
            for row in range(12)
        )
        ids.append(b_pair_start + k)
        ids.extend(
            w_decode_start + group * h + k
            for group in range(4)
        )
        ids.append(w_stop_start + k)
        blocks.append(
            np.asarray(ids, dtype=np.int64)
        )

    global_ids = np.asarray(
        [
            decode_step,
            b_decode,
            stop_step,
            b_stop,
        ],
        dtype=np.int64,
    )
    return blocks, global_ids


def _check(
    parent_a: ConsequenceAwareDirectGene,
    parent_b: ConsequenceAwareDirectGene,
    anchor: ConsequenceAwareDirectGene,
) -> None:
    if parent_a.hidden_dim != parent_b.hidden_dim:
        raise ValueError("Parent hidden dimensions must match")
    if parent_a.hidden_dim != anchor.hidden_dim:
        raise ValueError("Anchor hidden dimension must match parents")


def _trim_delta(
    delta: np.ndarray,
    keep_fraction: float,
) -> np.ndarray:
    keep_fraction = float(
        np.clip(keep_fraction, 0.0, 1.0)
    )
    if keep_fraction <= 0.0:
        return np.zeros_like(delta)
    if keep_fraction >= 1.0:
        return delta.copy()

    threshold = float(
        np.quantile(
            np.abs(delta),
            1.0 - keep_fraction,
        )
    )
    return np.where(
        np.abs(delta) >= threshold,
        delta,
        0.0,
    )


def _ties_pair_merge(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    keep_fraction: float,
) -> np.ndarray:
    a = _trim_delta(delta_a, keep_fraction)
    b = _trim_delta(delta_b, keep_fraction)

    elected = np.sign(a + b)
    a_ok = (a != 0.0) & (
        (np.sign(a) == elected)
        | (elected == 0.0)
    )
    b_ok = (b != 0.0) & (
        (np.sign(b) == elected)
        | (elected == 0.0)
    )

    numerator = (
        np.where(a_ok, a, 0.0)
        + np.where(b_ok, b, 0.0)
    )
    count = (
        a_ok.astype(np.float64)
        + b_ok.astype(np.float64)
    )
    return numerator / np.maximum(count, 1.0)


def recombine(
    parent_a: ConsequenceAwareDirectGene,
    parent_b: ConsequenceAwareDirectGene,
    anchor: ConsequenceAwareDirectGene,
    rng: np.random.Generator,
    *,
    operator: str | None = None,
) -> RecombinationResult:
    """
    Produce one child using one sampled model-merging operator.

    The anchor is the common V1.8 ancestor. Delta-based operators therefore
    recombine inherited changes rather than adding two complete parameter
    vectors.

    No external score is used here. Survival is decided later by the external
    multi-capability evaluator.
    """
    _check(parent_a, parent_b, anchor)

    if operator is None:
        operator = OPERATORS[
            int(rng.integers(0, len(OPERATORS)))
        ]
    if operator not in OPERATORS:
        raise ValueError(f"Unknown recombination operator: {operator}")

    a = parent_a.vector_data
    b = parent_b.vector_data
    base = anchor.vector_data
    h = parent_a.hidden_dim
    blocks, globals_ = hidden_block_indices(h)

    metadata: dict[str, Any] = {
        "operator": operator,
        "hidden_dim": h,
    }

    if operator == "parameter_blend":
        alpha = rng.uniform(
            0.0,
            1.0,
            size=a.shape,
        )
        child = alpha * a + (1.0 - alpha) * b
        metadata["alpha_mean"] = float(np.mean(alpha))
        metadata["alpha_std"] = float(np.std(alpha))

    elif operator == "block_pick":
        child = base.copy()
        sources: list[str] = []
        for ids in blocks:
            use_a = bool(rng.integers(0, 2))
            child[ids] = (
                a[ids] if use_a else b[ids]
            )
            sources.append(
                "A" if use_a else "B"
            )
        use_a = bool(rng.integers(0, 2))
        child[globals_] = (
            a[globals_]
            if use_a
            else b[globals_]
        )
        metadata["block_sources"] = sources
        metadata["global_source"] = (
            "A" if use_a else "B"
        )

    elif operator == "block_blend":
        child = base.copy()
        alphas: list[float] = []
        for ids in blocks:
            alpha = float(rng.uniform(0.0, 1.0))
            child[ids] = (
                alpha * a[ids]
                + (1.0 - alpha) * b[ids]
            )
            alphas.append(alpha)
        global_alpha = float(
            rng.uniform(0.0, 1.0)
        )
        child[globals_] = (
            global_alpha * a[globals_]
            + (1.0 - global_alpha) * b[globals_]
        )
        metadata["block_alphas"] = alphas
        metadata["global_alpha"] = global_alpha

    elif operator == "ancestor_delta":
        delta_a = a - base
        delta_b = b - base
        alpha_a = float(
            rng.uniform(0.25, 1.25)
        )
        alpha_b = float(
            rng.uniform(0.25, 1.25)
        )
        child = (
            base
            + alpha_a * delta_a
            + alpha_b * delta_b
        )
        metadata["alpha_a"] = alpha_a
        metadata["alpha_b"] = alpha_b

    elif operator == "ties_delta":
        keep_fraction = float(
            rng.uniform(0.35, 0.90)
        )
        scale = float(
            rng.uniform(0.75, 1.25)
        )
        merged = _ties_pair_merge(
            a - base,
            b - base,
            keep_fraction,
        )
        child = base + scale * merged
        metadata["keep_fraction"] = keep_fraction
        metadata["scale"] = scale

    elif operator == "dare_delta":
        drop_rate = float(
            rng.uniform(0.05, 0.50)
        )
        alpha_a = float(
            rng.uniform(0.50, 1.10)
        )
        alpha_b = float(
            rng.uniform(0.50, 1.10)
        )
        keep = max(1.0 - drop_rate, 1e-12)
        delta_a = a - base
        delta_b = b - base
        mask_a = (
            rng.random(a.shape)
            >= drop_rate
        )
        mask_b = (
            rng.random(b.shape)
            >= drop_rate
        )
        dare_a = delta_a * mask_a / keep
        dare_b = delta_b * mask_b / keep
        child = (
            base
            + alpha_a * dare_a
            + alpha_b * dare_b
        )
        metadata["drop_rate"] = drop_rate
        metadata["alpha_a"] = alpha_a
        metadata["alpha_b"] = alpha_b
        metadata["kept_a"] = float(
            np.mean(mask_a)
        )
        metadata["kept_b"] = float(
            np.mean(mask_b)
        )

    else:
        raise AssertionError(operator)

    return RecombinationResult(
        gene=ConsequenceAwareDirectGene(
            np.asarray(child, dtype=np.float64),
            hidden_dim=h,
        ),
        operator=operator,
        metadata=metadata,
    )


def offspring_family(
    parent_a: ConsequenceAwareDirectGene,
    parent_b: ConsequenceAwareDirectGene,
    anchor: ConsequenceAwareDirectGene,
    rng: np.random.Generator,
    *,
    children: int = 4,
    mutation_sigma: float = 0.0,
    mutation_rate: float = 0.05,
) -> list[RecombinationResult]:
    if children <= 0:
        raise ValueError("children must be positive")

    order = rng.permutation(len(OPERATORS))
    results: list[RecombinationResult] = []

    for child_idx in range(children):
        operator = OPERATORS[
            int(order[child_idx % len(order)])
        ]
        result = recombine(
            parent_a,
            parent_b,
            anchor,
            rng,
            operator=operator,
        )

        gene = result.gene
        metadata = dict(result.metadata)
        metadata["child_index"] = child_idx

        if mutation_sigma > 0.0:
            gene = gene.mutated(
                rng,
                sigma=mutation_sigma,
                mutation_rate=mutation_rate,
            )
            metadata["post_merge_mutation_sigma"] = (
                float(mutation_sigma)
            )
            metadata["post_merge_mutation_rate"] = (
                float(mutation_rate)
            )

        results.append(
            RecombinationResult(
                gene=gene,
                operator=result.operator,
                metadata=metadata,
            )
        )

    return results
