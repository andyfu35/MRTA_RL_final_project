from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl2d.gene_mrta_v18.direct_gene import (
    ConsequenceAwareDirectGene,
    OBS_DIM,
)


@dataclass(frozen=True)
class RouteTailDirectGene(ConsequenceAwareDirectGene):
    """
    V1.13 route-tail decoder.

    Parameter layout stays exactly compatible with V1.8-V1.11 (148 scalars
    for hidden_dim=8). The only decoder-semantic change is step
    normalization: a planning round can contain up to T task insertions, so
    step is normalized by task count rather than robot count.

    Rows are intentionally allowed to remain open across decoder steps.
    """

    @classmethod
    def from_v18(
        cls,
        gene: ConsequenceAwareDirectGene,
    ) -> "RouteTailDirectGene":
        return cls(
            np.asarray(
                gene.vector_data,
                dtype=np.float64,
            ).copy(),
            hidden_dim=gene.hidden_dim,
        )

    def action_logits(
        self,
        observations: np.ndarray,
        eligible: np.ndarray,
        row_open: np.ndarray,
        col_open: np.ndarray,
        step: int,
    ) -> tuple[np.ndarray, float]:
        obs = np.asarray(
            observations,
            dtype=np.float64,
        )
        eligible = np.asarray(
            eligible,
            dtype=bool,
        )

        if (
            obs.ndim != 3
            or obs.shape[-1] != OBS_DIM
        ):
            raise ValueError(
                "observations must have shape "
                f"(R,T,{OBS_DIM}), got {obs.shape}"
            )

        r_count, t_count, _ = (
            obs.shape
        )
        if eligible.shape != (
            r_count,
            t_count,
        ):
            raise ValueError(
                "eligible shape mismatch"
            )
        if row_open.shape != (
            r_count,
        ):
            raise ValueError(
                "row_open shape mismatch"
            )
        if col_open.shape != (
            t_count,
        ):
            raise ValueError(
                "col_open shape mismatch"
            )

        active = (
            eligible
            & row_open[:, None]
            & col_open[None, :]
        )
        (
            w_pair,
            b_pair,
            w_decode,
            b_decode,
            w_stop,
            b_stop,
        ) = self._unpack()

        pair_h = np.tanh(
            obs @ w_pair
            + b_pair
        )

        global_h = self._masked_mean(
            pair_h,
            active,
            axis=(0, 1),
        )
        robot_h = self._masked_mean(
            pair_h,
            active,
            axis=1,
        )
        task_h = self._masked_mean(
            pair_h,
            active,
            axis=0,
        )

        step_norm = float(step) / max(
            1,
            t_count,
        )

        decoder_input = np.concatenate(
            [
                pair_h,
                np.broadcast_to(
                    global_h,
                    pair_h.shape,
                ),
                np.broadcast_to(
                    robot_h[:, None, :],
                    pair_h.shape,
                ),
                np.broadcast_to(
                    task_h[None, :, :],
                    pair_h.shape,
                ),
                np.full(
                    (
                        r_count,
                        t_count,
                        1,
                    ),
                    step_norm,
                    dtype=np.float64,
                ),
            ],
            axis=-1,
        )
        logits = (
            decoder_input
            @ w_decode
            + b_decode
        )

        stop_input = np.concatenate(
            [
                global_h,
                np.asarray(
                    [step_norm],
                    dtype=np.float64,
                ),
            ]
        )
        stop_logit = float(
            stop_input
            @ w_stop
            + b_stop
        )

        return (
            np.where(
                active,
                logits,
                -np.inf,
            ),
            stop_logit,
        )
