from __future__ import annotations

import numpy as np

from marl2d.gene_mrta_v113.direct_gene import (
    RouteTailDirectGene,
    OBS_DIM,
)


class ScalableRouteTailGene(RouteTailDirectGene):
    """
    V1.20 decoder for variable-size candidate sets.

    The parameter layout remains exactly the same 148 scalars at hidden_dim=8.
    V1.20 may score only a candidate subset of the remaining tasks for
    scalability, so step normalization must use the *full original task count*
    rather than the current candidate-column count.
    """

    @classmethod
    def from_gene(
        cls,
        gene: RouteTailDirectGene,
    ) -> "ScalableRouteTailGene":
        return cls(
            np.asarray(
                gene.vector_data,
                dtype=np.float64,
            ).copy(),
            hidden_dim=gene.hidden_dim,
        )

    def action_logits_total(
        self,
        observations: np.ndarray,
        eligible: np.ndarray,
        row_open: np.ndarray,
        col_open: np.ndarray,
        *,
        step: int,
        total_tasks: int,
    ) -> tuple[np.ndarray, float]:
        obs = np.asarray(
            observations,
            dtype=np.float64,
        )
        eligible = np.asarray(
            eligible,
            dtype=bool,
        )
        row_open = np.asarray(
            row_open,
            dtype=bool,
        )
        col_open = np.asarray(
            col_open,
            dtype=bool,
        )

        if (
            obs.ndim != 3
            or obs.shape[-1] != OBS_DIM
        ):
            raise ValueError(
                f"observations must have shape (R,C,{OBS_DIM}), got {obs.shape}"
            )

        r_count, c_count, _ = obs.shape
        if eligible.shape != (r_count, c_count):
            raise ValueError("eligible shape mismatch")
        if row_open.shape != (r_count,):
            raise ValueError("row_open shape mismatch")
        if col_open.shape != (c_count,):
            raise ValueError("col_open shape mismatch")
        if total_tasks <= 0:
            raise ValueError("total_tasks must be positive")

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

        step_norm = float(step) / max(1, total_tasks)

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
                        c_count,
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
