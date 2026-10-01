from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBS_DIM = 8


@dataclass(frozen=True)
class DirectAssignmentGene:
    """
    Small permutation-equivariant autoregressive assignment policy.

    This Gene directly emits one (robot, task) action at a time.  There is no
    external greedy/Hungarian matcher.  After every selected pair, row/task
    masks are updated and policy logits are recomputed from the remaining
    joint assignment state.
    """

    vector_data: np.ndarray
    hidden_dim: int = 8

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector_data, dtype=np.float64)
        expected = self.parameter_count(self.hidden_dim)
        if vector.shape != (expected,):
            raise ValueError(
                f"Direct Gene vector must have shape ({expected},), got {vector.shape}"
            )
        object.__setattr__(self, "vector_data", vector)

    @staticmethod
    def parameter_count(hidden_dim: int = 8) -> int:
        # pair encoder W,b + decoder on [pair, global, robot, task, step] + bias
        return OBS_DIM * hidden_dim + hidden_dim + (4 * hidden_dim + 1) + 1

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        hidden_dim: int = 8,
        scale: float = 0.35,
    ) -> "DirectAssignmentGene":
        return cls(
            rng.normal(0.0, scale, size=cls.parameter_count(hidden_dim)),
            hidden_dim=hidden_dim,
        )

    def _unpack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        h = self.hidden_dim
        offset = 0
        w_pair = self.vector_data[offset : offset + OBS_DIM * h].reshape(
            OBS_DIM, h
        )
        offset += OBS_DIM * h
        b_pair = self.vector_data[offset : offset + h]
        offset += h
        w_decode = self.vector_data[offset : offset + 4 * h + 1]
        offset += 4 * h + 1
        b_decode = float(self.vector_data[offset])
        return w_pair, b_pair, w_decode, b_decode

    @staticmethod
    def _masked_mean(
        values: np.ndarray,
        mask: np.ndarray,
        axis: int | tuple[int, ...],
    ) -> np.ndarray:
        mask_f = mask.astype(np.float64)
        while mask_f.ndim < values.ndim:
            mask_f = mask_f[..., None]
        numerator = np.sum(values * mask_f, axis=axis)
        denominator = np.sum(mask_f, axis=axis)
        return numerator / np.maximum(denominator, 1.0)

    def action_logits(
        self,
        observations: np.ndarray,
        eligible: np.ndarray,
        row_open: np.ndarray,
        col_open: np.ndarray,
        step: int,
    ) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        eligible = np.asarray(eligible, dtype=bool)
        if obs.ndim != 3 or obs.shape[-1] != OBS_DIM:
            raise ValueError(
                f"observations must have shape (R,T,{OBS_DIM}), got {obs.shape}"
            )
        r_count, t_count, _ = obs.shape
        if eligible.shape != (r_count, t_count):
            raise ValueError("eligible shape mismatch")
        if row_open.shape != (r_count,) or col_open.shape != (t_count,):
            raise ValueError("open-mask shape mismatch")

        active = eligible & row_open[:, None] & col_open[None, :]
        w_pair, b_pair, w_decode, b_decode = self._unpack()
        pair_h = np.tanh(obs @ w_pair + b_pair)

        global_h = self._masked_mean(pair_h, active, axis=(0, 1))
        robot_h = self._masked_mean(pair_h, active, axis=1)
        task_h = self._masked_mean(pair_h, active, axis=0)
        step_norm = float(step) / max(1, r_count)

        global_grid = np.broadcast_to(global_h, pair_h.shape)
        robot_grid = np.broadcast_to(robot_h[:, None, :], pair_h.shape)
        task_grid = np.broadcast_to(task_h[None, :, :], pair_h.shape)
        step_grid = np.full(
            (r_count, t_count, 1),
            step_norm,
            dtype=np.float64,
        )
        decoder_input = np.concatenate(
            [pair_h, global_grid, robot_grid, task_grid, step_grid],
            axis=-1,
        )
        logits = decoder_input @ w_decode + b_decode
        return np.where(active, logits, -np.inf)

    def assign(
        self,
        observations: np.ndarray,
        eligible: np.ndarray,
        free: np.ndarray,
        task_available: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Emit the joint event assignment autoregressively.

        The argmax is policy decoding, not a separate matching optimizer:
        logits are recomputed after each action from the changed joint mask.
        """
        row_open = np.asarray(free, dtype=bool).copy()
        col_open = np.asarray(task_available, dtype=bool).copy()
        assignments: list[tuple[int, int]] = []
        task_count = observations.shape[1]

        for step in range(int(np.sum(row_open))):
            logits = self.action_logits(
                observations,
                eligible,
                row_open,
                col_open,
                step,
            )
            flat = int(np.argmax(logits))
            best = float(logits.flat[flat])
            if not np.isfinite(best):
                break
            robot = flat // task_count
            task = flat % task_count
            assignments.append((robot, task))
            row_open[robot] = False
            col_open[task] = False

        return assignments

    def crossed(
        self,
        other: "DirectAssignmentGene",
        rng: np.random.Generator,
    ) -> "DirectAssignmentGene":
        if other.hidden_dim != self.hidden_dim:
            raise ValueError("hidden dimensions must match")
        alpha = rng.uniform(0.0, 1.0, size=self.vector_data.shape)
        return DirectAssignmentGene(
            alpha * self.vector_data + (1.0 - alpha) * other.vector_data,
            hidden_dim=self.hidden_dim,
        )

    def mutated(
        self,
        rng: np.random.Generator,
        sigma: float,
        mutation_rate: float = 0.20,
    ) -> "DirectAssignmentGene":
        mask = rng.random(self.vector_data.shape) < mutation_rate
        delta = rng.normal(0.0, sigma, self.vector_data.shape) * mask
        return DirectAssignmentGene(
            self.vector_data + delta,
            hidden_dim=self.hidden_dim,
        )

    def key(self, decimals: int = 9) -> tuple[float, ...]:
        return tuple(np.round(self.vector_data, decimals=decimals).tolist())

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "autoregressive_direct_assignment_gene",
            "hidden_dim": self.hidden_dim,
            "parameter_count": self.parameter_count(self.hidden_dim),
            "parameters": self.vector_data.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DirectAssignmentGene":
        return cls(
            np.asarray(data["parameters"], dtype=np.float64),
            hidden_dim=int(data["hidden_dim"]),
        )
