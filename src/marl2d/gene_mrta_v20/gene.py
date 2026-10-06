from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TASK_DIM = 5
ROBOT_DIM = 4


@dataclass(frozen=True)
class SetAssignmentGene:
    """Shared variable-size set policy for MRTA allocation and ordering."""

    vector_data: np.ndarray
    hidden_dim: int = 8

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector_data, dtype=np.float64)
        expected = self.parameter_count(self.hidden_dim)
        if vector.shape != (expected,):
            raise ValueError(
                f"Gene vector must have shape ({expected},), got {vector.shape}"
            )
        object.__setattr__(self, "vector_data", vector)

    @staticmethod
    def decoder_dim(hidden_dim: int) -> int:
        return 6 * hidden_dim + 2

    @staticmethod
    def parameter_count(hidden_dim: int = 8) -> int:
        task_encoder = TASK_DIM * hidden_dim + hidden_dim
        robot_encoder = ROBOT_DIM * hidden_dim + hidden_dim
        decoder = SetAssignmentGene.decoder_dim(hidden_dim) + 1
        return task_encoder + robot_encoder + decoder

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        hidden_dim: int = 8,
        scale: float = 0.35,
    ) -> "SetAssignmentGene":
        return cls(
            rng.normal(0.0, scale, size=cls.parameter_count(hidden_dim)),
            hidden_dim=hidden_dim,
        )

    def mutated(
        self,
        rng: np.random.Generator,
        sigma: float = 0.12,
        mutation_rate: float = 0.20,
    ) -> "SetAssignmentGene":
        mask = rng.random(self.vector_data.shape) < mutation_rate
        delta = rng.normal(0.0, sigma, self.vector_data.shape) * mask
        return SetAssignmentGene(self.vector_data + delta, self.hidden_dim)

    def key(self, decimals: int = 9) -> tuple[float, ...]:
        return tuple(np.round(self.vector_data, decimals=decimals).tolist())

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "variable_set_mrta_gene_v20",
            "task_schema": ["x", "y", "priority", "deadline", "service_time"],
            "robot_schema": [
                "x",
                "y",
                "accumulated_distance",
                "estimated_finish_time",
            ],
            "hidden_dim": self.hidden_dim,
            "parameter_count": self.parameter_count(self.hidden_dim),
            "parameters": self.vector_data.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SetAssignmentGene":
        return cls(
            np.asarray(data["parameters"], dtype=np.float64),
            hidden_dim=int(data["hidden_dim"]),
        )
