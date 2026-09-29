from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBSERVATION_NAMES = (
    "distance_norm",
    "robot_load_norm",
    "competition_norm",
    "slack_norm",
)


@dataclass(frozen=True)
class Gene:
    """Four-parameter shared bidder for homogeneous robots.

    No bias is used because allocation depends only on relative bid ordering.
    Adding the same constant to every robot-task pair cannot change argmax.
    """

    weights: np.ndarray

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.shape != (4,):
            raise ValueError(f"Gene weights must have shape (4,), got {w.shape}")
        object.__setattr__(self, "weights", w)

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 1.0) -> "Gene":
        return cls(rng.normal(0.0, scale, size=4))

    def bid(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        if obs.shape[-1] != 4:
            raise ValueError(f"Expected observation dimension 4, got {obs.shape}")
        return obs @ self.weights

    def crossed(self, other: "Gene", rng: np.random.Generator) -> "Gene":
        alpha = rng.uniform(0.0, 1.0, size=4)
        return Gene(alpha * self.weights + (1.0 - alpha) * other.weights)

    def mutated(
        self,
        rng: np.random.Generator,
        sigma: float,
        mutation_rate: float = 0.35,
    ) -> "Gene":
        mask = rng.random(4) < mutation_rate
        delta = rng.normal(0.0, sigma, size=4) * mask
        return Gene(self.weights + delta)

    def vector(self) -> np.ndarray:
        return self.weights.copy()

    def key(self, decimals: int = 10) -> tuple[float, ...]:
        return tuple(np.round(self.vector(), decimals=decimals).tolist())

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_names": list(OBSERVATION_NAMES),
            "weights": self.weights.tolist(),
        }
