from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBSERVATION_NAMES = (
    "euclidean_distance_norm",
    "path_cost_norm",
    "service_time_norm",
    "priority_norm",
    "deadline_remaining_norm",
    "battery_remaining_norm",
    "robot_workload_norm",
    "competition_norm",
)


@dataclass(frozen=True)
class Gene:
    """Eight-parameter battery-aware time-optimal shared bidder for homogeneous robots."""

    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (8,):
            raise ValueError(f"Gene weights must have shape (8,), got {weights.shape}")
        object.__setattr__(self, "weights", weights)

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 1.0) -> "Gene":
        return cls(rng.normal(0.0, scale, size=8))

    def bid(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        if obs.shape[-1] != 8:
            raise ValueError(f"Expected observation dimension 8, got {obs.shape}")
        return obs @ self.weights

    def crossed(self, other: "Gene", rng: np.random.Generator) -> "Gene":
        alpha = rng.uniform(0.0, 1.0, size=8)
        return Gene(alpha * self.weights + (1.0 - alpha) * other.weights)

    def mutated(
        self,
        rng: np.random.Generator,
        sigma: float,
        mutation_rate: float = 0.35,
    ) -> "Gene":
        mask = rng.random(8) < mutation_rate
        delta = rng.normal(0.0, sigma, size=8) * mask
        return Gene(self.weights + delta)

    def vector(self) -> np.ndarray:
        return self.weights.copy()

    def key(self, decimals: int = 10) -> tuple[float, ...]:
        return tuple(np.round(self.weights, decimals=decimals).tolist())

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_names": list(OBSERVATION_NAMES),
            "weights": self.weights.tolist(),
        }
