from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np

from .reward_api import REWARD_API_VERSION, RewardContext, RewardOverride


@dataclass(frozen=True)
class LoadedRewardTerm:
    name: str
    mode: str
    order: int
    fn: Callable[[RewardContext], Any]


@dataclass(frozen=True)
class RewardEvaluation:
    total: np.ndarray
    terms: dict[str, np.ndarray]


class RewardEngine:
    """Loads reward terms from an external Python file.

    The file is loaded once at process start. Restart the node after editing
    reward.py so every training machine uses the exact same reward source.
    """

    def __init__(
        self,
        terms: list[LoadedRewardTerm],
        *,
        params: Mapping[str, Any] | None = None,
        source_path: str | Path | None = None,
        source_sha256: str | None = None,
    ) -> None:
        if not terms:
            raise ValueError("reward plugin registered no reward terms")
        self.terms = sorted(terms, key=lambda item: (item.order, item.name))
        self.params = dict(params or {})
        self.source_path = None if source_path is None else str(source_path)
        self.source_sha256 = source_sha256

    @staticmethod
    def _load_module(path: Path) -> ModuleType:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = f"_marl2d_reward_{digest[:16]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import reward plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> "RewardEngine":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"reward plugin not found: {path}")
        module = cls._load_module(path)
        api_version = int(getattr(module, "REWARD_API_VERSION", -1))
        if api_version != REWARD_API_VERSION:
            raise ValueError(
                f"reward plugin API version {api_version} is incompatible; "
                f"expected {REWARD_API_VERSION}"
            )

        terms: list[LoadedRewardTerm] = []
        for value in module.__dict__.values():
            if not callable(value) or not bool(
                getattr(value, "__marl2d_reward_term__", False)
            ):
                continue
            terms.append(
                LoadedRewardTerm(
                    name=str(getattr(value, "__marl2d_reward_name__")),
                    mode=str(getattr(value, "__marl2d_reward_mode__")),
                    order=int(getattr(value, "__marl2d_reward_order__")),
                    fn=value,
                )
            )
        return cls(
            terms,
            params=params,
            source_path=path,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    @staticmethod
    def _agent_array(
        value: Any,
        shape: tuple[int, int],
        *,
        name: str,
        dtype=np.float32,
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=dtype)
        try:
            arr = np.broadcast_to(arr, shape)
        except ValueError as exc:
            raise ValueError(
                f"reward term {name!r} returned shape {arr.shape}, "
                f"not broadcastable to {shape}"
            ) from exc
        return np.asarray(arr, dtype=dtype)

    def evaluate(self, context: RewardContext) -> RewardEvaluation:
        # Inject immutable-by-convention plugin parameters for this evaluation.
        ctx = RewardContext(
            **{
                **context.__dict__,
                "params": self.params,
            }
        )
        total = np.zeros(ctx.agent_shape, dtype=np.float32)
        breakdown: dict[str, np.ndarray] = {}

        for term in self.terms:
            result = term.fn(ctx)
            if term.mode == "add":
                values = self._agent_array(
                    result, ctx.agent_shape, name=term.name
                )
                total += values
                breakdown[term.name] = values.copy()
                continue

            if term.mode != "override":
                raise ValueError(
                    f"reward term {term.name!r} has unknown mode {term.mode!r}"
                )
            if not isinstance(result, RewardOverride):
                raise TypeError(
                    f"override reward term {term.name!r} must return RewardOverride"
                )
            mask = self._agent_array(
                result.mask,
                ctx.agent_shape,
                name=f"{term.name}.mask",
                dtype=bool,
            ).astype(bool, copy=False)
            values = self._agent_array(
                result.values,
                ctx.agent_shape,
                name=f"{term.name}.values",
            )
            before = total.copy()
            total[mask] = values[mask]
            breakdown[term.name] = np.where(mask, total - before, 0.0).astype(
                np.float32
            )

        if not np.all(np.isfinite(total)):
            raise ValueError("reward plugin produced non-finite rewards")
        return RewardEvaluation(total=total, terms=breakdown)
