from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import AGENT_IDS
from .coordinator import SynchronousCoordinator
from .exchange import MockPolicyExchange, PolicySet
from .policy import snapshot_model
from .worker import AgentWorker


class DistributedTrainer:
    """Single-process emulator of four future distributed PPO computers."""

    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        seed = int(cfg.get("seed", 0))
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.workers: dict[str, AgentWorker] = {}
        for agent_id in AGENT_IDS:
            self.workers[agent_id] = AgentWorker(agent_id, cfg, device=device)

        initial_policy_set = {
            agent_id: snapshot_model(self.workers[agent_id].model)
            for agent_id in AGENT_IDS
        }
        self.exchange = MockPolicyExchange(initial_policy_set, version=0)
        self.coordinator = SynchronousCoordinator(self.exchange)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
        safe: dict[str, float | int] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                safe[key] = int(value)
            else:
                safe[key] = float(value)
        return safe

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        checkpoint_path = Path(path) if path is not None else self.output_dir / "latest.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.coordinator.current_round,
                "policy_set": self.exchange.get_committed_policy_set(),
                "config": self.cfg,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def run(self, rounds: int | None = None) -> list[dict[str, Any]]:
        if rounds is None:
            rounds = int(self.cfg["training"]["rounds"])
        rounds = int(rounds)
        if rounds <= 0:
            return []

        records: list[dict[str, Any]] = []
        checkpoint_every = max(1, int(self.cfg["training"].get("checkpoint_every", 1)))

        for _ in range(rounds):
            current_round = self.coordinator.current_round
            next_version = current_round + 1
            frozen_policy_set = self.exchange.get_committed_policy_set()
            round_metrics: dict[str, dict[str, float | int]] = {}

            for agent_id in AGENT_IDS:
                snapshot, metrics = self.workers[agent_id].train_round(
                    frozen_policy_set,
                    round_index=current_round,
                )
                safe_metrics = self._json_safe_metrics(metrics)
                round_metrics[agent_id] = safe_metrics
                self.coordinator.submit_update(
                    agent_id,
                    next_version,
                    snapshot,
                    safe_metrics,
                )

            if not self.coordinator.try_commit(next_version):
                status = self.coordinator.ready_status(next_version)
                raise RuntimeError(f"Synchronous commit failed for round {next_version}: {status}")

            record = {
                "round": next_version,
                "agents": round_metrics,
            }
            records.append(record)
            self.history.append(record)
            self._append_metrics(record)

            self.save_checkpoint(self.output_dir / "latest.pt")
            if next_version % checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / f"round_{next_version:05d}.pt")

        return records


def load_checkpoint(path: str | Path) -> tuple[int, PolicySet, dict[str, Any]]:
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = int(data["version"])
    policy_set = data["policy_set"]
    cfg = data["config"]
    return version, policy_set, cfg
