from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from . import AGENT_IDS

PolicySnapshot = dict[str, torch.Tensor]
PolicySet = dict[str, PolicySnapshot]


def clone_snapshot(snapshot: PolicySnapshot) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in snapshot.items())


def clone_policy_set(policy_set: PolicySet) -> dict[str, OrderedDict[str, torch.Tensor]]:
    return {agent_id: clone_snapshot(policy_set[agent_id]) for agent_id in AGENT_IDS}


class MockPolicyExchange:
    """In-memory stand-in for the future ROS2 policy/status transport."""

    def __init__(self, initial_policy_set: PolicySet, version: int = 0) -> None:
        if set(initial_policy_set) != set(AGENT_IDS):
            raise ValueError("initial_policy_set must contain all four agents")
        self._version = int(version)
        self._committed = clone_policy_set(initial_policy_set)
        self._pending: dict[int, dict[str, OrderedDict[str, torch.Tensor]]] = {}
        self._metrics: dict[int, dict[str, dict[str, Any]]] = {}

    @property
    def version(self) -> int:
        return self._version

    def publish(
        self,
        agent_id: str,
        version: int,
        snapshot: PolicySnapshot,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if agent_id not in AGENT_IDS:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        expected = self._version + 1
        if int(version) != expected:
            raise ValueError(f"Expected update version {expected}, got {version}")
        pending = self._pending.setdefault(int(version), {})
        pending[agent_id] = clone_snapshot(snapshot)
        self._metrics.setdefault(int(version), {})[agent_id] = dict(metrics or {})

    def ready_status(self, version: int) -> dict[str, bool]:
        pending = self._pending.get(int(version), {})
        return {agent_id: agent_id in pending for agent_id in AGENT_IDS}

    def can_commit(self, version: int) -> bool:
        version = int(version)
        if version != self._version + 1:
            return False
        return all(self.ready_status(version).values())

    def commit(self, version: int) -> bool:
        version = int(version)
        if not self.can_commit(version):
            return False
        self._committed = clone_policy_set(self._pending[version])
        self._version = version
        del self._pending[version]
        return True

    def get_committed_policy_set(self) -> PolicySet:
        return clone_policy_set(self._committed)

    def get_round_metrics(self, version: int) -> dict[str, dict[str, Any]]:
        return {agent_id: dict(values) for agent_id, values in self._metrics.get(int(version), {}).items()}
