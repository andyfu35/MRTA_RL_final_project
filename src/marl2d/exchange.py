from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from . import AGENT_IDS

PolicySnapshot = dict[str, torch.Tensor]
PolicySet = dict[str, PolicySnapshot]


def clone_snapshot(snapshot: PolicySnapshot) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in snapshot.items())


def clone_policy_set(
    policy_set: PolicySet,
    agent_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, OrderedDict[str, torch.Tensor]]:
    ids = tuple(agent_ids or AGENT_IDS)
    return {agent_id: clone_snapshot(policy_set[agent_id]) for agent_id in ids}


class MockPolicyExchange:
    """In-memory stand-in for the future ROS2 policy/status transport."""

    def __init__(
        self,
        initial_policy_set: PolicySet,
        version: int = 0,
        agent_ids: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.agent_ids = tuple(agent_ids or AGENT_IDS)
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must contain unique agent identifiers")
        if set(initial_policy_set) != set(self.agent_ids):
            raise ValueError(f"initial_policy_set must contain exactly {self.agent_ids}")
        self._version = int(version)
        self._committed = clone_policy_set(initial_policy_set, self.agent_ids)
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
        if agent_id not in self.agent_ids:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        expected = self._version + 1
        if int(version) != expected:
            raise ValueError(f"Expected update version {expected}, got {version}")
        pending = self._pending.setdefault(int(version), {})
        pending[agent_id] = clone_snapshot(snapshot)
        self._metrics.setdefault(int(version), {})[agent_id] = dict(metrics or {})

    def ready_status(self, version: int) -> dict[str, bool]:
        pending = self._pending.get(int(version), {})
        return {agent_id: agent_id in pending for agent_id in self.agent_ids}

    def can_commit(self, version: int) -> bool:
        version = int(version)
        if version != self._version + 1:
            return False
        return all(self.ready_status(version).values())

    def commit(self, version: int) -> bool:
        version = int(version)
        if not self.can_commit(version):
            return False
        self._committed = clone_policy_set(self._pending[version], self.agent_ids)
        self._version = version
        del self._pending[version]
        return True

    def get_committed_policy_set(self) -> PolicySet:
        return clone_policy_set(self._committed, self.agent_ids)

    def get_round_metrics(self, version: int) -> dict[str, dict[str, Any]]:
        return {agent_id: dict(values) for agent_id, values in self._metrics.get(int(version), {}).items()}
