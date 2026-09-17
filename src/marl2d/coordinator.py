from __future__ import annotations

from typing import Any

from .exchange import MockPolicyExchange, PolicySnapshot


class SynchronousCoordinator:
    """Round barrier: no next policy set exists until all four workers are READY."""

    def __init__(self, exchange: MockPolicyExchange) -> None:
        self.exchange = exchange

    @property
    def current_round(self) -> int:
        return self.exchange.version

    def submit_update(
        self,
        agent_id: str,
        version: int,
        snapshot: PolicySnapshot,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.exchange.publish(agent_id, version, snapshot, metrics)

    def ready_status(self, version: int) -> dict[str, bool]:
        return self.exchange.ready_status(version)

    def try_commit(self, version: int) -> bool:
        return self.exchange.commit(version)
