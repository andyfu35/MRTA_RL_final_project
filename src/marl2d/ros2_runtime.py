from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .model_wire import build_policy_message, parse_policy_message
from .policy import ActorCritic, load_model_snapshot, snapshot_model
from .ppo import ppo_update
from .runtime_config import ClusterNode, RuntimeConfig
from .two_runner import (
    TWO_RUNNER_IDS,
    SharedTwoRunnerCollector,
    evaluate_two_runner,
)


class Ros2PolicyBus:
    """ROS2 transport for versioned model exchange and training status.

    rclpy is imported lazily so the core simulator/tests still work on macOS
    without a ROS2 installation.
    """

    def __init__(
        self,
        runtime: RuntimeConfig,
        local: ClusterNode,
    ) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from std_msgs.msg import String
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are unavailable. On Ubuntu, source "
                "/opt/ros/<distro>/setup.bash before starting marl2d_node."
            ) from exc

        self._rclpy = rclpy
        self._String = String
        if not rclpy.ok():
            rclpy.init(args=None)

        safe_name = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in local.node_id
        )
        self.node: Node = rclpy.create_node(f"marl2d_{safe_name}")
        self.runtime = runtime
        self.local = local
        self.expected_by_agent = {
            item.agent_id: item.node_id
            for item in runtime.training_nodes
        }
        self.policy_cache: dict[
            str, tuple[int, OrderedDict[str, torch.Tensor], dict[str, Any]]
        ] = {}
        self.peer_status: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None

        policy_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.policy_publishers = {}
        if local.train:
            topic = f"/marl2d/policy/{local.agent_id}"
            self.policy_publishers[local.agent_id] = (
                self.node.create_publisher(String, topic, policy_qos)
            )

        self.policy_subscriptions = []
        for agent_id in self.expected_by_agent:
            topic = f"/marl2d/policy/{agent_id}"
            self.policy_subscriptions.append(
                self.node.create_subscription(
                    String,
                    topic,
                    lambda msg, aid=agent_id: self._on_policy(
                        aid, msg.data
                    ),
                    policy_qos,
                )
            )

        self.status_publisher = self.node.create_publisher(
            String, "/marl2d/status", status_qos
        )
        self.status_subscription = self.node.create_subscription(
            String,
            "/marl2d/status",
            lambda msg: self._on_status(msg.data),
            status_qos,
        )

    def _on_policy(self, topic_agent: str, text: str) -> None:
        try:
            message, snapshot = parse_policy_message(
                text,
                expected_reward_sha256=self.runtime.reward_sha256,
                expected_config_sha256=self.runtime.config_sha256,
            )
            agent_id = str(message["agent_id"])
            node_id = str(message["node_id"])
            if agent_id != topic_agent:
                raise ValueError(
                    f"policy topic {topic_agent} carried agent {agent_id}"
                )
            expected_node = self.expected_by_agent.get(agent_id)
            if expected_node != node_id:
                raise ValueError(
                    f"agent {agent_id} expected from {expected_node}, "
                    f"received from {node_id}"
                )
            version = int(message["version"])
            old = self.policy_cache.get(agent_id)
            if old is None or version >= old[0]:
                self.policy_cache[agent_id] = (
                    version,
                    snapshot,
                    dict(message.get("metrics", {})),
                )
        except Exception as exc:
            self.last_error = f"policy receive error: {exc}"
            self.node.get_logger().error(self.last_error)

    def _on_status(self, text: str) -> None:
        try:
            message = json.loads(text)
            if int(message.get("schema", 0)) != 1:
                return
            if message.get("kind") != "status":
                return
            node_id = str(message["node_id"])
            if node_id not in {item.node_id for item in self.runtime.nodes}:
                return
            if message.get("reward_sha256") != self.runtime.reward_sha256:
                raise ValueError(
                    f"node {node_id} has different reward.py"
                )
            if message.get("config_sha256") != self.runtime.config_sha256:
                raise ValueError(
                    f"node {node_id} has different config.json"
                )
            self.peer_status[node_id] = message
        except Exception as exc:
            self.last_error = f"status receive error: {exc}"
            self.node.get_logger().error(self.last_error)

    def publish_status(
        self,
        *,
        phase: str,
        version: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema": 1,
            "kind": "status",
            "node_id": self.local.node_id,
            "ip": self.local.ip,
            "agent_id": self.local.agent_id,
            "train": self.local.train,
            "phase": str(phase),
            "version": int(version),
            "reward_sha256": self.runtime.reward_sha256,
            "config_sha256": self.runtime.config_sha256,
            "metrics": metrics or {},
            "unix_time": time.time(),
        }
        msg = self._String()
        msg.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.status_publisher.publish(msg)

    def publish_policy(
        self,
        *,
        version: int,
        snapshot: OrderedDict[str, torch.Tensor],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.local.train:
            raise RuntimeError("observer node cannot publish a policy")
        text = build_policy_message(
            node_id=self.local.node_id,
            agent_id=self.local.agent_id,
            version=version,
            snapshot=snapshot,
            reward_sha256=self.runtime.reward_sha256,
            config_sha256=self.runtime.config_sha256,
            metrics=metrics,
        )
        msg = self._String()
        msg.data = text
        self.policy_publishers[self.local.agent_id].publish(msg)
        # Cache the local policy immediately rather than waiting for loopback.
        self.policy_cache[self.local.agent_id] = (
            int(version),
            snapshot,
            dict(metrics or {}),
        )

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        self._rclpy.spin_once(
            self.node,
            timeout_sec=float(timeout_sec),
        )
        if self.last_error:
            raise RuntimeError(self.last_error)

    def wait_for_policy_set(
        self,
        version: int,
        *,
        timeout_s: float,
    ) -> dict[str, OrderedDict[str, torch.Tensor]]:
        deadline = time.monotonic() + float(timeout_s)
        expected_agents = set(self.expected_by_agent)
        while True:
            self.spin_once(0.1)
            ahead = {
                aid: item[0]
                for aid, item in self.policy_cache.items()
                if aid in expected_agents and item[0] > int(version)
            }
            if ahead:
                raise RuntimeError(
                    f"peer policy advanced beyond barrier version "
                    f"{version}: {ahead}"
                )
            ready = {
                aid
                for aid, item in self.policy_cache.items()
                if aid in expected_agents and item[0] == int(version)
            }
            if ready == expected_agents:
                return {
                    aid: self.policy_cache[aid][1]
                    for aid in sorted(expected_agents)
                }
            if time.monotonic() >= deadline:
                missing = sorted(expected_agents - ready)
                raise TimeoutError(
                    f"timed out waiting for policy version {version}; "
                    f"missing agents={missing}"
                )

    def close(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()


def _local_checkpoint_path(
    runtime: RuntimeConfig,
    local: ClusterNode,
) -> Path:
    root = Path(
        runtime.raw["training"].get(
            "output_dir", "runs/distributed"
        )
    )
    return root / local.node_id / "latest.pt"


def _save_local_checkpoint(
    runtime: RuntimeConfig,
    local: ClusterNode,
    *,
    version: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    collector: SharedTwoRunnerCollector,
) -> Path:
    path = _local_checkpoint_path(runtime, local)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "version": int(version),
            "agent_id": local.agent_id,
            "reward_sha256": runtime.reward_sha256,
            "config_sha256": runtime.config_sha256,
            "model": snapshot_model(model),
            "optimizer": optimizer.state_dict(),
            "collector_state": collector.snapshot_state(),
        },
        path,
    )
    return path


def _load_local_checkpoint(
    runtime: RuntimeConfig,
    local: ClusterNode,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    collector: SharedTwoRunnerCollector,
) -> int | None:
    if not bool(runtime.raw["training"].get("auto_resume", False)):
        return None
    path = _local_checkpoint_path(runtime, local)
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("reward_sha256") != runtime.reward_sha256:
        raise ValueError("checkpoint reward.py hash does not match")
    if data.get("config_sha256") != runtime.config_sha256:
        raise ValueError("checkpoint config.json hash does not match")
    if data.get("agent_id") != local.agent_id:
        raise ValueError("checkpoint agent_id does not match local node")
    load_model_snapshot(model, data["model"])
    optimizer.load_state_dict(data["optimizer"])
    collector.restore_state(data["collector_state"])
    return int(data["version"])


def run_training_node(
    runtime: RuntimeConfig,
    local: ClusterNode,
    *,
    device: str = "cpu",
) -> None:
    if not local.train:
        raise ValueError("run_training_node requires train=true")
    if local.agent_id not in TWO_RUNNER_IDS:
        raise NotImplementedError(
            f"training adapter for {local.agent_id!r} is not implemented"
        )

    cfg = runtime.two_runner_cfg
    dev = torch.device(device)
    agent_index = TWO_RUNNER_IDS.index(local.agent_id)
    torch.manual_seed(int(cfg["seed"]) + agent_index * 10_000)
    model = ActorCritic(
        18, 2, cfg["ppo"]["hidden_sizes"]
    ).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["ppo"]["learning_rate"]),
    )
    collector = SharedTwoRunnerCollector(
        cfg,
        device=device,
    )

    resumed_version = _load_local_checkpoint(
        runtime,
        local,
        model=model,
        optimizer=optimizer,
        collector=collector,
    )
    current_version = (
        int(resumed_version) if resumed_version is not None else 0
    )

    bus = Ros2PolicyBus(runtime, local)
    timeout_s = float(
        runtime.network.get("barrier_timeout_s", 120.0)
    )
    validation_node_id = str(
        runtime.network.get(
            "validation_node_id",
            runtime.training_nodes[0].node_id,
        )
    )

    try:
        bus.publish_status(
            phase="starting",
            version=current_version,
        )
        bus.publish_policy(
            version=current_version,
            snapshot=snapshot_model(model),
            metrics={
                "resumed": resumed_version is not None,
            },
        )

        rounds = int(cfg["training"]["rounds"])
        while current_version < rounds:
            frozen = bus.wait_for_policy_set(
                current_version,
                timeout_s=timeout_s,
            )
            bus.publish_status(
                phase="collecting",
                version=current_version,
            )

            batches, collection_metrics = collector.collect_round(
                frozen,
                current_version,
            )
            load_model_snapshot(
                model, frozen[local.agent_id]
            )
            model.train()
            update_metrics = ppo_update(
                model,
                optimizer,
                batches[local.agent_id],
                cfg["ppo"],
            )
            metrics = {
                **collection_metrics[local.agent_id],
                **update_metrics,
            }
            next_version = current_version + 1
            snapshot = snapshot_model(model)
            bus.publish_policy(
                version=next_version,
                snapshot=snapshot,
                metrics=metrics,
            )
            bus.publish_status(
                phase="updated",
                version=next_version,
                metrics=metrics,
            )

            committed = bus.wait_for_policy_set(
                next_version,
                timeout_s=timeout_s,
            )
            current_version = next_version

            if (
                local.node_id == validation_node_id
                and current_version
                % int(cfg["validation"]["every"])
                == 0
            ):
                bus.publish_status(
                    phase="validating",
                    version=current_version,
                )
                summary = evaluate_two_runner(
                    cfg,
                    committed,
                    episodes=int(
                        cfg["validation"]["episodes"]
                    ),
                    seed_start=int(
                        cfg["validation"]["seed_start"]
                    ),
                    device=device,
                )
                bus.publish_status(
                    phase="validated",
                    version=current_version,
                    metrics=summary,
                )
                print(
                    f"[validation v{current_version}] "
                    f"success={summary['team_success_rate']:.3f} "
                    f"collision={summary['any_collision_rate']:.3f} "
                    f"both_dead={summary['both_dead_rate']:.3f}",
                    flush=True,
                )

            checkpoint_every = max(
                1,
                int(
                    cfg["training"].get(
                        "checkpoint_every", 1
                    )
                ),
            )
            if current_version % checkpoint_every == 0:
                path = _save_local_checkpoint(
                    runtime,
                    local,
                    version=current_version,
                    model=model,
                    optimizer=optimizer,
                    collector=collector,
                )
                print(
                    f"[{local.node_id}] committed v{current_version} "
                    f"samples={int(metrics['samples'])} "
                    f"kl={float(metrics.get('approx_kl', 0.0)):.5f} "
                    f"checkpoint={path}",
                    flush=True,
                )

        bus.publish_status(
            phase="finished",
            version=current_version,
        )
    finally:
        bus.close()


def run_observer_node(
    runtime: RuntimeConfig,
    local: ClusterNode,
) -> None:
    bus = Ros2PolicyBus(runtime, local)
    print(
        f"[{local.node_id}] observer mode; monitoring "
        f"{[n.agent_id for n in runtime.training_nodes]}",
        flush=True,
    )
    last_heartbeat = 0.0
    try:
        while True:
            bus.spin_once(0.2)
            now = time.monotonic()
            if now - last_heartbeat >= float(
                runtime.network.get(
                    "heartbeat_period_s", 1.0
                )
            ):
                versions = {
                    aid: item[0]
                    for aid, item in bus.policy_cache.items()
                }
                bus.publish_status(
                    phase="observing",
                    version=max(versions.values(), default=0),
                    metrics={"policy_versions": versions},
                )
                last_heartbeat = now
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


def run_cluster_node(
    runtime: RuntimeConfig,
    local: ClusterNode,
    *,
    device: str = "cpu",
) -> None:
    os.environ["ROS_DOMAIN_ID"] = str(
        int(runtime.network.get("ros_domain_id", 42))
    )
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    if local.train:
        run_training_node(
            runtime, local, device=device
        )
    else:
        run_observer_node(runtime, local)
