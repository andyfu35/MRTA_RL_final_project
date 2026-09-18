from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import validate_two_runner_config
from .two_runner import TWO_RUNNER_IDS


@dataclass(frozen=True)
class ClusterNode:
    node_id: str
    ip: str
    agent_id: str
    train: bool


@dataclass(frozen=True)
class RuntimeConfig:
    source_path: Path
    raw: dict[str, Any]
    reward_path: Path
    reward_sha256: str
    config_sha256: str
    nodes: tuple[ClusterNode, ...]
    two_runner_cfg: dict[str, Any]

    @property
    def network(self) -> dict[str, Any]:
        return self.raw["network"]

    @property
    def training_nodes(self) -> tuple[ClusterNode, ...]:
        return tuple(node for node in self.nodes if node.train)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    source_path = Path(path).expanduser().resolve()
    raw_bytes = source_path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.json root must be an object")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("config.json schema_version must be 1")

    required = {
        "seed",
        "environment",
        "reward",
        "training",
        "joint_collection",
        "ppo",
        "validation",
        "evaluation",
        "final_test",
        "network",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"config.json missing sections: {sorted(missing)}")

    reward = raw["reward"]
    if not isinstance(reward, dict):
        raise ValueError("reward must be an object")
    reward_file = str(reward.get("file", "reward.py"))
    reward_path = (source_path.parent / reward_file).resolve()
    if not reward_path.is_file():
        raise FileNotFoundError(f"external reward file not found: {reward_path}")
    reward_sha256 = _sha256_bytes(reward_path.read_bytes())

    network = raw["network"]
    if not isinstance(network, dict):
        raise ValueError("network must be an object")
    raw_nodes = network.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
        raise ValueError("network.nodes must contain at least two computers")

    nodes: list[ClusterNode] = []
    seen_ids: set[str] = set()
    seen_ips: set[str] = set()
    seen_agents: set[str] = set()
    for item in raw_nodes:
        if not isinstance(item, dict):
            raise ValueError("each network.nodes entry must be an object")
        node = ClusterNode(
            node_id=str(item["id"]),
            ip=str(item["ip"]),
            agent_id=str(item["agent_id"]),
            train=bool(item.get("train", True)),
        )
        if node.node_id in seen_ids:
            raise ValueError(f"duplicate node id: {node.node_id}")
        if node.ip in seen_ips:
            raise ValueError(f"duplicate node ip: {node.ip}")
        if node.train and node.agent_id in seen_agents:
            raise ValueError(
                f"duplicate training agent_id: {node.agent_id}"
            )
        seen_ids.add(node.node_id)
        seen_ips.add(node.ip)
        if node.train:
            seen_agents.add(node.agent_id)
        nodes.append(node)

    # Current simulation adapter is Two-Runner. The ROS2 transport itself is
    # generic and the config can already list four physical computers, but
    # only runner_0/runner_1 can be marked train=true until blocker dynamics
    # and observations are implemented.
    training_agents = {
        node.agent_id for node in nodes if node.train
    }
    if training_agents != set(TWO_RUNNER_IDS):
        raise ValueError(
            "current Two-Runner training adapter requires exactly "
            f"{TWO_RUNNER_IDS} as train=true agents; other listed computers "
            "must use train=false until their agent adapter exists"
        )

    joint = raw["joint_collection"]
    parallel_envs = int(joint["parallel_envs"])
    rollout_steps = int(joint["rollout_steps"])
    samples_per_update = parallel_envs * rollout_steps

    two_runner_cfg = {
        "seed": int(raw["seed"]),
        "environment": raw["environment"],
        "two_runner_reward": {
            "plugin_path": str(reward_path),
            "params": dict(reward.get("params", {})),
        },
        "training": {
            "samples_per_update": samples_per_update,
            "rounds": int(raw["training"]["rounds"]),
            "checkpoint_every": int(
                raw["training"].get("checkpoint_every", 1)
            ),
        },
        "joint_collection": {
            "enabled": True,
            "parallel_envs": parallel_envs,
            "rollout_steps": rollout_steps,
        },
        # Historical independent workers remain constructible for the
        # trainer/checkpoint API, although distributed runtime uses the
        # shared fixed-horizon collector.
        "collection_profiles": {
            aid: {"parallel_envs": parallel_envs}
            for aid in TWO_RUNNER_IDS
        },
        "ppo": raw["ppo"],
        "validation": raw["validation"],
        "evaluation": raw["evaluation"],
        "final_test": raw["final_test"],
    }
    validate_two_runner_config(two_runner_cfg)

    if int(network.get("ros_domain_id", 0)) < 0:
        raise ValueError("network.ros_domain_id must be >= 0")
    if float(network.get("barrier_timeout_s", 120.0)) <= 0:
        raise ValueError("network.barrier_timeout_s must be positive")

    return RuntimeConfig(
        source_path=source_path,
        raw=raw,
        reward_path=reward_path,
        reward_sha256=reward_sha256,
        config_sha256=_sha256_bytes(raw_bytes),
        nodes=tuple(nodes),
        two_runner_cfg=two_runner_cfg,
    )


def local_ipv4_addresses(
    peer_ips: list[str] | tuple[str, ...] | None = None,
) -> set[str]:
    addresses = {"127.0.0.1"}
    try:
        for item in socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=socket.AF_INET,
        ):
            addresses.add(str(item[4][0]))
    except OSError:
        pass

    # This does not send traffic. connect() on UDP only asks the kernel which
    # local interface would route to a peer. Prefer configured LAN peers so
    # auto-identification works on an offline ROS2 network with no Internet.
    probes = list(peer_ips or ())
    probes.extend(["1.1.1.1", "8.8.8.8"])
    for probe in probes:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((probe, 9))
            addresses.add(str(sock.getsockname()[0]))
        except OSError:
            pass
        finally:
            sock.close()
    return addresses


def resolve_local_node(
    cfg: RuntimeConfig,
    *,
    explicit_node_id: str | None = None,
    local_ips: set[str] | None = None,
) -> ClusterNode:
    requested = explicit_node_id or os.environ.get("MARL2D_NODE_ID")
    if requested:
        matches = [node for node in cfg.nodes if node.node_id == requested]
        if len(matches) != 1:
            raise ValueError(
                f"node id {requested!r} is not present in config.json"
            )
        return matches[0]

    ips = (
        local_ipv4_addresses(
            tuple(node.ip for node in cfg.nodes)
        )
        if local_ips is None
        else set(local_ips)
    )
    matches = [node for node in cfg.nodes if node.ip in ips]
    if len(matches) != 1:
        raise RuntimeError(
            "could not uniquely identify this computer from network.nodes. "
            f"local IPv4 addresses={sorted(ips)}, matched="
            f"{[(x.node_id, x.ip) for x in matches]}. "
            "Fix the IP list or start with --node-id."
        )
    return matches[0]
