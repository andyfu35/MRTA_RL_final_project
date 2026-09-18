from __future__ import annotations

import base64
import hashlib
import io
import json
import zlib
from collections import OrderedDict
from typing import Any, Mapping

import torch


def snapshot_sha256(snapshot: Mapping[str, torch.Tensor]) -> str:
    payload = encode_snapshot(snapshot)
    return hashlib.sha256(payload).hexdigest()


def encode_snapshot(snapshot: Mapping[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    cpu_snapshot = OrderedDict(
        (str(key), value.detach().cpu())
        for key, value in snapshot.items()
    )
    torch.save(cpu_snapshot, buffer)
    return zlib.compress(buffer.getvalue(), level=6)


def decode_snapshot(payload: bytes) -> OrderedDict[str, torch.Tensor]:
    raw = zlib.decompress(payload)
    data = torch.load(
        io.BytesIO(raw),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(data, Mapping):
        raise ValueError("policy payload did not contain a state_dict mapping")
    return OrderedDict(
        (str(key), value.detach().cpu())
        for key, value in data.items()
    )


def build_policy_message(
    *,
    node_id: str,
    agent_id: str,
    version: int,
    snapshot: Mapping[str, torch.Tensor],
    reward_sha256: str,
    config_sha256: str,
    metrics: Mapping[str, Any] | None = None,
) -> str:
    payload = encode_snapshot(snapshot)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    message = {
        "schema": 1,
        "kind": "policy",
        "node_id": str(node_id),
        "agent_id": str(agent_id),
        "version": int(version),
        "reward_sha256": str(reward_sha256),
        "config_sha256": str(config_sha256),
        "payload_sha256": payload_sha256,
        "payload_b64": base64.b64encode(payload).decode("ascii"),
        "metrics": dict(metrics or {}),
    }
    return json.dumps(
        message,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_policy_message(
    text: str,
    *,
    expected_reward_sha256: str,
    expected_config_sha256: str,
) -> tuple[dict[str, Any], OrderedDict[str, torch.Tensor]]:
    message = json.loads(text)
    if int(message.get("schema", 0)) != 1:
        raise ValueError("unsupported policy message schema")
    if message.get("kind") != "policy":
        raise ValueError("message kind is not policy")
    if str(message.get("reward_sha256")) != expected_reward_sha256:
        raise ValueError(
            "peer reward.py hash differs from local reward.py"
        )
    if str(message.get("config_sha256")) != expected_config_sha256:
        raise ValueError(
            "peer config.json hash differs from local config.json"
        )
    payload = base64.b64decode(message["payload_b64"], validate=True)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != str(message["payload_sha256"]):
        raise ValueError("policy payload SHA256 mismatch")
    snapshot = decode_snapshot(payload)
    return message, snapshot
