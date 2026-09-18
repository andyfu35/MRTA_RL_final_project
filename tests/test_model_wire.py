import json

import pytest
import torch

from marl2d.model_wire import (
    build_policy_message,
    parse_policy_message,
)
from marl2d.policy import ActorCritic, snapshot_model


def snapshot():
    torch.manual_seed(3)
    return snapshot_model(ActorCritic(18, 2, [8]))


def test_policy_message_round_trip_preserves_version_hashes_and_weights():
    source = snapshot()
    text = build_policy_message(
        node_id="pc0",
        agent_id="runner_0",
        version=7,
        snapshot=source,
        reward_sha256="a" * 64,
        config_sha256="b" * 64,
        metrics={"success": 0.5},
    )
    message, decoded = parse_policy_message(
        text,
        expected_reward_sha256="a" * 64,
        expected_config_sha256="b" * 64,
    )
    assert message["version"] == 7
    assert message["agent_id"] == "runner_0"
    assert message["metrics"]["success"] == 0.5
    assert set(decoded) == set(source)
    for key in source:
        torch.testing.assert_close(decoded[key], source[key])


def test_policy_message_rejects_different_reward_file_hash():
    text = build_policy_message(
        node_id="pc0",
        agent_id="runner_0",
        version=1,
        snapshot=snapshot(),
        reward_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="reward.py"):
        parse_policy_message(
            text,
            expected_reward_sha256="c" * 64,
            expected_config_sha256="b" * 64,
        )


def test_policy_message_rejects_payload_corruption():
    text = build_policy_message(
        node_id="pc0",
        agent_id="runner_0",
        version=1,
        snapshot=snapshot(),
        reward_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    message = json.loads(text)
    message["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        parse_policy_message(
            json.dumps(message),
            expected_reward_sha256="a" * 64,
            expected_config_sha256="b" * 64,
        )
