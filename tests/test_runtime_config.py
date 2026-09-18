from pathlib import Path

import pytest

from marl2d.runtime_config import (
    load_runtime_config,
    resolve_local_node,
)


def test_deploy_config_loads_four_computers_and_external_reward():
    runtime = load_runtime_config(Path("deploy/config.json"))
    assert len(runtime.nodes) == 4
    assert {node.agent_id for node in runtime.training_nodes} == {
        "runner_0",
        "runner_1",
    }
    assert runtime.reward_path.name == "reward.py"
    assert len(runtime.reward_sha256) == 64
    assert len(runtime.config_sha256) == 64
    cfg = runtime.two_runner_cfg
    assert cfg["joint_collection"] == {
        "enabled": True,
        "parallel_envs": 64,
        "rollout_steps": 128,
    }
    assert cfg["training"]["samples_per_update"] == 8192
    assert cfg["two_runner_reward"]["plugin_path"].endswith(
        "deploy/reward.py"
    )


def test_same_config_can_identify_local_computer_from_ip_list():
    runtime = load_runtime_config(Path("deploy/config.json"))
    local = resolve_local_node(
        runtime,
        local_ips={"127.0.0.1", "192.168.1.102"},
    )
    assert local.node_id == "pc_runner_1"
    assert local.agent_id == "runner_1"
    assert local.train is True


def test_explicit_node_id_overrides_ip_auto_detection():
    runtime = load_runtime_config(Path("deploy/config.json"))
    local = resolve_local_node(
        runtime,
        explicit_node_id="pc_blocker_0",
        local_ips=set(),
    )
    assert local.ip == "192.168.1.103"
    assert local.train is False


def test_ambiguous_ip_identity_is_rejected():
    runtime = load_runtime_config(Path("deploy/config.json"))
    with pytest.raises(RuntimeError):
        resolve_local_node(
            runtime,
            local_ips={
                "192.168.1.101",
                "192.168.1.102",
            },
        )
