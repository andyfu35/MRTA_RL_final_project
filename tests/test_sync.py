import torch

from marl2d import AGENT_IDS
from marl2d.coordinator import SynchronousCoordinator
from marl2d.exchange import MockPolicyExchange
from marl2d.policy import ActorCritic, snapshot_model


def make_policy_set(seed=0):
    torch.manual_seed(seed)
    return {
        aid: snapshot_model(ActorCritic(21, 2, [8, 8]))
        for aid in AGENT_IDS
    }


def test_partial_ready_set_cannot_commit():
    initial = make_policy_set(1)
    exchange = MockPolicyExchange(initial, version=0)
    coordinator = SynchronousCoordinator(exchange)
    updates = make_policy_set(2)
    for aid in AGENT_IDS[:3]:
        coordinator.submit_update(aid, 1, updates[aid], {'loss': 0.1})
    assert coordinator.try_commit(1) is False
    assert coordinator.current_round == 0
    status = coordinator.ready_status(1)
    assert status['runner_0'] is True
    assert status['blocker_1'] is False


def test_all_four_matching_versions_commit_exactly_once():
    initial = make_policy_set(3)
    exchange = MockPolicyExchange(initial, version=0)
    coordinator = SynchronousCoordinator(exchange)
    updates = make_policy_set(4)
    for aid in AGENT_IDS:
        coordinator.submit_update(aid, 1, updates[aid], {'loss': 0.2})
    assert coordinator.try_commit(1) is True
    assert coordinator.current_round == 1
    committed = exchange.get_committed_policy_set()
    for aid in AGENT_IDS:
        for key in updates[aid]:
            torch.testing.assert_close(committed[aid][key], updates[aid][key])
    assert coordinator.try_commit(1) is False
