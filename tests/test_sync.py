import pytest
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


def test_exchange_supports_explicit_two_agent_set():
    ids = ('runner_0', 'runner_1')
    snapshot = {'w': torch.tensor([1.0])}
    exchange = MockPolicyExchange({aid: snapshot for aid in ids}, version=0, agent_ids=ids)

    assert exchange.ready_status(1) == {'runner_0': False, 'runner_1': False}
    exchange.publish('runner_0', 1, snapshot)
    assert exchange.ready_status(1) == {'runner_0': True, 'runner_1': False}
    assert exchange.commit(1) is False
    exchange.publish('runner_1', 1, snapshot)
    assert exchange.commit(1) is True
    assert exchange.version == 1
    assert set(exchange.get_committed_policy_set()) == set(ids)


def test_exchange_rejects_explicit_empty_agent_set():
    with pytest.raises(ValueError, match='agent_ids'):
        MockPolicyExchange({}, version=0, agent_ids=[])
