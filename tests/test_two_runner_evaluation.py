import torch

from marl2d.policy import ActorCritic, snapshot_model
from marl2d.two_runner import TWO_RUNNER_IDS, evaluate_two_runner, two_runner_validation_is_better
from test_two_runner_trainer import cfg


def zero_policy_set(c):
    result = {}
    for aid in TWO_RUNNER_IDS:
        model = ActorCritic(18, 2, c['ppo']['hidden_sizes'])
        for parameter in model.parameters():
            parameter.data.zero_()
        result[aid] = snapshot_model(model)
    return result


def test_evaluation_reports_required_team_metrics():
    c = cfg()
    summary = evaluate_two_runner(c, zero_policy_set(c), episodes=3, seed_start=40000, device='cpu')
    required = {
        'team_success_rate', 'timeout_rate', 'both_dead_rate', 'any_collision_rate',
        'runner_0_death_rate', 'runner_1_death_rate', 'runner_0_goal_count',
        'runner_1_goal_count', 'team_success_with_prior_death_rate',
        'mean_time_to_team_goal_s', 'mean_episode_reward_runner_0',
        'mean_episode_reward_runner_1', 'mean_min_clearance_runner_0_m',
        'mean_min_clearance_runner_1_m', 'mean_winner_path_efficiency',
    }
    assert required <= set(summary)
    assert summary['episodes'] == 3


def test_safety_constrained_validation_comparator():
    safe = {
        'team_success_rate': 0.30,
        'any_collision_rate': 0.08,
        'both_dead_rate': 0.05,
        'mean_team_episode_reward': 10.0,
    }
    unsafe_more_success = {
        'team_success_rate': 0.80,
        'any_collision_rate': 0.20,
        'both_dead_rate': 0.01,
        'mean_team_episode_reward': 99.0,
    }
    safe_better = {
        'team_success_rate': 0.35,
        'any_collision_rate': 0.09,
        'both_dead_rate': 0.10,
        'mean_team_episode_reward': 0.0,
    }
    assert not two_runner_validation_is_better(unsafe_more_success, safe)
    assert two_runner_validation_is_better(safe_better, safe)
