import math

import numpy as np

from marl2d.two_runner_env import TwoRunnerArena2D


def make_env_cfg(max_steps=20, obstacle_count=0):
    return {
        'width': 20.0,
        'height': 20.0,
        'dt': 0.1,
        'wheel_radius': 0.1,
        'wheel_base': 0.5,
        'max_wheel_linear_speed': 2.0,
        'robot_radius': 0.3,
        'body_length': 0.7,
        'body_width': 0.44,
        'wheel_length': 0.34,
        'wheel_width': 0.10,
        'goal': [18.0, 10.0],
        'goal_radius': 0.7,
        'max_steps': max_steps,
        'lidar_rays': 8,
        'lidar_range': 6.0,
        'reset_jitter': 0.0,
        'obstacles': {
            'count': obstacle_count,
            'min_width': 0.8,
            'max_width': 1.8,
            'min_height': 0.8,
            'max_height': 1.8,
            'min_spacing': 0.65,
            'spawn_clearance': 1.8,
            'goal_clearance': 2.0,
            'border_margin': 0.35,
            'max_attempts': 4000,
        },
    }


def make_reward_cfg():
    return {
        'team_success_bonus': 100.0,
        'collision_penalty': -100.0,
        'timeout_penalty': -20.0,
        'self_progress_scale': 3.0,
        'team_progress_scale': 2.0,
        'step_penalty': -0.01,
        'safety_distance': 0.5,
        'safety_scale': 3.0,
    }


def test_observation_is_18d_and_appends_teammate_xy_alive():
    env = TwoRunnerArena2D(1, make_env_cfg(), make_reward_cfg(), seed=7)
    obs = env.observe()
    assert obs.shape == (1, 2, 18)
    np.testing.assert_allclose(env.state[0, 0], [2.0, 7.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(env.state[0, 1], [2.0, 13.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs[0, 0, 15:17], [0.0, 0.3], atol=1e-6)
    np.testing.assert_allclose(obs[0, 1, 15:17], [0.0, -0.3], atol=1e-6)
    assert obs[0, 0, 17] == 1.0
    assert obs[0, 1, 17] == 1.0


def test_one_runner_collision_kills_only_that_runner_and_other_continues():
    env = TwoRunnerArena2D(1, make_env_cfg(), make_reward_cfg(), seed=1)
    env.set_state(np.array([[[0.31, 7.0, 0.0], [2.0, 13.0, 0.0]]], dtype=np.float32))
    before = env.state.copy()
    actions = np.array([[[-1.0, -1.0], [0.0, 0.0]]], dtype=np.float32)
    _, rewards, done, info = env.step(actions)

    assert info['collision'][0, 0]
    assert not info['collision'][0, 1]
    assert not info['alive_after'][0, 0]
    assert info['alive_after'][0, 1]
    assert rewards[0, 0] == -100.0
    assert not done[0]
    np.testing.assert_allclose(env.state[0, 0], before[0, 0])

    dead_pose = env.state[0, 0].copy()
    _, _, _, info2 = env.step(np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32))
    np.testing.assert_allclose(env.state[0, 0], dead_pose)
    np.testing.assert_allclose(env.prev_action[0, 0], [0.0, 0.0])
    assert not info2['new_death'][0, 0]


def test_either_runner_goal_is_team_success_and_both_get_bonus():
    env = TwoRunnerArena2D(1, make_env_cfg(), make_reward_cfg(), seed=2)
    env.set_state(np.array([[[17.55, 10.0, 0.0], [2.0, 13.0, 0.0]]], dtype=np.float32))
    _, rewards, done, info = env.step(np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32))

    assert done[0]
    assert info['team_success'][0]
    assert info['goal_reached'][0, 0]
    assert not info['goal_reached'][0, 1]
    np.testing.assert_allclose(rewards[0], [100.0, 100.0])


def test_runner_to_runner_collision_can_kill_both_and_terminate():
    env = TwoRunnerArena2D(1, make_env_cfg(), make_reward_cfg(), seed=3)
    env.set_state(np.array([[[5.0, 10.0, 0.0], [5.5, 10.0, math.pi]]], dtype=np.float32))
    _, rewards, done, info = env.step(np.zeros((1, 2, 2), dtype=np.float32))

    assert done[0]
    assert info['both_dead'][0]
    assert info['collision'][0].tolist() == [True, True]
    np.testing.assert_allclose(rewards[0], [-100.0, -100.0])


def test_timeout_penalizes_only_alive_runners():
    env = TwoRunnerArena2D(1, make_env_cfg(max_steps=1), make_reward_cfg(), seed=4)
    _, rewards, done, info = env.step(np.zeros((1, 2, 2), dtype=np.float32))
    assert done[0]
    assert info['timeout'][0]
    np.testing.assert_allclose(rewards[0], [-20.0, -20.0])


def make_record_reward_cfg(*, epsilon=0.0, self_scale=1.0, team_scale=0.0):
    cfg = make_reward_cfg()
    cfg.update({
        'progress_mode': 'record',
        'record_progress_epsilon': float(epsilon),
        'self_progress_scale': float(self_scale),
        'team_progress_scale': float(team_scale),
        'step_penalty': 0.0,
        'safety_scale': 0.0,
    })
    return cfg


def test_record_progress_allows_detour_without_negative_progress_and_cannot_replay_old_progress():
    env = TwoRunnerArena2D(1, make_env_cfg(), make_record_reward_cfg(), seed=11)
    start_best = env.best_goal_distance.copy()

    _, reward_closer, _, info_closer = env.step(
        np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    )
    first_improvement = float(info_closer['self_progress'][0, 0])
    assert first_improvement > 0.0
    assert reward_closer[0, 0] > 0.0
    assert env.best_goal_distance[0, 0] < start_best[0, 0]

    _, reward_detour, _, info_detour = env.step(
        np.array([[[-1.0, -1.0], [0.0, 0.0]]], dtype=np.float32)
    )
    assert info_detour['self_progress'][0, 0] == 0.0
    assert reward_detour[0, 0] == 0.0

    _, reward_return, _, info_return = env.step(
        np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    )
    assert info_return['self_progress'][0, 0] == 0.0
    assert reward_return[0, 0] == 0.0

    _, reward_new_record, _, info_new_record = env.step(
        np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    )
    assert info_new_record['self_progress'][0, 0] > 0.0
    assert reward_new_record[0, 0] > 0.0


def test_record_progress_epsilon_accumulates_small_improvements_until_threshold_is_crossed():
    env = TwoRunnerArena2D(
        1,
        make_env_cfg(),
        make_record_reward_cfg(epsilon=0.30),
        seed=12,
    )
    best0 = float(env.best_goal_distance[0, 0])

    _, reward1, _, info1 = env.step(
        np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    )
    assert info1['self_progress'][0, 0] == 0.0
    assert reward1[0, 0] == 0.0
    assert float(env.best_goal_distance[0, 0]) == best0

    _, reward2, _, info2 = env.step(
        np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    )
    assert info2['self_progress'][0, 0] > 0.30
    assert reward2[0, 0] > 0.30
    assert float(env.best_goal_distance[0, 0]) < best0


def test_record_team_progress_uses_largest_new_personal_record_not_sum_and_is_shared():
    env = TwoRunnerArena2D(
        1,
        make_env_cfg(),
        make_record_reward_cfg(self_scale=0.0, team_scale=1.0),
        seed=13,
    )
    _, rewards, _, info = env.step(
        np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    )
    r0 = float(info['self_progress'][0, 0])
    r1 = float(info['self_progress'][0, 1])
    expected = max(r0, r1)
    assert expected > 0.0
    assert np.isclose(float(info['team_progress'][0]), expected, atol=1e-6)
    np.testing.assert_allclose(rewards[0], [expected, expected], atol=1e-6)


def test_record_best_distance_is_reset_and_checkpointed_with_environment_state():
    env = TwoRunnerArena2D(1, make_env_cfg(), make_record_reward_cfg(), seed=14)
    env.step(np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32))
    snapshot = env.snapshot_state()
    saved_best = snapshot['best_goal_distance'].copy()

    env.best_goal_distance[:] = 999.0
    env.restore_state(snapshot)
    np.testing.assert_allclose(env.best_goal_distance, saved_best)

    env.reset_indices(np.array([True]))
    np.testing.assert_allclose(env.best_goal_distance, env._goal_distances(), atol=1e-6)
