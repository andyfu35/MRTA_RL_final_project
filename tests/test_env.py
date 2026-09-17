import math

import numpy as np

from marl2d.env import VectorArena2D


def make_cfg():
    return {
        'width': 20.0,
        'height': 20.0,
        'dt': 0.1,
        'wheel_radius': 0.1,
        'wheel_base': 0.5,
        'max_wheel_speed': 5.0,
        'robot_radius': 0.30,
        'body_length': 0.70,
        'body_width': 0.44,
        'wheel_length': 0.34,
        'wheel_width': 0.10,
        'goal': [18.0, 10.0],
        'goal_radius': 0.6,
        'max_steps': 200,
        'lidar_rays': 8,
        'lidar_range': 6.0,
        'reset_jitter': 0.0,
        'obstacles': {
            'count': 14,
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
        'runner': {'goal_bonus': 100.0, 'team_progress': 2.0, 'self_progress': 0.5, 'collision': -2.0, 'step': -0.01},
        'blocker': {'runner_progress': -2.0, 'timeout_bonus': 100.0, 'goal_failure': -100.0, 'collision': -2.0, 'step': -0.005},
    }


def test_reset_is_deterministic_for_same_seed():
    env_a = VectorArena2D(3, make_cfg(), make_reward_cfg(), seed=7)
    env_b = VectorArena2D(3, make_cfg(), make_reward_cfg(), seed=7)
    obs_a = env_a.reset(seed=99)
    obs_b = env_b.reset(seed=99)
    np.testing.assert_allclose(env_a.state, env_b.state)
    np.testing.assert_allclose(env_a.obstacles, env_b.obstacles)
    np.testing.assert_allclose(obs_a, obs_b)


def test_random_rectangles_have_expected_shape_and_stay_inside_map():
    cfg = make_cfg()
    env = VectorArena2D(4, cfg, make_reward_cfg(), seed=3)
    assert env.obstacles.shape == (4, cfg['obstacles']['count'], 4)
    for world in env.obstacles:
        for x, y, width, height in world:
            assert x - width / 2 >= cfg['obstacles']['border_margin'] - 1e-6
            assert x + width / 2 <= cfg['width'] - cfg['obstacles']['border_margin'] + 1e-6
            assert y - height / 2 >= cfg['obstacles']['border_margin'] - 1e-6
            assert y + height / 2 <= cfg['height'] - cfg['obstacles']['border_margin'] + 1e-6


def test_observation_has_expected_shape_21():
    env = VectorArena2D(5, make_cfg(), make_reward_cfg(), seed=1)
    obs = env.reset()
    assert obs.shape == (5, 4, 21)
    assert np.isfinite(obs).all()


def test_equal_wheel_speeds_move_straight():
    cfg = make_cfg()
    cfg['obstacles']['count'] = 0
    env = VectorArena2D(1, cfg, make_reward_cfg(), seed=1)
    state = np.array([[[2.0, 7.0, 0.0], [2.0, 13.0, 0.0], [14.5, 7.5, math.pi], [14.5, 12.5, math.pi]]], dtype=np.float32)
    env.set_state(state)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0] = [1.0, 1.0]
    env.step(actions)
    expected_dx = cfg['wheel_radius'] * cfg['max_wheel_speed'] * cfg['dt']
    np.testing.assert_allclose(env.state[0, 0, 0], 2.0 + expected_dx, atol=1e-6)


def test_opposite_wheel_speeds_rotate_in_place():
    cfg = make_cfg()
    cfg['obstacles']['count'] = 0
    env = VectorArena2D(1, cfg, make_reward_cfg(), seed=1)
    state = np.array([[[2.0, 7.0, 0.0], [2.0, 13.0, 0.0], [14.5, 7.5, math.pi], [14.5, 12.5, math.pi]]], dtype=np.float32)
    env.set_state(state)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0] = [-1.0, 1.0]
    env.step(actions)
    np.testing.assert_allclose(env.state[0, 0, :2], [2.0, 7.0], atol=1e-6)


def test_rectangle_collision_rolls_back_moving_robot():
    cfg = make_cfg()
    cfg['obstacles']['count'] = 0
    env = VectorArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.obstacles = np.array([[[2.50, 7.0, 0.30, 1.0]]], dtype=np.float32)
    state = np.array([[[2.0, 7.0, 0.0], [2.0, 13.0, 0.0], [14.5, 7.5, math.pi], [14.5, 12.5, math.pi]]], dtype=np.float32)
    env.set_state(state)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0] = [1.0, 1.0]
    _, _, _, info = env.step(actions)
    np.testing.assert_allclose(env.state[0, 0, :2], [2.0, 7.0], atol=1e-6)
    assert bool(info['collision'][0, 0]) is True


def test_reset_indices_regenerates_only_selected_worlds():
    env = VectorArena2D(3, make_cfg(), make_reward_cfg(), seed=23)
    before = env.obstacles.copy()
    env.reset_indices(np.array([False, True, False]))
    np.testing.assert_allclose(env.obstacles[0], before[0])
    np.testing.assert_allclose(env.obstacles[2], before[2])
    assert not np.allclose(env.obstacles[1], before[1])
