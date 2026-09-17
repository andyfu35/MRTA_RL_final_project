import math
import numpy as np

from marl2d.env import VectorArena2D


def make_cfg():
    return {
        'width': 10.0,
        'height': 6.0,
        'dt': 0.1,
        'wheel_radius': 0.1,
        'wheel_base': 0.4,
        'max_wheel_speed': 5.0,
        'robot_radius': 0.18,
        'goal': [9.0, 3.0],
        'goal_radius': 0.5,
        'max_steps': 200,
        'lidar_rays': 8,
        'lidar_range': 4.0,
        'reset_jitter': 0.0,
        'obstacles': [],
    }


def make_reward_cfg():
    return {
        'runner': {
            'goal_bonus': 100.0,
            'team_progress': 2.0,
            'self_progress': 0.5,
            'collision': -2.0,
            'step': -0.01,
        },
        'blocker': {
            'runner_progress': -2.0,
            'timeout_bonus': 100.0,
            'goal_failure': -100.0,
            'collision': -2.0,
            'step': -0.005,
        },
    }


def test_reset_is_deterministic_for_same_seed():
    env_a = VectorArena2D(3, make_cfg(), make_reward_cfg(), seed=7)
    env_b = VectorArena2D(3, make_cfg(), make_reward_cfg(), seed=7)
    obs_a = env_a.reset(seed=99)
    obs_b = env_b.reset(seed=99)
    np.testing.assert_allclose(env_a.state, env_b.state)
    np.testing.assert_allclose(obs_a, obs_b)


def test_observation_has_expected_shape_21():
    env = VectorArena2D(5, make_cfg(), make_reward_cfg(), seed=1)
    obs = env.reset()
    assert obs.shape == (5, 4, 21)
    assert np.isfinite(obs).all()


def test_equal_wheel_speeds_move_straight():
    cfg = make_cfg()
    env = VectorArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.reset()
    state = np.array([[[2.0, 2.0, 0.0], [2.0, 4.0, 0.0], [7.0, 2.0, math.pi], [7.0, 4.0, math.pi]]], dtype=np.float32)
    env.set_state(state)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0] = [1.0, 1.0]
    env.step(actions)
    expected_dx = cfg['wheel_radius'] * cfg['max_wheel_speed'] * cfg['dt']
    np.testing.assert_allclose(env.state[0, 0, 0], 2.0 + expected_dx, atol=1e-6)
    np.testing.assert_allclose(env.state[0, 0, 1], 2.0, atol=1e-6)


def test_opposite_wheel_speeds_rotate_in_place():
    cfg = make_cfg()
    env = VectorArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.reset()
    state = np.array([[[2.0, 2.0, 0.0], [2.0, 4.0, 0.0], [7.0, 2.0, math.pi], [7.0, 4.0, math.pi]]], dtype=np.float32)
    env.set_state(state)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0] = [-1.0, 1.0]
    env.step(actions)
    np.testing.assert_allclose(env.state[0, 0, :2], [2.0, 2.0], atol=1e-6)
    expected = (cfg['wheel_radius'] / cfg['wheel_base']) * (2 * cfg['max_wheel_speed']) * cfg['dt']
    np.testing.assert_allclose(env.state[0, 0, 2], expected, atol=1e-6)


def test_collision_rolls_back_moving_robot():
    cfg = make_cfg()
    cfg['obstacles'] = [[2.24, 2.0, 0.10]]
    env = VectorArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.reset()
    state = np.array([[[2.0, 2.0, 0.0], [2.0, 4.0, 0.0], [7.0, 2.0, math.pi], [7.0, 4.0, math.pi]]], dtype=np.float32)
    env.set_state(state)
    actions = np.zeros((1, 4, 2), dtype=np.float32)
    actions[0, 0] = [1.0, 1.0]
    _, _, _, info = env.step(actions)
    np.testing.assert_allclose(env.state[0, 0, :2], [2.0, 2.0], atol=1e-6)
    assert bool(info['collision'][0, 0]) is True
