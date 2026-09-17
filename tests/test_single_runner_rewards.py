import math

import numpy as np
import pytest

from marl2d.single_runner_env import SingleRunnerArena2D


def make_env_cfg(max_steps: int = 50):
    return {
        'width': 20.0,
        'height': 20.0,
        'dt': 0.1,
        'wheel_radius': 0.1,
        'wheel_base': 0.5,
        'max_wheel_linear_speed': 2.0,
        'robot_radius': 0.30,
        'body_length': 0.70,
        'body_width': 0.44,
        'wheel_length': 0.34,
        'wheel_width': 0.10,
        'goal': [18.0, 10.0],
        'goal_radius': 0.70,
        'max_steps': max_steps,
        'lidar_rays': 8,
        'lidar_range': 6.0,
        'reset_jitter': 0.0,
        'obstacles': {
            'count': 0,
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


def reward_cfg(mode: str):
    return {
        'mode': mode,
        'goal_bonus': 100.0,
        'collision_penalty': -100.0,
        'timeout_penalty': -20.0,
        'progress_scale': 5.0,
        'step_penalty': -0.01,
        'safety_distance': 0.5,
        'safety_scale': 0.5,
        'heading_scale': 0.02,
    }


def make_env(mode: str, max_steps: int = 50):
    env = SingleRunnerArena2D(1, make_env_cfg(max_steps=max_steps), reward_cfg(mode), seed=1)
    env.set_runner_state(np.array([[2.0, 10.0, 0.0]], dtype=np.float32))
    return env


def test_r0_is_sparse_before_terminal():
    env = make_env('R0')
    _, reward, done, _ = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    assert reward[0] == pytest.approx(0.0)
    assert not bool(done[0])


def test_r1_adds_progress_and_time_shaping():
    env = make_env('R1')
    _, reward, done, info = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    expected = reward_cfg('R1')['progress_scale'] * float(info['progress'][0]) + reward_cfg('R1')['step_penalty']
    assert reward[0] == pytest.approx(expected, abs=1e-6)
    assert reward[0] > 0.0
    assert not bool(done[0])


def test_r2_penalizes_small_obstacle_clearance_without_collision():
    r1 = make_env('R1')
    r2 = make_env('R2')
    obstacle = np.array([[[2.7, 10.0, 0.4, 1.0]]], dtype=np.float32)
    r1.obstacles = obstacle.copy()
    r2.obstacles = obstacle.copy()

    _, reward_r1, done_r1, info_r1 = r1.step(np.array([[0.0, 0.0]], dtype=np.float32))
    _, reward_r2, done_r2, info_r2 = r2.step(np.array([[0.0, 0.0]], dtype=np.float32))

    assert not bool(done_r1[0])
    assert not bool(done_r2[0])
    assert not bool(info_r2['collision'][0])
    assert info_r2['min_clearance'][0] == pytest.approx(0.2, abs=1e-6)
    assert reward_r2[0] < reward_r1[0]


def test_r3_adds_small_heading_shaping():
    r2 = make_env('R2')
    r3 = make_env('R3')
    action = np.array([[0.0, 0.0]], dtype=np.float32)
    _, reward_r2, _, info_r2 = r2.step(action)
    _, reward_r3, _, info_r3 = r3.step(action)

    assert info_r3['heading_error'][0] == pytest.approx(0.0, abs=1e-6)
    assert reward_r3[0] - reward_r2[0] == pytest.approx(reward_cfg('R3')['heading_scale'], abs=1e-6)


def test_collision_is_terminal_and_overrides_shaping_reward():
    env = make_env('R3')
    env.obstacles = np.array([[[2.50, 10.0, 0.30, 1.0]]], dtype=np.float32)
    _, reward, done, info = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    assert bool(done[0])
    assert bool(info['collision'][0])
    assert reward[0] == pytest.approx(-100.0)


def test_timeout_is_terminal_and_overrides_shaping_reward():
    env = make_env('R3', max_steps=1)
    _, reward, done, info = env.step(np.array([[0.0, 0.0]], dtype=np.float32))
    assert bool(done[0])
    assert bool(info['timeout'][0])
    assert reward[0] == pytest.approx(-20.0)


def test_goal_is_terminal_and_overrides_shaping_reward():
    env = make_env('R3')
    env.set_runner_state(np.array([[17.5, 10.0, 0.0]], dtype=np.float32))
    _, reward, done, info = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    assert bool(done[0])
    assert bool(info['goal_reached'][0])
    assert reward[0] == pytest.approx(100.0)


def test_invalid_reward_mode_is_rejected():
    with pytest.raises(ValueError, match='R0, R1, R2, R3'):
        SingleRunnerArena2D(1, make_env_cfg(), reward_cfg('RX'), seed=1)
