import numpy as np

from marl2d.single_runner_env import SingleRunnerArena2D


def make_env_cfg():
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
        'goal_radius': 0.60,
        'max_steps': 300,
        'lidar_rays': 8,
        'lidar_range': 6.0,
        'reset_jitter': 0.0,
        'obstacles': {
            'count': 8,
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
        'mode': 'R2',
        'goal_bonus': 100.0,
        'collision_penalty': -100.0,
        'timeout_penalty': -20.0,
        'progress_scale': 5.0,
        'step_penalty': -0.01,
        'safety_distance': 0.5,
        'safety_scale': 0.5,
        'heading_scale': 0.02,
    }


def test_single_runner_observation_shape_is_15():
    env = SingleRunnerArena2D(4, make_env_cfg(), make_reward_cfg(), seed=7)
    obs = env.reset(seed=7)
    assert obs.shape == (4, 15)
    assert np.isfinite(obs).all()


def test_single_runner_forward_progress_gets_positive_reward():
    cfg = make_env_cfg()
    cfg['obstacles']['count'] = 0
    env = SingleRunnerArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.set_runner_state(np.array([[2.0, 10.0, 0.0]], dtype=np.float32))
    _, reward, done, info = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    assert reward[0] > 0.0
    assert not bool(done[0])
    assert float(info['progress'][0]) > 0.0


def test_single_runner_goal_grants_bonus_and_terminates():
    cfg = make_env_cfg()
    cfg['obstacles']['count'] = 0
    env = SingleRunnerArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.set_runner_state(np.array([[17.5, 10.0, 0.0]], dtype=np.float32))
    _, reward, done, info = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    assert bool(done[0])
    assert bool(info['goal_reached'][0])
    assert reward[0] == make_reward_cfg()['goal_bonus']


def test_single_runner_rectangle_collision_rolls_back():
    cfg = make_env_cfg()
    cfg['obstacles']['count'] = 0
    env = SingleRunnerArena2D(1, cfg, make_reward_cfg(), seed=1)
    env.obstacles = np.array([[[2.50, 10.0, 0.30, 1.0]]], dtype=np.float32)
    env.set_runner_state(np.array([[2.0, 10.0, 0.0]], dtype=np.float32))
    _, _, _, info = env.step(np.array([[1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(env.runner_state[0, :2], [2.0, 10.0], atol=1e-6)
    assert bool(info['collision'][0])


def test_single_runner_episode_reset_changes_map_seed_reproducibly():
    env_a = SingleRunnerArena2D(2, make_env_cfg(), make_reward_cfg(), seed=99)
    env_b = SingleRunnerArena2D(2, make_env_cfg(), make_reward_cfg(), seed=99)
    first = env_a.map_seeds.copy()
    mask = np.array([True, False])
    env_a.reset_indices(mask)
    env_b.reset_indices(mask)
    assert int(env_a.map_seeds[0]) != int(first[0])
    assert int(env_a.map_seeds[1]) == int(first[1])
    np.testing.assert_array_equal(env_a.map_seeds, env_b.map_seeds)
    np.testing.assert_allclose(env_a.obstacles, env_b.obstacles)
