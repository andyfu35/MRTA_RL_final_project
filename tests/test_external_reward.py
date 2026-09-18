from pathlib import Path

import numpy as np

from marl2d.reward_loader import RewardEngine
from marl2d.two_runner_env import TwoRunnerArena2D
def make_env_cfg(max_steps=20):
    return {
        "width": 20.0,
        "height": 20.0,
        "dt": 0.1,
        "wheel_radius": 0.1,
        "wheel_base": 0.5,
        "max_wheel_linear_speed": 2.0,
        "robot_radius": 0.3,
        "body_length": 0.7,
        "body_width": 0.44,
        "wheel_length": 0.34,
        "wheel_width": 0.10,
        "goal": [18.0, 10.0],
        "goal_radius": 0.7,
        "max_steps": max_steps,
        "lidar_rays": 8,
        "lidar_range": 6.0,
        "reset_jitter": 0.0,
        "obstacles": {"count": 0},
    }


def make_reward_cfg():
    return {
        "team_success_bonus": 100.0,
        "collision_penalty": -100.0,
        "timeout_penalty": -20.0,
        "self_progress_scale": 3.0,
        "team_progress_scale": 2.0,
        "step_penalty": -0.01,
        "safety_distance": 0.5,
        "safety_scale": 3.0,
    }


def plugin_reward_cfg(tmp_path: Path | None = None):
    legacy = make_reward_cfg()
    return {
        "plugin_path": str(
            (Path("deploy") / "reward.py").resolve()
            if tmp_path is None
            else tmp_path / "reward.py"
        ),
        "params": legacy,
    }


def test_external_reward_template_matches_legacy_collision_semantics():
    env_cfg = make_env_cfg()
    legacy_env = TwoRunnerArena2D(
        1, env_cfg, make_reward_cfg(), seed=13
    )
    plugin_env = TwoRunnerArena2D(
        1, env_cfg, plugin_reward_cfg(), seed=13
    )
    state = np.array(
        [[[0.31, 7.0, 0.0], [2.0, 13.0, 0.0]]],
        dtype=np.float32,
    )
    legacy_env.set_state(state)
    plugin_env.set_state(state)
    action = np.array(
        [[[-1.0, -1.0], [0.0, 0.0]]],
        dtype=np.float32,
    )

    _, legacy_reward, _, _ = legacy_env.step(action)
    _, plugin_reward, _, info = plugin_env.step(action)

    np.testing.assert_allclose(plugin_reward, legacy_reward)
    assert info["reward_plugin_sha256"]
    assert "collision_terminal" in info["reward_breakdown"]


def test_external_reward_template_matches_legacy_success_override():
    env_cfg = make_env_cfg()
    legacy_env = TwoRunnerArena2D(
        1, env_cfg, make_reward_cfg(), seed=14
    )
    plugin_env = TwoRunnerArena2D(
        1, env_cfg, plugin_reward_cfg(), seed=14
    )
    state = np.array(
        [[[17.55, 10.0, 0.0], [2.0, 13.0, 0.0]]],
        dtype=np.float32,
    )
    legacy_env.set_state(state)
    plugin_env.set_state(state)
    action = np.array(
        [[[1.0, 1.0], [0.0, 0.0]]],
        dtype=np.float32,
    )

    _, legacy_reward, _, _ = legacy_env.step(action)
    _, plugin_reward, _, _ = plugin_env.step(action)
    np.testing.assert_allclose(plugin_reward, legacy_reward)


def test_adding_one_decorated_function_automatically_changes_reward(tmp_path: Path):
    plugin = tmp_path / "reward.py"
    plugin.write_text(
        """
import numpy as np
from marl2d.reward_api import REWARD_API_VERSION, RewardContext, reward_term, zeros

@reward_term
def constant_bonus(ctx: RewardContext):
    out = zeros(ctx)
    out[ctx.alive_before] = 2.5
    return out
""",
        encoding="utf-8",
    )
    engine = RewardEngine.from_file(plugin)
    assert [term.name for term in engine.terms] == ["constant_bonus"]
