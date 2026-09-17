# Single Runner Navigation Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible single-runner PPO navigation experiment on the 20 m x 20 m random-box arena, with normalized wheel actions limited to 2 m/s and evaluation metrics for success, time, collisions, and path efficiency.

**Architecture:** Keep the existing 4-agent mock-distributed trainer unchanged. Convert the shared differential-drive environment to linear wheel-speed limits and deterministic per-environment episode seed streams. Add a separate `SingleRunnerArena2D` plus `SingleRunnerTrainer` so Experiment 1 is isolated from the future 2-vs-2 reward design.

**Tech Stack:** Python 3.13 target, NumPy, PyTorch PPO, PyYAML, Pillow, pytest.

**Spec:** Existing project design in `docs/superpowers/specs/2026-09-17-marl2d-design.md`, plus the approved Experiment 1 decisions in the current conversation.

## Global Constraints

- Map stays 20 m x 20 m with random axis-aligned rectangular obstacles.
- PPO action remains normalized as `[left, right]` in `[-1, 1]`.
- Each wheel maps linearly to `[-2.0, 2.0]` m/s.
- `dt` remains 0.1 s, so maximum straight displacement is 0.2 m per step.
- Every episode must receive a different obstacle seed while remaining reproducible from a base seed.
- Existing 4-agent PPO, coordinator, reward weights, and mock network architecture remain functional.
- No ROS2 work in this change.
- No cloud CI; verification uses local `python -m pytest -q`.

---

### Task 1: Linear Wheel-Speed Limit and Episode Seed Stream

**Files:**
- Modify: `config/default.yaml`
- Modify: `src/marl2d/env.py`
- Modify: `src/marl2d/config.py`
- Modify: `tests/test_env.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: normalized action tensor shaped `[num_envs, 4, 2]`.
- Produces: wheel linear speed `v_left/right = action * max_wheel_linear_speed`, plus per-world `map_seeds` derived from base seed, environment index, and episode index.

- [ ] Write failing tests for 2 m/s mapping and independent reproducible episode seeds.
- [ ] Run focused tests and verify they fail for the missing behavior.
- [ ] Replace angular wheel-speed action scaling with `max_wheel_linear_speed` and deterministic episode seed derivation.
- [ ] Run focused tests and the full suite.

### Task 2: Single-Runner Environment

**Files:**
- Create: `src/marl2d/single_runner_env.py`
- Create: `tests/test_single_runner_env.py`

**Interfaces:**
- Consumes: `environment` config and `single_runner_reward` config.
- Produces: observation shape `[num_envs, 15]` with goal-relative position, heading, 8 LiDAR rays, collision flag, and previous action; action shape `[num_envs, 2]`.

- [ ] Write failing tests for observation shape, goal reward, progress reward, collision rollback, and episode map regeneration.
- [ ] Run focused tests and verify they fail because the environment does not exist.
- [ ] Implement `SingleRunnerArena2D` with the same map generator and 2 m/s action mapping.
- [ ] Run focused tests and the full suite.

### Task 3: Single-Runner PPO Training and Evaluation

**Files:**
- Create: `src/marl2d/single_runner.py`
- Create: `config/single_runner.yaml`
- Create: `tests/test_single_runner.py`

**Interfaces:**
- Produces: `SingleRunnerTrainer.run(rounds)`, `load_single_runner_checkpoint(path)`, and `evaluate_single_runner(...)`.
- Evaluation summary keys: `episodes`, `success_rate`, `mean_time_to_goal_s`, `mean_collisions`, `mean_path_efficiency`.

- [ ] Write failing tests for one PPO update, checkpoint output, held-out evaluation seeds, and metric definitions.
- [ ] Run focused tests and verify they fail because the trainer/evaluator does not exist.
- [ ] Implement rollout collection, GAE, PPO update, checkpointing, and deterministic held-out evaluation.
- [ ] Run focused tests and the full suite.

### Task 4: CLI, Renderer, and Python 3.13 Documentation

**Files:**
- Modify: `src/marl2d/cli.py`
- Modify: `src/marl2d/render.py`
- Modify: `README.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- CLI commands: `train-single`, `eval-single`, `render-single`.
- README install commands target Python 3.13 and use `python -m pytest -q`.

- [ ] Write failing CLI parser tests for the three new commands.
- [ ] Implement CLI wiring and single-runner GIF rendering.
- [ ] Update README with Python 3.13, 2 m/s action semantics, seed behavior, and Experiment 1 workflow.
- [ ] Run `python -m pytest -q` and a short local single-runner smoke train/eval/render.
