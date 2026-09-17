# MARL2D Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a single-machine, mock-distributed four-agent PPO training prototype for deterministic 2D differential-drive robots.

**Architecture:** A vectorized NumPy environment supplies deterministic 2D kinematics and fixed observations. Four independent PyTorch PPO workers collect equal sample counts with different collection profiles, exchange frozen policy snapshots through a mock transport, and synchronize on a round barrier. A renderer exports a top-view GIF.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, PyYAML, pytest, matplotlib, Pillow

**Spec:** `docs/superpowers/specs/2026-09-17-marl2d-design.md`

## Global Constraints
- No MuJoCo or rigid-body physics.
- Four independent policies; never average their weights.
- Opponent policies stay frozen for the full duration of one round.
- Every worker collects the same `samples_per_update` before one PPO update.
- YAML controls environment, rewards, PPO, collection profiles, and future network addresses.
- ROS2 is not required in this prototype; mock exchange must be replaceable later.

---

### Task 1: Configuration and deterministic environment
**Files:** create `config/default.yaml`, `src/marl2d/config.py`, `src/marl2d/env.py`, tests in `tests/test_config.py`, `tests/test_env.py`.
**Produces:** validated config loader, `VectorArena2D.reset()`, `VectorArena2D.step(actions)`, observations of shape `(num_envs, 4, 21)`.
- [ ] Write tests for equal-sample collection profile validation, straight-line motion, in-place rotation, collision rollback, deterministic reset, and observation shape.
- [ ] Run tests and verify they fail because implementation is absent.
- [ ] Implement minimal config/environment functionality.
- [ ] Run tests and keep all green.

### Task 2: Actor-critic and PPO update
**Files:** create `src/marl2d/policy.py`, `src/marl2d/ppo.py`, tests in `tests/test_ppo.py`.
**Produces:** tanh Gaussian actor-critic, policy snapshot/load, finite PPO update over a synthetic rollout.
- [ ] Write failing tests for action bounds, snapshot round-trip, and finite update metrics.
- [ ] Implement minimal model and PPO loss/update.
- [ ] Run tests and keep all green.

### Task 3: Rollout worker with heterogeneous batching
**Files:** create `src/marl2d/worker.py`, tests in `tests/test_worker.py`.
**Consumes:** config, environment, four policy snapshots.
**Produces:** `collect_round(...)` returning exactly `samples_per_update` transitions for the selected agent and `train_round(...)` returning updated snapshot/metrics.
- [ ] Write failing tests proving two different collection profiles return the same sample count and leave opponent snapshot parameters unchanged.
- [ ] Implement collection across sequential batches and GAE bootstrap boundaries.
- [ ] Run tests and keep all green.

### Task 4: Mock policy exchange and synchronous coordinator
**Files:** create `src/marl2d/exchange.py`, `src/marl2d/coordinator.py`, tests in `tests/test_sync.py`.
**Produces:** versioned policy-set commit barrier.
- [ ] Write failing tests proving partial READY states cannot commit and four matching next-version snapshots can commit exactly one round.
- [ ] Implement mock exchange and coordinator.
- [ ] Run tests and keep all green.

### Task 5: Training CLI and renderer
**Files:** create `src/marl2d/trainer.py`, `src/marl2d/render.py`, `src/marl2d/cli.py`, `src/marl2d/__main__.py`, `requirements.txt`, `README.md`, tests in `tests/test_trainer.py`.
**Produces:** `python -m marl2d train --rounds N`, metrics CSV/JSON, checkpoints, and `python -m marl2d render` GIF.
- [ ] Write a failing smoke test for one synchronized round.
- [ ] Implement trainer orchestration and checkpoint output.
- [ ] Implement evaluation renderer.
- [ ] Run the full test suite.
- [ ] Run a short multi-round training smoke test and verify finite metrics plus monotonically committed round numbers.
- [ ] Render one evaluation GIF.
