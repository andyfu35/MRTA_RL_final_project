# Experiment 2 Two-Runner Cooperative PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Experiment 2 with two independently trained cooperative PPO Runner policies, shared team success, individual death on collision, Safe3 warm start, persistent worlds, exact valid-sample accounting, deterministic validation, checkpoint/resume, and dedicated CLI commands.

**Architecture:** Add a dedicated two-runner environment and trainer instead of overloading the existing four-agent arena. Each worker simulates both runners from a frozen two-policy set but updates only its own policy; dead target agents stop producing actor samples while their pending trajectories wait for the team outcome so delayed team-success credit can be applied correctly. A generalized policy exchange/coordinator supports an explicit agent-id set while preserving existing four-agent behavior.

**Tech Stack:** Python 3.13, NumPy, PyTorch, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-experiment2-two-runner-cooperative-ppo-design.md`

## Global Constraints

- Keep Experiment 1 and existing four-agent behavior backward compatible.
- Two active agents are exactly `runner_0` and `runner_1`.
- Observation size is exactly 18; indices 0:15 preserve Experiment 1 ordering.
- Warm start from the finalized 15-input Safe3 policy by copying columns 0:15 and zero-initializing columns 15:18.
- No Experiment 1 optimizer state is reused for initial Experiment 2 training.
- Team success occurs when either alive Runner reaches the goal and gives both policies +100 episodic team credit.
- Individual collision gives that Runner -100 and marks it dead; the other Runner continues.
- Dead-period simulator steps never become PPO actor samples.
- Both workers consume exactly 8192 valid target-agent samples per PPO update.
- Persistent worlds and pending trajectories survive PPO update boundaries and full checkpoint/resume.
- Validation uses 40000-40199; confirmation 50000-50199; final 60000-60199.
- Safety-constrained best selection uses `any_collision_rate <= 0.10` then maximum team success.

---

### Task 1: Generalize policy exchange for two-agent policy sets

**Files:**
- Modify: `src/marl2d/exchange.py`
- Modify: `src/marl2d/coordinator.py` only if needed for wording/type hints
- Test: `tests/test_exchange.py`

**Interfaces:**
- Produces: `MockPolicyExchange(initial_policy_set, version=0, agent_ids=None)` where omitted `agent_ids` preserves existing four-agent behavior.
- Produces: `ready_status(version)` and `commit(version)` over the configured agent-id tuple.

- [ ] **Step 1: Write failing tests** proving a two-agent exchange accepts only `runner_0/runner_1`, reports both readiness states, commits only after both publish, and existing default four-agent tests remain valid.
- [ ] **Step 2: Run** `python -m pytest tests/test_exchange.py -q` and confirm the new two-agent test fails because exchange is hard-coded to `AGENT_IDS`.
- [ ] **Step 3: Implement minimal generalization** by storing `self.agent_ids = tuple(agent_ids or AGENT_IDS)` and using it consistently in validation, cloning, readiness, commit, and policy retrieval.
- [ ] **Step 4: Run** `python -m pytest tests/test_exchange.py -q` and confirm green.
- [ ] **Step 5: Commit** `feat: generalize policy exchange agent set`.

### Task 2: Add TwoRunnerArena2D and cooperative environment semantics

**Files:**
- Create: `src/marl2d/two_runner_env.py`
- Test: `tests/test_two_runner_env.py`

**Interfaces:**
- Produces: `TwoRunnerArena2D(num_envs, env_cfg, reward_cfg, seed=0)`.
- Properties: `observation_dim = 18`, `action_dim = 2`, `alive` with shape `[num_envs, 2]`.
- `step(actions)` consumes shape `[num_envs, 2, 2]` and returns `(obs, rewards, team_done, info)` where rewards shape is `[num_envs, 2]` and info includes per-agent collision/death, team success, timeout, goal owner, min clearance, self progress, team progress, alive-before/alive-after, and map seeds.

- [ ] **Step 1: Write failing observation tests** for 18-D output, preserved first 15 single-runner features, teammate relative XY, teammate alive flag, and two nominal start poses.
- [ ] **Step 2: Run** `python -m pytest tests/test_two_runner_env.py -q` and confirm import/behavior failure.
- [ ] **Step 3: Implement geometry/state** by reusing shared differential-drive helpers and obstacle generation while exposing only two active robots; dead robots remain at their rollback pose and receive forced zero commands.
- [ ] **Step 4: Write failing terminal tests** for one-agent death while teammate continues, both-dead termination, either Runner goal terminating team success, runner-to-runner collision, and timeout only penalizing alive agents.
- [ ] **Step 5: Implement reward/terminal semantics**: alive shaping `3*self_progress + 2*team_progress - 0.01 - 3*danger^2`; collision `-100`; team success tracked separately for delayed worker credit; timeout `-20` for alive agents only. Compute team progress over the alive set at the start of the step and avoid artificial jumps when the membership changes.
- [ ] **Step 6: Run** `python -m pytest tests/test_two_runner_env.py -q` and confirm green.
- [ ] **Step 7: Commit** `feat: add cooperative two-runner environment`.

### Task 3: Add 15-to-18 policy warm-start helper

**Files:**
- Modify: `src/marl2d/policy.py`
- Test: `tests/test_two_runner_policy_init.py`

**Interfaces:**
- Produces: `expand_observation_snapshot(snapshot, old_obs_dim=15, new_obs_dim=18)` returning a new snapshot compatible with an 18-input `ActorCritic`.

- [ ] **Step 1: Write failing test** that creates a 15-input model/snapshot, expands it, loads into an 18-input model, verifies first-layer columns 0:15 copied exactly, columns 15:18 are zero, all compatible parameters match, and action/value for `[old_obs, arbitrary_new3]` equal the old model because added columns are zero.
- [ ] **Step 2: Run** `python -m pytest tests/test_two_runner_policy_init.py -q` and confirm missing helper failure.
- [ ] **Step 3: Implement minimal snapshot expansion** without mutating the source snapshot.
- [ ] **Step 4: Run** the focused test and confirm green.
- [ ] **Step 5: Commit** `feat: warm start two-runner policies from single runner`.

### Task 4: Implement per-runner worker collection with delayed team credit

**Files:**
- Create: `src/marl2d/two_runner.py`
- Test: `tests/test_two_runner_worker.py`

**Interfaces:**
- Produces: `TWO_RUNNER_IDS = ("runner_0", "runner_1")`.
- Produces: `TwoRunnerWorker(agent_id, cfg, device="cpu")` with `collect_round(frozen_policy_set, round_index)` and `train_round(...)`.
- Internal finalized transition queue stores observation, action, old log-prob, reward, value, next value, done for the target policy only.
- Pending per-environment target trajectory remains open after target death until team outcome is known.

- [ ] **Step 1: Write failing tests** showing dead-period steps create no target actor samples and delayed team success adds +100 to the target agent's last valid transition even when it died earlier.
- [ ] **Step 2: Run** `python -m pytest tests/test_two_runner_worker.py -q` and confirm failure.
- [ ] **Step 3: Implement persistent collectors and pending trajectories** for the target worker, sampling both frozen policies while masking dead agents to zero actions.
- [ ] **Step 4: Write failing exact-sample test** with a small configured sample count proving the worker returns exactly the requested number of finalized valid target transitions, keeps surplus transitions queued, and reports simulator steps separately.
- [ ] **Step 5: Implement valid-sample queue and GAE construction** using only finalized target transitions; terminalize pending target trajectories only when team outcome is known.
- [ ] **Step 6: Run** focused worker tests and confirm green.
- [ ] **Step 7: Commit** `feat: collect exact valid samples for two-runner PPO`.

### Task 5: Implement synchronous trainer, warm start, checkpoint/resume, validation, and evaluation

**Files:**
- Continue: `src/marl2d/two_runner.py`
- Test: `tests/test_two_runner_trainer.py`
- Test: `tests/test_two_runner_evaluation.py`

**Interfaces:**
- Produces: `TwoRunnerTrainer(cfg, output_dir, device="cpu")`.
- Produces: `initialize_from_single_runner_checkpoint(path)` that loads model weights only and expands 15->18 for both workers with fresh optimizers.
- Produces: `resume_from_checkpoint(path)` for full Experiment 2 state.
- Produces: `evaluate_two_runner(cfg, policy_set, episodes, seed_start, device)`.
- Produces: `two_runner_validation_is_better(candidate, best, collision_limit=0.10)`.

- [ ] **Step 1: Write failing synchronization tests** proving both workers collect/update from the same frozen policy version and the next version commits only after both updates are present.
- [ ] **Step 2: Implement trainer commit loop** using generalized `MockPolicyExchange` with exactly two IDs.
- [ ] **Step 3: Write failing warm-start tests** proving both policies equal the expanded single-runner snapshot and optimizer state is fresh.
- [ ] **Step 4: Implement initial checkpoint loader** using `load_single_runner_checkpoint` plus `expand_observation_snapshot`.
- [ ] **Step 5: Write failing checkpoint/resume tests** for policies, optimizers, current round, collector worlds, alive masks, pending trajectories, finalized queues, observations, and RNG; verify next round reproducibility in a small deterministic config.
- [ ] **Step 6: Implement full-state serialization/restoration** with an Experiment 2 training-state version.
- [ ] **Step 7: Write failing evaluation tests** for team success, any-collision, both-dead, per-runner death/goal counts, prior-death success, time-to-goal, reward, clearance, winner path efficiency, version/config metadata.
- [ ] **Step 8: Implement deterministic evaluator and safety-constrained comparator**: eligible if `any_collision_rate <= 0.10`, maximize team success, then lower both-dead rate, then higher mean team reward; before any eligible model exists prefer lower collision then higher success.
- [ ] **Step 9: Run** `python -m pytest tests/test_two_runner_trainer.py tests/test_two_runner_evaluation.py -q` and confirm green.
- [ ] **Step 10: Commit** `feat: train and evaluate synchronous cooperative runners`.

### Task 6: Add config validation and dedicated CLI

**Files:**
- Create: `config/two_runner.yaml`
- Modify: `src/marl2d/config.py`
- Modify: `src/marl2d/cli.py`
- Test: `tests/test_two_runner_config.py`
- Test: `tests/test_cli_two_runner.py`

**Interfaces:**
- Produces: `load_two_runner_config(path)` / `validate_two_runner_config(cfg)`.
- Adds CLI `two-train` and `two-eval`.
- `two-train` requires exactly one of `--init-single-runner-checkpoint` for a new run or `--resume` for continuation; both are mutually exclusive.

- [ ] **Step 1: Write failing config/CLI tests** for 8192 samples, reward constants, validation seed 40000/200 episodes/every 5, argument parsing, and init-vs-resume exclusivity.
- [ ] **Step 2: Run** `python -m pytest tests/test_two_runner_config.py tests/test_cli_two_runner.py -q` and confirm failure.
- [ ] **Step 3: Add config** matching Experiment 1 environment/PPO values and the Experiment 2 reward/validation split.
- [ ] **Step 4: Add config validation and CLI handlers** printing per-runner sample/simulator-step/death metrics and validation metrics.
- [ ] **Step 5: Run** focused config/CLI tests and confirm green.
- [ ] **Step 6: Commit** `feat: add Experiment 2 config and CLI`.

### Task 7: Regression verification, README, review, and integration

**Files:**
- Modify: `README.md`
- Test: entire `tests/` suite

**Interfaces:**
- Documents exact smoke/train/eval commands and data-split rules.

- [ ] **Step 1: Update README** with Experiment 2 architecture, reward, death/team-credit semantics, warm-start checkpoint, validation/confirmation/final seed ranges, and commands.
- [ ] **Step 2: Run focused Experiment 2 suite**: `python -m pytest tests/test_two_runner_env.py tests/test_two_runner_policy_init.py tests/test_two_runner_worker.py tests/test_two_runner_trainer.py tests/test_two_runner_evaluation.py tests/test_two_runner_config.py tests/test_cli_two_runner.py -q`.
- [ ] **Step 3: Run full regression suite**: `python -m pytest -q`.
- [ ] **Step 4: Run one smoke train** with a temporary reduced-sample config or test helper to prove end-to-end checkpoint creation without changing formal config.
- [ ] **Step 5: Review branch diff** against the spec; verify no Experiment 1 reward/PPO/environment constants changed.
- [ ] **Step 6: Request code review** and fix all Critical/Important findings.
- [ ] **Step 7: Re-run full regression suite** after review fixes.
- [ ] **Step 8: Commit docs/final fixes** and open PR to `main`.
