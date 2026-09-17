# Experiment 2: Two-Runner Cooperative PPO Design

Date: 2026-09-17
Status: Design for implementation
Base branch at design time: `main` @ `d2334ccc5cd8bf7afc66a4b38196da49c2943a94`

## 1. Purpose

Experiment 2 introduces two cooperative Runner agents before adding Blockers. The experiment asks whether two independently trained PPO policies can learn a useful team strategy from a shared objective while preserving the safe navigation capability established in Experiment 1.

The team objective is:

> If either Runner reaches the goal, the team succeeds and both policies receive the team-success signal.

Experiment 2 is not intended to hard-code roles such as leader/follower, left/right lanes, sacrifice, or blocking. Those behaviors, if useful, must emerge from the reward and observations.

## 2. Non-goals

Experiment 2 does not add Blockers, ROS2 transport, centralized critics, MAPPO, parameter sharing, or a shared actor. It also does not change the differential-drive kinematics, random obstacle generator, map size, action scale, or PPO objective established by Experiment 1.

## 3. Training architecture

Use two independent policies:

- `runner_0` owns policy `pi_R0`
- `runner_1` owns policy `pi_R1`

Training is synchronous independent PPO.

For round `k`, both workers receive the same frozen policy set:

`P_k = {pi_R0^k, pi_R1^k}`

Each worker simulates both runners in its own parallel worlds but updates only its own target policy. The non-target runner is frozen for the entire collection round. When both workers have collected the configured number of valid target-agent samples and completed one PPO update, both publish `READY(k+1)`. The coordinator commits `P_(k+1)` only after both updates are ready.

There is no parameter averaging and no federated-learning step.

The existing `MockPolicyExchange` should be generalized to support an explicit agent-id set instead of assuming all four future 2v2 agents. Existing four-agent behavior must remain backward compatible.

## 4. Environment

Add a dedicated `TwoRunnerArena2D` rather than disabling Blockers inside the existing four-agent arena.

The environment contains exactly two active differential-drive robots and the same random rectangular obstacle maps used by Experiment 1.

Initial nominal poses:

- `runner_0`: `(2.0, 7.0, 0)`
- `runner_1`: `(2.0, 13.0, 0)`
- goal: `(18.0, 10.0)`

The existing reset position and heading jitter remain enabled.

Robot-to-wall, robot-to-obstacle, and runner-to-runner collision checks are active.

A dead Runner remains physically at its last valid pose and is treated as an inert obstacle for the surviving Runner. Its commanded action is forced to `[0, 0]`.

## 5. Observation

Each Runner receives an 18-dimensional local observation.

The first 15 values must preserve the Experiment 1 ordering exactly:

1. goal relative XY in the Runner body frame: 2
2. heading `sin(theta), cos(theta)`: 2
3. 8-ray lidar: 8
4. own collision flag: 1
5. previous left/right action: 2

Append three cooperative values:

6. teammate relative XY in the Runner body frame: 2
7. teammate alive flag: 1

Total: `15 + 2 + 1 = 18`.

The teammate position remains the last physical position after teammate death, while `teammate_alive=0` identifies that the teammate can no longer act.

No `self_alive` input is required because a dead agent is no longer queried for a policy action.

## 6. Experiment 1 warm start

Both Experiment 2 policies initialize from the finalized Experiment 1 policy:

`runs/exp1_r2_safe30_r70_to_r100/round_00080.pt`

Experiment 1 has 15 observation inputs while Experiment 2 has 18.

For the first linear layer:

- copy the trained weights for columns `0:15`
- initialize new columns `15:18` to zero
- copy the first-layer bias

Copy all remaining compatible parameters exactly:

- subsequent backbone layers
- actor mean head
- critic head
- `log_std`

This ensures the initial Experiment 2 deterministic behavior matches the learned Experiment 1 navigation behavior because the appended cooperative inputs initially have zero weight.

Do not restore the Experiment 1 Adam optimizer. Experiment 2 starts new optimizers because the task and network input size changed.

The training CLI must accept an explicit initialization checkpoint path. A resumed Experiment 2 checkpoint must not require the Experiment 1 file again.

## 7. Reward

The reward has two concepts:

1. shared team objective
2. individual navigation/safety shaping

For a Runner `i` that is alive at the start of timestep `t`, before terminal events:

`r_i = 3 * self_progress_i + 2 * team_progress - 0.01 - 3 * danger_i^2`

where:

- `self_progress_i = d_i(t) - d_i(t+1)`
- `A_t` is the set of Runners alive at the start of timestep `t`
- `team_progress = min_{j in A_t} d_j(t) - min_{j in A_t} d_j(t+1)`
- `danger_i = max(0, (0.5 - clearance_i) / 0.5)`

Using the same start-of-step alive set on both sides of `team_progress` prevents a death event from creating an artificial positive or negative progress jump merely because team membership changed. On the next timestep, a dead Runner is absent from `A_t`, so it cannot freeze the surviving Runner's team-progress signal from its final position.

`clearance_i` includes walls, static obstacles, and the teammate body, whether the teammate is alive or dead.

The `3 + 2` split keeps the progress-reward scale close to Experiment 1 for the currently leading Runner while giving both policies a direct cooperative signal. Individual progress remains present to reduce trivial free-riding.

### 7.1 Collision / individual death

When Runner `i` collides with a wall, obstacle, or teammate:

- add collision penalty `-100` to Runner `i`
- mark Runner `i` dead
- stop requesting actions from Runner `i`
- force its simulator command to `[0, 0]`
- keep its body at the collision rollback / last valid pose
- the other Runner continues if alive

The collision does not terminate the team episode unless both Runners are dead.

### 7.2 Team success

If either alive Runner reaches the goal:

- `team_success = true`
- both policies receive a `+100` team-success credit for that episode
- the team episode terminates

This includes a Runner that died earlier in the same episode: the dead Runner must still receive the shared team-success credit because the research objective is explicitly team-level, not individual goal ownership.

The Runner that physically reached the goal is tracked separately for metrics but does not receive an additional private goal bonus beyond the shared `+100`.

### 7.3 Both dead

If both Runners die before either reaches the goal:

- `team_failure = true`
- the episode terminates immediately
- no additional team-failure penalty is added because each Runner already received its own collision penalty

### 7.4 Timeout

If `max_steps` is reached without team success:

- each Runner still alive at timeout receives `-20`
- a Runner that already died receives no additional timeout penalty
- the episode terminates

### 7.5 Terminal precedence

Within one simulator step use:

`team goal > collision/death > timeout`

so a goal reached on the same step is counted as team success.

## 8. Delayed team credit for a dead Runner

A dead Runner must not generate fake PPO action transitions while its teammate continues.

Each worker therefore maintains pending per-environment episode data for its target agent. Valid transitions are recorded only while the target agent is alive, including the collision transition that kills it.

If the target agent dies before the team episode ends, its valid trajectory remains pending until the team outcome is known.

When the team episode ends:

- on team success, add the shared `+100` episodic team-success credit to the target agent's final valid transition even if that agent died earlier
- on target-agent collision, retain the `-100` collision term on that transition
- on timeout, apply `-20` only if the target agent was still alive at timeout
- mark the target trajectory terminal for GAE/return construction

No dead-period dummy actions are included in PPO actor samples.

The `+100` credit is intentionally an episode-level shared terminal credit and is not reduced simply because the target Runner died several simulator steps before its teammate reached the goal. This matches the stated Experiment 2 objective that either Runner reaching the goal is a high-score outcome for both models. Individual collision cost remains separate, so a dead Runner can receive both `-100` collision cost and `+100` team-success credit in the same episode.

## 9. Sample-count semantics and persistent worlds

Each worker must update from exactly `8192` valid target-agent samples per PPO round.

Dead-period simulator steps do not count toward the dead target worker's sample total. A worker may therefore need to simulate more environment timesteps to gather 8192 valid samples.

Environment instances persist across PPO round boundaries. A PPO update boundary does not reset an unfinished team episode.

Because pending episode trajectories may cross a PPO boundary, collection state required for exact resume includes:

- environment state
- map seed and episode count
- alive mask
- previous actions
- pending target-agent trajectories / pending delayed team credit state
- collector observation state
- RNG state

If a worker has more finalized valid samples than needed for the current 8192-sample batch, excess finalized samples remain queued for the next round rather than being discarded.

## 10. Collection profiles

The default Experiment 2 configuration preserves the heterogeneous-machine idea while making the sample count the invariant:

- `runner_0`: 32 parallel worlds
- `runner_1`: 16 parallel worlds
- both workers: exactly 8192 valid target-agent samples per PPO update

Because death creates variable-length target trajectories, Experiment 2 does not require the old fixed identity `parallel_envs * rollout_steps * batches = samples_per_update`. Each worker advances its persistent worlds in chunks until its finalized valid-sample queue contains at least 8192 samples, consumes exactly 8192, and carries any excess into the next round.

This preserves the intended future behavior: faster machines can use more parallel worlds, slower machines can use fewer, but every policy performs the same number of PPO updates from the same number of valid samples.

## 11. Checkpoint and resume

Experiment 2 checkpoint stores:

- policy version / committed round
- both policy snapshots
- both optimizer states
- Experiment 2 config
- observation/action dimensions
- per-worker persistent collector worlds
- per-worker alive masks
- pending per-environment target trajectories
- finalized-sample queues not yet consumed
- collector observations
- RNG states
- current best-validation summary
- training-state version

Full resume must reproduce the next round exactly when using the same device/backend and deterministic settings.

The same explicit `reset best validation` concept used in Experiment 1 should be available when forking a new Experiment 2 run from an Experiment 2 checkpoint.

## 12. Validation and checkpoint selection

Use a fixed deterministic validation set separate from Experiment 1 and from the final test.

Default validation:

- episodes: 200
- seed start: 40000
- run every 5 committed rounds

Reserve:

- diagnostic/confirmation set: 50000-50199
- untouched final set: 60000-60199

Do not inspect the final set until the Experiment 2 model and checkpoint-selection rule are locked.

Best-checkpoint selection is safety-constrained:

1. checkpoints with `any_collision_rate <= 0.10` are eligible
2. among eligible checkpoints, maximize `team_success_rate`
3. break ties with lower `both_dead_rate`
4. then higher mean team episode reward

If no checkpoint yet satisfies the safety constraint, choose lower `any_collision_rate`, then higher `team_success_rate`, so `best.pt` still exists during early training.

## 13. Evaluation metrics

Every deterministic evaluation reports at least:

- `team_successes`
- `team_success_rate`
- `timeouts`
- `timeout_rate`
- `both_dead_count`
- `both_dead_rate`
- `any_collision_count`
- `any_collision_rate`
- `runner_0_death_rate`
- `runner_1_death_rate`
- `runner_0_goal_count`
- `runner_1_goal_count`
- `team_success_with_prior_death_count`
- `team_success_with_prior_death_rate`
- `mean_time_to_team_goal_s`
- `mean_episode_reward_runner_0`
- `mean_episode_reward_runner_1`
- `mean_min_clearance_runner_0_m`
- `mean_min_clearance_runner_1_m`
- `mean_winner_path_efficiency`
- policy version
- reward/config metadata

Training metrics additionally report for each worker:

- exact valid sample count
- simulator steps required to obtain those samples
- completed team episodes
- target-agent death count
- target-agent goal contribution count
- stochastic action standard deviation
- PPO loss / value loss / entropy / approximate KL

These metrics are diagnostic; no role-balance reward is added merely to force equal goal counts.

## 14. CLI

Add dedicated commands instead of overloading the single-runner commands.

Initial training:

```bash
python -m marl2d two-train \
  --config config/two_runner.yaml \
  --init-single-runner-checkpoint runs/exp1_r2_safe30_r70_to_r100/round_00080.pt \
  --rounds 100 \
  --output runs/exp2_two_runner \
  --device cpu
```

Resume:

```bash
python -m marl2d two-train \
  --config config/two_runner.yaml \
  --resume runs/exp2_two_runner/latest.pt \
  --rounds 50 \
  --output runs/exp2_two_runner \
  --device cpu
```

Deterministic evaluation:

```bash
python -m marl2d two-eval \
  --checkpoint runs/exp2_two_runner/best.pt \
  --episodes 200 \
  --seed-start 40000 \
  --device cpu
```

`--resume` and `--init-single-runner-checkpoint` are mutually exclusive.

## 15. Configuration

Add `config/two_runner.yaml` based on the finalized Experiment 1 environment and PPO settings.

Initial reward values:

- team success bonus: `100`
- collision penalty: `-100`
- timeout penalty: `-20`
- self progress scale: `3`
- team progress scale: `2`
- step penalty: `-0.01`
- safety distance: `0.50`
- safety scale: `3.0`

PPO begins with the same core hyperparameters as Experiment 1 unless an Experiment 2 ablation later changes one factor explicitly.

## 16. Implementation boundaries

Expected primary additions/changes:

- `config/two_runner.yaml`
- `src/marl2d/two_runner_env.py`
- `src/marl2d/two_runner.py` for workers/trainer/evaluation/checkpoint logic
- `src/marl2d/policy.py` helper for 15->18 input warm-start expansion
- `src/marl2d/exchange.py` generalization from hard-coded four-agent IDs to configured IDs
- `src/marl2d/coordinator.py` wording/generalization only as needed
- `src/marl2d/config.py` Experiment 2 config validation
- `src/marl2d/cli.py` `two-train` / `two-eval`
- tests for environment semantics, warm start, delayed team credit, exact valid sample count, synchronous commit, checkpoint resume, validation selection, CLI, and evaluation metrics

Do not refactor unrelated Experiment 1 code.

## 17. Required tests before merge

At minimum verify:

1. observation is exactly 18-D and first 15 values match the equivalent single-runner features when teammate effects are excluded
2. 15->18 warm start produces the same action/value as the Experiment 1 model because appended cooperative input weights are zero
3. either Runner reaching the goal terminates the team episode and credits both target policies with team success
4. one Runner collision kills only that Runner and the other continues
5. dead Runner action is forced to zero and dead periods create no actor samples
6. dead Runner can still receive delayed shared team-success credit
7. both dead terminates immediately without an extra duplicate failure penalty
8. timeout penalizes only Runners alive at timeout
9. runner-to-runner collision is detected
10. alive-set changes do not create artificial team-progress jumps
11. each worker consumes exactly 8192 valid samples per update
12. both policies are frozen consistently during a collection round and commit only after both updates are ready
13. persistent worlds and pending trajectories survive checkpoint/resume
14. full resume reproduces the next round
15. validation seeds and final seeds remain disjoint
16. safety-constrained `best.pt` selection follows the specified rule
17. existing Experiment 1 and four-agent tests remain green

## 18. Success criterion for Experiment 2 v1

Experiment 2 v1 is considered technically successful when:

- both independent policies train under the synchronous frozen-policy protocol
- team success is learned above the single-policy baseline without violating the collision constraint used for model selection
- neither policy is forced into a predefined role
- the two policy returns both receive the shared success signal whenever either Runner reaches the goal
- deterministic held-out evaluation shows that the behavior generalizes to unseen obstacle maps

Only after this baseline is stable should Experiment 3 introduce two Blockers and adversarial interaction.
