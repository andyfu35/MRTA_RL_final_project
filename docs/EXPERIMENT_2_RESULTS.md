# Experiment 2 Experimental History and Results

Date: 2026-09-18

This document records the Experiment 2 two-Runner PPO research sequence, the reason for each ablation, the controlled variables, the observed validation results, and the conclusions supported by the experiments completed so far.

## 1. Shared Experiment 2 setup

Experiment 2 trains two independent Runner policies in the same two-Robot cooperative navigation task before Blockers are introduced.

Shared task semantics:

- Runner 0 and Runner 1 each own an independent Actor-Critic PPO policy.
- The team succeeds when either Runner reaches the goal.
- An individual collision kills only the colliding Runner.
- The teammate remains active after an individual death.
- Both policies can receive shared team-success credit.
- Observation size is 18-D:
  - goal-relative XY in body frame
  - sin/cos heading
  - 8 lidar rays
  - own collision flag
  - previous left/right action
  - teammate-relative XY
  - teammate-alive flag
- Both policies warm-start from the Experiment 1 Safe3 Round80 checkpoint:
  `runs/exp1_r2_safe30_r70_to_r100/round_00080.pt`
- Network: hidden sizes [64, 64].
- PPO learning rate: 3e-4.
- gamma: 0.99.
- GAE lambda: 0.95.
- clip range: 0.2.
- entropy coefficient: 0.005.
- PPO epochs: 4.
- minibatch size: 256.
- validation uses held-out seeds 40000-40199.
- confirmation seeds 50000-50199 and final-test seeds 60000-60199 remain reserved.

The original alive-step reward was:

```
r_i =
  3 * self_progress_i
+ 2 * team_progress
- 0.01
- 3 * danger_i^2
```

Terminal rewards:

- team success: +100
- collision: -100
- timeout for an alive Runner: -20

## 2. Experiment A - Original 8,192-sample baseline

Config: `config/two_runner.yaml`

### Purpose

Establish the first two-Runner cooperative PPO baseline.

### Training schedule

- 8,192 valid transitions / policy / PPO update
- 100 rounds
- Runner 0: 32 parallel worlds
- Runner 1: 16 parallel worlds
- total used transitions / policy: 819,200
- progress shaping: instantaneous Euclidean distance change

### Important validation checkpoints

| Round | Team success | Any collision | Both dead | Notes |
| ---: | ---: | ---: | ---: | --- |
| 50 | 52.5% | 36.5% | 6.0% | success starts improving |
| 60 | **61.0%** | 30.0% | 5.0% | highest baseline success |
| 65 | 51.0% | **11.0%** | 5.0% | strongest safety/success trade-off |
| 70 | 47.5% | 16.0% | 8.5% | performance begins to drift |
| 75 | 35.5% | 30.5% | - | large regression |
| 80 | 40.5% | 27.5% | - | partial recovery |
| 100 | 43.0% | 34.0% | 5.0% | final checkpoint |

### Result

The baseline demonstrated that the two independent policies can learn cooperative task completion, but validation performance is strongly non-monotonic. A good joint policy can appear at one checkpoint and then regress after further PPO updates.

The 8,192 number is raw transitions rather than independent trajectories. With episodes lasting hundreds of steps, each PPO update is based on only tens of independent team episodes. This motivated a controlled large-batch test.

## 3. Experiment B - Large-Batch controlled ablation

Config: `config/two_runner_large_batch.yaml`

Run directory: `runs/exp2_two_runner_large_batch_32768`

### Hypothesis

A PPO update may be too noisy when only 8,192 transitions are collected. Increasing the amount of experience per update may make successful behavior less dependent on a small number of stochastic trajectories.

### Controlled change

Only the batch schedule was changed:

| Setting | Samples/update/policy | Rounds | Total samples/policy |
| --- | ---: | ---: | ---: |
| Original baseline | 8,192 | 100 | 819,200 |
| Large Batch | 32,768 | 25 | 819,200 |

PPO hyperparameters, environment, reward, observation, and world counts were unchanged.

Total PPO minibatch optimizer steps also remained equal at 12,800 over the whole experiment.

### Result summary

- average validation team success: approximately **48.9%**
- average validation any-collision rate: approximately **49.8%**
- highest validation team success: **67.5% at Round 23**
- Round 23 collision rate: **48.0%**
- Round 23 both-dead rate: **6.0%**
- final Round 25 team success: **62.0%**
- final Round 25 collision rate: **53.5%**
- final Round 25 both-dead rate: **8.5%**
- last-five-round mean team success: approximately **61.9%**
- last-five-round mean collision rate: approximately **51.5%**

### Interpretation

Increasing the batch size materially improved late-stage task success compared with the original baseline:

```
Original Round100: 43.0% success
Large Batch Round25: 62.0% success
```

Peak success also increased from 61.0% to 67.5%.

However, the policy still oscillated strongly and collision rate became high. Large Batch therefore improved sample quality/stability but did not solve policy regression by itself.

The new diagnostics also showed that successful trajectories were not rare in many updates. In some rounds roughly 60-70% of completed training episodes were team successes. This weakens the hypothesis that regression is caused only by "almost never seeing a good trajectory."

## 4. Experiment C - Record-Progress reward

Config: `config/two_runner_record_progress.yaml`

Run directory: `runs/exp2_two_runner_record_progress`

### Hypothesis

Instantaneous Euclidean shaping penalizes a useful obstacle detour whenever the Runner temporarily becomes farther from the goal.

Record-Progress stores each Runner's best goal distance during the current episode and pays progress only when that historical record is improved.

```
record_progress_i =
  best_distance_i - current_distance_i, if a new record is made
  0, otherwise
```

Temporary movement away from the goal therefore receives no progress penalty. Returning to an already visited distance cannot replay reward.

Team record progress is the maximum new personal record improvement made by either Runner on that step.

### Controlled settings

The 32,768 x 25 Large-Batch schedule was retained. PPO settings, reward scales, world counts, terminal rewards, and observations remained unchanged.

The local full regression suite passed:

```
84 passed in 2.19s
```

### Result summary

- average validation team success: approximately **43.9%**
- average validation any-collision rate: approximately **43.6%**
- highest validation team success: **61.0% at Round 24**
- Round 24 collision rate: **45.0%**
- Round 24 both-dead rate: **4.5%**
- final Round 25 team success: **58.5%**
- final Round 25 collision rate: **41.0%**
- final Round 25 both-dead rate: **6.5%**
- lowest collision checkpoint: Round 6
  - success: 40.0%
  - collision: 32.5%
  - both dead: 4.0%
- last-five-round mean team success: approximately **53.8%**
- last-five-round mean collision rate: approximately **43.0%**

### Interpretation

Record-Progress changed the learned policy in the expected safety direction:

- collision rate decreased relative to the Large-Batch instantaneous-Euclidean experiment
- both-dead rate was often low late in training
- the policy was not simply frozen or refusing to move because late-round success still reached 56-61%

However, task success was lower than Large Batch and strong checkpoint-to-checkpoint oscillation remained. For example, early validation moved:

```
51.0% -> 27.5% -> 53.0% -> 25.5%
```

Therefore, penalizing temporary Euclidean detours was a real shaping problem, but it was not the main cause of PPO regression.

Record-Progress also creates long spans of zero progress reward while a Runner is still inside its historical best-distance contour, so it does not provide dense directional credit throughout the detour.

## 5. Experiment D - Geodesic-Progress reward

Config: `config/two_runner_geodesic_progress.yaml`

Run directory: `runs/exp2_two_runner_geodesic_progress`

### Hypothesis

Instead of suppressing Euclidean detour penalties, use shortest traversable path distance through the obstacle map. A physically useful detour can then receive positive progress immediately even when Euclidean goal distance temporarily increases.

For Runner i:

```
p_i(t) = D_geo_i(t) - D_geo_i(t+1)
```

The 18-D observation remains unchanged. Geodesic information is privileged training-only reward information.

### Implementation

- grid resolution: 0.20 m
- static obstacles inflated by robot radius
- walls reduced by robot radius
- SciPy sparse Dijkstra computes a reverse shortest-path field from the goal
- distance field is generated on map creation/restore
- per-step reward uses interpolation rather than running A*
- teammate position is intentionally excluded from the geodesic field
- fallback to Euclidean distance exists only if a local geodesic query has no finite field value
- worker records `geodesic_fallback_rate`

### Validation results by round

| Round | Success | Collision | Both dead |
| ---: | ---: | ---: | ---: |
| 1 | **55.5%** | 36.0% | 13.5% |
| 2 | 53.5% | 44.5% | 13.5% |
| 3 | 38.5% | 47.5% | 14.5% |
| 4 | 39.0% | 51.5% | 13.5% |
| 5 | 36.0% | 53.5% | 12.0% |
| 6 | 32.0% | 61.0% | 11.5% |
| 7 | 25.0% | 60.5% | 22.0% |
| 8 | 30.0% | 59.0% | 19.0% |
| 9 | 28.5% | 46.0% | 12.5% |
| 10 | 37.5% | 40.5% | 8.5% |
| 11 | 32.5% | 46.5% | 15.0% |
| 12 | 38.0% | 53.5% | 13.0% |
| 13 | 39.5% | 50.5% | 19.5% |
| 14 | 34.5% | 49.5% | 20.0% |
| 15 | 42.0% | 35.5% | 7.0% |
| 16 | 47.5% | 36.0% | 5.0% |
| 17 | 41.5% | 52.0% | 3.5% |
| 18 | 46.5% | 51.0% | 6.0% |
| 19 | 51.5% | 40.0% | 3.5% |
| 20 | 46.0% | 41.0% | 6.0% |
| 21 | 39.0% | **34.5%** | 10.5% |
| 22 | 46.5% | 37.5% | **3.0%** |
| 23 | 38.5% | 44.0% | 8.5% |
| 24 | 42.0% | 42.5% | 3.5% |
| 25 | 31.5% | 44.5% | 5.5% |

### Aggregate result

- mean validation success over 25 rounds: **39.7%**
- mean validation collision rate: **46.34%**
- mean both-dead rate: **10.8%**
- highest success: **55.5% at Round 1**
- lowest collision: **34.5% at Round 21**
- final Round 25:
  - success: **31.5%**
  - collision: **44.5%**
  - both dead: **5.5%**
- last-five-round mean success: **39.5%**
- last-five-round mean collision: **40.6%**
- `geo_fb=0.000` for both workers in every reported round

### Interpretation

The geodesic implementation was actually active throughout training. The fallback rate staying at zero rules out a simple explanation that the experiment silently reverted to Euclidean progress because the grid was disconnected.

Despite providing an obstacle-aware progress signal, Geodesic-Progress did not improve the learned policy in this setup:

- peak success was below Large Batch and Record-Progress
- final success was substantially lower
- the highest success occurred at Round 1, before prolonged geodesic training
- later training often degraded the warm-start policy rather than improving it

Therefore, the current evidence does **not** support the hypothesis that replacing Euclidean progress with a global geodesic distance scalar is sufficient to fix the regression.

A plausible interpretation is that the reward oracle knows the global shortest-path potential, while the policy observation remains local: relative goal position plus eight lidar rays and teammate state. The policy receives reward for moving along a globally good path but is not directly given the global route or geodesic direction. This is a hypothesis for future testing, not a result established by this experiment.

Another important observation is that early PPO updates still had large approximate KL values, for example:

```
Round 1 R0 KL = 0.04357
Round 2 R0 KL = 0.04230
Round 6 R1 KL = 0.03231
```

Large one-update policy movement remains a viable explanation for why a good warm-start policy can be damaged quickly.

## 6. Cross-experiment comparison

| Experiment | Progress shaping | Samples/update | Rounds | Peak success | Final success | Representative collision result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A. Original | instant Euclidean | 8,192 | 100 | 61.0% | 43.0% | 11.0% at R65 |
| B. Large Batch | instant Euclidean | 32,768 | 25 | **67.5%** | **62.0%** | 48.0% at peak R23 |
| C. Record Progress | best-so-far Euclidean | 32,768 | 25 | 61.0% | 58.5% | 41.0% final; 32.5% minimum |
| D. Geodesic Progress | obstacle-aware shortest path | 32,768 | 25 | 55.5% | 31.5% | 44.5% final; 34.5% minimum |

## 7. What the experiments currently support

### Supported by the observations

1. **Per-update data amount matters.**
   The 32,768-sample Large-Batch experiment achieved substantially better late-stage success than the original 8,192-sample baseline at the same total used sample count.

2. **Instant Euclidean shaping biases the task toward aggressive direct progress.**
   Record-Progress reduced collision rates while preserving meaningful task success.

3. **Simply removing detour penalties does not eliminate regression.**
   Record-Progress still showed severe validation oscillation.

4. **Simply replacing Euclidean distance with geodesic distance does not solve the problem.**
   Geodesic-Progress underperformed both Large Batch and Record-Progress despite zero geodesic fallback.

5. **The training process remains capable of damaging an already useful policy.**
   Across experiments, high-performing checkpoints were frequently followed by materially worse checkpoints.

### Not yet established

The current experiments do not yet establish which of the following is the dominant remaining cause:

- PPO update overshoot / excessive KL per commit
- long-horizon credit assignment
- independent multi-agent non-stationarity
- reward-scale/value-function mismatch after changing shaping
- mismatch between privileged global reward information and local policy observations

## 8. Current most useful checkpoints

For analysis and future controlled experiments, preserve at least:

### Original baseline

- `round_00060.pt`: highest observed baseline success
- `round_00065.pt`: strongest observed baseline safety/success compromise

### Large Batch

- `round_00023.pt`: 67.5% validation success
- `round_00025.pt`: 62.0% final validation success

### Record Progress

- `round_00006.pt`: lowest observed collision in this experiment
- `round_00024.pt`: 61.0% validation success
- `round_00025.pt`: 58.5% final validation success

### Geodesic Progress

- `round_00001.pt`: 55.5% validation success
- `round_00019.pt`: 51.5% success with 40.0% collision and 3.5% both-dead
- `round_00021.pt`: lowest collision at 34.5%
- `round_00025.pt`: final checkpoint

Do not select a final Experiment 2 model using seeds 60000-60199 until the model-selection rule is frozen.

## 9. Recommended next controlled test

The strongest unresolved signal is large one-update KL movement. The next clean ablation should preserve the best-performing Large-Batch setup and add only a PPO KL guard / early stop, rather than changing multiple reward and credit parameters at the same time.

Example comparison:

```
Large Batch:
32768 samples/update
instant Euclidean progress
no KL stop

vs.

Large Batch + KL guard:
32768 samples/update
instant Euclidean progress
target_kl around 0.015
```

This directly tests whether policy regression is caused by PPO overshoot while keeping the highest-performing reward/batch configuration fixed.
