# Experiment 2 Record-Progress Reward Ablation

## Motivation

The original shaping reward uses instantaneous distance change:

```
progress = d_previous - d_current
```

That makes a necessary detour look bad whenever the Runner temporarily moves away from the goal. In obstacle navigation this can discourage going around a wall even when the detour is required for eventual success.

The record-progress variant instead keeps a per-Runner best-so-far goal distance for the current episode.

## Definition

For Runner `i`:

```
improvement_i = best_distance_i - current_distance_i
```

A new progress reward is emitted only when the improvement is positive and reaches the configured epsilon threshold:

```
record_progress_i =
    improvement_i, if improvement_i >= epsilon
    0,             otherwise
```

When the threshold is crossed:

```
best_distance_i = current_distance_i
```

Sub-threshold improvements do not update the record, so they accumulate instead of being lost.

The shared team progress is:

```
team_record_progress = max(record_progress_0, record_progress_1)
```

This means:
- moving farther from the goal gives zero progress reward, not a negative progress reward;
- returning to an already visited best distance gives zero progress reward;
- oscillating cannot repeatedly farm the same progress reward;
- a dead Runner's historical record does not block the surviving teammate from earning new team progress;
- either Runner making a genuinely new distance record produces shared team progress.

## Reward used in this experiment

For an alive Runner:

```
r_i =
  3 * personal_record_progress_i
+ 2 * team_record_progress
- 0.01
- 3 * danger_i^2
```

Terminal semantics remain unchanged:
- individual collision: -100 and that Runner dies;
- timeout: -20 for Runners still alive;
- either Runner reaches the goal: team success +100.

## Controlled settings

This experiment inherits the large-batch schedule:
- 32,768 valid samples per policy per PPO update;
- 25 rounds;
- Runner 0: 32 parallel worlds;
- Runner 1: 16 parallel worlds;
- validation every round on seeds 40000-40199;
- all PPO hyperparameters unchanged.

Only the progress-shaping definition changes.

## Run

```bash
python -m marl2d two-train \
  --config config/two_runner_record_progress.yaml \
  --init-single-runner-checkpoint runs/exp1_r2_safe30_r70_to_r100/round_00080.pt \
  --output runs/exp2_two_runner_record_progress \
  --device cpu 2>&1 | tee runs/exp2_two_runner_record_progress/train.log
```

Keep confirmation seeds 50000-50199 and final-test seeds 60000-60199 untouched until checkpoint selection is locked.
