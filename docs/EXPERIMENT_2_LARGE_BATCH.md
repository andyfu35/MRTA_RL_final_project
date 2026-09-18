# Experiment 2 Large-Batch Controlled Ablation

## Question

The 8,192-transition baseline showed strong validation oscillation. A plausible cause is that one PPO update is based on too few independent team episodes, so rare stochastic successes can dominate one update and disappear from the next.

This ablation changes only the amount of data used per PPO update.

## Controlled comparison

| Setting | Samples/update/policy | Rounds | Total used samples/policy |
| --- | ---: | ---: | ---: |
| Baseline | 8,192 | 100 | 819,200 |
| Large batch | 32,768 | 25 | 819,200 |

Environment, reward, collection world counts, network, learning rate, gamma, GAE lambda, PPO clip, entropy coefficient, epochs, and minibatch size are unchanged.

Because minibatch size and epochs are unchanged, total optimizer minibatch steps are also matched:

- baseline: 100 * 4 * (8192 / 256) = 12,800
- large batch: 25 * 4 * (32768 / 256) = 12,800

Validation runs every large-batch round on seeds 40000-40199. Checkpoints are also written every round.

## New diagnostics

Each Runner now records, per PPO update:

- completed team episodes
- team-success / timeout / both-dead episode counts
- sample-pool success/failure transition counts
- selected success/failure transition counts
- selected unique trajectory count
- selected success/failure trajectory counts
- mean target-agent trajectory length for success/failure outcomes
- theoretical terminal-success GAE weight reaching the start of successful target trajectories
- selected success/failure advantage mean and standard deviation
- sample-pool size and discarded surplus
- existing PPO KL, entropy, value loss and policy loss

These metrics make it possible to distinguish:

1. too few successful trajectories per update,
2. successful trajectories being diluted by long failure episodes,
3. long-horizon terminal credit becoming negligible,
4. overly aggressive PPO updates,
5. true multi-agent non-stationarity.

## Run

```bash
python -m marl2d two-train \
  --config config/two_runner_large_batch.yaml \
  --init-single-runner-checkpoint runs/exp1_r2_safe30_r70_to_r100/round_00080.pt \
  --output runs/exp2_two_runner_large_batch_32768 \
  --device cpu 2>&1 | tee runs/exp2_two_runner_large_batch_32768/train.log
```

The CLI now prints each Experiment 2 round immediately instead of waiting for all rounds to finish.

## Sample-matched comparison points

| Baseline round | Large-batch round | Cumulative used samples/policy |
| ---: | ---: | ---: |
| 20 | 5 | 163,840 |
| 40 | 10 | 327,680 |
| 60 | 15 | 491,520 |
| 80 | 20 | 655,360 |
| 100 | 25 | 819,200 |

Do not use final-test seeds 60000-60199 for tuning.

## Completed result

The formal 25-round run completed successfully.

Key validation results:

- mean team success: approximately **48.9%**
- mean any-collision rate: approximately **49.8%**
- peak team success: **67.5% at Round 23**
- Round 23 collision: **48.0%**
- Round 23 both-dead: **6.0%**
- final Round 25 success: **62.0%**
- final Round 25 collision: **53.5%**
- final Round 25 both-dead: **8.5%**
- last-five-round mean success: approximately **61.9%**
- last-five-round mean collision: approximately **51.5%**

Conclusion: increasing per-update data from 8,192 to 32,768 improved late-stage and peak success, but did not remove checkpoint oscillation and produced a relatively aggressive high-collision policy. See `docs/EXPERIMENT_2_RESULTS.md` for the cross-experiment comparison.
