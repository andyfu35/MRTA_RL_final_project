# Experiment 2 Large-Batch + Target-KL Guard

## Question

The best-performing Experiment 2 setting so far is the 32,768-sample Large-Batch run, but its validation curve still oscillates strongly. Early rounds also showed large approximate KL values. This experiment tests whether PPO update overshoot is damaging useful policies.

## Controlled comparison

Parent experiment:

```
config/two_runner_large_batch.yaml
```

New experiment:

```
config/two_runner_large_batch_kl_guard.yaml
```

Everything remains identical except:

```
ppo:
  target_kl: 0.015
```

Unchanged settings include:

- 32,768 valid samples / policy / PPO update
- 25 rounds
- Runner 0: 32 parallel worlds
- Runner 1: 16 parallel worlds
- instantaneous Euclidean progress reward
- learning rate 3e-4
- gamma 0.99
- GAE lambda 0.95
- clip range 0.2
- entropy coefficient 0.005
- at most 4 PPO epochs
- minibatch size 256
- validation every round on seeds 40000-40199

## KL guard

Before each minibatch optimizer step, the current policy is compared with the frozen rollout policy on that minibatch.

The trust-region guard uses:

```
guard_kl = mean((exp(log_ratio) - 1) - log_ratio)
```

where:

```
log_ratio = log pi_new(a|s) - log pi_old(a|s)
```

This non-negative approximation is used only for the early-stop decision.

If:

```
guard_kl > 0.015
```

the minibatch that first exceeds the threshold is NOT optimized, and the rest of the current PPO update is stopped.

The historical signed `approx_kl` metric is retained for continuity with earlier experiment logs.

## New diagnostics

Each PPO update reports:

- `guard_kl`: mean KL guard value over checks
- `max_guard_kl`: maximum checked value
- `target_kl`: configured threshold
- `optimizer_steps`: actual applied minibatch optimizer steps
- `epochs_completed`: fully completed PPO epochs
- `early_stopped`: 1 when target-KL stopped the update

The Large-Batch maximum without a guard is:

```
4 * (32768 / 256) = 512 optimizer steps per policy update
```

With the KL guard, this number may be lower when the policy moves too far.

## What would support the hypothesis?

Compared with the original Large-Batch run, evidence for the overshoot hypothesis would include:

- fewer sharp validation regressions after strong checkpoints
- lower checkpoint-to-checkpoint success variance
- equal or better late-stage success
- early stopping occurring mainly on rounds with large policy movement
- actual optimizer-step counts below 512 when the guard activates

If success falls because almost every update stops extremely early, the 0.015 threshold may be too restrictive.

## Run

```bash
git fetch origin
git checkout experiment/exp2-large-batch-kl-guard
git pull origin experiment/exp2-large-batch-kl-guard

source .venv/bin/activate
pip install -e .
python -m pytest -q

mkdir -p runs/exp2_two_runner_large_batch_kl_guard

python -m marl2d two-train \
  --config config/two_runner_large_batch_kl_guard.yaml \
  --init-single-runner-checkpoint runs/exp1_r2_safe30_r70_to_r100/round_00080.pt \
  --output runs/exp2_two_runner_large_batch_kl_guard \
  --device cpu 2>&1 | tee runs/exp2_two_runner_large_batch_kl_guard/train.log
```

Do not use confirmation seeds 50000-50199 or final-test seeds 60000-60199 until model selection is frozen.
