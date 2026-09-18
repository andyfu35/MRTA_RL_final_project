# Experiment 2 Fresh Shared-Joint Rollout PPO

## Motivation

The previous 32,768-sample Large-Batch run collected a large on-policy rollout and then reused the same data for four PPO epochs before collecting again. This improves sample efficiency, but the policy can move substantially while the data remain fixed.

This experiment changes the interaction/update rhythm:

```
collect fresh joint rollout
-> one PPO epoch for R0
-> one PPO epoch for R1
-> synchronous commit
-> immediately recollect with the new joint policy
```

The goal is to reduce policy-to-data staleness and directly observe the effect of each committed update in the next rollout.

## Shared joint worlds

Unlike the previous architecture, R0 and R1 no longer collect on separate world pools.

A single collector owns:

- 64 joint worlds
- both R0 and R1 in every world
- one shared team episode outcome
- separate per-agent action/log-prob/value/advantage trajectories

Both policies therefore learn from the same map seeds and the same team episodes.

When either Runner dies, it stops producing actor samples. Its valid trajectory remains pending until the team outcome is known so delayed team-success credit is still applied correctly.

## Freshness schedule

Formal configuration:

- 8,192 selected transitions / policy / round
- 100 rounds
- 819,200 selected transitions / policy total
- 64 shared joint worlds
- minibatch size 256
- PPO epochs = 1
- 32 optimizer steps maximum / policy / round
- no target-KL guard
- after the one epoch, both new policies are committed together
- the next rollout is generated only by the newly committed policy set

Each selected transition is therefore used in exactly one PPO epoch.

For comparison:

```
Large Batch:
32768 samples -> 4 epochs -> up to 512 optimizer steps -> recollect

Fresh Joint:
8192 samples -> 1 epoch -> 32 optimizer steps -> recollect
```

Both still use 819,200 selected transitions per policy over the full formal experiment.

## Unchanged

- instantaneous Euclidean progress reward
- environment and obstacle distribution
- 18-D observation
- learning rate 3e-4
- gamma 0.99
- GAE lambda 0.95
- PPO clip 0.2
- entropy coefficient 0.005
- minibatch size 256
- validation seeds 40000-40199

## Run

```bash
git fetch origin
git checkout experiment/exp2-fresh-joint-rollout
git pull origin experiment/exp2-fresh-joint-rollout

source .venv/bin/activate
pip install -e .
python -m pytest -q

mkdir -p runs/exp2_two_runner_fresh_joint

python -m marl2d two-train \
  --config config/two_runner_fresh_joint.yaml \
  --init-single-runner-checkpoint runs/exp1_r2_safe30_r70_to_r100/round_00080.pt \
  --output runs/exp2_two_runner_fresh_joint \
  --device cpu 2>&1 | tee runs/exp2_two_runner_fresh_joint/train.log
```
