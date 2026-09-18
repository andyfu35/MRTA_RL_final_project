# Experiment 2 Fresh Shared-Joint Rollout PPO

## Motivation

The previous 32,768-sample Large-Batch run collected a large on-policy rollout and then reused the same data for four PPO epochs before collecting again. This improves sample efficiency, but the policy can move substantially while the data remain fixed.

This experiment changes the interaction/update rhythm and trains the two Runner policies **from scratch** with fresh random Actor-Critic weights:

```
random policy initialization
-> collect fresh joint rollout
-> one PPO epoch for R0
-> one PPO epoch for R1
-> synchronous commit
-> immediately recollect with the new joint policy
```

No Experiment 1 or previous Experiment 2 policy weights are inherited. The goal is to test whether this new training architecture can learn navigation directly, without carrying behavior learned under the older update regime.

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
- initialization: random weights; no warm-start checkpoint

## Run

```bash
git fetch origin
git checkout experiment/exp2-fresh-joint-rollout
git pull origin experiment/exp2-fresh-joint-rollout

source .venv/bin/activate
pip install -e .
python -m pytest -q

mkdir -p runs/exp2_two_runner_fresh_joint_scratch

python -m marl2d two-train \
  --config config/two_runner_fresh_joint.yaml \
  --from-scratch \
  --output runs/exp2_two_runner_fresh_joint_scratch \
  --device cpu 2>&1 | tee runs/exp2_two_runner_fresh_joint_scratch/train.log
```
