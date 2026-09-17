# Experiment 1 - Single Runner Navigation

## Goal

Verify that one shared PPO policy can learn goal-directed obstacle avoidance in a 20 m x 20 m arena before adding any multi-agent competition.

## Environment

- Differential-drive kinematics only; no rigid-body physics noise.
- Action: left/right normalized wheel commands in `[-1, 1]`.
- Wheel ground-speed limit: `2.0 m/s` per wheel.
- 14 random axis-aligned rectangular obstacles per episode.
- 8-ray LiDAR observation plus relative goal, heading, collision flag, previous action.
- Each parallel world has an independent, reproducible map-seed stream.

## PPO Collection

Formal baseline:

- 32 parallel worlds.
- 256 steps/world/update.
- 8192 samples/update.
- Stochastic action sampling during training.
- Deterministic action during evaluation.

## Reward Ablation

- R0: terminal rewards only.
- R1: R0 + progress + time penalty.
- R2: R1 + safety-distance shaping. This is the formal baseline.
- R3: R2 + small heading shaping.

R2 shaping before terminal states:

`5 * (d_prev - d_now) - 0.01 - 0.5 * max(0, (0.5 - clearance) / 0.5)^2`

Terminal rewards override shaping:

- Goal: +100
- Collision: -100 and terminate
- Timeout: -20

R3 additionally adds `0.02 * cos(heading_error)`.

## Evaluation

Use held-out map seeds starting at 10000. Track:

- Success rate
- Collision rate
- Mean time to goal
- Mean path efficiency
- Mean episode reward
- Mean minimum clearance

Do not compare reward variants on different evaluation seed sets.
