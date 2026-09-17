# MARL2D Prototype Design

## Goal
Build a single-machine prototype of a future four-computer ROS2 distributed training system for four independent PPO agents controlling 2D differential-drive robots. Two runners cooperate to get either runner to a goal; two blockers cooperate to prevent this.

## Scope for this prototype
- No MuJoCo and no rigid-body physics.
- Deterministic 2D differential-drive kinematics.
- Four independent PPO policies: `runner_0`, `runner_1`, `blocker_0`, `blocker_1`.
- One local process emulates four training computers.
- Synchronous round-based updates: all four policies train exactly once from the same frozen policy-set version, then all four new policies are committed together.
- Each virtual computer can use a different collection profile while collecting the same number of samples per update.
- Rewards, environment parameters, training hyperparameters, and future node addresses live in YAML.
- A transport boundary isolates policy exchange so mock exchange can later be replaced by ROS2.
- Produce a 2D top-view GIF for inspection.

## Environment
Robot state is `(x, y, theta)`. Each action is `(left_wheel, right_wheel)` in `[-1, 1]` and is converted to wheel angular velocity using `max_wheel_speed`.

For wheel radius `r`, wheel base `L`, and timestep `dt`:

`v = r/2 * (omega_r + omega_l)`

`theta_dot = r/L * (omega_r - omega_l)`

`x_next = x + v*cos(theta)*dt`

`y_next = y + v*sin(theta)*dt`

`theta_next = wrap(theta + theta_dot*dt)`

Robot-obstacle and robot-robot collision are geometric. A colliding move is rolled back to the previous position and the collision flag is set.

## Observation
Each agent receives a fixed-size vector containing:
- goal position relative to itself in its body frame: 2
- own heading as `sin(theta), cos(theta)`: 2
- each of the other three robots as relative body-frame `(dx, dy)`: 6
- 8 deterministic lidar-like range values against walls and circular obstacles: 8
- collision flag: 1
- previous wheel action: 2

Total: 21 values.

All distances are normalized by arena size / lidar range as appropriate.

## Rewards
Runner reward combines:
- team progress based on the closest runner's distance to goal
- individual progress
- success bonus when either runner reaches the goal
- collision penalty
- step penalty

Blocker reward combines:
- the negative of runner team progress
- timeout success bonus if the runners fail before episode limit
- collision penalty
- small shaping reward for staying between runners and goal / near the closest runner

All weights are configurable in YAML.

## PPO
Each agent has its own actor-critic network and optimizer. Continuous wheel actions use a tanh-squashed Gaussian policy. GAE computes advantages and PPO clipped objective performs each update.

## Synchronous training
For round `k`, all workers receive a frozen snapshot `P^k` containing all four policies. Each worker collects exactly `samples_per_update` for only its trainable agent while the other three agents act using `P^k`.

Different workers may collect those samples differently, e.g.:
- 32 envs x 32 steps x 1 batch = 1024
- 16 envs x 32 steps x 2 batches = 1024
- 8 envs x 32 steps x 4 batches = 1024
- 4 envs x 32 steps x 8 batches = 1024

After its PPO update, each worker publishes policy `k+1` to the mock exchange and reports READY. The coordinator commits round `k+1` only when all four versions are present. No worker can train round `k+1` before the commit.

## Future ROS2 boundary
`PolicyExchange` is an interface. The prototype uses `MockPolicyExchange`. Later a ROS2 implementation will move policy snapshots/status/commit messages over ROS2 while retaining the same trainer/coordinator behavior. YAML already contains host addresses for the four future computers and coordinator.

## Success criteria
- Deterministic kinematics unit tests pass.
- Collision rollback works.
- Observation shape is stable at 21.
- Each virtual worker collects the configured equal sample count despite different parallelism profiles.
- A coordinator round cannot commit until four updated policies are present.
- A short smoke training run completes multiple synchronized rounds with finite PPO losses/returns.
- Evaluation can render a top-view GIF.
