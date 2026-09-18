# Experiment 2 Geodesic-Progress PPO

## Hypothesis

Straight-line Euclidean progress can punish a correct obstacle detour because the Runner may have to move temporarily farther from the goal.

This experiment replaces only the progress metric with an obstacle-aware shortest traversable path distance while preserving the large-batch PPO schedule and all terminal/safety rewards.

## Progress definition

For Runner `i`:

```
p_i(t) = D_geo_i(t) - D_geo_i(t+1)
```

where `D_geo` is shortest-path distance through the static map for the center of a robot with radius 0.30 m.

Team progress preserves the original Experiment 2 semantics: among the Runners alive at the start of the step,

```
p_team = min(D_geo_alive(t)) - min(D_geo_alive(t+1))
```

The reward remains:

```
r_i =
  3 * p_i
+ 2 * p_team
- 0.01
- 3 * danger_i^2
```

Terminal rewards are unchanged:
- team success: +100
- individual collision: -100 and that Runner dies
- timeout for an alive Runner: -20

## Distance-field implementation

A new geodesic field is generated once when a static obstacle map is created or restored.

- grid resolution: 0.20 m
- static walls are reduced by robot radius
- rectangular obstacles are inflated by robot radius
- graph includes axis, diagonal, and conservative sqrt(5) moves
- diagonal/longer moves require crossed cells to be free, preventing corner cutting
- shortest-path field is computed from the goal with SciPy sparse Dijkstra
- per-step reward only performs distance-field interpolation; it does not run A*
- fields are cached with a bounded 256-map LRU, which is especially useful because validation reuses the same held-out map seeds each round

The teammate is deliberately NOT inserted into the geodesic map. Robot-to-robot interaction remains handled by collision and safety shaping. This prevents a moving teammate from changing the navigation potential every step.

If the coarse grid cannot connect a physically valid narrow corridor, a query with no finite local field value falls back to Euclidean distance rather than injecting infinity/NaN into PPO.

## Controlled comparison

The experiment keeps the Large-Batch settings unchanged:

- 32,768 valid samples per policy per PPO update
- 25 rounds
- 819,200 total used samples per policy
- Runner 0: 32 parallel worlds
- Runner 1: 16 parallel worlds
- PPO learning rate: 3e-4
- gamma: 0.99
- GAE lambda: 0.95
- clip: 0.2
- epochs: 4
- minibatch: 256
- validation every round on seeds 40000-40199

The observation remains exactly 18-D. The global shortest-path field is privileged training-only reward information and is not exposed to either policy.

## Run

```bash
git checkout experiment/exp2-geodesic-progress
git pull origin experiment/exp2-geodesic-progress
source .venv/bin/activate
pip install -e .

python -m pytest -q

mkdir -p runs/exp2_two_runner_geodesic_progress

python -m marl2d two-train \
  --config config/two_runner_geodesic_progress.yaml \
  --init-single-runner-checkpoint runs/exp1_r2_safe30_r70_to_r100/round_00080.pt \
  --output runs/exp2_two_runner_geodesic_progress \
  --device cpu 2>&1 | tee runs/exp2_two_runner_geodesic_progress/train.log
```

Do not use confirmation seeds 50000-50199 or final-test seeds 60000-60199 until checkpoint selection is locked.
