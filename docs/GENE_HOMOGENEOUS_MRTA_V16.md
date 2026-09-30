# Gene-based Homogeneous MRTA V1.6

V1.6 adds exactly one new source of environment complexity: finite robot battery.
V1 through V1.5 remain untouched.

## Research question

Can a shared Gene Bank allocate tasks under heterogeneous remaining battery while
retaining completion, route efficiency, priority satisfaction, deadline
satisfaction, and workload balance?

## Inherited world

- 100 x 100 continuous plane
- 4 homogeneous robots
- 20 tasks
- robot speed = 4
- service time = Uniform(2, 35) seconds
- priority = Uniform(0.1, 1.0)
- soft deadline = Uniform(25, 50) seconds
- 10 static square obstacles
- deterministic 8-connected grid A*
- episode horizon = 50 seconds
- event-based task allocation

## New V1.6 battery model

Each robot has the same nominal battery capacity but a different initial state:

```text
battery_capacity = 70 energy units
initial_battery_i ~ Uniform(35, 70)
energy_per_distance = 1
```

Travel energy is:

```text
energy_required_ij =
    AStarPathLength_ij * energy_per_distance
```

Service time does not consume battery in V1.6. This keeps Battery as the only new
complexity dimension and avoids adding an additional service-power model.

A robot-task pair is eligible only when all inherited constraints are satisfied
and:

```text
energy_required_ij <= battery_remaining_i
```

After assignment:

```text
battery_remaining_i -= energy_required_ij
```

Battery is therefore a hard feasibility constraint. Robots are never allowed to
run below zero energy.

## Observation

The shared Gene becomes eight-dimensional:

1. `euclidean_distance_norm`
2. `path_cost_norm`
3. `service_time_norm`
4. `priority_norm`
5. `deadline_remaining_norm`
6. `battery_remaining_norm`
7. `robot_workload_norm`
8. `competition_norm`

where:

```text
battery_remaining_norm =
    battery_remaining / battery_capacity
```

The bid remains linear and interpretable:

```text
bid_ij =
    w_euclidean   * euclidean_distance_norm
  + w_path        * path_cost_norm
  + w_service     * service_time_norm
  + w_priority    * priority_norm
  + w_deadline    * deadline_remaining_norm
  + w_battery     * battery_remaining_norm
  + w_workload    * robot_workload_norm
  + w_competition * competition_norm
```

No separate `energy_required_norm` feature is added because travel energy is
already proportional to A* path cost. Adding both would duplicate the same
information.

## Capability axes

V1.6 keeps the five existing capability archives:

```text
completion
efficiency
priority_satisfaction
deadline_satisfaction
balance
```

Battery is intentionally not added as a sixth optimization axis.

Directly maximizing remaining battery would create an obvious loophole:

```text
do fewer tasks -> consume less energy -> retain more battery
```

Instead, Battery changes which assignments are feasible. Its effect is measured
through the existing task-performance axes plus battery diagnostics.

## Battery diagnostics

Every evaluation also reports:

```text
mean_initial_battery
mean_final_battery
battery_remaining_fraction
energy_consumed
battery_blocked_pair_events
robot_final_batteries
```

`battery_blocked_pair_events` counts robot-task pairs that were time-feasible
but rejected because the robot did not have enough remaining energy.

This is a diagnostic only; it is not a reward.

## Baselines

All V1.5 baselines remain and are subject to exactly the same hard battery
feasibility rule.

V1.6 additionally adds:

### Max Battery Margin

For every feasible robot-task pair:

```text
battery_margin_ij =
    battery_remaining_i
    - energy_required_ij
```

The baseline chooses the pair with the largest post-task battery reserve.

It is deliberately simple and interpretable. It represents a battery-conservative
heuristic rather than an optimized weighted combination.

The strongest general baseline remains expected to be `shortest_path_time`,
now evaluated under the same battery constraint.

## Important interpretation

A positive `w_battery` means the Gene tends to assign work to robots with more
remaining energy.

A negative `w_battery` may instead indicate a strategy that preserves high
battery robots for future expensive tasks and uses lower-battery robots when a
task is still feasible.

Therefore coefficient sign alone is not treated as proof of battery reasoning.
The stronger evidence is performance under battery constraints, battery
diagnostics, and later a battery-observation ablation.

## Initial calibration target

The first V1.6 smoke should verify that Battery is neither irrelevant nor
catastrophic.

Useful diagnostics are:

```text
battery_blocked_pair_events > 0
completion remains in a usable range
battery_remaining_fraction is neither near 0 nor near 1 for every policy
```

If almost no pair is battery-blocked, the battery range is too generous.

If completion collapses because most robots cannot reach tasks, the battery range
is too restrictive.

## Default protocol

Single seed:

```text
100 generations
128 genes
8 random training worlds / generation
64 fixed probe worlds
128 fixed validation worlds
8 genes / capability archive
seed = 7
```

Formal suite:

```text
seeds = 7, 17, 27, 37, 47
```

## Mac commands

Update:

```bash
git fetch origin
git switch experiment/gene-homogeneous-mrta-v1
git pull
```

Smoke:

```bash
bash tools/run_gene_mrta_v16_mac.sh smoke
```

Single 100-generation run:

```bash
bash tools/run_gene_mrta_v16_mac.sh single
```

Formal five-seed run:

```bash
bash tools/run_gene_mrta_v16_mac.sh full
```

## Progression

```text
V1.2  Variable service time
V1.3  Priority
V1.4  Deadline
V1.5  Obstacles / A* path cost
V1.6  Battery
```

V1.6 completes the planned first full homogeneous-MRTA environment stack.


## Hungarian quality-time benchmark

V1.6 includes a separate benchmark that does not alter training.

The primary comparison is:

```text
Gene Greedy
vs
Same Gene + Hungarian
```

Both methods use exactly the same 8D Gene score matrix. The only difference is
the matching step.

- `Gene Greedy`: repeatedly selects the current highest feasible bid.
- `Same Gene + Hungarian`: finds the maximum-total-bid feasible batch matching
  for the current event.

This makes the comparison suitable for a quality-versus-computation-time study.

Hungarian optimality here is intentionally limited to the current event's
assignment matrix. It is not claimed to be the globally optimal solution of the
entire sequential MRTA episode, because future robot positions, battery states,
busy times, and available tasks depend on earlier assignments.

Two additional centralized baselines are reported:

```text
Hungarian Path-Time
Hungarian Priority-per-Path-Time
```

All methods use the same horizon and battery feasibility constraints.

### Timing protocol

World generation and A* path precomputation are performed before timing because
they are common inputs to all compared assignment policies.

The benchmark reports:

```text
decision_ms_per_world
decision_us_per_event
matcher_us_per_event
wall_ms_per_world
```

It also performs a matcher-only scaling sweep at:

```text
4, 8, 16, 32, 64, 100 robots
tasks = 5 * robots
```

The main quality-retention comparison is:

```text
Gene Greedy metric / Same-Gene Hungarian metric
```

This directly measures how much episode-level task performance is retained when
using the cheaper greedy assignment instead of exact per-event Hungarian
matching under the same learned score.

### Command

Using the latest V1.6 multiseed suite automatically:

```bash
bash tools/run_gene_mrta_v16_hungarian_benchmark_mac.sh
```

Or explicitly:

```bash
bash tools/run_gene_mrta_v16_hungarian_benchmark_mac.sh \
  runs/gene_mrta_v16_suite/multiseed_20260930_100639 \
  reference
```

The second argument may be one of:

```text
reference
completion
efficiency
priority_satisfaction
deadline_satisfaction
balance
```

For the first paper-facing comparison, use `reference`. To study near-optimal
task throughput specifically, rerun with `completion`.
