# Gene-based Homogeneous MRTA V1.6-T

V1.6-T does not change the V1.6 environment. It adds one new capability axis
whose purpose is to evolve a very fast Gene policy toward the same time-oriented
assignment behavior measured by event-based Hungarian Path-Time.

## Research question

Can an offline-evolved shared Gene compress near-Hungarian time-oriented
assignment behavior into a low-cost online linear bidder plus greedy matching?

## Environment

Exactly inherited from V1.6:

- 4 homogeneous robots
- 20 tasks
- 100 x 100 world
- variable service time
- priority
- soft deadline
- static obstacles and A* path cost
- finite heterogeneous remaining battery
- hard horizon and battery feasibility
- 8D Gene observation

For the same seed, V1.6-T must generate the same robots, tasks, service times,
priorities, deadlines, obstacles, path table, and initial battery values as
V1.6.

## New capability axis: Time Optimality

For every completed task j:

```text
time_contribution_j =
    1 - finish_time_j / episode_horizon
```

The episode metric is:

```text
time_optimality =
    sum(time_contribution_j for completed tasks)
    / total_task_count
```

Unfinished tasks contribute zero.

Therefore:

```text
0 <= time_optimality <= completion <= 1
```

This avoids the trivial loophole in raw total-time minimization:

```text
do no tasks -> zero total time
```

When two policies complete the same number of tasks, maximizing
`time_optimality` favors earlier task completion and therefore lower cumulative
completion time.

At one event, all currently free robots share the same current time. For a
fixed-cardinality feasible matching:

```text
maximize sum(1 - finish_time / H)
<=> minimize sum(path_time + service_time)
```

so the new axis is directly aligned with Hungarian Path-Time assignment.

## Six independent archives

V1.6-T uses:

```text
completion
efficiency
priority_satisfaction
deadline_satisfaction
balance
time_optimality
```

There is still no scalar weighted reward.

The Gene remains 8D. No Hungarian result is inserted into the observation and
Hungarian is not used for parent selection.

## Hungarian regret diagnostic

The separate V1.6-T Hungarian benchmark evaluates the trained Gene on held-out
worlds.

At every actual Gene decision event, it computes a counterfactual Hungarian
Path-Time matching on the exact same current state.

Define:

```text
U_G = sum(1 - finish_time / H) for the Gene assignments
U_H = sum(1 - finish_time / H) for the Hungarian assignments
local_regret = (U_H - U_G) / U_H
```

The reported mean local regret is diagnostic only. It is excluded from the
measured Gene runtime and does not influence evolution.

The benchmark also reports:

```text
local_hungarian_time_retention = 1 - local_regret
```

and episode-level:

```text
Gene time_optimality / Hungarian Path-Time time_optimality
```

## Runtime comparison

The benchmark compares:

```text
Gene Greedy
Same Gene + Hungarian
Greedy Path-Time
Hungarian Path-Time
Hungarian Priority-per-Path-Time
```

World generation and A* path precomputation are excluded from policy decision
timing because they are common inputs.

The main runtime outputs are:

```text
decision_ms_per_world
decision_us_per_event
matcher_us_per_event
wall_ms_per_world
```

The offline Hungarian regret diagnostic is subtracted from runtime timing.

A matcher scaling sweep is also run at:

```text
robots = 4, 8, 16, 32, 64, 100
tasks = 5 * robots
```

## Experiment sequence

First validate the new metric and archive:

```bash
bash tools/run_gene_mrta_v16t_mac.sh smoke
```

If the time archive separates from the existing five archives:

```bash
bash tools/run_gene_mrta_v16t_mac.sh single
```

Then:

```bash
bash tools/run_gene_mrta_v16t_mac.sh full
```

After a full multiseed suite, evaluate the time specialist:

```bash
bash tools/run_gene_mrta_v16t_hungarian_benchmark_mac.sh \
  <v16t_multiseed_suite_dir> \
  time_optimality
```

The benchmark defaults to `time_optimality` if the second argument is omitted.

## Interpretation boundary

Hungarian is optimal only for the current event assignment matrix under the
selected Path-Time objective.

It is not claimed to be the globally optimal solution of the entire sequential
MRTA episode, because each assignment changes later robot position, remaining
battery, busy time, and task availability.

The paper-facing claim should therefore be framed as:

```text
near event-level Hungarian time-assignment quality
with lower online computation
```

rather than global MRTA optimality.
