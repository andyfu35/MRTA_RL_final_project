# Gene-based Homogeneous MRTA V1.2

V1.2 adds exactly one new source of task heterogeneity: variable service time.
V1 and V1.1 remain untouched.

## Research question

Can a shared gene allocator learn that the nearest task is not always the best task when different tasks occupy a robot for different amounts of time?

## World

- continuous 100 x 100 plane
- 4 homogeneous robots
- 20 tasks
- no priority
- no deadline
- no obstacle
- no battery model
- constant robot speed = 2 distance units / second
- task service time sampled uniformly from 2 to 15 seconds
- episode horizon = 50 seconds
- event-based simulation, no small-dt physical integration

For robot i and task j:

```text
travel_time_ij = distance_ij / robot_speed
assignment_time_ij = travel_time_ij + service_time_j
```

A task can be assigned only when it can finish before the episode horizon.

## Observation

Each free robot-task pair produces exactly four normalized observations:

1. `distance_norm`
2. `service_time_norm`
3. `robot_workload_norm`
4. `competition_norm`

The gene output remains one scalar bid:

```text
bid_ij =
    w_distance * distance_norm
  + w_service * service_time_norm
  + w_workload * robot_workload_norm
  + w_competition * competition_norm
```

The same four-parameter gene is shared by every robot.

## Workload

V1.2 no longer treats one long task and one short task as equal load.

Robot workload is:

```text
workload_i = sum(travel_time + service_time)
```

over assignments completed by that robot.

## Competition

Competition only counts robots that are:

1. free at the current allocation event;
2. able to finish that task before the episode horizon;
3. closer to the task than the candidate robot.

This avoids counting unavailable robots as competitors.

## Capability axes

### Completion

```text
completion = completed_tasks / total_tasks
```

### Efficiency

For each completed assignment:

```text
route_value = 1 - distance / world_diagonal
```

Then:

```text
efficiency = sum(route_value) / total_tasks
```

Unfinished tasks therefore contribute zero.

### Balance

Jain fairness is computed over accumulated robot workload:

```text
fairness =
    (sum(workload_i)^2)
    /
    (N * sum(workload_i^2))
```

The capability axis is completion weighted:

```text
balance = completion * fairness
```

This prevents a policy from obtaining an excellent balance score simply by making every robot do equally little work.

## Baselines

V1.2 reports four policies.

### Uninformed

All bids are tied.

### Nearest

```text
min distance
```

### Shortest service

```text
min service_time
```

### Shortest total time

```text
min(distance / robot_speed + service_time)
```

Because distance and service time are normalized differently, the baseline gene coefficients are analytically converted so its bid ordering is exactly equivalent to shortest total assignment time.

The most important comparison is:

```text
Gene vs Shortest Total Time
```

If Gene only rediscovers travel plus service duration, it should not materially outperform this baseline. Improvement beyond it would indicate that workload and competition context add useful allocation behavior.

## Training protocol

Default single-seed experiment:

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

Training worlds change every generation.
Probe worlds are fixed and are not used for parent selection.
Validation worlds are not used until the final report.

## Mac commands

Update the branch:

```bash
git fetch origin
git switch experiment/gene-homogeneous-mrta-v1
git pull
```

Smoke:

```bash
bash tools/run_gene_mrta_v12_mac.sh smoke
```

One complete seed:

```bash
bash tools/run_gene_mrta_v12_mac.sh single
```

Five-seed formal experiment:

```bash
bash tools/run_gene_mrta_v12_mac.sh full
```

Results:

```text
runs/gene_mrta_v12/
runs/gene_mrta_v12_suite/
```

The key final outputs are:

```text
summary.json
history.csv
aggregate_summary.json
per_seed.csv
```
