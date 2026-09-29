# Gene-based Homogeneous MRTA V1.3

V1.3 adds exactly one new source of task heterogeneity: task priority.
V1, V1.1, and V1.2 remain untouched.

## Research question

Can a shared gene allocator trade task priority against travel distance, service time,
current robot workload, and robot-task competition better than fixed priority heuristics?

## Inherited V1.2 world

- 100 x 100 continuous plane
- 4 homogeneous robots
- 20 tasks
- robot speed = 4 distance units / second
- service time = Uniform(2, 35) seconds
- episode horizon = 50 seconds
- no deadline
- no obstacle
- no battery model
- event-based simulation

V1.2 established that Gene allocation can outperform nearest, shortest-service,
and shortest-total-time heuristics when service time varies.

## New V1.3 task property

Each task receives:

```text
priority_j ~ Uniform(0.1, 1.0)
```

Priority has no effect on travel time or service time. It only describes task value.

## Observation

The pairwise observation becomes five-dimensional:

1. `distance_norm`
2. `service_time_norm`
3. `priority_norm`
4. `robot_workload_norm`
5. `competition_norm`

The shared gene is still a single interpretable linear bidder:

```text
bid_ij =
    w_distance    * distance_norm
  + w_service     * service_time_norm
  + w_priority    * priority_norm
  + w_workload    * robot_workload_norm
  + w_competition * competition_norm
```

## Capability axes

V1.3 keeps the original three V1.2 capabilities and adds one independent priority axis.

### Completion

```text
completion = completed_tasks / total_tasks
```

### Efficiency

```text
efficiency =
    sum(1 - distance/world_diagonal over completed tasks)
    / total_tasks
```

### Priority satisfaction

```text
priority_satisfaction =
    sum(priority of completed tasks)
    / sum(priority of all tasks)
```

This distinguishes completing many low-value tasks from completing high-value tasks.

### Balance

```text
workload_i = sum(travel_time + service_time)

fairness =
    (sum(workload_i)^2)
    /
    (N * sum(workload_i^2))

balance = completion * fairness
```

## Independent Gene Bank archives

Evolution does not collapse the objectives into one weighted scalar reward.

Each generation preserves independent archives for:

```text
completion
efficiency
priority_satisfaction
balance
```

Parents are sampled across these capability archives.

The fixed probe set is never used for parent selection.
A probe hall of fame only prevents good reporting/model-selection candidates from being lost.
Validation remains untouched until the final report.

## Baselines

V1.3 reports:

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
min(distance / speed + service_time)
```

### Highest priority

```text
max priority
```

### Priority per time

```text
max(
    priority
    /
    (distance / speed + service_time)
)
```

The most important V1.3 comparison is Gene versus `priority_per_time`.

If the Gene merely rediscovers "get the most priority per unit assignment time",
it should perform similarly to this heuristic. Any consistent improvement beyond it
would indicate that workload and competition context add useful MRTA behavior.

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

Five-seed suite:

```text
7, 17, 27, 37, 47
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
bash tools/run_gene_mrta_v13_mac.sh smoke
```

Single 100-generation run:

```bash
bash tools/run_gene_mrta_v13_mac.sh single
```

Formal five-seed run:

```bash
bash tools/run_gene_mrta_v13_mac.sh full
```

## Experiment progression

```text
V1.2  Variable service time
  -> V1.3  Priority
  -> V1.4  Deadline
  -> V1.5  Obstacles / path cost
  -> V1.6  Battery
```

Only one new source of complexity is introduced at each step.
