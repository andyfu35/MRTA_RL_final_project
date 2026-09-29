# Gene-based Homogeneous MRTA V1.4

V1.4 adds exactly one new source of task complexity: individual task deadlines.
V1 through V1.3 remain untouched.

## Research question

Can a shared Gene Bank learn deadline-aware task scheduling while preserving
completion, route efficiency, priority satisfaction, and workload balance?

## Inherited world

- 100 x 100 continuous plane
- 4 homogeneous robots
- 20 tasks
- robot speed = 4
- service time = Uniform(2, 35) seconds
- priority = Uniform(0.1, 1.0)
- episode horizon = 50 seconds
- no obstacles
- no battery model
- event-based simulation

## New V1.4 task property

Each task receives an absolute soft deadline:

```text
deadline_j ~ Uniform(25, 50) seconds
```

A task does not disappear when its deadline passes.

A late task can still be completed and can still contribute to:

- completion
- efficiency
- priority satisfaction
- balance

But it contributes zero to deadline satisfaction.

This keeps Deadline as an independent capability instead of silently mixing it
into Completion.

## Observation

The shared Gene becomes six-dimensional:

1. `distance_norm`
2. `service_time_norm`
3. `priority_norm`
4. `deadline_remaining_norm`
5. `robot_workload_norm`
6. `competition_norm`

where:

```text
deadline_remaining_norm =
    clip((deadline_j - current_time) / episode_time, 0, 1)
```

The bid remains linear and interpretable:

```text
bid_ij =
    w_distance    * distance_norm
  + w_service     * service_time_norm
  + w_priority    * priority_norm
  + w_deadline    * deadline_remaining_norm
  + w_workload    * robot_workload_norm
  + w_competition * competition_norm
```

A negative deadline coefficient favors tasks with less remaining time.

## Capability axes

V1.4 maintains independent archives for five capabilities:

```text
completion
efficiency
priority_satisfaction
deadline_satisfaction
balance
```

### Deadline satisfaction

```text
deadline_satisfaction =
    number of tasks completed by their own deadline
    / total number of tasks
```

Therefore:

```text
deadline_satisfaction <= completion
```

Late completion is still useful to the system, but does not receive deadline credit.

## Deadline baselines

All V1.3 baselines are retained.

V1.4 adds:

### Earliest Deadline First

```text
choose minimum deadline
```

### Least Laxity First

For robot-task pair (i, j):

```text
duration_ij =
    distance_ij / robot_speed
    + service_time_j

laxity_ij =
    deadline_j
    - current_time
    - duration_ij
```

Among assignments that can still meet their deadline, smaller non-negative
laxity is preferred.

If no on-time pair remains, the baseline falls back toward shorter assignment
duration.

The key deadline comparison is:

```text
Deadline specialist Gene
vs
Least Laxity
```

## Specialist-aware formal reporting

Starting in V1.4, the multi-seed suite reports both:

- reference/maximin Gene
- completion specialist
- efficiency specialist
- priority specialist
- deadline specialist
- balance specialist

This avoids confusing the reference Gene with a capability specialist.

The aggregate output includes:

```text
reference
specialists
baselines
delta_reference_minus_baseline
specialist_own_axis
```

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
bash tools/run_gene_mrta_v14_mac.sh smoke
```

Single 100-generation run:

```bash
bash tools/run_gene_mrta_v14_mac.sh single
```

Formal five-seed run:

```bash
bash tools/run_gene_mrta_v14_mac.sh full
```

## Progression

```text
V1.2  Variable service time
V1.3  Priority
V1.4  Deadline
V1.5  Obstacles / path cost
V1.6  Battery
```

V1.4 is the first stage where the task allocator must reason about explicit
time pressure, so the problem becomes MRTA plus scheduling rather than only
allocation.
