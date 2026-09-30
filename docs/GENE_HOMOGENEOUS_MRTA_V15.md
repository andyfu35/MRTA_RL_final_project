# Gene-based Homogeneous MRTA V1.5

V1.5 adds exactly one new source of environment complexity: static obstacles and
obstacle-aware path cost. V1 through V1.4 remain untouched.

## Research question

Can a shared Gene Bank exploit actual obstacle-aware navigation cost instead of
Euclidean distance while retaining completion, efficiency, priority satisfaction,
deadline satisfaction, and workload balance?

## Inherited world

- 100 x 100 continuous plane
- 4 homogeneous robots
- 20 tasks
- robot speed = 4
- service time = Uniform(2, 35) seconds
- priority = Uniform(0.1, 1.0)
- soft deadline = Uniform(25, 50) seconds
- episode horizon = 50 seconds
- no battery model
- event-based task allocation

## New V1.5 environment property

Each world receives static non-overlapping square obstacles:

```text
obstacle count = 10
side length ~ Uniform(12, 20)
```

Robot initial positions and task positions are protected from obstacle overlap.

Obstacle layouts are regenerated deterministically from the world seed until all
robot/task points belong to one traversable connected component.

## Deterministic path planner

Navigation cost is computed on a deterministic 8-connected grid:

```text
grid resolution = 5 world units
planner = A*
diagonal corner cutting = disabled
```

For efficiency, each world precomputes path cost from:

```text
initial robot nodes -> all task nodes
task nodes          -> all task nodes
```

After a robot completes a task, its current navigation node becomes that task.
The allocator therefore performs no repeated A* search during Gene evaluation.

Actual travel time is:

```text
travel_time_ij = AStarPathLength_ij / robot_speed
```

not Euclidean distance divided by speed.

## Observation

The shared Gene becomes seven-dimensional:

1. `euclidean_distance_norm`
2. `path_cost_norm`
3. `service_time_norm`
4. `priority_norm`
5. `deadline_remaining_norm`
6. `robot_workload_norm`
7. `competition_norm`

The bid remains linear and interpretable:

```text
bid_ij =
    w_euclidean   * euclidean_distance_norm
  + w_path        * path_cost_norm
  + w_service     * service_time_norm
  + w_priority    * priority_norm
  + w_deadline    * deadline_remaining_norm
  + w_workload    * robot_workload_norm
  + w_competition * competition_norm
```

Path cost normalization uses the same spatial scale as Euclidean distance:

```text
path_cost_norm = clip(AStarPathLength / world_diagonal, 0, 1)
```

This keeps Euclidean distance and actual path cost on comparable feature scales.

Competition is path-aware: a robot counts as a closer competitor only when its
actual A* path to the task is shorter.

## Capability axes

V1.5 intentionally does not add a sixth capability archive.

Obstacles change the environment and navigation cost; they are not a separate
objective.

The five archives remain:

```text
completion
efficiency
priority_satisfaction
deadline_satisfaction
balance
```

Efficiency now uses actual path length:

```text
efficiency =
    sum(
        1 - clip(AStarPathLength / world_diagonal, 0, 1)
        for completed tasks
    )
    / total_tasks
```

The report also records:

```text
detour_ratio =
    total_actual_path_length
    / total_euclidean_length
```

so obstacle impact is directly observable.

## Baselines

All earlier baselines remain for continuity.

V1.5 adds three path-aware baselines:

### Nearest Path

```text
min AStarPathLength
```

### Shortest Path Time

```text
min(
    AStarPathLength / robot_speed
    + service_time
)
```

This is the main V1.5 overall baseline.

### Priority per Path Time

```text
max(
    priority
    /
    (AStarPathLength / robot_speed + service_time)
)
```

The previous `nearest`, `shortest_total_time`, and `priority_per_time`
continue to use Euclidean travel estimates. This creates a controlled comparison
between obstacle-unaware and obstacle-aware heuristics.

Least Laxity uses the real path-aware assignment duration because deadline
feasibility must reflect actual travel time.

## Formal comparisons

The main V1.5 questions are:

```text
Reference Gene vs Shortest Path Time
Efficiency specialist vs Nearest Path / Shortest Path Time
Priority specialist vs Priority per Path Time
Deadline specialist vs path-aware Least Laxity / Shortest Path Time
```

A useful learned behavior would also show a meaningful negative `w_path` and
possibly a weaker Euclidean-distance coefficient once path cost becomes
available.

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
bash tools/run_gene_mrta_v15_mac.sh smoke
```

Single 100-generation run:

```bash
bash tools/run_gene_mrta_v15_mac.sh single
```

Formal five-seed run:

```bash
bash tools/run_gene_mrta_v15_mac.sh full
```

## Progression

```text
V1.2  Variable service time
V1.3  Priority
V1.4  Deadline
V1.5  Obstacles / A* path cost
V1.6  Battery
```

V1.5 is the first stage where Euclidean closeness is no longer equivalent to
navigation cost.


## V1.5 calibration note

The first obstacle smoke used 8 obstacles with side length 10-18 and produced
only a small detour effect (typically about 1-4 percent). To make the V1.5
research question meaningful without changing the Gene Bank logic, the
calibrated default is:

```text
obstacle count = 10
side length ~ Uniform(12, 20)
grid resolution = 5
path_cost_norm = clip(path_length / world_diagonal, 0, 1)
```

The calibration target is not a fixed score; it is to create a visible gap
between Euclidean-only and path-aware heuristics while keeping all generated
robot/task nodes connected.
