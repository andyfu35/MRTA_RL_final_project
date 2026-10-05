# V1.17 Clean Two-Stage Gene MRTA Training

## Goal

Train a new 148-parameter Route-Tail Gene using the lessons from V1-V1.16
without inheriting old Policy parameters or old capability labels.

The protocol is intentionally split into two scientific stages.

## Stage A: complete known task semantics

Stage A must include all task information and objective dimensions that are
known before looking at hard-world failures.

Canonical static task record:

```
Task_j = {
    task_id,
    position: [x, y],
    service_time,
    priority,
    deadline,
}
```

Canonical static robot record:

```
Robot_i = {
    robot_id,
    start_position: [x, y],
    initial_battery,
}
```

Global environment:

```
episode_horizon
robot_speed
battery_capacity
energy_per_distance
obstacles
grid_resolution
```

A* path tables are derived preprocessing, not Task fields.

Dynamic Route-Tail pair observations remain derived from the current virtual
state. Static Task records must not contain competition, future reachability,
opportunity cost, path distance, residual battery, or other state-dependent
features.

### Stage-A capability axes

The first-stage Gene Bank uses six independent base axes:

1. completion
2. time_retention
3. path_efficiency
4. priority_satisfaction
5. deadline_satisfaction
6. workload_balance

Definitions:

```
completion = completed_tasks / total_tasks

time_retention = T_gene / T_star

path_efficiency =
    sum_completed(1 - clip(A*_distance / map_diagonal, 0, 1))
    / total_tasks

priority_satisfaction =
    completed_priority / total_priority

deadline_satisfaction =
    on_time_tasks / total_tasks

workload_balance =
    completion * Jain(robot_workloads)
```

There is no weighted scalar reward.

Battery is intentionally not a capability axis. It is a hard feasibility
constraint because maximizing remaining battery directly has the trivial
"do no work" loophole. Under the current environment, travel energy is
proportional to A* path length, so path_efficiency already rewards lower
travel/energy use while completion prevents the no-work loophole.

Stage A is not allowed to use continuation-preservation, fleet reserve,
tail-10%, or any other hard-world-derived axis merely because previous versions
used them. Those mechanisms must be rediscovered from Stage-A failure analysis
if the new run exhibits the corresponding failure.

## Stage B: evidence-driven hard-world retraining

After Stage A is frozen:

1. evaluate on a large fixed development set;
2. rank worlds by exact time retention and the six Stage-A axes;
3. inspect the hard tail;
4. classify failure mechanisms using state/action traces;
5. add only capability axes supported by repeated failures;
6. retrain/fuse from the frozen Stage-A Gene Bank;
7. select a final multi-capability Gene;
8. evaluate once on protected final worlds.

Examples of possible Stage-B axes include continuation preservation or fleet
option reserve, but V1.17 does not pre-register them as active axes before
Stage-A failure evidence exists.

## Data-layer separation

The V1.17 implementation separates:

1. canonical static data: RobotSpec / TaskSpec / WorldSpec;
2. derived pair observation: Robot x Task x dynamic Route-Tail features;
3. capability evaluation: external scores used by the Gene Bank.

This separation prevents task records from silently embedding policy-dependent
or future-looking information.

## Current implementation

Implemented:

- `src/marl2d/gene_mrta_v117/schema.py`
- `src/marl2d/gene_mrta_v117/capabilities.py`
- schema/capability tests
- Mac test launcher

Next implementation step after this schema/axis freeze:

- clean random 148-parameter Stage-A population;
- six independent specialist archives;
- mutation + capability-fusion mating;
- exact-T* development bank;
- hard-world trace exporter for Stage B.
