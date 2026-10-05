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


## Global objective specialists

Stage A distinguishes raw metrics from Gene-Bank capability axes.

The two primary exact-ceiling specialists are:

### Global time optimality

For a Gene:

    T_gene = (1/N) * sum_completed(1 - F_j/H)

For the exact time MILP oracle:

    T_star = max feasible T

Capability:

    global_time_optimality = T_gene / T_star

Therefore:

    global_time_optimality = 1

means the Gene has reached the proven global optimum for the time objective on
that world.

This objective is not makespan. It jointly rewards completing more tasks and
finishing them earlier, avoiding the "do fewer tasks to reduce total time"
loophole.

### Global priority optimality

Raw priority satisfaction is:

    P_gene = sum_priority(completed tasks) / sum_priority(all tasks)

A second exact MILP oracle maximizes that same feasible priority objective:

    P_star = max feasible priority satisfaction

Capability:

    global_priority_optimality = P_gene / P_star

Thus the Gene Bank can preserve a time specialist and a priority specialist as
different valid solutions instead of scalarizing them.

### Revised Stage-A axes

The active Stage-A archives are now:

1. completion
2. global_time_optimality
3. path_efficiency
4. global_priority_optimality
5. deadline_satisfaction
6. workload_balance

Raw time_optimality and priority_satisfaction remain recorded diagnostics, but
their exact-ceiling ratios are the capability scores used for the two global
objective archives.

The final multi-capability Gene can later be selected/fused with explicit
retention gates, for example requiring both the global-time and global-priority
capabilities to remain within a chosen fraction of their specialist ceilings.


## Stage-A trainer implementation

Stage A is now executable.

The training line starts from a fully random 148-parameter Route-Tail
population. No V1.13, V1.10, or other previous Policy parameters are loaded.

Implemented training mechanics:

- six independent Stage-A specialist archives;
- random initial population;
- normal mutation children;
- parent sampling proportional to weakest retained capability quality squared;
- 5% uniform exploration by default;
- complementary-capability mating;
- stronger mating parent pressure using q^10 by default;
- screen-world evaluation before full evaluation;
- per-axis top normal candidates promoted to full evaluation;
- mating candidates ranked by retained parent capabilities;
- 95% parent-retention plus current-capability-ceiling gate for inherited
  multi-capability labels;
- active bank retains specialist archives plus certified mating hybrids;
- checkpoint is written after every generation.

Stage-A exact oracle bank contains, for every training world:

- T_star: proven global Time-Utility optimum;
- P_star: proven global Priority-Satisfaction optimum.

Smoke protocol:

    bash tools/run_gene_mrta_v117_mac.sh smoke

This runs:

1. all V1.17 unit tests;
2. 2R/10T x 2 dual-oracle worlds;
3. a 5-generation random Stage-A evolution.

The smoke is only for wiring and capability-separation validation. It is not a
scientific result.

After smoke validation, the initial formal protocol is:

    bash tools/run_gene_mrta_v117_mac.sh oracle-formal
    bash tools/run_gene_mrta_v117_mac.sh train-formal

Formal defaults currently use:

- 4R/20T;
- 32 exact dual-oracle worlds;
- random population 256;
- 50 generations;
- 16 Genes per specialist archive;
- 128 normal mutation children/generation;
- 32 mating pairs x 4 children;
- screen batch 8 worlds;
- certification threshold 0.95.

These defaults remain adjustable after the smoke results are inspected.


## Final Stage-A capability scale before formal training

Smoke #2 confirmed both specialist separation and real mating-based fusion.

To make capability scores comparable across worlds, all linear task objectives
now use exact per-world ceilings.

The final Stage-A axes are:

1. global_completion_optimality = C / C*
2. global_time_optimality = T / T*
3. global_path_efficiency = E / E*
4. global_priority_optimality = P / P*
5. global_deadline_optimality = D / D*
6. workload_balance = (C/C*) * Jain(workload)

Interpretation:

For axes 1-5, a score of 1.0 means the Policy reaches the proven global optimum
of that objective on the evaluated world.

Definitions:

    C = completed_tasks / N

    T = (1/N) * sum_completed(1 - finish/H)

    E = (1/N) * sum_completed(
            1 - clip(A*_incoming_distance / map_diagonal, 0, 1)
        )

    P = sum_priority(completed) / sum_priority(all)

    D = on_time_tasks / N

The exact oracle bank now solves five independent objective MILPs per world:

    C*, T*, E*, P*, D*

This allows the Gene Bank to retain genuinely distinct global specialists such
as fastest-time, highest-priority, shortest-path, or best-deadline Genes without
scalarizing those objectives.

Jain workload fairness remains the only structural axis without a MILP ceiling,
because the current Jain ratio is nonlinear and should not be replaced by a
different linear surrogate merely to force a common normalization scheme.


## Capability provenance semantics

Before formal training, capability provenance is explicitly separated.

A Gene may be strong on multiple axes for two different reasons:

1. it independently ranks inside multiple specialist archives;
2. a mating child explicitly passes the 95% dual inheritance gate for the
   union of parent capabilities.

These must not be conflated.

The trainer now stores:

    archive_capabilities
    inherited_capabilities
    capabilities

where:

    capabilities =
        archive_capabilities U inherited_capabilities

Archive overlap is valid evidence that a Gene performs strongly on several
objectives, but it is not evidence that mating transmitted those abilities.

A true capability-fusion result requires:

- origin == "mating";
- inherited_capabilities contains at least two axes;
- each inherited axis passed both:
  - retention versus the relevant best parent;
  - retention versus the current capability ceiling.

The generation log therefore reports:

    max_inherited_capabilities
    best_fusion_gene

and includes that Gene's parents, mating operator, inherited capability list,
archive memberships, and scores.

This provenance split replaces the earlier ambiguous interpretation of
max_capabilities.


## Stage-A freeze decision

The final provenance smoke validates the pre-registered Stage-A mechanism.

The capability set is frozen:

    global_completion_optimality = C/C*
    global_time_optimality       = T/T*
    global_path_efficiency       = E/E*
    global_priority_optimality   = P/P*
    global_deadline_optimality   = D/D*
    workload_balance             = (C/C*) * Jain(workload)

No hard-world-derived axis is active in Stage A.

The final smoke demonstrates:

- distinct Time and Priority specialists;
- exact-ceiling normalization for all five linear task objectives;
- genuine mating inheritance tracked separately from archive membership;
- a mating-derived child retaining three capabilities under the 95% dual gate.

The smoke does not require all six capabilities, or Time+Priority, to fuse
within five generations. Those are discovery outcomes for the formal
evolution, not smoke acceptance criteria.

### Formal run reliability

The exact oracle bank is world-level resumable.

If interrupted after some 4R/20T worlds finish, rerunning oracle-formal skips
completed seeds and resumes from the first unfinished world.

The Stage-A evolutionary run is generation-level resumable.

Every completed generation atomically checkpoints:

- active Gene records and parameters;
- archive capability provenance;
- inherited capability provenance;
- generation history;
- RNG state;
- training configuration.

The formal launcher uses a fixed run directory, so rerunning train-formal
continues the same run.

Status:

    bash tools/run_gene_mrta_v117_mac.sh status-formal

Formal sequence:

    bash tools/run_gene_mrta_v117_mac.sh oracle-formal
    bash tools/run_gene_mrta_v117_mac.sh train-formal

Recommended on macOS for long runs:

    caffeinate -dimsu bash tools/run_gene_mrta_v117_mac.sh oracle-formal

followed by:

    caffeinate -dimsu bash tools/run_gene_mrta_v117_mac.sh train-formal


## Post-Stage-A Pareto retention

After the frozen Stage-A experiment, Gene Bank retention changes from
axis-specific archive bookkeeping to a unified Pareto archive.

The capability measurements themselves do not change.

For Gene g:

    c(g) =
    [
        global_completion_optimality,
        global_time_optimality,
        global_path_efficiency,
        global_priority_optimality,
        global_deadline_optimality,
        workload_balance
    ]

The Bank retains the nondominated set in this six-dimensional space.

This means the system defines the capability measurements but does not define
which combinations should exist. Time specialists, Priority specialists,
balanced knees, and other trade-offs emerge as geometry of the Pareto front.

Single-axis best Genes and maximin Genes are analysis/deployment views only.
They are never admission rules.

This Pareto Bank is used for:

1. post-Stage-A mating;
2. unseen-world evaluation;
3. hard-world mining;
4. later Stage-B evolution;
5. final Fusion-2.
