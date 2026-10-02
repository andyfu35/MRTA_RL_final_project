# V1.9 Robust Gene Bank

## Design decision from Bottom-10 failure analysis

The V1.8 Bottom-10 development analysis produced:

- continuation_collapse: 10/10
- fleet_reserve_risk: 7/10
- hard_for_all: 5/10
- v18_regression_vs_v17: 3/10
- immediate_future_imbalance: 0/10

Therefore V1.9 does not add an immediate-time capability axis. The evidence
supports three new robustness capabilities.

The V1.8 policy architecture is unchanged:

- 12D pair observation
- hidden dimension 8
- 148 parameters
- autoregressive Direct Assignment
- learned STOP/WAIT
- no external assignment matcher
- no MILP action supervision

## Independent Gene Bank axes

V1.9 maintains four independent archives:

1. mean_time
2. hard_world_time
3. continuation_preservation
4. fleet_option_reserve

There is no weighted scalar sum across these axes.

Parents are sampled by first selecting an archive uniformly, then selecting a
Gene within that archive. This gives each capability equal reproductive
access regardless of its numerical score scale.

## Mean-time axis

The existing global-time capability:

[
A_{mean}(G)
=
rac{1}{|B|}
sum_{win B}
rac{T_G(w)}{T^*(w)}
]

where B is the scheduled random oracle-training batch.

## Hard-world time axis

The fixed Bottom-10 98M development worlds are:

- 98000042
- 98000000
- 98000086
- 98000003
- 98000037
- 98000020
- 98000062
- 98000025
- 98000034
- 98000078

The hard-world capability is:

[
A_{hard}(G)
=
rac{1}{10}
sum_{win H}
rac{T_G(w)}{T^*(w)}
]

This axis intentionally creates specialists for the observed lower-tail
structures. Generalization is not inferred from 98M; it will later be tested
on the untouched 99M range.

## Continuation-preservation axis

For each event t define normalized fleet option mass O as the sum, over
remaining tasks, of the best one-step future T utility available from any
robot.

Let U_t be the normalized immediate T utility collected by the event.

[
C_t
=
clip
left(
rac{U_t + O_{after}}{O_{before}},
0,
1
ight)
]

The episode capability is the mean C_t over events with positive pre-decision
option mass.

This axis rewards decisions that capture current T utility while preserving
future task-chain value.

## Fleet-option-reserve axis

For every remaining task j let d_j be the number of robots that can still
feasibly complete it under time and battery constraints.

Define task reserve contribution:

[
q_j = rac{min(d_j,2)}{2}
]

Thus:

- stranded task: 0
- singleton task: 0.5
- task with at least two feasible robots: 1

Normalized reserve mass Q is:

[
Q = rac{1}{N}sum_j q_j
]

For an event completing A_t assignments:

[
R_t
=
clip
left(
rac{A_t/N + Q_{after}}{Q_{before}},
0,
1
ight)
]

The episode fleet-option-reserve capability is the mean R_t over valid
events.

This rewards preserving cross-robot alternatives and avoiding unnecessary
creation of singleton or stranded tasks.

## Training

Long-run default:

- generations: 1000
- population: 256
- archive size: 16 per axis
- HOF size: 16 per axis
- oracle batch schedule: 16 -> 32 -> 64
- mutation sigma: 0.12 -> 0.015
- immigrants: 10%
- V1.8 final Gene is the bootstrap seed

The three mechanism axes are evaluated on both the current oracle-training
batch and the fixed hard-world set. The hard-time axis is evaluated directly
on the fixed Bottom-10 exact T* values.

## Experimental sequence

1. Smoke-test the V1.9 implementation.
2. Run the V1.9 long evolution.
3. Evaluate all four saved specialists on the complete 98M development set.
4. Before opening any 99M results, select and freeze the V1.9 candidate using
   the predefined development criteria.
5. Evaluate the frozen V1.9 candidate once on untouched
   99,000,000-99,000,099.

The 99M set is not used for architecture design, training, or candidate
selection.
