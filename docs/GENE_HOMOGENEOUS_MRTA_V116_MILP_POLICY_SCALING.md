# V1.16 MILP vs Frozen Gene Policy Scaling Benchmark

Status as of 2026-10-04:

PLANNED / IMPLEMENTATION IN PROGRESS.

## Objective

Directly compare the current frozen 148-parameter homogeneous route-tail Gene
Policy against the existing exact MILP reference on the same MRTA worlds.

The benchmark asks two questions simultaneously:

1. How far is the Policy solution from the proven global optimum?
2. How do Policy and MILP computation times scale as robot/task count grows?

The intended outcome is a joint quality/runtime frontier, not another Policy
training experiment.

## Frozen Policy

Use the same mature four-capability V1.13 Policy selected throughout V1.15:

    7ee7c18fca2280022ac5

Parameter count:

    148

No retraining, fine-tuning, or scale-specific selection is allowed.

## Exact MILP

Reuse:

    src/marl2d/gene_mrta_v16t/global_optimal_core.py

Solver:

SciPy scipy.optimize.milp backed by HiGHS.

MILP objective:

    T = (1/N) * sum_completed (1 - finish_time/H)

This is exactly the existing external time-optimality capability reference.

Oracle rule remains:

Oracle does not teach the action; it defines the capability ceiling.

## Fair comparison protocol

For every seed and R/T scale:

1. generate one obstacle-aware world;
2. build one shared A* path table;
3. give the identical World object to both methods;
4. time the frozen Gene route-tail decoder;
5. time MILP formulation + HiGHS solve;
6. compare objective values only when the MILP proof status is interpreted
   correctly.

Shared geometry/A* preprocessing is recorded separately.

Primary method time excludes shared preprocessing:

- Policy solve time;
- MILP formulation + solve total time.

Also report end-to-end time by adding the same shared preprocessing to each.

## Optimality-gap rule

If HiGHS proves optimality:

    exact_retention = T_policy / T_star

    exact_relative_gap = (T_star - T_policy) / T_star

    exact_absolute_gap = T_star - T_policy

Only these rows are called exact optimality-gap comparisons.

If HiGHS does not prove optimality within the time budget:

- do NOT call the MILP incumbent the optimum;
- record the incumbent score if one exists;
- record the MILP dual upper bound if available;
- report a guaranteed Policy-retention lower bound:

    T_policy / T_upper_bound

when the upper bound is finite and positive.

Thus MILP timeout itself remains scientifically useful instead of being
discarded or mislabelled.

## Runtime measurements

Per world record:

- geometry generation seconds;
- A* path preprocessing seconds;
- Policy solve seconds;
- MILP total call seconds;
- MILP internal solver seconds;
- estimated MILP formulation overhead;
- Policy end-to-end seconds;
- MILP end-to-end seconds;
- MILP / Policy speedup ratio.

## MILP problem-size diagnostics

For R robots and N tasks, the existing formulation contains approximately:

Binary variables:

    R*N*(N+1)

Continuous finish-time variables:

    R*N

Total variables:

    R*N*(N+2)

With all paths finite, constraints are approximately:

    R*N*(N+3) + 2R + N

At fixed N=5R, formulation size already grows approximately cubically with
fleet scale before branch-and-bound search is considered.

The benchmark records these counts per scale.

## Initial scaling ladder

Keep five tasks per robot and preserve spatial density as in V1.15.

Initial smoke:

- 2R / 10T
- 3R / 15T
- 4R / 20T

one world each.

Formal/staged ladder after smoke:

- 2R / 10T
- 3R / 15T
- 4R / 20T
- 5R / 25T
- 6R / 30T
- 8R / 40T

Use multiple worlds per scale.

The ladder should stop after a scale has zero proven-optimal MILP worlds under
the frozen solve-time budget, because that already identifies the exact-MILP
practical bottleneck.

## Time budgets

Smoke MILP budget:

    60 s / world

Main ladder MILP budget:

    300 s / world

The 300 s budget matches the frozen compute budget used in the V1.15 scaling
study and the earlier exact-oracle work.

Do not silently increase the time budget after seeing results.

## Primary aggregate outputs

For every scale report:

- MILP optimal proof rate;
- mean/median Policy runtime;
- mean/median MILP runtime;
- runtime speedup MILP/Policy;
- mean exact retention on proven-optimal worlds;
- mean exact relative gap;
- minimum exact retention;
- Policy completion and raw T;
- MILP incumbent/upper-bound diagnostics on non-proven worlds;
- problem-size counts.

## Interpretation categories

A. MILP exact and Policy near-optimal:
Policy preserves quality while solving much faster.

B. MILP exact but Policy gap grows:
Policy runtime advantage remains, but scale generalization quality degrades.

C. MILP cannot prove optimality while Policy remains fast:
exact optimization is the bottleneck; Policy becomes useful for large online
allocation even when exact quality cannot be measured.

D. Both become slow:
identify the crossover scale and distinguish MILP combinatorial growth from
Policy autoregressive pair-rescoring growth.

## Data policy

Use a new dedicated seed namespace beginning at:

    116000000

Do not use 95M, 98M, or protected 99M for this scaling benchmark.

99M remains untouched.

## Documentation rule

Every completed run must be persisted in:

- AI_PROJECT_CONTEXT.md
- docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md
- this document
- README current status

before changing the next benchmark stage.
