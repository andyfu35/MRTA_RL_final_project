# V1.15 Scalability and Limit Stress-Test Plan

Status as of 2026-10-04:

PLANNED. IMPLEMENTATION MUST START AFTER V1.14.3 OFFLINE ANALYSIS IS RECORDED.

## Objective

Test the practical and algorithmic limits of the current homogeneous
route-tail Gene policy when both robot count and task count become much larger.

This is not primarily a mating experiment.

The core question is:

Can one fixed 148-parameter homogeneous Policy Gene allocate and sequence large
numbers of tasks for increasingly large robot fleets without changing network
size?

## Why this is important

The current Policy has fixed parameter count:

148 parameters

but its input at each decision is an R x T pair tensor.

Therefore Policy parameter count does not grow with fleet size.

However computational work does grow with R and T, and the task-to-task path
table may grow quadratically with T.

The stress test must separate:

1. Policy / autoregressive assignment scaling;
2. environment and A* path-precomputation scaling;
3. solution-quality scaling.

Otherwise a path-planner bottleneck could be mistaken for a Gene-policy limit.

## Frozen Policy for the first scaling stage

The first scaling experiment is zero-shot.

Do not retrain separately for every fleet size.

Select one mature four-capability Policy Gene deterministically from the frozen
V1.13+ bank by maximizing its minimum normalized retention across the four
Policy capability axes.

Freeze that Gene for the entire zero-shot scaling suite.

This asks whether the architecture itself is size-general.

Only after zero-shot failure patterns are measured may scale-specific
retraining be considered.

## Scaling ladder

Baseline robot/task ratio is:

4 robots / 20 tasks = 5 tasks per robot.

Primary constant-density ladder:

- 4 robots / 20 tasks
- 8 robots / 40 tasks
- 16 robots / 80 tasks
- 32 robots / 160 tasks
- 64 robots / 320 tasks
- 128 robots / 640 tasks, only if the previous stage remains tractable

Task-dense stress cases after the primary ladder:

- 8 robots / 80 tasks
- 16 robots / 160 tasks
- 32 robots / 320 tasks
- 64 robots / 640 tasks, if feasible

Do not jump directly to the largest case.
Each stage records the first computational or behavioral failure.

## Spatial-density scaling

Keeping a 100 x 100 map while multiplying robots/tasks would artificially
increase spatial density.

For the constant 5-task-per-robot ladder, scale map side approximately as:

L(R) = 100 * sqrt(R / 4)

This keeps area proportional to fleet size.

To preserve obstacle density, obstacle count should also scale approximately
with area/fleet scale while keeping obstacle-side distribution and clearance
unchanged.

The first implementation must record the exact scaling formula.

## Two benchmark layers

### V1.15A Full-System Scaling

Uses the real obstacle-aware environment and path precomputation.

Measure separately:

- world generation time;
- A* / path-table precomputation time;
- path-table memory;
- Policy allocation time;
- total planning time;
- peak resident memory;
- assigned task count;
- completion;
- raw time utility;
- continuation;
- fleet reserve;
- queue-depth diagnostics.

This reveals the actual system limit.

### V1.15B Policy-Only Scaling

If full-system scaling fails because path precomputation dominates, run an
isolated Policy-decoder benchmark using already-available/synthetic pair
features or simplified distance tables.

This isolates the scaling law of the 148-parameter route-tail decoder itself.

Do not silently replace V1.15A with V1.15B.
Report both bottlenecks separately.

## Computational quantities to record

For every R/T case:

- R
- T
- R*T initial pair count
- maximum decoder steps
- actual decoder steps
- parameter count
- policy planning latency
- latency per decoder step
- latency per evaluated pair
- world/path preprocessing latency
- peak memory / RSS if available
- path table size
- assigned task count
- unassigned task count
- mean queue depth
- max queue depth

Expected decoder work is superlinear in task count because an autoregressive
decision recomputes pair context repeatedly.

The benchmark should measure the empirical exponent rather than assume one.

## Quality metrics at large scale

Exact MILP T* is not expected to remain practical at large R/T.

Therefore do not fabricate an oracle-normalized score where no exact oracle is
available.

For all scales report raw external metrics:

- completion;
- raw time utility T;
- continuation preservation;
- fleet option reserve;
- workload fairness / balance if useful;
- feasibility / assignment coverage.

For small sizes where exact MILP remains feasible, keep T/T* as an anchor.

For larger sizes use fixed heuristic/reference baselines only if explicitly
defined and never call them exact optimum.

## Failure definition

A scale is considered computationally failed if any fixed benchmark limit is
hit, for example:

- out-of-memory;
- path precomputation cannot finish under the chosen timeout;
- Policy allocation exceeds the chosen planning-time budget.

A scale is behaviorally failed if quality collapses even though computation
still finishes.

The benchmark must distinguish computational failure from behavioral failure.

## Repetitions

Use multiple world seeds per R/T case.

The first smoke may use 3 worlds per scale.

Formal scaling should use at least 20 worlds per scale where runtime permits.

Do not tune the Policy on these formal scaling worlds before reporting
zero-shot results.

## Primary deliverables

1. scaling_results.csv
2. timing_breakdown.csv
3. memory_breakdown.csv
4. per-world JSONL
5. scaling_summary.json
6. plots after results exist:
   - planning latency vs R*T;
   - path-precompute latency vs T;
   - memory vs T;
   - completion vs scale;
   - time utility vs scale;
   - continuation/reserve vs scale.

## Scientific interpretation

Possible outcomes include:

1. quality stays stable but computation explodes:
   architecture generalizes behaviorally; systems optimization is the limit.

2. computation stays manageable but quality drops:
   fixed Policy lacks scale generalization.

3. both remain stable over a large range:
   strong evidence for size-general homogeneous Gene allocation.

4. path table fails first:
   MRTA policy is not yet the limiting component; routing infrastructure must
   be redesigned before claiming an algorithmic Policy limit.

## Protected benchmark policy

99M remains unrelated to this scaling stress suite and stays protected.

Scaling tests should use new dedicated seed namespaces and must be recorded in
AI_PROJECT_CONTEXT.md and the experiment ledger.
