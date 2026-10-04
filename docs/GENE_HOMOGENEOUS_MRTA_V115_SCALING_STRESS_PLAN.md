# V1.15 Scalability and Limit Stress-Test Plan

Status as of 2026-10-04:

V1.15A EXTREME LADDER COMPLETED. FULL-SYSTEM LIMIT IS A* PATH PRECOMPUTATION BETWEEN 64R/320T AND 128R/640T.

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


## Implemented V1.15A benchmark

Files:

- src/marl2d/gene_mrta_v115/scaling.py
- tests/test_gene_mrta_v115_scaling.py
- tools/run_gene_mrta_v115_scaling_mac.sh

Frozen Policy selection:

The benchmark loads the completed V1.13 checkpoint and deterministically selects
one active currently four-capability Gene by maximizing minimum normalized
retention across the four V1.13 axes.

The selected Gene is converted to the V1.13 route-tail decoder and frozen for
all scale cases.

No scale-specific retraining occurs.

Dedicated scaling seed namespace starts at:

115000000

### Implemented scale modes

Tests:

    bash tools/run_gene_mrta_v115_scaling_mac.sh tests

Initial smoke:

    bash tools/run_gene_mrta_v115_scaling_mac.sh smoke

Smoke cases:

- 4R/20T
- 8R/40T
- 16R/80T

one world each.

Primary extreme ladder after smoke:

    bash tools/run_gene_mrta_v115_scaling_mac.sh ladder3

Cases:

- 4R/20T
- 8R/40T
- 16R/80T
- 32R/160T
- 64R/320T
- 128R/640T

three worlds each, stopping after a scale has zero successful worlds.

Task-dense smoke:

    bash tools/run_gene_mrta_v115_scaling_mac.sh dense-smoke

Fixed-map crowding diagnostic:

    bash tools/run_gene_mrta_v115_scaling_mac.sh fixed-map-smoke

### Failure separation

Failures are explicitly labeled:

- geometry
- path
- policy

A path/A* timeout therefore cannot be misreported as a Policy failure.

### Outputs

Each run writes:

- scaling_results.csv
- scaling_case_summary.csv
- timing_breakdown.csv
- memory_breakdown.csv
- scaling_summary.json

### Scale-normalization diagnostic

The current Policy's original distance features normalize by the map diagonal.

When spatial density is preserved by increasing map size, local physical
nearest-neighbor distances may remain comparable while normalized distances
shrink because the global map diagonal grows.

V1.15A deliberately does not change this observation definition.

Every world records:

- mean nearest physical robot-task distance;
- median nearest physical robot-task distance;
- mean nearest robot-task distance divided by the current map diagonal.

Therefore a future quality collapse can be separated into:

- computational/path bottleneck;
- Policy compute bottleneck;
- observation-scale distribution shift.

No result is claimed until the tests and smoke are actually run.


## Completed V1.15A initial smoke

Date: 2026-10-04

Tests:

    4 passed

Frozen Policy:

    7ee7c18fca2280022ac5

Parameter count:

    148

Smoke run:

    runs/gene_mrta_v115_scaling/gene_mrta_v115_scaling_20261004_121633_seed115000000

### 4R / 20T

- world size = 100
- obstacles = 10
- path preprocessing = 0.031005708 s
- Policy planning = 0.002770000 s
- decoder steps = 7
- pair slots scored = 476
- completion = 0.35
- raw time utility = 0.183477256
- continuation = 0.828880100
- reserve = 0.862087259
- mean queue depth = 1.75
- max queue depth = 2
- mean nearest physical robot-task distance = 37.1913
- normalized nearest distance = 0.262982
- RSS peak = 70.95 MB

### 8R / 40T

- world size = 141.421356
- obstacles = 20
- path preprocessing = 0.177800708 s
- Policy planning = 0.016564500 s
- decoder steps = 21
- pair slots scored = 5040
- completion = 0.525
- raw time utility = 0.271732244
- continuation = 0.905777154
- reserve = 0.946279206
- mean queue depth = 2.625
- max queue depth = 4
- mean nearest physical robot-task distance = 28.1493
- normalized nearest distance = 0.140746
- RSS peak = 71.02 MB

### 16R / 80T

- world size = 200
- obstacles = 40
- path preprocessing = 1.278738291 s
- Policy planning = 0.044244333 s
- decoder steps = 30
- pair slots scored = 31440
- completion = 0.375
- raw time utility = 0.198862617
- continuation = 0.905288690
- reserve = 0.945038368
- mean queue depth = 1.875
- max queue depth = 3
- mean nearest physical robot-task distance = 28.3613
- normalized nearest distance = 0.100272
- RSS peak = 71.30 MB

### Smoke interpretation

Structural result:

PASS.

The same frozen 148-parameter Policy Gene runs zero-shot at all three tested
fleet/task sizes.

The dominant early computational cost is path/A* preprocessing, not the Policy
decoder.

Observed path preprocessing:

0.0310 s -> 0.1778 s -> 1.2787 s

Observed Policy planning:

0.00277 s -> 0.01656 s -> 0.04424 s

At 16R/80T, path preprocessing is already about 29 times the Policy planning
latency for this world.

Path-table storage itself remains small:

0.00366 MB -> 0.01465 MB -> 0.05859 MB.

Therefore the initial systems bottleneck is A* computation time rather than path
table memory.

The normalization diagnostic also behaves as expected:

physical nearest distance is similar for 8R/40T and 16R/80T
(about 28.15 vs 28.36), while normalized nearest distance falls
(about 0.1407 -> 0.1003) because the global map diagonal grows.

This confirms an observation-distribution shift exists under map scaling and
must be kept visible when interpreting larger zero-shot cases.

Behavioral quality must NOT be inferred from this one-world-per-scale smoke.
The 4R/20T, 8R/40T and 16R/80T completion values are single-world observations,
not scale-level means.

Next frozen action:

run the three-world extreme ladder with the same Policy and no architecture
changes.

99M remains untouched.


## Completed V1.15A extreme ladder

Date: 2026-10-04

Command:

    bash tools/run_gene_mrta_v115_scaling_mac.sh ladder3

Tests before run:

    4 passed

Frozen Policy Gene:

    7ee7c18fca2280022ac5

Parameter count:

    148

Run:

    runs/gene_mrta_v115_scaling/gene_mrta_v115_scaling_20261004_121902_seed115010000

Each successful scale used three worlds.

### Mean results by scale

4R / 20T:
- path preprocessing = 0.025383 s
- Policy = 0.003340 s
- completion = 0.400000
- raw time utility = 0.192855
- balance = 0.390321
- continuation = 0.817841
- reserve = 0.843393
- mean queue depth = 2.000
- RSS peak = 70.28 MB

8R / 40T:
- path preprocessing = 0.190411 s
- Policy = 0.012502 s
- completion = 0.408333
- raw time utility = 0.214036
- balance = 0.385639
- continuation = 0.891299
- reserve = 0.904090
- mean queue depth = 2.0417
- RSS peak = 70.48 MB

16R / 80T:
- path preprocessing = 1.125755 s
- Policy = 0.048175 s
- completion = 0.395833
- raw time utility = 0.209933
- balance = 0.361391
- continuation = 0.931922
- reserve = 0.942528
- mean queue depth = 1.9792
- RSS peak = 71.99 MB

32R / 160T:
- path preprocessing = 7.546574 s
- Policy = 0.247902 s
- completion = 0.410417
- raw time utility = 0.216213
- balance = 0.387233
- continuation = 0.954358
- reserve = 0.965232
- mean queue depth = 2.0521
- RSS peak = 84.97 MB

64R / 320T:
- path preprocessing = 55.205867 s
- Policy = 1.246522 s
- completion = 0.400000
- raw time utility = 0.212930
- balance = 0.369836
- continuation = 0.975437
- reserve = 0.980648
- mean queue depth = 2.000
- RSS peak = 130.38 MB

128R / 640T:
- 0 / 3 worlds completed;
- all three failed in failure_stage = path;
- each hit the 300 s path-precomputation timeout;
- Policy planning was never reached.

### Primary V1.15A conclusion

The full-system practical limit in the tested implementation is not the
148-parameter Policy Gene.

The first hard limit is obstacle-aware A* path precomputation.

64R / 320T succeeds on 3/3 worlds.
128R / 640T fails on 3/3 worlds before Policy inference begins.

At 64R / 320T:

- A* preprocessing ~55.21 s;
- Policy allocation ~1.25 s;
- path preprocessing is about 44 times slower than Policy allocation;
- path-table storage is only 0.9375 MB.

Therefore the observed bottleneck is CPU/path-search time, not path-table
storage capacity.

### Empirical scaling trend

Across 20, 40, 80, 160, and 320 tasks while robot count scales proportionally:

- path-precomputation time is approximately proportional to T^2.75 over this
  measured range;
- Policy planning time is approximately proportional to T^2.14 over this
  measured range.

These are descriptive empirical fits, not theoretical complexity proofs.

The observed path-time doubling ratios are roughly:

7.50, 5.91, 6.70, 7.32.

The observed Policy-time doubling ratios are roughly:

3.74, 3.85, 5.15, 5.03.

The measured 128R / 640T A* timeout is therefore consistent with the prior
scaling curve.

### Behavioral zero-shot result

From 4R/20T through 64R/320T, the frozen Gene's three-world mean completion is:

0.4000, 0.4083, 0.3958, 0.4104, 0.4000.

Raw time utility is:

0.1929, 0.2140, 0.2099, 0.2162, 0.2129.

Mean queue depth remains approximately two tasks per robot:

2.0000, 2.0417, 1.9792, 2.0521, 2.0000.

This is strong structural evidence that the fixed-size Policy does not
immediately lose allocation behavior as fleet/task count grows to 64R/320T.

Because only three worlds were used per scale, this is still a scaling
diagnostic rather than a publication-grade behavioral generalization claim.

Continuation and reserve rise toward 1 as the problem size grows. Those
metrics include quantities normalized by the number of remaining tasks/options
and are not assumed scale-invariant. Their absolute cross-scale increase must
not be interpreted as proof that large-scale behavior is inherently better.

### Observation scaling diagnostic

Mean physical nearest robot-task distance remains around 25-30 for the larger
cases, while diagonal-normalized distance falls strongly:

- 4R/20T: 0.20994
- 8R/40T: 0.14497
- 16R/80T: 0.09578
- 32R/160T: 0.06484
- 64R/320T: 0.04529

Thus the current Policy receives a significant feature-distribution shift as
map size grows, yet completion and raw time utility remain approximately
stable through 64R/320T.

### Next frozen action

V1.15A has identified the full-system bottleneck.

Do not increase the A* timeout merely to claim a larger full-system scale.

Proceed to V1.15B Policy-Only Scaling using Euclidean/precomputed synthetic path
tables so that the same route-tail decoder can be tested beyond 128R/640T
without obstacle-A* preprocessing.

V1.15B must remain clearly labeled as Policy-only and must not replace the
V1.15A full-system result.

99M remains untouched.


## Implemented V1.15B Policy-only scaling

Status:

TESTS PASSED (7/7) AND POLICY-ONLY SMOKE PASSED THROUGH 128R/640T.

Purpose:

V1.15A found that the full-system limit occurs in obstacle-aware A* path
precomputation before the Policy decoder can be evaluated at 128R/640T.

V1.15B therefore removes obstacle-A* preprocessing while preserving:

- the same frozen V1.13 Policy Gene;
- the same 148 Policy parameters;
- the same route-tail autoregressive decoder;
- the same robot/task feature semantics where possible;
- the same world-size scaling;
- no scale-specific retraining.

The only path-model substitution is:

dense vectorized Euclidean robot/task and task/task distance table.

This makes V1.15B a Policy-only scalability benchmark, not a replacement for
the V1.15A full-system result.

Files:

- src/marl2d/gene_mrta_v115/policy_only.py
- tests/test_gene_mrta_v115b_policy_only.py
- tools/run_gene_mrta_v115b_policy_only_mac.sh

### V1.15B execution order

Tests:

    bash tools/run_gene_mrta_v115b_policy_only_mac.sh tests

Smoke:

    bash tools/run_gene_mrta_v115b_policy_only_mac.sh smoke

Smoke cases:

- 64R/320T
- 128R/640T

one world each.

If smoke passes:

    bash tools/run_gene_mrta_v115b_policy_only_mac.sh ladder3

Policy-only ladder:

- 64R/320T
- 128R/640T
- 256R/1280T
- 512R/2560T

three worlds each.

Only if 512R/2560T remains tractable:

    bash tools/run_gene_mrta_v115b_policy_only_mac.sh extreme1

Extreme diagnostic:

- 1024R/5120T
- one world
- Policy timeout 600 s

### V1.15B measurements

For each case:

- Euclidean table construction time;
- path-table entries and memory;
- Policy planning time;
- decoder steps;
- pair slots scored;
- Policy time per decoder step;
- Policy time per pair slot;
- assignment/completion;
- raw time utility;
- balance;
- queue depth;
- process RSS peak.

The summary also fits empirical timing exponents against task count.

### Memory implementation note

The Euclidean table uses the identity:

||x-y||^2 = ||x||^2 + ||y||^2 - 2 x dot y

instead of constructing a full (nodes x tasks x 2) displacement tensor.

This reduces temporary memory and makes the extreme Policy-only cases more
meaningful.

### Interpretation guardrail

Behavioral scores from V1.15B cannot be compared directly with V1.15A because
V1.15B removes obstacle detours.

V1.15B answers:

How far can the fixed 148-parameter route-tail Policy decoder scale if the
routing-preprocessing bottleneck is removed?

It does not answer:

How large can the full obstacle-aware system scale?

That answer remains:

64R/320T succeeds; 128R/640T is blocked by A* preprocessing under the frozen
300 s timeout.

99M remains untouched.


## Completed V1.15B smoke

Date: 2026-10-04

Tests:

    7 passed

Run:

    runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_141132_seed115100000

Frozen Policy:

    7ee7c18fca2280022ac5

Parameter count:

    148

### 64R / 320T

- Euclidean table build = 0.002422 s
- path table entries = 122,880
- path table = 0.9375 MB
- Policy = 1.970497 s
- decoder steps = 134
- pair slots scored = 2,174,016
- Policy / decoder step = 0.014705 s
- Policy / pair slot = 9.0639e-7 s
- completion = 0.41875
- raw time utility = 0.215997
- balance = 0.387412
- mean queue depth = 2.09375
- RSS peak = 130.97 MB

### 128R / 640T

- Euclidean table build = 0.001655 s
- path table entries = 491,520
- path table = 3.75 MB
- Policy = 14.836715 s
- decoder steps = 255
- pair slots scored = 16,744,320
- Policy / decoder step = 0.058183 s
- Policy / pair slot = 8.8607e-7 s
- completion = 0.3984375
- raw time utility = 0.209678
- balance = 0.367342
- mean queue depth = 1.99219
- RSS peak = 209.28 MB

### Primary smoke conclusion

V1.15B confirms that 128R/640T is not beyond the fixed Policy's current
computational capability.

With A* removed, the exact same 148-parameter route-tail Policy successfully
plans 255 task assignments at 128R/640T in about 14.84 s.

The V1.15A 128R/640T failure is therefore attributable to obstacle-aware path
precomputation under the frozen 300 s timeout, not to Policy inference.

Behavior also does not collapse in this one-world diagnostic:

- completion changes from 0.41875 at 64R/320T to 0.39844 at 128R/640T;
- raw time utility changes from 0.21600 to 0.20968;
- mean queue depth stays near 2.

These are single-world observations and are not formal quality estimates.

### Decoder scaling interpretation

When R and T both double:

- initial pair tensor size grows 4x;
- decoder steps grow from 134 to 255 (~1.90x);
- total pair slots scored grow from 2.174M to 16.744M (~7.70x);
- Policy time grows from 1.9705 s to 14.8367 s (~7.53x);
- Policy time per pair slot remains nearly constant (~0.9 microseconds).

This strongly indicates that the current implementation cost is dominated by
the number of pair slots repeatedly rescored, rather than by increased neural
parameter cost.

With approximately constant tasks per robot and decoder steps proportional to
T, the current autoregressive implementation has an expected leading work term
on the order of:

R * T * decoder_steps

and therefore approximately:

O(T^3)

when R is proportional to T.

The two-point empirical exponent reported by the smoke is:

2.91254

which is consistent with this near-cubic interpretation, but is not itself a
theoretical proof.

### Staged next experiment

Extrapolating the measured 64->128 ratio predicts approximately:

- 256R/1280T: about 110-120 s Policy time;
- 512R/2560T: well above 300 s if the same scaling continues.

Therefore the next protocol is intentionally staged:

1. ladder3:
   64R/320T, 128R/640T, 256R/1280T, three worlds each;
2. only after that, extreme512:
   512R/2560T, one world, 300 s Policy timeout;
3. extreme1024 is attempted only if justified by the 512 result or after a
   separate decoder optimization study.

This prevents wasting three full timeout periods at a scale already predicted
to exceed the frozen compute budget.

99M remains untouched.
