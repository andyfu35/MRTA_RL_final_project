# V1.16 MILP vs Frozen Gene Policy Scaling Benchmark

Status as of 2026-10-04:

SMOKE AND THREE-WORLD LADDER COMPLETED. EXACT-MILP PRACTICAL WALL OBSERVED BETWEEN 5R/25T AND 6R/30T UNDER 300 S.

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


## Implemented files

- src/marl2d/gene_mrta_v116/milp_policy_scaling.py
- tests/test_gene_mrta_v116_milp_policy_scaling.py
- tools/run_gene_mrta_v116_milp_policy_mac.sh

The existing MILP result object was also extended to expose:

- mip_node_count
- time_optimality_upper_bound

so timeout worlds can retain useful proof-bound information.

## Launcher modes

Tests only:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh tests

Initial smoke:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh smoke

Smoke:
- 2R/10T
- 3R/15T
- 4R/20T
- one world each
- MILP limit 60 s/world

Staged 3-world ladder:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh ladder3

Cases:
- 2R/10T
- 3R/15T
- 4R/20T
- 5R/25T
- 6R/30T
- 8R/40T

MILP limit:
300 s/world.

The ladder stops after a scale has zero proven-optimal MILP worlds.

Small-scale 10-world comparison:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh small10

Cases:
2R/10T, 3R/15T, 4R/20T.

This mode is intended only after the smoke/ladder identify a reasonable exact
MILP range.

No result is claimed until these commands are actually executed.


## Completed smoke and ladder3 results

Date: 2026-10-04

Tests:

    5 passed

Smoke run:

    runs/gene_mrta_v116_milp_policy/gene_mrta_v116_milp_policy_20261004_201534_seed116000000

Ladder run:

    runs/gene_mrta_v116_milp_policy/gene_mrta_v116_milp_policy_20261004_201755_seed116010000

Frozen Gene:

    7ee7c18fca2280022ac5

Parameters:

    148

### Smoke

2R/10T:
- Policy = 0.001932 s
- MILP = 0.095153 s
- exact retention = 0.953380
- MILP/Policy time ratio = 49.24x

3R/15T:
- Policy = 0.003714 s
- MILP = 0.054417 s
- exact retention = 1.000000
- MILP/Policy time ratio = 14.65x

4R/20T:
- Policy = 0.007082 s
- MILP = 9.687818 s
- exact retention = 0.938533
- MILP/Policy time ratio = 1367.92x

Smoke structural result:

PASS.

### Three-world ladder aggregate

2R/10T:
- MILP proof rate = 3/3
- mean Policy time = 0.001377 s
- mean MILP time = 0.195471 s
- mean exact retention = 0.991087
- mean exact relative gap = 0.008913
- minimum exact retention = 0.973260
- mean per-world MILP/Policy ratio = 129.46x

3R/15T:
- MILP proof rate = 3/3
- mean Policy time = 0.002730 s
- mean MILP time = 7.901246 s
- mean exact retention = 0.946564
- mean exact relative gap = 0.053436
- minimum exact retention = 0.907421
- mean per-world MILP/Policy ratio = 2644.40x

4R/20T:
- MILP proof rate = 3/3
- mean Policy time = 0.003387 s
- mean MILP time = 20.814807 s
- mean exact retention = 0.977449
- mean exact relative gap = 0.022551
- minimum exact retention = 0.944180
- mean per-world MILP/Policy ratio = 5053.38x

5R/25T:
- MILP proof rate = 3/3
- mean Policy time = 0.004702 s
- mean MILP time = 41.553477 s
- median MILP time = 7.509629 s
- one world required 112.500683 s
- mean exact retention = 0.942129
- mean exact relative gap = 0.057871
- minimum exact retention = 0.930872
- mean per-world MILP/Policy ratio = 9060.85x

6R/30T:
- MILP proof rate = 0/3
- all three MILP runs reached the frozen 300 s time limit
- mean Policy time = 0.010127 s
- mean MILP call time = 300.050995 s
- mean per-world MILP/Policy ratio = 30,883.87x
- exact retention unavailable because optimality was not proved
- mean guaranteed Policy-retention lower bound from the MILP dual upper bound = 0.620293
- Policy / MILP incumbent ratios are approximately 0.9819, 1.0000, and 0.9815
- mean Policy / incumbent ratio is approximately 0.9878

The Policy-incumbent ratio is NOT an exact optimality metric because the MILP
incumbent is only the best feasible solution found before timeout.

### Primary result

Under the frozen 300 s exact-solve budget, the observed MILP practical
optimality-proof wall occurs between:

    5R/25T and 6R/30T

for the tested obstacle-aware task distribution.

The frozen 148-parameter Policy remains millisecond-scale at this boundary.

At 5R/25T the Policy retains about 94.2% of the proven optimum on average while
being orders of magnitude faster.

At 6R/30T exact optimality can no longer be established in any of the three
tested worlds, while the Policy still returns a solution in roughly 10 ms.

### Important runtime variability

MILP runtime is highly world-dependent.

Examples:

- 3R/15T ranges from about 0.47 s to 20.98 s in the three-world ladder;
- 4R/20T ranges from about 0.70 s to 59.51 s;
- 5R/25T ranges from about 4.65 s to 112.50 s.

Therefore fleet/task size alone does not determine MILP hardness.

Formal reporting must include proof rate and runtime distributions, not only
mean runtime.

### Next validation step

The current n=3 ladder is enough to identify the qualitative bottleneck but is
not a publication-grade estimate of the transition.

A new launcher mode is added:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh boundary5

It tests:

- 4R/20T
- 5R/25T
- 6R/30T

with five new worlds per scale, a frozen 300 s MILP limit, and new 116100000
seed namespace.

This will strengthen estimates of:

- exact retention before the MILP wall;
- MILP proof probability near the wall;
- runtime heavy-tail behavior;
- Policy quality relative to MILP incumbents after exact proof becomes
  impractical.

Do not proceed to 8R/40T exact MILP until this boundary replication is
interpreted.

99M remains untouched.


## Exact-unlimited follow-up

Decision after the 300 s ladder:

Do not treat the 300 s timeout as the end of the exact-reference experiment.

The 300 s ladder answers the practical-online question:

> At what scale does exact MILP stop being operationally useful under a fixed
> compute budget?

The new unlimited mode answers a different scientific question:

> What is the true global optimum T* for a hard world, even if proving it takes
> much longer than 300 s?

### Exact solver semantics

`solve_global_time_optimum` now accepts:

```python
time_limit=None
```

When it is `None`:

- the SciPy/HiGHS options dictionary contains no `time_limit` key;
- presolve stays enabled;
- `mip_rel_gap=0.0` stays unchanged;
- solver precision is not relaxed;
- only HiGHS status 0 with a solution is accepted as optimal;
- any other termination raises an error in unlimited mode;
- an incumbent is never used as T*.

### Progress during long runs

Long exact runs expose two progress mechanisms:

1. native HiGHS display can be enabled with `--milp-solver-display`;
2. the benchmark prints an independent line like:

```
MILP_ALIVE 6R/30T world=1 seed=116050000 elapsed=330.0s
```

every 30 s by default.

Therefore a silent solver does not look like a frozen process.

### Resume and durability

Every completed world is appended immediately to:

```
per_world.jsonl
```

and the file is flushed and fsynced.

The runner supports a fixed `--run-dir` and resume is enabled by default.

On restart:

- already completed finite-budget rows may be skipped;
- for exact-unlimited runs, a row is skipped only if
  `status=ok` AND `milp_optimal=true`;
- a previous timeout/non-optimal row is not considered an exact completed
  result.

Thus Ctrl+C or a reboot does not discard earlier completed worlds.

### Mac commands

Regression of already tractable exact sizes:

```bash
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-regression
```

First unlimited hard world:

```bash
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

Default:

- case = 6R/30T
- worlds = 1
- seed = 116050000
- fixed run dir =
  `runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000`

This seed corresponds to the first 6R/30T world in the existing ladder seed
mapping, so the experiment directly revisits a known 300 s timeout.

After one world proves optimal, more same-scale worlds can be requested without
changing code:

```bash
V116_EXACT_WORLDS=3 bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

The fixed run directory causes world 1 to be skipped after it is already
proven, and worlds 2-3 continue.

No 7R/35T or 8R/40T unlimited exact claim should be made before the first
6R/30T exact result is obtained and interpreted.


## Completed first exact-unlimited 6R/30T world

Command:

```bash
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

Tests:

9 passed.

Run:

```
runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000
```

World:

```
6R / 30T
seed = 116050000
```

HiGHS result:

```
Status = Optimal
Gap = 0%
T* = 0.28074964073648145
```

Frozen Policy:

```
T_policy = 0.2756685668779121
exact_absolute_gap = 0.005081073858569374
exact_relative_gap = 0.018098238149977172
exact_retention = 0.9819017618500229
```

Both MILP and Policy complete 14 tasks.

Runtime:

```
Policy = 0.008538166999642272 s
MILP solver = 717.154159042002 s
MILP total = 717.1704191659992 s
MILP / Policy = 83995.82945567204x
```

MILP search diagnostics:

```
nodes = 276137
LP iterations = 8274628
peak RSS = 290.953125 MB
```

### Most important interpretation

The final optimum value was already present as the best incumbent by roughly
138.5 s.

However, the solver did not prove that no better solution existed until about
717.1 s.

So on this world:

- solution discovery happened relatively early;
- exact proof dominated the remaining compute;
- the 300 s practical timeout was mainly a proof-certification failure, not a
  failure to find the eventual optimum.

This distinction is important for reporting.

The 300 s result still correctly demonstrates loss of real-time exactness.
The unlimited result establishes the true capability ceiling.

For this first hard exact world, the 148-parameter Policy is 98.19% of the
global optimum while using about 1/83,996 of the MILP method time.

Do not report 98.19% as the mean 6R/30T retention yet because n=1.

Next:

```bash
V116_EXACT_WORLDS=3 bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

Resume should skip the already-proven seed 116050000 and solve seeds 116050001
and 116050002.


## 6R/30T multi-seed exact worst-case study

After the first 6R/30T world reached a proven optimum, the next goal is to
measure distribution-level variability.

For each exact seed:

    relative_gap = (T* - T_policy) / T*
    retention = T_policy / T*

Primary worst-case quantities:

    max_exact_relative_gap
    min_exact_retention
    max_milp_total_seconds

The summary also records the seed responsible for each extreme.

The first formal batch is 10 seeds:

    116050000 through 116050009

Run:

    V116_EXACT_WORLDS=10 bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited

Seed 116050000 is already complete and should be skipped automatically by
resume.

The final report should include per-seed T*, T_policy, retention, relative gap,
Policy time, MILP time, and MILP/Policy ratio, plus mean/median/max runtime and
the maximum observed exact gap.

The phrase "worst case" must be qualified as "worst observed among the tested
seeds"; this is not a proof of the theoretical worst possible MRTA instance.


## Completed 10-seed 6R/30T exact-unlimited study

Run:

```
runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000
```

Seeds:

```
116050000 .. 116050009
```

All 10 MILP runs ended with proven global optimum.

### Quality summary

```
mean exact retention       = 0.9694540924627978
mean exact relative gap    = 0.030545907537202176
mean exact absolute gap    = 0.006215299617823469
minimum exact retention    = 0.9152010318146683
maximum exact relative gap = 0.08479896818533172
maximum exact absolute gap = 0.017600480877557306
worst-gap seed             = 116050009
```

Worst observed seed:

```
seed        = 116050009
T_policy    = 0.18995488511570233
T*          = 0.20755536599325963
retention   = 0.9152010318146683
relative gap= 0.08479896818533172
```

Thus the worst observed Policy error across the 10 tested 6R/30T worlds is
approximately 8.48%, corresponding to a minimum observed retention of 91.52%.

### Runtime summary

```
mean Policy time   = 0.007979583600172192 s
median Policy time = 0.00851533350032696 s
max Policy time    = 0.011506959002872463 s

mean MILP time     = 453.48357761249974 s
median MILP time   = 231.51708522899935 s
max MILP time      = 1734.6120088330026 s
slowest MILP seed  = 116050002
```

Maximum exact proof time:

```
1734.612 s = 28.910 minutes
```

Mean per-world MILP/Policy method-time ratio:

```
47884.56x
```

Five of the ten worlds exceed the previous 300 s practical exact-proof budget.

### Main conclusion

At fixed 6R/30T, MILP runtime varies by roughly three orders of magnitude
across worlds even though the variable/constraint counts are unchanged.

The frozen 148-parameter Gene remains millisecond-scale and achieves:

- mean exact retention about 96.95%;
- minimum observed retention 91.52%;
- maximum observed gap 8.48%;
- maximum Policy time below 12 ms.

The 8.48% number is the maximum observed exact gap over these 10 seeds, not a
mathematical worst-case guarantee.

### Objective nuance

At seed 116050005, the Policy completes 13 tasks and the globally T-optimal
MILP solution completes 12, but the Policy still has a lower T score.

This is expected because the oracle objective is time utility, not raw
completion count. More completed tasks do not necessarily imply a larger T if
they finish sufficiently late.
