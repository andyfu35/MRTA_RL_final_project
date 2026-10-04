# AI PROJECT CONTEXT — Gene Homogeneous MRTA / SEGB

Last updated: 2026-10-04
Repository: andyfu35/MRTA_RL_final_project
Active branch: experiment/gene-homogeneous-mrta-v1
Author: 傅獻德 (Hsien-Te Fu)

THIS FILE IS THE REQUIRED ENTRY POINT FOR ANY NEW AI CONVERSATION.

Before changing code or proposing the next experiment:

1. Read this file completely.
2. Read docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md.
3. Read the latest version-specific document.
4. Verify whether the latest planned experiment was actually executed.
5. Never infer test success from code existence alone.
6. Never inspect the protected 99M final benchmark before the current procedure and candidate are frozen.

---

# 1. What this research is doing

This branch studies a Gene-based homogeneous Multi-Robot Task Allocation system inspired by the Self-Evolving Gene Bank idea.

The central research direction is:

Policy Gene Bank
+
Recombination Gene Bank

Policy Genes learn/evolve how homogeneous robots allocate and sequence tasks.

Recombination Genes learn/evolve how Policy Genes should mate.

The target is not to discover one universal crossover formula for all domains.
The target is a self-improving system that can discover an effective mating rule for its current problem distribution.

Core principles:

- no weighted scalar capability reward;
- keep independent capability axes;
- preserve specialists/Pareto conflicts;
- one homogeneous Policy Gene shared by all robots;
- no global world model;
- MILP never teaches actions;
- exact oracle only defines capability ceilings.

Canonical oracle statement:

Oracle does not teach the action; it defines the capability ceiling.

---

# 2. Current fixed MRTA environment

- world 100 x 100
- 4 homogeneous robots
- 20 tasks
- robot speed 4
- service time Uniform(2,35)
- priority Uniform(0.1,1.0)
- deadline Uniform(25,50)
- horizon H = 50
- 10 static square obstacles
- obstacle side 12–20
- obstacle clearance 4
- deterministic 8-connected A*
- no diagonal corner cutting
- grid resolution 5
- battery capacity 70
- initial battery Uniform(35,70)
- energy/distance 1
- horizon + battery are hard feasibility constraints

Primary time metric:

T = (1/N) * sum_completed (1 - F_j/H)

Exact MILP reference:

T*

Later time retention is based on:

T_gene / T*

---

# 3. Current Policy architecture

Current Policy architecture is inherited from V1.8 and V1.13.

Observation dimension = 12.

Features:

1. Euclidean distance
2. obstacle-aware path distance
3. service time
4. priority
5. deadline remaining
6. battery remaining
7. workload
8. competition
9. self future reachability
10. self best future time utility
11. other-robot opportunity cost
12. residual battery

hidden_dim = 8
parameter count = 148

Policy is autoregressive Direct Assignment with learned STOP/WAIT.

V1.13 changed the semantics from one task per robot per planning round to route-tail multi-task append.

A robot may receive:

R_i -> [T_a, T_b, T_c, ...]

in one planning call.

After appending a task, the robot row remains active.
Only the selected task closes.

The robot virtual state is updated:

- tail node
- tail position
- tail time
- battery
- workload

The next candidate task is evaluated from the current route tail.

Current route planning is append-only.
Arbitrary middle insertion is not yet implemented.

---

# 4. Current Policy capability axes

Current V1.13+ Policy Gene Bank axes:

1. mean_time
2. tail10_time
3. continuation_preservation
4. fleet_option_reserve

No weighted scalar sum.

Continuation:

C_t = clip((U_t + O_after)/O_before, 0, 1)

Fleet reserve uses feasible robot-owner counts for remaining tasks.

Capability certification threshold:

0.95 of current active capability ceiling.

Mating admission requires both:

parent retention >= 0.95
and
generation capability ceiling retention >= 0.95

This prevents repeated 95%-of-parent degradation.

---

# 5. Key completed results

## V1.8 frozen 98M benchmark

100/100 MILPs proven optimal.

V1.8 mean T/T*:

96.93972395246712%

Other means:

- V1.7: 96.1490%
- V1.6-T-O: 95.4079%
- Hungarian: 94.1866%

V1.8 exact matches: 34/100.

This was the frozen V1.8 publication benchmark.

After V1.9 started, 98M became development/diagnostic data.

## V1.9

Bottom-10 analysis:

- continuation collapse 10/10
- fleet reserve risk 7/10
- hard for all 5/10
- V1.8 regression vs V1.7 3/10

Result:

specialist discovery worked, but capability fusion was the bottleneck.

## V1.10 mating Pilot50

Run:

runs/gene_mrta_v110_mating/gene_mrta_v110_mating_20261003_163423_seed7

First certified four-capability Gene appeared by Gen2.

Final Gen49:

- mean_time = 0.9779050005301373
- tail10 = 0.9143206459612623
- continuation = 0.7785539293423699
- reserve = 0.83879703612327
- certified capabilities = 4

This proved 148-parameter Policies could fuse all four capability axes.

## V1.11 law discovery smoke

Interpretable seven-coefficient parent-swap-symmetric recombination law was searched.

Smoke selected law_006 and beat three hand-designed baselines on n=2 held-out pairs in min dual retention.

This is only a proof of feasibility, not statistical evidence or a universal law claim.

## V1.13 route-tail Pilot50

Run:

runs/gene_mrta_v113_route_tail_evolution/gene_mrta_v113_route_tail_20261003_203157_seed7

Zero-shot structural smoke:

5/5 worlds had multi-task robots.

Total assigned tasks = 37.

Pilot50 final:

- mean_time = 0.977920253212843
- tail10 = 0.9262063481487696
- continuation = 0.8328325738169616
- reserve = 0.8757350433050963
- certified capabilities = 4
- multi-task behavior persisted
- qmean approximately 1.82
- qmax approximately 2.70

Tail10 improved about +4.00 percentage points from Gen0 while queue depth stayed approximately stable.

Interpretation:

Route-Tail Multi-Task Architecture = PASS on the 95M development distribution.

## V1.14 self-evolving Recombination Bank

Run:

runs/gene_mrta_v114_self_recombination/gene_mrta_v114_self_recombination_20261003_214207_seed7

Final Policy best:

- mean_time = 0.9799838029221177
- tail10 = 0.9262063481487696
- continuation = 0.8336073765542017
- reserve = 0.8760983169402837

Final Recombination Pareto size = 4.

However, the most-used rule remained the all-gates-off center law for almost the whole run:

Rterms = 0

Center rule by Gen49:

54 accepted / 295 generated.

Two confounds were identified:

1. neutral center-law clones could have different old Gene IDs even when the formula was identical;
2. small-sample rules could look like specialists due to lucky point estimates.

Therefore:

V1.14 system-level Recombination Bank = PASS.

Autonomous mating-law discovery = NOT YET PASS.

---

# 6. Current version: V1.14.1

STATUS:

COMPLETED SINGLE-SEED PAIRED SMOKE + PAIRED50 FOR seed=7.

Regression tests: 15 passed.

Paired-smoke structural invariants passed:
- adaptive Rcenter <= 1, observed exactly 1;
- center Rbank = 1;
- center Rcenter = 1.

Formal paired50 is complete.

V1.14.1 fixes V1.14 before any stronger claim.

Implemented fixes:

## Phenotype canonicalization

If a gate is off, its coefficient is forced to zero.

Bank identity is based on effective equation phenotype, not mutation sigma.

Equivalent formulas cannot occupy duplicate slots.

Center law can occupy at most one bank phenotype.

## Evidence-aware rule selection

Rule selection/specialist ranking uses a lower Beta-posterior quantile.

Default:

q = 0.10

Formal specialist minimum evidence:

generated >= 32

Low-n rules remain provisional.

## Exploration reserve

Formal adaptive bank:

- bank limit 32
- exploration slots 8
- specialist size 6 per axis
- Pareto limit 12
- 25% uniform rule exploration

## Separate RNG streams

policy_rng = seed

rule_rng = seed + 114100003

This is required for clean paired comparison.

## Paired control

Adaptive arm:

Policy mutation + adaptive Recombination Gene Bank.

Center arm:

Policy mutation + fixed center law only.

Both start from the same V1.13 checkpoint and use the same Policy evolution budget.

The comparison remains four independent Policy-axis differences:

adaptive - center for:

- mean_time
- tail10_time
- continuation_preservation
- fleet_option_reserve

No aggregate weighted score.

Current launcher:

tools/run_gene_mrta_v1141_paired_mac.sh

---

# 7. Completed V1.14.1 paired50 result

Adaptive run:

runs/gene_mrta_v1141_paired_seed7/adaptive/gene_mrta_v1141_recombination_20261003_231301_seed7

Center-only run:

runs/gene_mrta_v1141_paired_seed7/center/gene_mrta_v1141_recombination_20261004_013416_seed7

Comparison:

runs/gene_mrta_v1141_paired_seed7/comparison_50.json

Adaptive final:
- mean_time = 0.9797388961714149
- tail10_time = 0.9273987323265199
- continuation_preservation = 0.8334576354631884
- fleet_option_reserve = 0.8757350433050963

Center final:
- mean_time = 0.9795402913092378
- tail10_time = 0.9273987323265199
- continuation_preservation = 0.8345021458732124
- fleet_option_reserve = 0.8762071604639012

Adaptive minus center:
- mean_time = +0.0001986048621770431
- tail10_time = 0
- continuation_preservation = -0.0010445104100239577
- fleet_option_reserve = -0.0004721171588049078

Interpretation:
- adaptive has a tiny mean-time advantage;
- tail10 is tied;
- center is better on continuation and reserve;
- no consistent adaptive advantage is demonstrated;
- do not scalarize these four axes into one result.

Adaptive final Recombination Bank:
- Rcenter = 1;
- mature Pareto size = 1;
- all four mature specialists = canonical center phenotype c7723fa1e0127975e49e;
- center phenotype in adaptive arm: generated 884, screen_selected 189, accepted 189, four-capability accepted 183.

This shows V1.14 center dominance was not merely caused by neutral-clone duplication.

Remaining comparison caveat:
center mode generates four deterministic center children per parent pair, so its nominal 128-child budget can contain repeated Policy children, whereas adaptive can produce multiple distinct children from one pair. A publication-grade next control should equalize unique-child opportunity.

---



## V1.14.2 formal128 result

Run:

runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7

Design:

- 128 unique frozen parent pairs;
- same parents for adaptive and center;
- one child per pair per condition;
- full 100-world 95M evaluation;
- no mutation/bank feedback;
- mating rule is the treatment difference.

Counts:

- non-center adaptive pairs = 115/128
- exact identical adaptive/center children = 19/128

Adaptive - Center:

mean_time:
- delta +0.00064842
- W/T/L 47/38/43

tail10_time:
- delta +0.00104946
- W/T/L 40/54/34

continuation_preservation:
- delta +0.00029748
- W/T/L 62/29/37

fleet_option_reserve:
- delta -0.00006791
- W/T/L 51/31/46

Dominance:
- tie 29
- adaptive 19
- neither 67
- center 13

Dual gate:
- adaptive 98/128
- center 95/128

Four-capability:
- adaptive 90/128
- center 87/128

Exploratory sign-test p-values on non-tied pairs:
- mean 0.7520
- tail 0.5614
- continuation 0.01543
- reserve 0.6849

Continuation is nominally significant alone, but Bonferroni over four axes gives approximately 0.0617, so no corrected multi-axis superiority claim is justified yet.

Current interpretation:

Adaptive is not universally better than center.
It shows small positive mean/tail changes, a clearer continuation advantage, essentially neutral/slightly negative reserve change, and +3 dual-gate / +3 four-capability children.

The next step is NOT a new formula family.
Analyze pair_results.jsonl to identify which non-center rules and parent contexts create the continuation and four-capability gains.

# 8. Protected data policy

95M:

development data.

98M:

historical V1.8 final set, later intentionally reused for V1.9 diagnosis, therefore development/diagnostic for later versions.

99M:

CURRENTLY UNTOUCHED AND PROTECTED.

Do not inspect 99M during V1.14.1 development.

V1.14.2 seed=7 formal128 is complete. It is still 95M development evidence and is not sufficient to justify 99M or a larger formula grammar.

Next:
1. analyze V1.14.2 pair_results.jsonl by non-center rule ID and parent capability context;
2. determine whether continuation/four-capability gains concentrate in specific rules/contexts;
3. only if a context-conditioned signal is present, freeze that selection hypothesis;
4. then repeat paired seeds;
5. use independent 96M development-validation if required;
6. freeze procedure/candidate;
7. only then evaluate 99M.

---

# 9. Documentation protocol

This is now mandatory.

Before starting any new architecture-changing version, update:

1. AI_PROJECT_CONTEXT.md
2. docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md
3. the latest version-specific document

Every completed experiment must record:

- exact run directory;
- exact architecture;
- exact seed/data namespace;
- exact capability definitions;
- test result;
- generation schedule;
- final metrics;
- relevant intermediate milestones;
- known failures/confounds;
- interpretation limits;
- next frozen action.

Do not allow conversation-only results to become the sole source of project state.

---

# 10. Files to read for full history

Full ledger:

docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md

Current version:

docs/GENE_HOMOGENEOUS_MRTA_V1141_RECOMBINATION_CORRECTION.md

Important historical docs:

docs/GENE_HOMOGENEOUS_MRTA_V18_PUBLICATION_FREEZE.md
docs/GENE_HOMOGENEOUS_MRTA_V19_FAILURE_PROTOCOL.md
docs/GENE_HOMOGENEOUS_MRTA_V19_ROBUST_BANK.md
docs/GENE_HOMOGENEOUS_MRTA_V110_MATING.md
docs/GENE_HOMOGENEOUS_MRTA_V111_LAW_DISCOVERY.md
docs/GENE_HOMOGENEOUS_MRTA_V113_ROUTE_TAIL.md
docs/GENE_HOMOGENEOUS_MRTA_V114_SELF_EVOLVING_RECOMBINATION.md

If anything in a conversation conflicts with these files, inspect the actual repo/run outputs before changing the experiment.


# 11. Immediate next step: V1.14.3 offline rule/context analysis

STATUS:

COMPLETED. V1.14 RECOMBINATION SERIES FROZEN FOR NOW.

Purpose:

Do not train again yet. Reuse the completed V1.14.2 formal128 pair_results.jsonl to identify which non-center Recombination Gene IDs and parent capability contexts produced:

- continuation wins;
- dual-gate rescues;
- four-capability rescues;
- adaptive Pareto dominance;
- exact center-equivalent children.

Implementation:

- src/marl2d/gene_mrta_v1143/analyze.py
- tests/test_gene_mrta_v1143_rule_context_analysis.py
- tools/run_gene_mrta_v1143_analysis_mac.sh
- docs/GENE_HOMOGENEOUS_MRTA_V1143_RULE_CONTEXT_ANALYSIS.md

Command:

bash tools/run_gene_mrta_v1143_analysis_mac.sh

This is descriptive analysis of the already-used 95M development assay. It may define a future context-conditioned mating hypothesis but does not validate that hypothesis.

After V1.14.3 is interpreted and recorded, freeze the V1.14 mating research line for now.

# 12. Next major phase: V1.15 scalability / limit stress test

The user explicitly wants to increase robot count and task count after V1.14.3 and test the algorithm's practical limit.

Planned primary constant 5 tasks/robot ladder:

- 4R / 20T
- 8R / 40T
- 16R / 80T
- 32R / 160T
- 64R / 320T
- 128R / 640T only if previous stage remains tractable

Then task-dense cases:

- 8R / 80T
- 16R / 160T
- 32R / 320T
- 64R / 640T if feasible

First scaling stage is ZERO-SHOT with one frozen mature four-capability Policy Gene. Do not retrain per scale before measuring architecture generalization.

Measure separately:

1. world/path preprocessing cost;
2. Policy decoder/allocation cost;
3. memory;
4. behavior quality.

The full-system test must preserve spatial density by scaling map area with fleet size.

Large-scale cases should not fabricate exact MILP-normalized scores when exact MILP is infeasible. Report raw completion, raw time utility, continuation, reserve, queue diagnostics, and compute metrics.

If A* / task-to-task path-table preprocessing fails before Policy allocation, record that as an infrastructure bottleneck, not a Policy-Gene failure.

Full plan:

docs/GENE_HOMOGENEOUS_MRTA_V115_SCALING_STRESS_PLAN.md

99M remains protected.


## V1.14.3 completed result and freeze

Source pairs = 128; non-center pairs = 115.

Non-center results:
- mean_time +0.00072172, W/T/L 47/25/43
- tail10 +0.00116809, W/T/L 40/41/34
- continuation +0.00033111, W/T/L 62/16/37
- reserve -0.00007559, W/T/L 51/18/46
- four-capability rescue = 3
- four-capability loss = 0
- candidate rule/context groups = 8
- identical non-center children = 6

Leading descriptive rule:
6858fcfc2c9540c2f0fd
- n=18
- active terms=2
- continuation W/T/L=11/1/6
- continuation mean delta=+0.0004531036432003304
- four-cap rescue-minus-loss=+1

Conclusion:
non-center mating can be useful in specific parent contexts, especially for continuation and occasional four-capability rescue, but there is no validated globally superior non-center law.

Decision:
freeze V1.14 mating research now. Do not tune further on 95M.

Proceed directly to V1.15 fleet/task scalability stress testing.


## V1.15A implementation status

V1.14.3 is complete and V1.14 is frozen.

V1.15A full-system zero-shot scaling is now implemented.

Files:
- src/marl2d/gene_mrta_v115/scaling.py
- tests/test_gene_mrta_v115_scaling.py
- tools/run_gene_mrta_v115_scaling_mac.sh

First commands:
1. bash tools/run_gene_mrta_v115_scaling_mac.sh tests
2. bash tools/run_gene_mrta_v115_scaling_mac.sh smoke

Smoke cases:
4R/20T, 8R/40T, 16R/80T, one world each.

If smoke is structurally valid, next:
bash tools/run_gene_mrta_v115_scaling_mac.sh ladder3

The ladder continues through 32R/160T, 64R/320T and 128R/640T unless a scale has zero successful worlds.

The benchmark freezes one mature four-capability V1.13 Policy Gene; no scale-specific retraining is allowed in V1.15A.

Failure stages are separated into geometry, path, and policy.

Outputs include detailed timing/memory CSVs and raw quality metrics.

99M remains untouched.


## V1.15A initial smoke result

Run:
runs/gene_mrta_v115_scaling/gene_mrta_v115_scaling_20261004_121633_seed115000000

Frozen Gene:
7ee7c18fca2280022ac5

Parameters:
148

4R/20T:
- path 0.0310 s
- Policy 0.00277 s
- completion 0.35
- time utility 0.18348
- continuation 0.82888
- reserve 0.86209

8R/40T:
- path 0.1778 s
- Policy 0.01656 s
- completion 0.525
- time utility 0.27173
- continuation 0.90578
- reserve 0.94628

16R/80T:
- path 1.27874 s
- Policy 0.04424 s
- completion 0.375
- time utility 0.19886
- continuation 0.90529
- reserve 0.94504

Interpretation:
- smoke structural PASS;
- same frozen 148-parameter Gene runs zero-shot through 16R/80T;
- A* path preprocessing is already the dominant compute cost;
- path-table memory remains small;
- normalized distance shrinks with map size while physical nearest distance remains similar, confirming observation-scale distribution shift;
- one world per scale is not enough for behavioral scaling claims.

Next command:
bash tools/run_gene_mrta_v115_scaling_mac.sh ladder3


## V1.15A extreme ladder result

Run:
runs/gene_mrta_v115_scaling/gene_mrta_v115_scaling_20261004_121902_seed115010000

Frozen Gene:
7ee7c18fca2280022ac5

Parameters:
148

Successful zero-shot scales:
- 4R/20T: 3/3
- 8R/40T: 3/3
- 16R/80T: 3/3
- 32R/160T: 3/3
- 64R/320T: 3/3

128R/640T:
- 0/3 successful
- all three failure_stage=path
- all three hit 300 s A* path-precompute timeout
- Policy inference was never reached

Three-world mean path / Policy time:
- 4/20: 0.02538 s / 0.00334 s
- 8/40: 0.19041 s / 0.01250 s
- 16/80: 1.12575 s / 0.04817 s
- 32/160: 7.54657 s / 0.24790 s
- 64/320: 55.20587 s / 1.24652 s

Three-world mean completion:
- 4/20: 0.4000
- 8/40: 0.4083
- 16/80: 0.3958
- 32/160: 0.4104
- 64/320: 0.4000

Three-world mean raw time utility:
- 0.1929, 0.2140, 0.2099, 0.2162, 0.2129

Mean queue depth remains approximately 2 across all successful scales.

Empirical descriptive timing fits over 20..320 tasks:
- A* path preprocessing ~ T^2.75
- Policy planning ~ T^2.14

Interpretation:
the first observed full-system scaling limit is A* preprocessing, not the 148-parameter Policy.
At 64R/320T path preprocessing is ~44x slower than Policy planning.
Path-table memory is still below 1 MB at 64R/320T.

Continuation/reserve increase with scale and are not treated as scale-invariant absolute quality evidence.

Next:
implement/run V1.15B Policy-only scaling with Euclidean/precomputed path tables to test the route-tail decoder beyond 128R/640T.

99M remains untouched.


## V1.15B implementation status

V1.15A full-system extreme ladder is complete.

Full-system result:
- 64R/320T succeeds 3/3;
- 128R/640T fails 3/3 in A* path preprocessing after 300 s;
- Policy inference is not reached at 128R/640T.

V1.15B Policy-only scaling is now implemented.

It replaces obstacle-A* path preprocessing with a dense vectorized Euclidean
distance table while retaining the same frozen 148-parameter route-tail Policy.

Files:
- src/marl2d/gene_mrta_v115/policy_only.py
- tests/test_gene_mrta_v115b_policy_only.py
- tools/run_gene_mrta_v115b_policy_only_mac.sh

Execution:
1. bash tools/run_gene_mrta_v115b_policy_only_mac.sh tests
2. bash tools/run_gene_mrta_v115b_policy_only_mac.sh smoke
3. if valid, bash tools/run_gene_mrta_v115b_policy_only_mac.sh ladder3
4. only if 512R/2560T remains tractable, consider extreme1 (1024R/5120T).

V1.15B is compute-isolation only; its behavioral scores are not directly
comparable with obstacle-aware V1.15A.

99M remains untouched.
