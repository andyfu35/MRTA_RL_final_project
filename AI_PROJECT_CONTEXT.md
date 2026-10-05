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

V1.15B Policy-only scaling tests and smoke are complete.

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


## V1.15B smoke result

Tests: 7 passed.

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_141132_seed115100000

64R/320T:
- Policy 1.9705 s
- pair slots 2,174,016
- completion 0.41875
- time utility 0.215997
- queue mean 2.09375
- RSS 130.97 MB

128R/640T:
- Policy 14.8367 s
- pair slots 16,744,320
- completion 0.39844
- time utility 0.209678
- queue mean 1.99219
- RSS 209.28 MB

Both succeeded with the same frozen 148-parameter Gene.

This confirms the V1.15A 128R/640T failure was A* preprocessing, not Policy inference.

64->128:
- pair slots x7.70
- Policy time x7.53
- Policy time per pair slot remains approximately constant.

The measured two-point Policy exponent is 2.91254.
With R proportional to T and decoder steps approximately proportional to T,
the current autoregressive implementation is expected to approach O(T^3)
pair-scoring work.

Next:
bash tools/run_gene_mrta_v115b_policy_only_mac.sh ladder3

The updated ladder3 stops at 256R/1280T with 3 worlds.
If successful, run:
bash tools/run_gene_mrta_v115b_policy_only_mac.sh extreme512

512R/2560T is now a one-world extreme test because current scaling predicts it
may exceed the frozen 300 s Policy timeout.

99M remains untouched.


## V1.15B extreme512 result

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_175225_seed115120000

512R/2560T:
- 0/1 successful
- failure_stage=policy
- Policy timeout at 300.0555 s
- Euclidean table build 0.02927 s
- path table 60.0 MB
- RSS peak 961.14 MB
- initial pair count 1,310,720

This is a clean Policy-compute limit because routing-table construction is negligible.

Current known Policy-only bracket under the frozen 300 s budget:
- 128R/640T succeeds in 14.84 s
- 512R/2560T fails at 300 s

256R/1280T has not yet been measured in the three-world ladder.

Next frozen command:
bash tools/run_gene_mrta_v115b_policy_only_mac.sh ladder3

Do not increase the 512 timeout yet. Use 256R/1280T to tighten the limit bracket first.

99M remains untouched.


## V1.15B ladder3 result

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_180029_seed115110000

All scales completed 3/3 worlds.

64R/320T:
- Policy 1.447824 s
- pair slots 2,092,501
- completion 0.397917
- time utility 0.212987
- queue mean 1.98958
- RSS 144.90 MB

128R/640T:
- Policy 9.910586 s
- pair slots 16,986,752
- completion 0.406250
- time utility 0.213276
- queue mean 2.03125
- RSS 238.17 MB

256R/1280T:
- Policy 81.533248 s
- pair slots 134,082,304
- completion 0.399219
- time utility 0.212191
- queue mean 1.99609
- RSS 516.41 MB

Empirical Policy exponent:
2.9077150801970753

Interpretation:
- near-cubic repeated pair rescoring is confirmed over three scales;
- zero-shot behavioral metrics remain remarkably stable through 256R/1280T;
- fixed 148 Policy parameters are not the source of scaling cost.

Current 300 s bracket:
- 256R/1280T succeeds 3/3 at mean 81.53 s
- 512R/2560T fails at 300 s

Next:
bash tools/run_gene_mrta_v115b_policy_only_mac.sh probe384

384R/1920T is predicted around 265 s from the measured exponent and is the most informative next probe.

99M remains untouched.


## V1.15B probe384 result

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_191351_seed115140000

384R/1920T:
- SUCCESS
- Policy 296.5072 s
- decoder steps 758
- pair slots 448,687,488
- completion 0.394792
- time utility 0.211622
- balance 0.364720
- queue mean 1.97396
- RSS 1038.42 MB
- Euclidean table only 0.02464 s

This is only 3.4928 s below the frozen 300 s Policy timeout.

Largest measured successful Policy-only scale:
384R/1920T.

Updated descriptive exponent using 64/128/256/384:
~2.9707.

Next:
bash tools/run_gene_mrta_v115b_policy_only_mac.sh probe388

388R/1940T is predicted around 306 s and is intended as the final tight
threshold-localization probe.

After threshold localization, move to decoder-compute optimization rather than
increasing Policy parameter count.

99M remains untouched.


## V1.15 closeout

V1.15 scaling study is complete and frozen.

Final observed full-system result:
- 64R/320T succeeds 3/3;
- 128R/640T fails 3/3 in obstacle-aware A* preprocessing before Policy inference.

Final observed Policy-only result:
- 256R/1280T succeeds 3/3, mean Policy 81.53 s;
- 384R/1920T succeeds in 296.5072 s;
- 388R/1940T fails at Policy timeout 300.0242 s;
- 512R/2560T also fails at Policy timeout.

Largest observed successful Policy-only scale under the frozen 300 s budget:
384R/1920T.

Important caveat:
384 and 388 are different one-world probes. Treat 384-388 as an observed
practical wall, not a mathematically exact threshold.

Behavior remains approximately stable through the largest successful scales:
completion ~0.40, raw time utility ~0.21, mean queue depth ~2.

Main bottlenecks:
- full system: A* path preprocessing;
- Policy-only: near-cubic repeated pair rescoring.

The 148 Policy parameters are not the scaling bottleneck.

Do not test 385/386/387.
Freeze V1.15 and move to a decoder-compute optimization study.

Recommended next version:
V1.16 Efficient Route-Tail Decoder.

Primary goal:
reduce inference complexity while preserving the exact frozen 148-parameter
Gene and, initially, the original decision semantics.

99M remains untouched.


## V1.16 MILP vs Policy scaling

User decision:
skip decoder optimization for now and directly compare the frozen Gene Policy
against exact MILP as robot/task count grows.

Primary questions:
1. exact best-solution gap;
2. computation time;
3. scale at which either method becomes impractical.

Frozen Policy:
7ee7c18fca2280022ac5, 148 parameters.

MILP:
reuse src/marl2d/gene_mrta_v16t/global_optimal_core.py and the exact T objective.

Fairness:
same obstacle-aware world and same A* path table for both methods.
Shared preprocessing is timed separately.

Exact gap is reported only when HiGHS proves optimal.
MILP timeouts use incumbent/dual-bound diagnostics and are never labelled exact.

New seed namespace:
116M.

Initial smoke:
2R/10T, 3R/15T, 4R/20T, one world each.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V116_MILP_POLICY_SCALING.md

99M remains untouched.


## V1.16 implementation status

MILP-vs-frozen-Policy smoke and three-world ladder are complete.

Code:
src/marl2d/gene_mrta_v116/milp_policy_scaling.py

Tests:
tests/test_gene_mrta_v116_milp_policy_scaling.py

Launcher:
tools/run_gene_mrta_v116_milp_policy_mac.sh

Existing MILP core now exposes incumbent dual upper bound and node count.

First commands:
1. bash tools/run_gene_mrta_v116_milp_policy_mac.sh tests
2. bash tools/run_gene_mrta_v116_milp_policy_mac.sh smoke

Do not claim any V1.16 result before user terminal output exists.


## V1.16 smoke and ladder result

Tests:
5 passed.

Smoke:
2R/10T, 3R/15T, 4R/20T all MILP-optimal.

Ladder run:
runs/gene_mrta_v116_milp_policy/gene_mrta_v116_milp_policy_20261004_201755_seed116010000

Three-world aggregates:

2R/10T:
- MILP proof 3/3
- Policy 0.001377 s
- MILP 0.195471 s
- exact retention 0.991087

3R/15T:
- MILP proof 3/3
- Policy 0.002730 s
- MILP 7.901246 s
- exact retention 0.946564

4R/20T:
- MILP proof 3/3
- Policy 0.003387 s
- MILP 20.814807 s
- exact retention 0.977449

5R/25T:
- MILP proof 3/3
- Policy 0.004702 s
- MILP 41.553477 s mean
- exact retention 0.942129
- one MILP world took 112.50 s

6R/30T:
- MILP proof 0/3
- all hit 300 s
- Policy 0.010127 s
- MILP/Policy per-world ratio mean 30,883.87x
- exact retention unavailable
- Policy / MILP incumbent ratio mean approximately 0.9878

Primary result:
observed exact-MILP practical proof wall under 300 s lies between 5R/25T and 6R/30T.

MILP runtime is strongly world-dependent/heavy-tailed.

Next:
bash tools/run_gene_mrta_v116_milp_policy_mac.sh boundary5

boundary5 uses five new worlds each at 4R/20T, 5R/25T, 6R/30T.

Do not run 8R/40T exact MILP yet.

99M remains untouched.


## V1.16 exact-unlimited MILP extension

User decision on 2026-10-04:

The 300 s boundary is not the final exact reference. MILP must be allowed to
continue until HiGHS actually proves the global optimum so that the true
Policy-vs-optimum gap is known.

Implementation status:

IMPLEMENTED, NOT YET EXECUTED ON MAC.

Core changes:

- `solve_global_time_optimum(..., time_limit=None)` is supported;
- when `time_limit=None`, the HiGHS `time_limit` option is omitted entirely;
- `mip_rel_gap=0.0` remains frozen;
- unlimited mode accepts a result as exact only when HiGHS returns
  `status == 0` with a solution;
- an unlimited run that terminates without optimal proof raises an error;
- incumbent solutions are never relabelled as optimum.

Long-run observability / durability:

- optional native HiGHS display;
- elapsed-time `MILP_ALIVE` heartbeat every 30 s by default;
- each completed world is appended immediately to `per_world.jsonl`;
- JSONL writes are flushed and fsynced;
- a fixed `--run-dir` can be reused;
- resume is enabled by default and skips only completed worlds;
- in exact-unlimited mode, only rows with `milp_optimal=true` are skippable.

New launcher modes:

```
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-regression
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

`exact-regression` uses the same ladder-equivalent seeds for 4R/20T and
5R/25T.

`exact-unlimited` starts with one 6R/30T world using seed 116050000, which is
the first 6R/30T world from the previous ladder namespace, and therefore
directly revisits a known 300 s timeout instance rather than selecting a new
easier world.

Default exact-unlimited run directory:

```
runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000
```

To extend to additional 6R/30T worlds after the first exact result:

```
V116_EXACT_WORLDS=3 bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

The same fixed run directory and resume semantics preserve already completed
worlds.

Scientific rule remains unchanged:

Oracle does not teach the action; it defines the capability ceiling.

99M remains untouched.


## V1.16 exact-unlimited completed first hard world

Run:

runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000

Tests before run:

9 passed.

Frozen Gene:

7ee7c18fca2280022ac5

Parameters:

148.

Exact hard world:

- case = 6R/30T
- seed = 116050000
- MILP time limit = None
- mip_rel_gap = 0.0
- HiGHS status = Optimal
- MILP optimum T* = 0.28074964073648145
- Policy T = 0.2756685668779121
- exact absolute gap = 0.005081073858569374
- exact relative gap = 0.018098238149977172
- exact retention = 0.9819017618500229
- MILP completed tasks = 14
- Policy completed tasks = 14
- Policy solve = 0.008538166999642272 s
- MILP solver = 717.154159042002 s
- MILP total = 717.1704191659992 s
- MILP/Policy method-time ratio = 83995.82945567204x
- MILP nodes = 276137
- LP iterations = 8274628
- RSS peak = 290.953125 MB

Important solver-dynamics observation:

The final best solution value 0.2807496407 was already found around 138.5 s,
but global optimality was not proved until about 717.1 s.

Therefore most of the remaining MILP runtime on this world was optimality
certification, not discovery of a better incumbent.

This sharpens the V1.16 conclusion:

- under a 300 s operational budget, 6R/30T is already beyond the reliable exact-proof regime;
- nevertheless, the frozen 148-parameter Policy is only about 1.81% below the true global optimum on this first hard exact world;
- the Policy returns in about 8.54 ms;
- exact MILP proof requires about 11.95 minutes;
- the observed method-time ratio is about 84,000x.

Regression exact-unlimited checks also passed:

4R/20T, seed 116030000:
- T_policy = T* = 0.21914552147298996
- exact retention = 1.0
- MILP total = 0.7096404160001839 s
- Policy = 0.0037326669989852235 s

5R/25T, seed 116040000:
- T* = 0.18523070152476162
- T_policy = 0.1769690430944892
- exact retention = 0.9553980071215785
- exact relative gap = 0.04460199287842149
- MILP total = 109.60991570900296 s
- Policy = 0.004382666000310564 s

The first 6R/30T exact world is now complete.
Do not yet generalize 98.19% retention to the whole 6R/30T distribution from n=1.

Recommended next action:
extend exact-unlimited to 3 worlds at 6R/30T using resume so world 1 is skipped.


## V1.16 6R/30T multi-seed worst-case exact benchmark

Next experiment decision:

Run multiple exact-unlimited 6R/30T seeds to measure not only mean Policy
quality but the worst observed exact error and exact-proof runtime.

Per-seed exact error:

    e_s = (T*_s - T_policy,s) / T*_s

Primary worst-case statistic:

    e_max = max_s e_s

Equivalent worst observed retention:

    retention_min = min_s T_policy,s / T*_s

Runtime statistics:

- mean / median / max MILP total seconds;
- slowest MILP seed;
- mean / median / max Policy seconds;
- slowest Policy seed;
- per-seed MILP / Policy time ratio.

The V1.16 case summary now records:

- max_exact_relative_gap;
- max_exact_absolute_gap;
- worst_exact_gap_seed;
- worst_exact_gap_policy_score;
- worst_exact_gap_optimum_score;
- max_milp_total_seconds;
- slowest_milp_seed;
- max_policy_seconds;
- slowest_policy_seed.

Recommended first formal batch:

10 exact seeds at 6R/30T: 116050000 through 116050009.

Existing seed 116050000 is already proven optimal and will be skipped via resume.

Command:

    V116_EXACT_WORLDS=10 bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited

Do not call the resulting maximum a theoretical upper bound. It is the maximum
observed exact gap over the tested seed set.


## V1.16 6R/30T 10-seed exact-unlimited result

Status:

COMPLETED.

Run:

runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000

Seeds:

116050000 through 116050009.

Tests before run:

10 passed.

All 10 worlds reached proven global optimality.

Aggregate:

- exact proof rate = 10/10 = 100%;
- mean exact retention = 0.9694540924627978;
- mean exact relative gap = 0.030545907537202176;
- mean exact absolute gap = 0.006215299617823469;
- minimum exact retention = 0.9152010318146683;
- maximum observed exact relative gap = 0.08479896818533172;
- maximum observed exact absolute gap = 0.017600480877557306;
- worst exact-gap seed = 116050009.

Worst observed quality world:

seed 116050009:
- T_policy = 0.18995488511570233;
- T* = 0.20755536599325963;
- exact retention = 0.9152010318146683;
- relative gap = 0.08479896818533172;
- Policy completed tasks = 11;
- MILP completed tasks = 13.

Runtime aggregate:

- mean Policy = 0.007979583600172192 s;
- median Policy = 0.00851533350032696 s;
- max Policy = 0.011506959002872463 s;
- mean MILP total = 453.48357761249974 s;
- median MILP total = 231.51708522899935 s;
- max MILP total = 1734.6120088330026 s;
- slowest MILP seed = 116050002;
- mean MILP / Policy ratio = 47884.560790352734x.

The slowest exact proof is about 28.91 minutes.

Five of ten exact MILP worlds require more than the original 300 s practical
budget, while the Policy remains below 12 ms for every tested seed.

Distribution summary:

- 10/10 retention >= 90%;
- 9/10 retention >= 95%;
- 5/10 retention >= 98%;
- 2/10 retention >= 99%;
- 9/10 exact relative gap <= 5%.

Important interpretation:

At fixed 6R/30T, MILP hardness varies extremely across worlds.
Observed exact proof time ranges from about 2.48 s to 1734.61 s despite identical
R/T dimensions and the same formulation size.

The frozen 148-parameter Policy remains consistently millisecond-scale and has
mean exact retention about 96.95%, with worst observed retention 91.52% over
these 10 seeds.

The maximum observed 8.48% gap is empirical over these seeds, not a theoretical
worst-case bound over all possible worlds.

A notable objective effect occurs at seed 116050005: Policy completes 13 tasks
while the exact T-optimal MILP completes 12, yet Policy T is lower. This is
consistent with the benchmark objective optimizing completion time utility T,
not raw task count.


## V1.16 5R/25T plotting dataset

Next plotting dataset is a 10-seed exact-unlimited 5R/25T benchmark using the
same frozen V1.13 Gene and the same V1.16 MILP/Policy fairness rules.

Launcher:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-5r25

Defaults:

- case = 5R/25T;
- seeds = 116040000 through 116040009;
- worlds = 10;
- MILP time limit = none;
- mip_rel_gap = 0.0;
- exact proof required;
- heartbeat = 30 s;
- fixed run directory =
  runs/gene_mrta_v116_exact_unlimited_5r25_seed116040000.

This dataset is intended to be directly comparable with the completed 10-seed
6R/30T exact dataset for plotting quality and runtime scaling.


## V1.17 clean two-stage retraining line

A new clean training line has started. The purpose is to rebuild a Policy Gene
using lessons from V1-V1.16 without inheriting old Policy parameters or old
capability labels.

Stage A is designed from complete known task semantics before hard-world
inspection.

Canonical Task:

    task_id
    position [x,y]
    service_time
    priority
    deadline

Canonical Robot:

    robot_id
    start_position [x,y]
    initial_battery

Global environment:

    episode horizon
    robot speed
    battery capacity
    energy per distance
    obstacles
    grid resolution

A* path tables and all route-tail pair features are derived data, not static
Task fields.

Stage-A independent capability axes:

1. completion
2. time_retention = T_gene / T_star
3. path_efficiency
4. priority_satisfaction
5. deadline_satisfaction
6. workload_balance

Battery remains a hard feasibility constraint rather than a capability axis.
With the current energy model, energy is proportional to A* path distance, so
path_efficiency supplies the travel/energy pressure without the trivial
remaining-battery no-work loophole.

Stage B is evidence-driven:

- freeze Stage-A model/bank;
- evaluate a large fixed development set;
- extract hard worlds;
- classify repeated failure mechanisms;
- add only evidence-supported robustness axes;
- retrain/fuse a final multi-capability model.

Continuation preservation, fleet option reserve, tail-10 time, and similar
robustness axes are intentionally NOT active in Stage A. They may be introduced
in Stage B only if the new Stage-A failure analysis supports them.

Implemented:

- src/marl2d/gene_mrta_v117/schema.py
- src/marl2d/gene_mrta_v117/capabilities.py
- tests/test_gene_mrta_v117_schema.py
- tools/run_gene_mrta_v117_mac.sh
- docs/GENE_HOMOGENEOUS_MRTA_V117_TWO_STAGE_TRAINING.md


## V1.17 global objective specialist refinement

Stage-A capability design was refined so that the Gene Bank can explicitly
preserve different global-objective specialists.

Primary exact-ceiling axes:

    global_time_optimality = T_gene / T_star

where

    T = (1/N) * sum_completed(1 - finish/H)

and T_star is a proven global MILP optimum.

Also:

    global_priority_optimality = P_gene / P_star

where P_gene is completed-priority / total-priority and P_star is the proven
maximum feasible priority satisfaction from a separate MILP objective.

A score of 1.0 on either axis therefore has a clear interpretation: the Gene
has reached the global optimum for that capability objective on the evaluated
world.

Revised Stage-A axes:

- completion
- global_time_optimality
- path_efficiency
- global_priority_optimality
- deadline_satisfaction
- workload_balance

This allows selecting a global-time specialist or a global-priority specialist
from the same Gene Bank before capability fusion.

Implemented:

- exact V1.17 priority MILP oracle;
- normalized global-time and global-priority capability scoring;
- unit tests for the priority oracle and 1.0 optimality semantics.


## V1.17 Stage-A trainer now implemented

The new clean retraining line can now execute end to end.

Files:

- src/marl2d/gene_mrta_v117/stage_a_oracle_bank.py
- src/marl2d/gene_mrta_v117/stage_a_train.py
- tests/test_gene_mrta_v117_stage_a.py

Important constraints:

- Stage A starts from random 148-parameter Route-Tail Genes;
- no previous Policy parameters are loaded;
- no V1.13 robustness labels are imported;
- Time and Priority each have their own exact MILP capability ceiling;
- conflicting specialists are intentionally preserved in separate archives.

Smoke command:

    bash tools/run_gene_mrta_v117_mac.sh smoke

The smoke builds 2 exact 2R/10T dual-oracle worlds and runs 5 generations with
a small random population to validate specialist separation and mating.

Formal launcher modes also exist but should not be started until smoke output is
reviewed:

    bash tools/run_gene_mrta_v117_mac.sh oracle-formal
    bash tools/run_gene_mrta_v117_mac.sh train-formal


## V1.17 Stage-A smoke #1 findings

The first executable smoke passed 7 tests and completed 2 exact 2R/10T
dual-oracle worlds plus 5 generations of random Stage-A evolution.

Key positive result:

Time and Priority specialists separated immediately and remained different.
Priority capability improved from about 0.9100 to 0.9674 while the best Time
specialist remained at about 0.9905.

The smoke also exposed two pre-formal design issues.

First, raw completion stayed at 0.4 while both exact Time and Priority oracles
completed 4/10 tasks. Raw completion therefore conflates Policy quality with
world feasibility. V1.17 now adds an exact completion ceiling C* and uses:

    global_completion_optimality = C_gene / C_star

Workload balance now uses completion retention times Jain fairness rather than
raw completion times Jain fairness.

Second, random-from-scratch Stage A has no shared trained ancestor, so
ancestor_delta, TIES-delta, and DARE-delta mating are not semantically valid.
Stage-A mating is now restricted to anchor-free:

- parameter_blend
- block_pick
- block_blend

Generation logs now report the most-capable Gene's origin, operator, parents,
capabilities, and scores, so actual mating-based fusion can be distinguished
from simple multi-archive membership.

Run smoke again after pulling before formal training.


## V1.17 Stage-A smoke #2: true fusion confirmed

Second smoke:

- tests = 9 passed;
- exact C* = 0.4 for both 2R/10T worlds;
- global completion retention reaches 1.0;
- global time retention reaches 1.0 by generation 1;
- best global priority retention = 0.9338964620;
- Time and Priority specialist IDs remain distinct.

True mating fusion is now directly observed.

Generation 1 record:

    812e74f49dfb20a82ae8

origin:

    mating

operator:

    parameter_blend

parents:

    cdc2746ad444bce89452
    3eab704335a0fbf5c989

It carries five archive/certified capabilities including global Time and global
Priority, with time retention ~0.98334 and priority retention ~0.93390.

This confirms the intended specialist -> crossover -> multi-capability Gene
mechanism on the clean random-from-scratch line.

Pre-formal refinement:

Raw Path Efficiency and Deadline are now also replaced by exact-ceiling
retention axes:

    global_path_efficiency = E / E*
    global_deadline_optimality = D / D*

The exact Stage-A oracle bank therefore contains C*, T*, E*, P*, and D* for
every world.

Only workload balance remains non-oracle because the Jain-based objective is
nonlinear; it is represented as completion retention times Jain fairness.

Run one more smoke after pulling before formal training.


## V1.17 smoke #3 and capability-provenance correction

Smoke #3 passed 11 tests and validated exact C*, T*, E*, P*, D* ceilings.

Two smoke worlds:

117000000:
C*=0.4, T*=0.2242209550768175, E*=0.33060942664538956,
P*=0.45197901256124035, D*=0.4.

117000001:
C*=0.4, T*=0.19496278602896533, E*=0.3537811670895189,
P*=0.5098102611285834, D*=0.4.

Training behavior:

- C/C*=1.0;
- D/D*=1.0;
- E/E* reaches 0.9906102433;
- P/P* reaches 0.9723705458;
- T/T*=0.9905165213 in this run;
- Time and Priority specialists remain distinct.

Correction:

Earlier smoke #2 was described as proving true mating capability fusion.
That claim was too strong because the Record.capabilities field mixed archive
membership with passed mating inheritance.

V1.17 now separates:

- archive_capabilities;
- inherited_capabilities;
- combined capabilities.

Only inherited_capabilities is used to identify and preserve true mating
fusion. Logs now expose max_inherited_capabilities and best_fusion_gene.

Formal Stage-A training must wait for one final smoke using this provenance
split.
