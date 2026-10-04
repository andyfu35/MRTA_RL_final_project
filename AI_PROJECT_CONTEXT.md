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

# 8. Protected data policy

95M:

development data.

98M:

historical V1.8 final set, later intentionally reused for V1.9 diagnosis, therefore development/diagnostic for later versions.

99M:

CURRENTLY UNTOUCHED AND PROTECTED.

Do not inspect 99M during V1.14.1 development.

Current seed=7 evidence is not sufficient to justify 99M or a larger formula grammar.

Next:
1. fix unique-child control asymmetry;
2. rerun paired control;
3. if adaptive remains promising, repeat paired seeds;
4. then use independent 96M development-validation if required;
5. freeze procedure/candidate;
6. only then evaluate 99M.

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
