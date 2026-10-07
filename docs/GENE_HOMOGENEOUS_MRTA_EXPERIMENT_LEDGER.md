# Gene Homogeneous MRTA / SEGB Experiment Ledger

Last updated: 2026-10-04
Repository: andyfu35/MRTA_RL_final_project
Active branch: experiment/gene-homogeneous-mrta-v1
Author: 傅獻德 (Hsien-Te Fu)

This file is the long-form experiment ledger for the Gene-based homogeneous MRTA research track.
It is intended to preserve the complete reasoning chain between conversations.

IMPORTANT FOR ANY NEW AI CONVERSATION:
1. Read /AI_PROJECT_CONTEXT.md first.
2. Read this ledger second.
3. Read the latest version-specific document before changing code.
4. Do not assume a planned experiment was executed unless this ledger explicitly records a completed result.
5. Do not inspect the protected 99M final benchmark until the current procedure and candidate are frozen.

---

# 1. Research objective

The research goal is not merely to solve one MRTA instance.

The target is a Self-Evolving Gene Bank style system in which:

- one homogeneous Policy Gene is shared by all robots;
- multiple independent external capability axes judge behavior;
- conflicting capabilities are preserved rather than scalarized into one weighted reward;
- Policy Genes are stored, selected, mutated, recombined, admitted, and pruned by a Gene Bank;
- the system eventually also evolves how Policy Genes reproduce.

The current long-term architecture is therefore:

Policy Gene Bank
+
Recombination Gene Bank

The Policy Gene Bank answers:

What allocation behavior should survive?

The Recombination Gene Bank answers:

How should surviving Policy Genes reproduce?

The desired research contribution is a self-improving evolutionary system, not a claim that one universal crossover formula exists for every domain.

---

# 2. Non-negotiable research principles

## 2.1 No weighted scalar capability reward

Capability axes remain independent.

Do not replace the Gene Bank with a weighted score such as:

0.4 * mean_time + 0.3 * tail + ...

Reproductive weights may use worst declared-capability retention, but this is not a weighted scalar reward.

## 2.2 Oracle role

Exact MILP is an external capability reference only.

Canonical statement:

Oracle does not teach the action; it defines the capability ceiling.

The oracle must never provide:

- action labels;
- imitation targets;
- assignment demonstrations;
- online matching decisions.

Multiple assignments may achieve the same optimum. Compare capability value, not assignment identity.

## 2.3 No global world model

The project does not introduce a global learned world model.

Each Policy Gene remains a direct policy parameter vector.

## 2.4 Preserve conflicting skills

The Gene Bank must be capable of retaining specialists and Pareto-conflicting capabilities.

Do not collapse all specialist behavior into a single mandatory generalist.

## 2.5 Protected benchmark discipline

Current data-status rule:

- 95M: development scenario bank used by V1.10 onward.
- 98M: historical publication-final benchmark for frozen V1.8; after V1.9 failure analysis it became development/diagnostic data.
- 99M: protected untouched final benchmark for the current research line.

Do not inspect, probe, tune on, or use 99M before the architecture, procedure, and candidate are frozen.

---

# 3. Fixed MRTA environment after V1.6

The current core environment is:

- world: 100 x 100 continuous plane;
- homogeneous robots R = 4;
- tasks T = 20;
- robot speed = 4;
- task service time Uniform(2, 35);
- task priority Uniform(0.1, 1.0);
- soft deadline Uniform(25, 50);
- episode horizon H = 50;
- 10 static non-overlapping square obstacles;
- obstacle side length Uniform(12, 20);
- obstacle clearance = 4;
- deterministic 8-connected A*;
- grid resolution = 5;
- diagonal corner cutting disabled;
- battery capacity = 70;
- initial battery Uniform(35, 70);
- energy per distance = 1;
- service time consumes no battery;
- pair eligibility requires both horizon feasibility and battery feasibility.

Task/robot points are generated in traversable connected space.

Path tables are precomputed for:

- initial robot nodes to tasks;
- task nodes to tasks.

This allows route-tail planning without rerunning A* during every Gene evaluation.

---

# 4. Core external metrics

## Completion

C = completed_tasks / total_tasks

## Route efficiency

For a completed task:

route_value = 1 - path_distance / path_cost_scale

The normalized episode efficiency counts unfinished tasks as zero.

## Priority satisfaction

P = completed_priority / total_priority

## Deadline satisfaction

D = tasks_completed_before_own_deadline / total_tasks

## Balance

Jain fairness is computed over robot workload.

Balance = Completion * Jain(workload)

## Time utility / time optimality

For a completed task j with finish time F_j:

u_j = 1 - F_j / H

Episode time utility:

T = (1 / N) * sum over completed tasks of u_j

Therefore:

0 <= T <= Completion <= 1

The exact MILP time reference is T*.

The primary time-retention quantity used in later experiments is:

T_G / T*

---

# 5. Version evolution

## V1 - first homogeneous Gene MRTA

Environment:

- no obstacles;
- no battery;
- fixed service time;
- 4 robots, 20 tasks;
- event-based simulator.

Observation:

- distance;
- robot load;
- competition;
- slack.

Policy:

- four linear weights plus an ineffective shared bias;
- same Gene shared by all robots;
- external greedy global matching.

Capabilities:

- completion;
- efficiency;
- balance.

Result:

The first run showed a small advantage over a nearest-task heuristic, but exposed calibration confounds.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V1.md

## V1.1 - calibration

Changes:

- removed ineffective shared bias;
- four effective parameters;
- corrected competition definition;
- corrected efficiency loophole;
- fixed probe worlds;
- untouched fixed validation worlds;
- five-seed suite.

Key metric correction:

efficiency = sum(route_value completed) / total_tasks

so unfinished tasks contribute zero and efficiency <= completion.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V11.md

## V1.2 - variable service time

Added:

service_time Uniform(2, 35)

Robot speed calibrated to 4.

Observation became:

- distance;
- service time;
- robot workload;
- competition.

Workload became accumulated travel time + service time.

Result recorded in later version documentation:

V1.2 established that Gene allocation could outperform nearest, shortest-service, and shortest-total-time heuristics under variable service time.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V12.md

## V1.3 - task priority

Added:

priority Uniform(0.1, 1.0)

Observation added priority.

Capabilities became:

- completion;
- efficiency;
- priority satisfaction;
- balance.

No weighted reward.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V13.md

## V1.4 - soft deadline

Added:

deadline Uniform(25, 50)

A late task may still be completed but gets zero deadline credit.

Capabilities:

- completion;
- efficiency;
- priority satisfaction;
- deadline satisfaction;
- balance.

Observation added deadline remaining.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V14.md

## V1.5 - static obstacles and A*

Added:

- 10 square obstacles;
- deterministic grid A*;
- obstacle-aware path cost;
- precomputed path table.

Observation became seven-dimensional, including both Euclidean distance and A* path distance.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V15.md

## V1.6 - finite battery

Added:

- battery capacity 70;
- initial battery Uniform(35,70);
- energy/distance = 1;
- hard battery feasibility.

Observation became eight-dimensional:

1. Euclidean distance
2. path distance
3. service
4. priority
5. deadline remaining
6. battery remaining
7. workload
8. competition

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V16.md

## V1.6-T - time capability

Environment unchanged.

Added independent time-optimality axis:

T = (1/N) sum_completed (1 - F_j/H)

Goal:

learn a low-cost Gene whose allocation behavior approaches time-oriented assignment references without receiving oracle actions.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V16T.md

## V1.6-T-O - MILP oracle-guided capability evolution

Added external capability:

global_optimality_retention = mean(T_gene / T_star)

MILP is never used online.

Historical seed namespaces included separate training/probe/held-out regions.

This version still used the linear bidder + external matching path.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V16TO.md

## V1.7 - Direct Assignment

Major architectural change:

remove the external learned-path matcher.

Policy sees the full R x T x 8 tensor.

Autoregressive decoder:

- score robot-task pairs;
- choose one pair;
- mask selected robot and selected task;
- recompute context;
- learned STOP/WAIT action;
- zero/subset assignment allowed.

hidden_dim = 8
parameter count = 116

No Hungarian/greedy matcher is used in deployment.

MILP remains external capability reference only.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V17.md

## V1.7-T - Direct-Time scaling

Controlled time-only scaling of the V1.7 architecture.

Used cached exact T* oracle worlds.

Known result motivating V1.8:

- 64-world oracle probe retention approximately 96.52%;
- 20-world 97M held-out retention approximately 94.98%.

Failure traces showed that STOP/WAIT was not the main bottleneck.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V17T_SCALE.md

## V1.8 - Consequence-Aware Direct Assignment

Hypothesis:

V1.7 mainly lacked information about assignment consequences, not decoder capacity.

Kept:

- hidden_dim 8;
- same direct decoder;
- learned STOP/WAIT.

Observation 8D -> 12D.

Added:

9. self_future_reachability
10. self_future_best_time_utility
11. other_robot_opportunity_cost
12. residual_battery

parameter count:

116 -> 148

Exact lift from V1.7 is possible by zeroing the new input weights.

Frozen V1.8 run:

runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7

### V1.8 final 98M benchmark

100/100 worlds were proven MILP-optimal.

V1.8:

- mean = 96.93972395246712%
- std = 3.6242 percentage points
- median = 98.1012%
- minimum = 82.7159%
- P10 = 91.5574%
- P25 = 94.6710%
- max = 100%
- bootstrap 95% CI of mean = [96.2191%, 97.6318%]

Comparison means:

- V1.7 mean = 96.1490%
- V1.6-T-O mean = 95.4079%
- Hungarian mean = 94.1866%

V1.8 paired differences:

versus V1.7:
- mean +0.79068 percentage points
- CI95 [+0.1018, +1.5107]
- W/T/L = 33/43/24
- Cohen dz = 0.2178
- Wilcoxon p = 0.0584
- interpretation: trend / modest improvement, not a strong significance claim.

versus V1.6-T-O:
- mean +1.53182 percentage points
- CI95 [+0.7239, +2.3449]
- Wilcoxon p = 0.0002826

versus Hungarian:
- mean +2.75313 percentage points
- CI95 [+1.7854, +3.7466]
- p approximately 7.19e-7

Exact T* matches:

- V1.8: 34
- V1.7: 24
- V1.6-T-O: 18
- Hungarian: 11

Oracle solve timing:

- mean 27.63 s
- median 3.69 s
- max 408.81 s

After this benchmark was finalized, 98M was intentionally reused for V1.9 failure analysis and is no longer untouched for later versions.

Primary documents:
docs/GENE_HOMOGENEOUS_MRTA_V18.md
docs/GENE_HOMOGENEOUS_MRTA_V18_PUBLICATION_FREEZE.md

## V1.9 - Failure-derived robust Gene Bank

Bottom-10 analysis of V1.8 on 98M found:

- continuation_collapse: 10/10
- fleet_reserve_risk: 7/10
- hard_for_all: 5/10
- v18_regression_vs_v17: 3/10
- immediate_future_imbalance: 0/10

Therefore V1.9 retained the same 12D / 148-parameter architecture and introduced four Gene Bank axes:

1. mean_time
2. hard_world_time
3. continuation_preservation
4. fleet_option_reserve

Hard fixed worlds were the V1.8 bottom-10 98M worlds.

Continuation:

C_t = clip((U_t + O_after) / O_before, 0, 1)

Fleet reserve:

For remaining task j with feasible owner count d_j:

q_j = min(d_j,2)/2

Q = (1/N) sum_j q_j

The episode reserve axis measures retention of future multi-robot task feasibility.

Result:

Specialists could be discovered, but independent specialist archives did not naturally fuse into one strong generalist.

This identified the next bottleneck:

specialist-to-generalist capability fusion.

Primary documents:
docs/GENE_HOMOGENEOUS_MRTA_V19_FAILURE_PROTOCOL.md
docs/GENE_HOMOGENEOUS_MRTA_V19_ROBUST_BANK.md

## V1.10 - Evolutionary Mating

Goal:

explicitly add mating alongside ordinary mutation to fuse capability specialists.

Policy unchanged:

- 12D;
- hidden 8;
- 148 parameters;
- direct assignment;
- learned STOP/WAIT.

### Frozen development scenario bank

Dedicated 95M namespace.

Procedure:

- 500 candidate worlds;
- policy-independent descriptors;
- standardized farthest-point sampling;
- select 100 diverse worlds;
- exact MILP T* cached.

One selected world, seed 95000034, failed exact proof after 900 seconds and was removed.

Descriptor-nearest unused replacement:

seed 95000442

Frozen bank version:

v110_diverse_100_v2_frozen

No 98M/99M seeds included.

### Population

Per generation:

128 normal mutation
+
128 mating
=
256 children

Normal parent pressure:

P proportional to Q^2

Mating parent pressure:

P proportional to Q^10

with 5% uniform exploration.

Quality for declared capability set C_G:

Q(G) = min over a in C_G of S_a(G)/B_a

clipped to [0,1].

### Recombination operators

Six hand-designed operators:

1. parameter_blend
2. block_pick
3. block_blend
4. ancestor_delta
5. ties_delta
6. dare_delta

Four children are produced per parent pair, using four distinct operators in that family.

Mating-child mutation was disabled in the first pilot.

### Inheritance gate

Required child capabilities:

C_C = C_A union C_B

Full admission requires, for every required axis:

parent retention >= 0.95
and
current generation ceiling retention >= 0.95

This prevents repeated 95%-of-parent decay.

### Certification

Declared capability labels are separate from current certification.

A declared capability a is currently certified only when:

S_a(G)/B_a_current >= 0.95

Success is measured by certified capability count, not tag count.

### V1.10 Pilot50 result

Run:

runs/gene_mrta_v110_mating/gene_mrta_v110_mating_20261003_163423_seed7

Gen 0:

- Bank 57
- hybrid 16
- accepted 16
- max certified capability count 2
- mean 0.9741
- tail 0.8993
- continuation 0.7737
- reserve 0.8309

Gen 1:

- first certified 3-cap hybrid.

Gen 2:

- first certified 4-cap hybrid.
- two notable accepted four-cap children:
  - a7ad33c888c271b5936e by parameter_blend
  - c54074219e136ee775a4 by ties_delta

Certified four-capability status persisted afterward.

Final Gen 49:

- mean_time = 0.9779050005301373
- tail10_time = 0.9143206459612623
- continuation_preservation = 0.7785539293423699
- fleet_option_reserve = 0.83879703612327
- certified capability count = 4

Operator funnel over 6400 generated mating children:

parameter_blend:
- generated 1075
- selected 175
- accepted 174
- generated-to-accepted = 0.1618604651

block_pick:
- 1068 / 154 / 152
- acceptance = 0.1423220974

block_blend:
- 1059 / 198 / 198
- acceptance = 0.1869688385

ancestor_delta:
- 1050 / 80 / 77
- acceptance = 0.0733333333

ties_delta:
- 1082 / 142 / 141
- acceptance = 0.1303142329

dare_delta:
- 1066 / 51 / 49
- acceptance = 0.0459662289

Do not claim one operator is universally superior from this single seed.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V110_MATING.md

## V1.11 - Empirical Recombination Law Discovery

Goal:

use recombination outcomes to search for an interpretable mixing law.

This version searches coefficients inside a manually defined symmetric law family.

For functional group k:

Delta_A^(k) = theta_A^(k) - theta_0^(k)
Delta_B^(k) = theta_B^(k) - theta_0^(k)

Parent contribution:

alpha_k = sigmoid(
    beta_r * r_k
    + beta_q * (Q_A-Q_B)
    + beta_h * ((H_A-H_B)/4)
)

Scale:

eta_k = 0.5 + sigmoid(
    gamma_0
    + gamma_c * c_k
    + gamma_s * s_k
    + gamma_m * m_k
)

Child:

theta_C^(k)
=
theta_0^(k)
+
eta_k [
    alpha_k Delta_A^(k)
    + (1-alpha_k) Delta_B^(k)
]

Seven coefficients:

- beta_norm_ratio
- beta_quality_diff
- beta_capability_diff
- gamma_bias
- gamma_cosine
- gamma_sign_agreement
- gamma_magnitude

The law is parent-swap symmetric.

### V1.11 smoke

Tests passed.

Smoke setup:

- 6 discovery pairs
- 8 laws
- 10 development worlds
- 2 held-out parent pairs
- full100 held-out evaluation.

Selected law_006:

beta_norm_ratio = 2.0523582511733975
beta_quality_diff = -0.3926412285036207
beta_capability_diff = 2.7181892936803624
gamma_bias = -1.1873306270250525
gamma_cosine = -0.6100233703361213
gamma_sign_agreement = -2.1270340653846125
gamma_magnitude = 1.322449589266319

Discovery:

- success rate = 1.0
- mean min dual retention = 0.9768225
- P10 = 0.96701277

Held-out n = 2:

Discovered law:
- acceptance = 1.0
- mean min dual retention = 0.9714228940

Block blend:
- 0.9658875747

Parameter blend:
- 0.9670835221

TIES:
- 0.9616134773

Paired min-retention delta of discovered law:

- vs block blend: +0.0055353194, 2W/0T/0L
- vs parameter blend: +0.0043393719, 2W/0T/0L
- vs TIES: +0.0098094168, 2W/0T/0L

This smoke was too small for statistical claims and had no four-capability parent pairs.

V1.11 conclusion:

useful proof that recombination can be parameterized and empirically searched, but not a universal formula discovery.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V111_LAW_DISCOVERY.md

## V1.12

V1.12 was conceptually reserved for self-evolving recombination.

It was not used as the final implementation version number because route-tail multi-task assignment became the next implemented architectural change.

Do not assume a completed V1.12 experiment exists.

## V1.13 - Route-Tail Multi-Task Assignment

Research requirement changed:

each robot must be able to receive multiple ordered tasks in one planning call.

Old direct assignment behavior:

once robot i received one task in the current autoregressive round, robot row i was masked.

V1.13 change:

- do not mask selected robot row;
- only close the selected task column;
- update that robot's virtual route tail;
- recompute observations;
- the same robot may be selected again.

Per robot virtual state:

- tail_node
- tail_position
- tail_time
- battery_remaining
- workload

If task j is appended:

tail_time_i
=
tail_time_i
+
path(tail_i,j)/speed
+
service_j

battery_i
=
battery_i
-
path(tail_i,j) * energy_per_distance

The next task candidate is evaluated from the end of the already planned route.

The 12D observation and 148 parameters remain unchanged.

Decoder step normalization changed from robot count to task count because up to T tasks may be selected in one planning call.

Current scope:

append-only route planning.

Arbitrary insertion into the middle of an existing queue is not implemented.

### V1.13 structural smoke

Tests:

9 passed.

V1.10 four-capability source Gene:

4b47774f9cfebff5c594

Zero-shot route-tail smoke on five frozen 95M worlds:

World seed 95000284:
- routes = ((1,), (10,4,16), (6,9), (18,))
- lengths = [1,3,2,1]
- completed = 7
- T = 0.190214

95000442:
- routes = ((6,12), (8,19), (3,9), (14,15,13))
- lengths = [2,2,2,3]
- completed = 9
- T = 0.249890

95000291:
- completed = 5
- max queue depth = 2

95000377:
- completed = 7
- max queue depth = 3

95000263:
- completed = 9
- max queue depth = 3

All five worlds had at least one multi-task robot.

Total assigned tasks = 37.

This proved the new semantics worked before retraining.

### V1.13 Route-Tail Evolution Pilot50

Warm-start:

V1.10 Policy parameters.

Important semantic reset:

old V1.10 capability labels/scores were not trusted after changing route semantics.
All warm-start Genes were re-evaluated on the 100 frozen worlds before V1.13 archive construction.

Same four capability axes:

- mean_time
- tail10_time
- continuation_preservation
- fleet_option_reserve

Pilot50 result:

Gen 0:
- mean = 0.9680
- tail10 = 0.8862
- continuation = 0.8303
- reserve = 0.8670
- qmean = 1.85
- qmax = 2.80
- multi = 1.00
- certified capability count = 3

Gen 1:
- certified capability count = 4

Four-capability certification then persisted through Gen49.

Final Gen49:

- mean_time = 0.977920253212843
- tail10_time = 0.9262063481487696
- continuation_preservation = 0.8328325738169616
- fleet_option_reserve = 0.8757350433050963

Improvement from Gen0:

- mean: about +0.99 percentage points
- tail10: about +4.00 percentage points
- continuation: about +0.25 percentage points
- reserve: about +0.87 percentage points

Queue behavior remained stable:

- qmean about 1.82
- qmax about 2.70
- multi = 1.00

Interpretation:

performance improvement did not come from simply making queues longer.
The stronger evidence is improved assignment / task ordering while multi-task behavior remained stable.

V1.13 result status:

Route-Tail Multi-Task Architecture = PASS on the 95M development distribution.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V113_ROUTE_TAIL.md

## V1.14 - Self-Evolving Recombination Gene Bank

Goal:

evolve how Policy Genes mate.

Two populations:

1. Policy Gene Bank
2. Recombination Gene Bank

Recombination genotype:

- seven continuous coefficients;
- seven binary feature gates;
- one self-adaptive mutation sigma.

A gate removes a term from the formula.

Rule mutation:

sigma_R' = clip(sigma_R * exp(N(0,tau)))

and coefficients mutate with N(0,sigma_R').

Recombination capability axes:

1. screen_yield
2. acceptance_yield
3. retention_quality
4. four_capability_yield

No weighted mating reward.

### V1.14 tests and smoke

15 tests passed.

Three-generation smoke completed.

### V1.14 Pilot50

Started from the completed V1.13 Policy Bank.

Gen0:

- Policy bank 187
- Recombination bank 32
- certified capability count 4
- mean 0.9779
- tail10 0.9262
- continuation 0.8328
- reserve 0.8757

Final Gen49:

- mean_time = 0.9799838029221177
- tail10_time = 0.9262063481487696
- continuation_preservation = 0.8336073765542017
- fleet_option_reserve = 0.8760983169402837
- certified capability count = 4
- multi = 1.00

Final specialists:

- screen_yield: 2292dcab9d5e0e727004
- acceptance_yield: 2292dcab9d5e0e727004
- retention_quality: 90fce80ad0b13e05b820
- four_capability_yield: 2292dcab9d5e0e727004

Final Pareto size = 4.

Run:

runs/gene_mrta_v114_self_recombination/gene_mrta_v114_self_recombination_20261003_214207_seed7

### V1.14 critical finding

The most-used / most-accepted rule remained an all-gates-off center law for almost the entire run:

Rterms = 0

For the center law:

alpha = 0.5
eta = 1

so:

theta_C
=
theta_0
+
0.5 Delta_A
+
0.5 Delta_B

By Gen49 the repeatedly logged center rule had:

54 accepted / 295 generated

about 18.3%.

This is not automatically evidence that the center law is truly optimal.

Two confounds were identified:

1. neutral center-law clones:
   disabled coefficients and mutation sigma could produce different Gene IDs while the actual formula was identical;

2. small-sample specialist bias:
   a rule with a few lucky outcomes could outrank a mature rule using point-estimate yield.

Therefore:

System-level Recombination Gene Bank = PASS.

Autonomous mating-law discovery = NOT YET PASS.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V114_SELF_EVOLVING_RECOMBINATION.md

## V1.14.1 - Phenotype-Canonical, Evidence-Aware Recombination

STATUS AS OF 2026-10-04:

COMPLETED SINGLE-SEED PAIRED SMOKE AND PAIRED50 FOR seed=7.

Regression tests:

15 passed.

The paired-smoke and paired50 results below are now completed development evidence.

Purpose:

fix V1.14 confounds before making a claim about autonomous mating-law discovery.

### Fix 1: phenotype canonicalization

If gate g_i = 0:

the corresponding coefficient is forced to exactly zero.

Bank identity uses only:

- canonical effective coefficients;
- gates.

mutation_sigma is excluded from phenotype identity.

Therefore equivalent formulas cannot occupy multiple bank slots.

The center law can occupy at most one phenotype slot.

A separate genotype_fingerprint still records mutation sigma differences for diagnostics.

### Fix 2: correct formula display

Actual implementation uses:

(H_A - H_B) / 4

V1.14.1 prints this normalization explicitly.

### Fix 3: evidence-aware rule ranking

Mature rule ranking uses a lower posterior quantile rather than a point estimate.

Default quantile:

q = 0.10

Examples:

screen yield posterior:
Beta(1 + selected, 3 + generated - selected)

acceptance yield:
Beta(1 + accepted, 9 + generated - accepted)

four-capability yield:
Beta(0.5 + accepted_4cap, 9.5 + generated - accepted_4cap)

Retention is handled as a bounded fractional-success process.

### Fix 4: minimum evidence

Formal specialist requirement:

generated >= 32

Immature rules remain provisional and may still be explored.

### Fix 5: exploration reserve

Formal adaptive bank:

- bank limit 32
- exploration slots 8
- specialist size per axis 6
- Pareto limit 12
- uniform rule exploration 25%

### Fix 6: separate random streams

policy_rng = seed

rule_rng = seed + 114100003

This prevents adaptive rule mutation from merely shifting the normal Policy mutation random stream.

### Fix 7: paired control

Two conditions start from the exact same V1.13 checkpoint.

Adaptive:
Policy mutation + self-evolving Recombination Gene Bank.

Center:
Policy mutation + fixed all-gates-off center law.

Both use:

- same Policy architecture;
- same Policy parameter count 148;
- same 95M worlds;
- same Policy mutation schedule;
- same number of normal children;
- same number of mating children;
- same parent selection;
- same seed.

Comparison is per Policy axis only:

Adaptive minus Center for:

- mean_time
- tail10_time
- continuation_preservation
- fleet_option_reserve

No weighted aggregate comparison is created.

Primary document:
docs/GENE_HOMOGENEOUS_MRTA_V1141_RECOMBINATION_CORRECTION.md


### V1.14.1 paired-smoke result

Adaptive smoke:

- Gen0: Rbank 10, Rpareto 0, Rmature 0, Rcenter 1.
- Gen1: Rbank 12, Rpareto 1, Rmature 1, Rcenter 1.
- Gen2: Rbank 12, Rpareto 2, Rmature 2, Rcenter 1.
- final Policy best:
  - mean_time 0.977920253212843
  - tail10_time 0.9262063481487696
  - continuation_preservation 0.8328795356499624
  - fleet_option_reserve 0.8757350433050963
- final adaptive smoke Pareto size = 2.

Center smoke:

- Rbank = 1 throughout.
- Rcenter = 1 throughout.
- final Policy best exactly matched adaptive smoke.
- final Pareto size = 1.

Therefore the phenotype-canonical structural invariants passed.

### V1.14.1 formal paired50 result

Common bootstrap:

runs/gene_mrta_v113_route_tail_evolution/gene_mrta_v113_route_tail_20261003_203157_seed7/checkpoint.json

Scenario bank:

runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json

Adaptive:

runs/gene_mrta_v1141_paired_seed7/adaptive/gene_mrta_v1141_recombination_20261003_231301_seed7

Center:

runs/gene_mrta_v1141_paired_seed7/center/gene_mrta_v1141_recombination_20261004_013416_seed7

Comparison:

runs/gene_mrta_v1141_paired_seed7/comparison_50.json

Adaptive final Policy best:

- mean_time = 0.9797388961714149
- tail10_time = 0.9273987323265199
- continuation_preservation = 0.8334576354631884
- fleet_option_reserve = 0.8757350433050963

Center final Policy best:

- mean_time = 0.9795402913092378
- tail10_time = 0.9273987323265199
- continuation_preservation = 0.8345021458732124
- fleet_option_reserve = 0.8762071604639012

Adaptive - Center:

- mean_time = +0.0001986048621770431
- tail10_time = 0
- continuation_preservation = -0.0010445104100239577
- fleet_option_reserve = -0.0004721171588049078

Percentage-point interpretation:

- mean_time approximately +0.01986 pp adaptive;
- tail10 exact tie;
- continuation approximately -0.10445 pp adaptive;
- reserve approximately -0.04721 pp adaptive.

Because the project explicitly keeps these capability axes independent, this result must not be converted into one weighted scalar verdict.

### Recombination-bank finding

Phenotype canonicalization worked:

Rcenter = 1 throughout the adaptive formal run.

The adaptive run developed up to roughly twenty mature rule phenotypes and transient Pareto fronts of size 2–3, but finished with mature Pareto size 1.

All four final mature recombination specialists were:

c7723fa1e0127975e49e

This is the canonical all-gates-off center law.

Its adaptive-arm evidence:

- active terms = 0;
- generated = 884;
- screen_selected = 189;
- accepted = 189;
- four-capability accepted = 183.

Thus the V1.14 center-law dominance survives removal of neutral phenotype clones and low-evidence specialist promotion.

Current interpretation:

- V1.14.1 correction mechanics = PASS.
- Adaptive Recombination Bank implementation = PASS.
- Adaptive > center claim = NOT SUPPORTED for seed=7.
- Non-center mature formula superiority = NOT SUPPORTED for seed=7.

### New control caveat found

The center-only arm uses one deterministic center rule.

The formal launcher still uses:

32 parent pairs x 4 children/pair = 128 mating children.

For a given parent pair, the four center-law children can be identical because the rule is deterministic.

Adaptive mode can generate different rule phenotypes for those four children.

Therefore equal nominal offspring count does not imply equal unique-child search diversity.

This asymmetry tends to handicap the center control, not the adaptive arm.

Despite that handicap, center matched or exceeded adaptive on tail, continuation, and reserve, while adaptive was only slightly higher on mean time.

Before publication-grade claims, the next paired control should match unique-child opportunity, e.g. one deterministic center child per distinct parent pair with enough distinct pairs to reach the same unique candidate budget, or explicit deduplication followed by equalized unique screening budget.

Do not inspect 99M.



## V1.14.2 - Frozen-Parent Matched-Pair Recombination Assay

STATUS AS OF 2026-10-04:

TESTS PASSED (14/14). MATCHED-PAIR SMOKE PASSED. FORMAL128 COMPLETED FOR seed=7.

V1.14.1 left one causal ambiguity: after adaptive and center Policy Banks evolve independently, their parent populations diverge, so later-generation offspring are not born from identical parent pairs.

In addition, deterministic center mating can waste nominal offspring budget on duplicate children when the same parent pair is repeated.

V1.14.2 removes both issues by freezing the parent population.

Frozen sources:

- completed V1.13 Policy checkpoint;
- completed V1.14.1 adaptive Recombination checkpoint;
- frozen V1.8 common ancestor;
- frozen 95M 100-world development bank.

Exactly 128 unique unordered parent pairs are sampled once and saved as a manifest.

For each pair:

Center:
- one canonical center child.

Adaptive:
- one child using one rule sampled from the frozen V1.14.1 adaptive rule bank with the evidence-aware rule-selection protocol.

Both children are evaluated on all 100 worlds.

No Policy mutation, no Policy-bank admission feedback, no generational drift, and no primary 25-world screening.

The paired report records independently:

- mean_time;
- tail10_time;
- continuation_preservation;
- fleet_option_reserve;
- per-axis adaptive-minus-center differences;
- per-axis win/tie/loss;
- dual 0.95 inheritance-gate success;
- four-capability certification;
- adaptive-dominates-center count;
- center-dominates-adaptive count;
- neither-dominates count;
- exact-identical-child count;
- sampled rule usage;
- non-center-rule subset results.

No weighted aggregate score is allowed.

Primary document:

docs/GENE_HOMOGENEOUS_MRTA_V1142_MATCHED_PAIR_ASSAY.md

99M remains untouched.


### V1.14.2 completed smoke

Tests:

14 passed.

Smoke run:

runs/gene_mrta_v1142_matched_pair_smoke/gene_mrta_v1142_matched_pair_20261004_101048_seed7

Smoke design:

- 8 unique unordered frozen parent pairs;
- same pair manifest for adaptive and center;
- one child per pair per condition;
- full 100-world 95M evaluation;
- no Policy mutation or bank drift.

Results:

- non-center adaptive pairs = 6
- identical adaptive/center children = 2

Adaptive minus center:

mean_time:
- delta +0.00027892
- W/T/L 4/3/1

tail10_time:
- delta -0.00015336
- W/T/L 2/4/2

continuation_preservation:
- delta +0.00046004
- W/T/L 4/3/1

fleet_option_reserve:
- delta -0.00052722
- W/T/L 1/3/4

Dominance:
- tie 3
- adaptive 1
- neither 4

Dual gate:
- adaptive 5/8
- center 5/8

Four-capability:
- adaptive 4/8
- center 4/8

Interpretation:

Structural assay PASS.

The smoke is too small for a scientific comparison, but demonstrates that exact-parent matched treatment, non-center adaptive sampling, center-equivalent identical children, symmetric gate evaluation, and direct full100 child evaluation all work.

Next frozen action:

run formal128 with no code/protocol changes.



### V1.14.2 formal128 result

Run:

runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7

Formal matched-parent result:

- pairs = 128
- non-center adaptive pairs = 115
- exact identical children = 19

Adaptive - Center:

mean_time:
- +0.00064842
- W/T/L 47/38/43

tail10_time:
- +0.00104946
- W/T/L 40/54/34

continuation_preservation:
- +0.00029748
- W/T/L 62/29/37

fleet_option_reserve:
- -0.00006791
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

Exploratory exact sign tests on W/L only:

- mean p 0.7520
- tail p 0.5614
- continuation p 0.01543
- reserve p 0.6849

With four tested Policy axes, continuation does not survive a simple Bonferroni 0.05 threshold (adjusted approximately 0.0617).

Interpretation:

Adaptive shows a reproducible directional continuation advantage and small positive mean/tail changes, but not universal Pareto superiority. Reserve is nearly neutral/slightly negative.

Adaptive also produced 3 additional dual-gate children and 3 additional four-capability children.

This supports the narrower statement that learned non-center mating can help some same-parent offspring, especially continuation, while preserving multi-capability fusion.

It does not support a claim that the adaptive rule bank is globally superior.

Next frozen action:

analyze pair_results.jsonl by adaptive rule phenotype and parent capability context before changing the formula family or touching 99M.


---

# 6. Current code architecture

Current Policy architecture:

- pair observation: 12 dimensions;
- hidden_dim = 8;
- parameters = 148;
- shared homogeneous Policy Gene;
- autoregressive direct robot-task selection;
- learned STOP/WAIT;
- route-tail append semantics;
- one robot may receive multiple ordered tasks in a single planning call;
- no external matcher;
- no oracle action labels.

Current Recombination architecture in V1.14.1:

- seven continuous equation coefficients;
- seven binary structural gates;
- one self-adaptive sigma;
- phenotype canonicalization;
- evidence-aware posterior lower-bound selection;
- mature specialist threshold;
- Pareto preservation;
- provisional exploration slots;
- adaptive versus center-only paired control.

---

# 7. Important run directories

Frozen V1.8:

runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7

Completed V1.10:

runs/gene_mrta_v110_mating/gene_mrta_v110_mating_20261003_163423_seed7

Completed V1.13:

runs/gene_mrta_v113_route_tail_evolution/gene_mrta_v113_route_tail_20261003_203157_seed7

Completed V1.14:

runs/gene_mrta_v114_self_recombination/gene_mrta_v114_self_recombination_20261003_214207_seed7

Current V1.14.1 paired output root when run with seed 7:

runs/gene_mrta_v1141_paired_seed7/

Expected files after paired50:

adaptive/.../summary.json
center/.../summary.json
comparison_50.json

---

# 8. Current commands

Before running any V1.14.1 experiment:

git pull

Tests:

bash tools/run_gene_mrta_v1141_paired_mac.sh tests

Paired structural smoke:

bash tools/run_gene_mrta_v1141_paired_mac.sh paired-smoke

Only if paired-smoke is structurally correct:

bash tools/run_gene_mrta_v1141_paired_mac.sh paired50

Expected adaptive smoke invariants:

Rcenter <= 1

Expected center-only invariants:

Rbank = 1
Rcenter = 1

Formal V1.14.1 pilot invariants:

- 99M untouched;
- same V1.13 checkpoint in both arms;
- same Policy seed;
- separate policy_rng/rule_rng;
- same Policy mutation budget;
- no scalarized Policy or recombination reward.

---

# 9. Current unanswered scientific question

The seed=7 V1.14.1 paired experiment did not demonstrate a consistent adaptive advantage over the canonical center law.

The immediate unresolved question is now whether this remains true after matching UNIQUE child opportunity between the adaptive and deterministic-center arms.

Preferred next experiment:

- preserve the V1.13 bootstrap;
- preserve the frozen 95M development bank;
- preserve Policy architecture and all four capability axes;
- preserve separate policy_rng/rule_rng;
- make center and adaptive arms receive equal unique Policy-child opportunity before screening;
- compare four axes separately;
- keep 99M untouched.

Do not expand the formula grammar or move to symbolic expression trees until this remaining control asymmetry is resolved.

---

# 10. Planned interpretation after V1.14.1 paired50

For each Policy axis compute:

Delta_a
=
S_a(adaptive)
-
S_a(center)

Inspect:

- mean_time
- tail10_time
- continuation_preservation
- fleet_option_reserve

Also inspect the adaptive mature Recombination specialists:

- phenotype_id;
- active term count;
- gates;
- effective coefficients;
- mutation sigma;
- generated trials;
- screen selections;
- accepted children;
- four-capability accepted children;
- posterior lower-bound evidence score;
- complete equation.

Single-seed positive differences are descriptive only.

If promising:

1. repeat paired experiment across multiple fixed seeds;
2. keep the protocol frozen;
3. create/use an independent 96M development-validation bank;
4. only after procedure/candidate freeze evaluate 99M once.

---

# 11. Known pitfalls

## Pitfall A: capability tag drift

Declared capability labels are not proof of current capability.

Always report current certification relative to current active ceiling.

## Pitfall B: parent-relative decay

Do not admit a child merely because it retains 95% of a weaker parent.

V1.10+ dual gate requires both parent retention and generation-ceiling retention.

## Pitfall C: operator statistics

Generated -> accepted is different from selected -> accepted.

Do not confuse screen effectiveness with raw operator quality.

## Pitfall D: center-law neutral clones

Fixed in V1.14.1 through phenotype canonicalization.

## Pitfall E: lucky low-n specialists

Fixed in V1.14.1 through:

- minimum evidence threshold;
- posterior lower-bound scoring.

## Pitfall F: RNG contamination in paired controls

Fixed in V1.14.1 by separating policy_rng and rule_rng.

## Pitfall G: route length as fake improvement

Queue depth is diagnostic only.

V1.13 improvement was valuable because mean/tail improved while qmean/qmax remained approximately stable.

## Pitfall H: final benchmark contamination

Never inspect 99M during development.

---

# 12. Version-specific source locations

Policy / environment history:

src/marl2d/gene_mrta_v16t/
src/marl2d/gene_mrta_v17/
src/marl2d/gene_mrta_v18/
src/marl2d/gene_mrta_v19/

Mating:

src/marl2d/gene_mrta_v110/

Law discovery:

src/marl2d/gene_mrta_v111/

Route-tail multi-task:

src/marl2d/gene_mrta_v113/

First self-evolving recombination:

src/marl2d/gene_mrta_v114/

Current corrected self-evolving recombination:

src/marl2d/gene_mrta_v1141/

Current paired launcher:

tools/run_gene_mrta_v1141_paired_mac.sh

---

# 13. Handoff checklist for a new conversation

A new conversation should be able to continue the experiment if it knows only this repository.

Before modifying code:

1. Read AI_PROJECT_CONTEXT.md.
2. Read this ledger.
3. Read docs/GENE_HOMOGENEOUS_MRTA_V1141_RECOMBINATION_CORRECTION.md.
4. Check the active branch is experiment/gene-homogeneous-mrta-v1.
5. V1.14.1 paired-smoke and paired50 are completed for seed=7.
6. The result does not show a consistent adaptive advantage over center.
7. The next control should equalize unique-child opportunity before multi-seed replication.
8. Record the new control design in this ledger/context before implementation.
9. Never use 99M during this development step.

This documentation update rule is now part of the experimental workflow:

Every completed experiment or architecture-changing decision must update:

- AI_PROJECT_CONTEXT.md
- docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md
- the relevant version-specific document

before beginning the next experimental version.


## V1.14.3 - Offline Rule / Parent-Context Analysis

STATUS AS OF 2026-10-04:

COMPLETED. V1.14 SERIES FROZEN FOR NOW.

This is the final analysis step before the project moves from mating research to fleet/task scalability testing.

No new Policy or Recombination training occurs.

Input:

completed V1.14.2 formal128 pair_results.jsonl.

Outputs:

- v1143_rule_context_analysis.json
- v1143_rule_table.csv
- v1143_parent_context_table.csv
- v1143_rule_context_table.csv

Questions:

- which non-center rule IDs produce continuation gains;
- which rules rescue four-capability certification relative to center;
- which parent capability contexts favor non-center mating;
- which rule/context combinations are repeatable enough to justify a later context-conditioned selection hypothesis;
- why some non-center rules collapse to the exact same Policy child as center.

Command:

bash tools/run_gene_mrta_v1143_analysis_mac.sh

Guardrail:

this reuses 95M development evidence and may generate hypotheses only.

After interpretation, freeze the V1.14 research line for now.

Primary document:

docs/GENE_HOMOGENEOUS_MRTA_V1143_RULE_CONTEXT_ANALYSIS.md

## V1.15 - Scalability / Extreme Fleet and Task Stress Test

STATUS AS OF 2026-10-04:

V1.15A EXTREME LADDER COMPLETED. FULL-SYSTEM LIMIT IS A* PATH PRECOMPUTATION BETWEEN 64R/320T AND 128R/640T.

User objective:

increase robot count and task count substantially and test the practical limit of the algorithm.

Primary constant-ratio ladder:

- 4R/20T
- 8R/40T
- 16R/80T
- 32R/160T
- 64R/320T
- 128R/640T if feasible

Task-dense follow-up:

- 8R/80T
- 16R/160T
- 32R/320T
- 64R/640T if feasible

The first stage is zero-shot with one frozen mature four-capability Policy Gene.

Do not retrain per scale before measuring zero-shot size generalization.

Measure separately:

- world generation;
- obstacle/A* path preprocessing;
- path-table memory;
- Policy allocation latency;
- Policy decoder steps;
- peak memory;
- completion;
- raw time utility;
- continuation;
- reserve;
- queue depth and assignment coverage.

Map area and obstacle count should scale to preserve spatial density.

The benchmark must distinguish:

computational failure
vs
behavioral failure.

If path preprocessing fails first, do not mislabel it as a Policy limit.

No exact MILP score is required at sizes where exact MILP is no longer practical.

Primary document:

docs/GENE_HOMOGENEOUS_MRTA_V115_SCALING_STRESS_PLAN.md

99M remains protected.


### V1.14.3 completed result

Tests: 3 passed.

Source:
runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7

Counts:
- 128 source pairs
- 115 non-center pairs
- 3 four-cap rescues
- 0 four-cap losses
- 8 candidate rule/context groups
- 6 identical non-center children

Non-center Adaptive - Center:
- mean +0.00072172, 47/25/43
- tail +0.00116809, 40/41/34
- continuation +0.00033111, 62/16/37
- reserve -0.00007559, 51/18/46

Leading descriptive rule:
6858fcfc2c9540c2f0fd, n=18, active terms=2,
continuation +0.0004531036432003304, W/T/L 11/1/6,
four-cap rescue-minus-loss +1.

Decision:
V1.14 recombination series is frozen for now. No further 95M formula tuning.
Proceed to V1.15 scalability stress testing.



### V1.15A implementation

Full-system zero-shot scaling benchmark implemented.

Code:
src/marl2d/gene_mrta_v115/scaling.py

Launcher:
tools/run_gene_mrta_v115_scaling_mac.sh

Tests:
tests/test_gene_mrta_v115_scaling.py

The benchmark freezes one mature four-capability V1.13 Policy Gene and measures
the same 148-parameter route-tail Policy without retraining as R/T increase.

Initial smoke:
4R/20T, 8R/40T, 16R/80T.

Extreme ladder:
4R/20T -> 8R/40T -> 16R/80T -> 32R/160T -> 64R/320T -> 128R/640T.

Failure stage is recorded separately as geometry/path/policy.

Outputs include:
scaling_results.csv, scaling_case_summary.csv, timing_breakdown.csv,
memory_breakdown.csv, scaling_summary.json.



### V1.15A initial zero-shot smoke result

Tests: 4 passed.

Run:
runs/gene_mrta_v115_scaling/gene_mrta_v115_scaling_20261004_121633_seed115000000

Frozen Policy Gene:
7ee7c18fca2280022ac5

Parameter count:
148

4R/20T:
path 0.0310 s, policy 0.00277 s, completion 0.35,
time utility 0.18348, continuation 0.82888, reserve 0.86209.

8R/40T:
path 0.1778 s, policy 0.01656 s, completion 0.525,
time utility 0.27173, continuation 0.90578, reserve 0.94628.

16R/80T:
path 1.27874 s, policy 0.04424 s, completion 0.375,
time utility 0.19886, continuation 0.90529, reserve 0.94504.

Conclusion:
structural zero-shot scaling PASS through 16R/80T.
A* path preprocessing is already the dominant runtime bottleneck.
One world per scale is diagnostic only, not behavioral evidence.

Next frozen action:
run ladder3 unchanged.



### V1.15A extreme ladder result

Run:
runs/gene_mrta_v115_scaling/gene_mrta_v115_scaling_20261004_121902_seed115010000

Frozen Gene:
7ee7c18fca2280022ac5

Successful:
4R/20T, 8R/40T, 16R/80T, 32R/160T, 64R/320T all 3/3.

128R/640T:
0/3, all failed in path preprocessing after 300 s; Policy never ran.

Mean path preprocessing:
0.02538, 0.19041, 1.12575, 7.54657, 55.20587 s.

Mean Policy:
0.00334, 0.01250, 0.04817, 0.24790, 1.24652 s.

Mean completion:
0.4000, 0.4083, 0.3958, 0.4104, 0.4000.

Mean raw time utility:
0.1929, 0.2140, 0.2099, 0.2162, 0.2129.

Mean queue depth stays approximately 2.

Empirical timing exponents:
path ~ T^2.75, Policy ~ T^2.14 over measured range.

Conclusion:
full-system bottleneck = obstacle-aware A* preprocessing.
Behavioral zero-shot scaling remains stable through 64R/320T on this three-world diagnostic.
Proceed to V1.15B Policy-only scaling rather than increasing A* timeout.



### V1.15B Policy-only scaling implementation

V1.15A established the full-system limit:

- 64R/320T succeeds 3/3;
- 128R/640T fails 3/3 during obstacle-aware A* path preprocessing;
- Policy inference is never reached at 128R/640T.

V1.15B is implemented to isolate the route-tail Policy decoder.

It replaces obstacle-A* preprocessing with a dense vectorized Euclidean
robot/task + task/task distance table while keeping:

- the same frozen mature V1.13 Policy Gene;
- 148 Policy parameters;
- the same autoregressive route-tail decoder;
- no scale-specific retraining.

Code:
src/marl2d/gene_mrta_v115/policy_only.py

Tests:
tests/test_gene_mrta_v115b_policy_only.py

Launcher:
tools/run_gene_mrta_v115b_policy_only_mac.sh

Planned Policy-only ladder:
64R/320T -> 128R/640T -> 256R/1280T -> 512R/2560T.

Optional extreme diagnostic:
1024R/5120T if the previous scale remains tractable.

V1.15B is a compute-isolation experiment; behavioral scores are not directly
comparable with obstacle-aware V1.15A.

99M remains untouched.



### V1.15B policy-only smoke result

Tests:
7 passed.

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_141132_seed115100000

Frozen Policy:
7ee7c18fca2280022ac5

Parameters:
148

64R/320T:
- Euclidean table 0.002422 s
- Policy 1.970497 s
- pair slots 2,174,016
- completion 0.41875
- raw time utility 0.215997
- balance 0.387412
- queue mean 2.09375
- RSS peak 130.97 MB

128R/640T:
- Euclidean table 0.001655 s
- Policy 14.836715 s
- pair slots 16,744,320
- completion 0.3984375
- raw time utility 0.209678
- balance 0.367342
- queue mean 1.99219
- RSS peak 209.28 MB

Conclusion:
128R/640T is computationally tractable for the frozen 148-parameter Policy when
A* preprocessing is removed. Therefore the V1.15A 128R/640T full-system failure
is attributable to obstacle-aware path preprocessing, not Policy inference.

64->128:
- initial pair count x4
- decoder steps x1.90
- total pair slots x7.70
- Policy time x7.53
- time per pair slot remains approximately constant

The smoke-reported two-point exponent is 2.91254, consistent with near-cubic
pair-scoring work when R and T scale proportionally and decoder steps scale with T.

Protocol update:
- ladder3 now tests 64/320, 128/640, 256/1280 with 3 worlds each;
- extreme512 tests 512/2560 with one world and 300 s Policy timeout;
- extreme1024 is deferred unless justified.



### V1.15B extreme512 result

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_175225_seed115120000

512R/2560T:
- tests 7 passed before run
- one world
- failure_stage=policy
- Policy timeout 300.0555 s
- Euclidean table build 0.02927 s
- table entries 7,864,320
- table memory 60.0 MB
- RSS peak 961.14 MB
- initial pairs 1,310,720

Interpretation:
the routing preprocessing bottleneck has been removed, so this is the first
clean Policy-side hard limit under the frozen 300 s compute budget.

Known bracket:
128R/640T succeeds; 512R/2560T fails.

Next:
run ladder3 and measure 256R/1280T with three worlds before adding intermediate
scales.



### V1.15B ladder3 result

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_180029_seed115110000

All 64R/320T, 128R/640T, and 256R/1280T cases completed 3/3 worlds.

Three-world mean Policy time:
- 64/320: 1.447824 s
- 128/640: 9.910586 s
- 256/1280: 81.533248 s

Mean completion:
- 0.397917
- 0.406250
- 0.399219

Mean raw time utility:
- 0.212987
- 0.213276
- 0.212191

Mean queue depth:
- 1.98958
- 2.03125
- 1.99609

Mean pair slots:
- 2.0925M
- 16.9868M
- 134.0823M

Empirical Policy exponent:
2.9077150801970753.

Conclusion:
zero-shot behavior remains stable through 256R/1280T while compute approaches
near-cubic scaling from repeated pair rescoring.

Combined with the prior 512R/2560T 300 s timeout, the current Policy-only
300 s limit lies between 256R/1280T and 512R/2560T.

Next probe:
384R/1920T, one world, 300 s timeout.



### V1.15B probe384 result

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_191351_seed115140000

384R/1920T:
- success
- Policy 296.5072 s
- decoder steps 758
- pair slots 448,687,488
- completion 0.394792
- raw time utility 0.211622
- balance 0.364720
- queue mean 1.97396
- RSS peak 1038.42 MB
- Euclidean table build 0.02464 s

This is only 3.4928 s below the frozen 300 s Policy timeout.

Updated descriptive scaling exponent using 64/128/256/384:
~2.9707.

Interpretation:
384R/1920T is the largest measured successful Policy-only scale and is
effectively at the practical 300 s compute wall.

Next:
probe388 = 388R/1940T, one world, same 300 s timeout.

After threshold localization, move to decoder-compute optimization rather than
increasing Policy size.



### V1.15B probe388 and V1.15 closeout

Run:
runs/gene_mrta_v115b_policy_only/gene_mrta_v115b_policy_only_20261004_193511_seed115150000

388R/1940T:
- failure_stage=policy
- Policy timeout 300.0242 s
- Euclidean table 0.02346 s
- path table 34.46 MB
- RSS peak 985.81 MB
- initial pairs 752,720

Largest observed successful Policy-only case:
384R/1920T at 296.5072 s.

Because 384 and 388 use different single-world seeds, this is an observed
practical wall rather than an exact deterministic threshold.

V1.15 final conclusions:
- full-system limit: A* preprocessing blocks 128R/640T;
- Policy-only behavior remains stable through 384R/1920T;
- Policy-only 300 s compute wall is observed near 384-388 robots at 5 tasks/robot;
- current Policy inference approaches cubic scaling because of repeated pair
  rescoring;
- 148 Policy parameters are not the bottleneck.

V1.15 is frozen.
Do not chase 385/386/387.

Next planned architecture study:
V1.16 Efficient Route-Tail Decoder, preserving the frozen Gene while reducing
inference dataflow/computation.



### V1.16 MILP vs Policy scaling plan

V1.15 is frozen.

Next user-requested phase:
compare the current frozen 148-parameter route-tail Gene directly against the
existing exact MILP on identical worlds.

Measure simultaneously:
- exact objective gap when MILP proves optimal;
- Policy and MILP compute time;
- MILP proof rate;
- problem-size growth;
- practical bottleneck scale.

Frozen Gene:
7ee7c18fca2280022ac5.

Objective:
T=(1/N) sum_completed (1-F_j/H).

New seeds:
116M namespace.

Initial smoke:
2R/10T, 3R/15T, 4R/20T.

No training occurs.



### V1.16 implementation

Implemented, not yet run.

Code:
src/marl2d/gene_mrta_v116/milp_policy_scaling.py

Launcher:
tools/run_gene_mrta_v116_milp_policy_mac.sh

Tests:
tests/test_gene_mrta_v116_milp_policy_scaling.py

The benchmark records exact optimality gap only for MILP-proven worlds and
uses dual-bound retention lower bounds for non-proven worlds.

First smoke:
2R/10T, 3R/15T, 4R/20T, one world each, 60 s MILP budget.



### V1.16 smoke and ladder result

Tests:
5 passed.

Ladder:
2R/10T through 6R/30T, three worlds per scale, frozen 300 s MILP budget.

2R/10T:
MILP 3/3 exact, Policy retention 0.9911, Policy 1.38 ms, MILP 0.195 s mean.

3R/15T:
MILP 3/3 exact, Policy retention 0.9466, Policy 2.73 ms, MILP 7.90 s mean.

4R/20T:
MILP 3/3 exact, Policy retention 0.9774, Policy 3.39 ms, MILP 20.81 s mean.

5R/25T:
MILP 3/3 exact, Policy retention 0.9421, Policy 4.70 ms, MILP 41.55 s mean.
One world required 112.50 s.

6R/30T:
MILP 0/3 exact; all reached 300 s.
Policy mean 10.13 ms.
Mean MILP/Policy ratio 30,883.87x.
Policy remained approximately 98.78% of the best MILP incumbent on average,
but this is not an exact-optimality claim.

Conclusion:
observed MILP proof wall = between 5R/25T and 6R/30T under the frozen 300 s budget.

Next:
boundary5, five new worlds each at 4R/20T, 5R/25T, 6R/30T.



## V1.16 exact-unlimited extension

STATUS:

IMPLEMENTED, NOT YET RUN TO COMPLETION.

After the 6R/30T ladder produced 0/3 optimal proofs under the frozen 300 s
budget, the user explicitly decided that MILP must not stop merely because the
online/practical budget is exceeded. The exact reference should continue until
global optimality is actually proved.

The practical 300 s result is still retained as a runtime result. The new
unlimited experiment serves a different purpose: establish the true capability
ceiling T* for selected hard worlds.

Implemented rules:

- `time_limit=None` means no HiGHS time-limit option is supplied;
- exact relative gap / retention still require `status == optimal`;
- `mip_rel_gap=0.0`;
- no precision relaxation;
- no incumbent-as-optimum interpretation;
- native solver output can be enabled;
- independent elapsed-time heartbeat confirms the process is alive;
- completed worlds are fsynced to JSONL immediately;
- fixed run directories support resume;
- exact resume skips only rows already proven optimal.

Regression launcher:

```
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-regression
```

Hard-world launcher:

```
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

Default hard world:

- 6R/30T
- seed 116050000
- 1 world initially
- no MILP time limit
- 30 s heartbeat
- HiGHS display enabled
- frozen Gene 7ee7c18fca2280022ac5
- 148 Policy parameters

The first goal is to obtain one real 6R/30T global optimum. Only after that
result is understood should more seeds or 7R/35T / 8R/40T unlimited exact
experiments be considered.


## V1.16 first 6R/30T exact-unlimited result

Status:

FIRST HARD EXACT WORLD COMPLETED.

Run:

runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000

Pre-run tests:

9 passed.

World:

6R/30T, seed 116050000.

Exact result:

- HiGHS status Optimal;
- T* = 0.28074964073648145;
- Policy T = 0.2756685668779121;
- absolute gap = 0.005081073858569374;
- relative gap = 0.018098238149977172;
- retention = 0.9819017618500229;
- both methods complete 14 tasks.

Runtime:

- Policy = 0.008538166999642272 s;
- MILP solver = 717.154159042002 s;
- MILP total = 717.1704191659992 s;
- MILP / Policy = 83995.82945567204x.

Search effort:

- 276137 branch-and-bound nodes;
- 8274628 LP iterations;
- peak RSS 290.953125 MB.

Key interpretation:

The final incumbent value was found at about 138.5 s, but the dual bound did
not close to zero gap until about 717.1 s.

Thus the main cost was proving optimality rather than finding the final best
solution.

This validates the distinction between:

1. practical MILP usefulness under a fixed 300 s budget; and
2. offline exact-oracle certification with unlimited solve time.

At 6R/30T the Policy is millisecond-scale and within 1.81% of the proven
global optimum for this first hard world, while exact certification takes
about 11.95 minutes.

This is n=1 exact evidence only. The next frozen action is to run the remaining
two worlds in the same 6R/30T namespace via resume before considering 7R/35T or
8R/40T.


## V1.16 6R/30T multi-seed worst-case exact protocol

The next exact study will use multiple seeds at fixed 6R/30T.

Goal:

quantify the largest observed Policy-vs-global-optimum error and the slowest
exact MILP proof, not only the mean.

For each seed s:

    relative_gap_s = (T*_s - T_policy,s) / T*_s
    retention_s = T_policy,s / T*_s

Report:

- mean exact retention;
- minimum exact retention;
- mean exact relative gap;
- maximum exact relative gap;
- seed with maximum exact relative gap;
- mean / median / maximum MILP total time;
- seed with maximum MILP time;
- mean / median / maximum Policy time;
- exact proof rate.

Initial batch size:

10 seeds, 116050000 through 116050009.

The already completed seed 116050000 is reused and resume-skipped rather than
recomputed.

Command:

    V116_EXACT_WORLDS=10 bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited

Interpretation rule:

maximum observed gap over 10 seeds is an empirical worst case, not a guaranteed
global worst-case bound over all possible worlds.


## V1.16 6R/30T 10-seed exact benchmark result

STATUS:

COMPLETED.

Run:

runs/gene_mrta_v116_exact_unlimited_6r30_seed116050000

Seeds:

116050000..116050009.

Pre-run tests:

10 passed.

Exact proof:

10/10 worlds proven globally optimal.

Quality:

- mean retention = 0.9694540924627978;
- mean exact relative gap = 0.030545907537202176;
- mean exact absolute gap = 0.006215299617823469;
- minimum retention = 0.9152010318146683;
- maximum observed relative gap = 0.08479896818533172;
- maximum observed absolute gap = 0.017600480877557306;
- worst-gap seed = 116050009.

Worst-gap seed 116050009:

- T_policy = 0.18995488511570233;
- T* = 0.20755536599325963;
- Policy completed 11 tasks;
- MILP completed 13 tasks;
- retention = 91.520103%;
- gap = 8.479897%.

Runtime:

- mean Policy = 7.9796 ms;
- median Policy = 8.5153 ms;
- max Policy = 11.5070 ms;
- mean MILP = 453.4836 s = 7.558 min;
- median MILP = 231.5171 s = 3.859 min;
- max MILP = 1734.6120 s = 28.910 min;
- slowest MILP seed = 116050002;
- mean per-world MILP/Policy ratio = 47,884.56x.

Five of ten MILP worlds exceed 300 s exact-proof time.

Observed MILP exact runtime spans approximately 2.48 s to 1734.61 s for the
same 6R/30T problem dimension, confirming strong instance-dependent hardness.

Policy quality frequencies:

- 10/10 retention >= 90%;
- 9/10 retention >= 95%;
- 5/10 retention >= 98%;
- 2/10 retention >= 99%;
- 9/10 gap <= 5%.

Interpretation:

The frozen 148-parameter Policy preserves about 96.95% of the true optimum on
average and at least 91.52% on all 10 tested worlds while remaining under 12 ms.

This supports a strong practical quality/runtime result, but the 8.48% maximum
must be described as the maximum observed gap among 10 seeds, not a theoretical
worst-case bound.

Special case:

seed 116050005 has Policy completed_tasks=13 versus MILP completed_tasks=12,
while the Policy time objective remains lower. This is not a contradiction:
the exact oracle optimizes T=(1/N) sum(1-F_j/H), not completed-task count alone.


## V1.16 5R/25T exact plotting protocol

A matched 10-seed exact-unlimited 5R/25T dataset will be generated for direct
comparison with the completed 6R/30T 10-seed benchmark.

Command:

    bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-5r25

Default seed namespace:

116040000 through 116040009.

The same frozen 148-parameter V1.13 Gene, world generation rules, A* path
preprocessing, time objective, exact MILP semantics, and summary metrics are
used.

Primary plotting metrics:

- per-seed exact retention;
- per-seed exact relative gap;
- Policy solve time;
- MILP total exact-proof time;
- MILP/Policy runtime ratio;
- mean/median/max MILP time;
- mean/min retention and max observed exact gap.


## V1.17 structured two-stage retraining

Status:

IMPLEMENTATION STARTED.

Research structure:

Stage A:
train a clean 148-parameter Route-Tail model from complete known task semantics
using six independent base capability archives.

Axes:

- completion;
- time_retention;
- path_efficiency;
- priority_satisfaction;
- deadline_satisfaction;
- workload_balance.

Stage B:
after Stage A is frozen, identify hard worlds and add only robustness axes
supported by repeated failure mechanisms.

No old V1.13 capability labels are imported into this new line.

Static Task data is frozen as:

    {task_id, position, service_time, priority, deadline}

Static Robot data is frozen as:

    {robot_id, start_position, initial_battery}

A* paths, competition, future reachability, opportunity cost, residual battery,
and route-tail state remain derived observations.

This explicitly separates:

- task/world data;
- Policy observation;
- external capability evaluation.

The first code layer and tests are now present under gene_mrta_v117.


## V1.17 Stage-A global objective specialists

The Stage-A archive definition has been refined.

Two axes now use exact per-world capability ceilings:

    global_time_optimality = T_gene / T_star

and

    global_priority_optimality = P_gene / P_star

T_star and P_star are globally proven MILP optima for different objectives.
MILP provides only the objective ceilings/routes for verification and never
supplies action labels to the Gene.

This design intentionally allows the bank to retain distinct specialists:

- a Gene that is globally strong on completion-time utility;
- a Gene that is globally strong on completed priority.

The remaining base archives are completion, path efficiency, deadline
satisfaction, and workload balance.

This preserves conflicts instead of hiding them inside a weighted scalar sum.


## V1.17 Stage-A executable protocol

The clean Stage-A evolutionary trainer is implemented.

Random initialization:

148 parameters, hidden_dim=8, Route-Tail decoder, no prior checkpoint.

Specialists:

- completion;
- global_time_optimality;
- path_efficiency;
- global_priority_optimality;
- deadline_satisfaction;
- workload_balance.

Evolution:

- normal mutation uses weakest-capability q^2 parent pressure;
- mating uses complementary capability parents and q^10 pressure;
- 5% uniform exploration is retained;
- candidate screening precedes full-world evaluation;
- capability inheritance requires 95% retention versus relevant parent
  capability and 95% versus the current capability ceiling.

Smoke:

    bash tools/run_gene_mrta_v117_mac.sh smoke

Formal training is intentionally deferred until smoke behavior is inspected.


## V1.17 Stage-A smoke #1

Run:

runs/gene_mrta_v117_stage_a/gene_mrta_v117_stage_a_20261005_085638_seed117

Tests:

7 passed.

Oracle smoke:

2R/10T, seeds 117000000 and 117000001.

Exact time / priority ceilings:

- seed 117000000:
  - T* = 0.2242209550768175
  - P* = 0.45197901256124035
  - both exact objectives complete 4 tasks
- seed 117000001:
  - T* = 0.19496278602896533
  - P* = 0.5098102611285834
  - both exact objectives complete 4 tasks

Stage-A observations over 5 generations:

- global-time specialist and global-priority specialist were different Gene IDs
  in every logged generation;
- best global_time_optimality = 0.9905165213414664 from generation 0;
- best global_priority_optimality improved from 0.9100312097338881 to
  0.9673984386344753;
- path_efficiency improved from 0.33681726591232225 to 0.33990888335670366;
- max archive-membership capability count increased from 4 to 5.

Important interpretation:

The smoke validates objective-specialist separation and the basic evolutionary
wiring. However, max_capabilities alone does not prove mating-based fusion,
because a normal Gene may simultaneously rank inside several specialist
archives.

Two design issues were identified before formal training:

1. raw completion = 0.4 cannot distinguish a weak Gene from a world whose exact
   maximum feasible completion is only 4/10;
2. old ancestor-delta / TIES / DARE recombination assumes a shared trained
   ancestor and is not justified for a random-from-scratch V1.17 population.

Corrections implemented after smoke #1:

- add exact C* completion oracle;
- replace completion axis with global_completion_optimality = C/C*;
- normalize workload balance using completion retention times Jain fairness;
- restrict Stage-A clean mating to parameter_blend, block_pick, and block_blend;
- add explicit most-capable Gene origin/operator/parents to generation logs;
- add time_priority_same_specialist diagnostic;
- bump oracle-bank/checkpoint semantics;
- remove repeated pytest execution inside smoke recursion.

A second smoke is required before formal training.


## V1.17 Stage-A smoke #2

Run:

runs/gene_mrta_v117_stage_a/gene_mrta_v117_stage_a_20261005_090202_seed117

Tests:

9 passed.

Oracle smoke:

2R/10T, seeds 117000000 and 117000001.

Exact completion ceilings:

- seed 117000000: C* = 0.4
- seed 117000001: C* = 0.4

This confirms that the previous raw completion score 0.4 was not a weak Policy
result; the tested worlds themselves permit at most 4/10 completed tasks.

Important Stage-A results:

- global_completion_optimality reaches 1.0;
- global_time_optimality reaches 0.9905165 at generation 0 and a proven
  1.0 archive best by generation 1;
- global_priority_optimality reaches 0.9338965;
- Time and Priority specialists remain different Gene IDs in all five logged
  generations;
- a true mating-derived five-capability Gene appears at generation 1.

First true fusion Gene:

record_id = 812e74f49dfb20a82ae8
origin = mating
operator = parameter_blend

Parents:

- cdc2746ad444bce89452
- 3eab704335a0fbf5c989

Certified/archive capabilities:

- deadline_satisfaction
- global_completion_optimality
- global_priority_optimality
- global_time_optimality
- workload_balance

Scores:

- completion retention = 1.0
- time retention = 0.9833370
- priority retention = 0.9338965
- deadline raw score = 0.4
- workload balance = 0.9900686

This is direct evidence that the clean random-lineage crossover can fuse
conflicting specialists without loading a previous Policy checkpoint.

Final pre-formal capability refinement:

The smoke also shows that raw Deadline=0.4 and raw Path Efficiency~=0.339 do
not reveal distance to their world-specific ceilings.

Therefore Stage A now uses exact ceilings for all linear task objectives:

- global_completion_optimality = C / C*
- global_time_optimality = T / T*
- global_path_efficiency = E / E*
- global_priority_optimality = P / P*
- global_deadline_optimality = D / D*

Workload balance remains a structural nonlinear capability:

    completion_retention * Jain(workload)

A fake MILP ceiling is not introduced for Jain fairness.

A third smoke with the full exact-axis bank is required before starting formal
Stage-A training.


## V1.17 Stage-A smoke #3

Run:

runs/gene_mrta_v117_stage_a/gene_mrta_v117_stage_a_20261005_091017_seed117

Tests:

11 passed.

Exact oracle ceilings on both 2R/10T smoke worlds now include all five linear
task objectives:

seed 117000000:
- C* = 0.4
- T* = 0.2242209550768175
- E* = 0.33060942664538956
- P* = 0.45197901256124035
- D* = 0.4

seed 117000001:
- C* = 0.4
- T* = 0.19496278602896533
- E* = 0.3537811670895189
- P* = 0.5098102611285834
- D* = 0.4

Observed Stage-A behavior:

- global_completion_optimality = 1.0;
- global_deadline_optimality = 1.0;
- best global_path_efficiency improves to 0.9906102433109147;
- best global_priority_optimality improves from 0.9100312097338881 to
  0.9723705458253016;
- best global_time_optimality remains 0.9905165213414664 in this random run;
- Time and Priority specialists remain different in every generation.

Important correction to smoke #2 interpretation:

The previous checkpoint/log field named "capabilities" mixed two different
provenance concepts:

1. current specialist archive membership;
2. capabilities explicitly inherited by a mating child after the 95% dual
   inheritance gate.

Therefore origin=mating plus capabilities>1 is not, by itself, sufficient proof
of mating-based capability inheritance. The prior statement that smoke #2
"proved true five-capability fusion" is reclassified as suggestive but not yet
formally proven.

The trainer is now corrected to track separately:

- archive_capabilities;
- inherited_capabilities;
- capabilities = union used for active Gene behavior.

Hybrid retention is based only on inherited_capabilities, not generic archive
overlap.

Generation logs now additionally report:

- max_inherited_capabilities;
- best_fusion_gene;
- inherited_capabilities and archive_capabilities separately.

Two new tests ensure:

- archive membership is not mislabeled as mating inheritance;
- a mating Gene with no inherited capabilities is not retained merely as a
  fusion candidate.

One additional smoke is required before formal Stage-A freeze.


## V1.17 Stage-A freeze gate passed

Final provenance smoke:

runs/gene_mrta_v117_stage_a/gene_mrta_v117_stage_a_20261005_091817_seed117

Tests before the smoke:

13 passed.

The five exact base ceilings remained valid:

- C*
- T*
- E*
- P*
- D*

Observed best capability values during the 5-generation smoke:

- global_completion_optimality = 1.0
- global_time_optimality = 0.9999999999999997 by generation 3
- global_path_efficiency = 0.9930842458608933
- global_priority_optimality = 0.9723705458253018
- global_deadline_optimality = 1.0
- workload_balance = 0.9983580546544989

Time and Priority specialists were distinct in every generation.

Capability provenance is now verified.

Generation 1 reports:

    max_inherited_capabilities = 3

with an explicit best_fusion_gene produced by block_pick and carrying three
inherited axes.

Generation 2 onward retains a mating-derived Gene with inherited:

- global_completion_optimality
- global_deadline_optimality
- global_time_optimality

This confirms genuine inheritance under the 95% dual gate, not archive overlap.

It does not yet demonstrate Time+Priority inheritance in the same child.
That is not required from a 5-generation smoke; the formal run is responsible
for discovering richer capability combinations.

Stage-A capability definitions are now FROZEN:

1. C/C*
2. T/T*
3. E/E*
4. P/P*
5. D/D*
6. (C/C*) * Jain(workload)

No further Stage-A capability-axis changes are planned before the formal run.

Long-run reliability was added before formal execution:

- exact oracle bank resumes at completed-world granularity;
- the bank is atomically fsync-written after every completed world;
- Stage-A training resumes from the latest completed generation;
- RNG state and capability provenance are checkpointed;
- formal training uses a fixed run directory;
- status-formal prints the latest checkpoint summary.

Formal defaults:

- 4R/20T
- 32 exact oracle worlds
- 256 random initial Genes
- 50 generations
- 16 Genes per axis archive
- 128 mutation children/generation
- 32 mating pairs x 4 children
- 8 screen worlds
- 95% capability inheritance threshold


## V1.17 post-Stage-A pipeline

Stage A formal 4R/20T x 32 exact-oracle worlds completed 50 generations.

Final specialist ceilings at generation 49:

- global_completion_optimality = 0.9440577651515143
- global_time_optimality = 0.9848937195581706
- global_path_efficiency = 0.9338499169797052
- global_priority_optimality = 0.942236871706114
- global_deadline_optimality = 0.9214409722222221
- workload_balance = 0.9084818733160203

Time and Priority specialists remained distinct through generation 49.

Maximum inherited capability count briefly reached 5 at generation 2, then the
formal population stabilized around 4-capability hybrids. This is treated as a
capability-fusion bottleneck rather than a reason to redefine Stage-A axes.

The next protocol is frozen as:

1. Fusion-1:
   mating-only consolidation from the final Stage-A bank;
2. Unseen exact evaluation:
   run mature specialists and Fusion-1 hybrids on new 4R/20T worlds;
3. Hard-world mining:
   select worlds where even the best hybrid has a weak worst capability;
4. Hard-world targeted training:
   define additional Stage-B robustness capability only after trace evidence;
5. Fusion-2:
   final mating between Stage-A specialists, Fusion-1 hybrids, and Stage-B
   targeted specialists.

Fusion-1 intentionally disables ordinary mutation and uses only clean
parameter/block crossover operators.

Default Fusion-1:

- 20 mating rounds;
- 64 parent pairs / round;
- 4 children / pair;
- 8-world screening;
- 32 children promoted to full 32-world evaluation;
- 95% full-union inheritance gate;
- no post-mating mutation.

Unseen development bank default:

- 64 new 4R/20T worlds;
- seed namespace 117200000+;
- exact C*, T*, E*, P*, D*;
- resumable oracle generation.

Unseen audit candidate cohort:

- one current specialist per Stage-A axis;
- top 12 Fusion-1 hybrids;
- hard worlds ranked primarily by the best available hybrid's weakest
  capability retention, secondarily by the whole candidate frontier's weakest
  capability.

The hard-world audit exports:

- unseen_audit.json
- hard_world_bank.json

Stage-B targeted capability definitions are deliberately not pre-registered.
They must be justified by the unseen hard-world traces.


## V1.17 Fusion-1 v1 result and correction

Fusion-1 v1 completed 20 mating rounds from the frozen Stage-A bank.

Per round:

- 64 parent pairs;
- 256 children screened;
- 32 children full-evaluated.

Across all 20 rounds:

- 5,120 children screened;
- 640 children full-evaluated;
- accepted_full_union_children = 0 in every round;
- max_inherited_capabilities remained 4;
- Stage-A specialist ceilings did not improve;
- the best retained fusion Gene remained the pre-existing
  Completion + Path + Priority + Balance hybrid.

This is interpreted as a gating/search-path failure, not evidence that
capability fusion is impossible.

The v1 gate required a child to retain the ENTIRE union of both parents at
>=95% of both the strongest relevant parent and the current specialist ceiling.
For mature parents whose union often contains 5-6 capabilities, this creates a
cliff: a child retaining 4 strong capabilities but missing one axis receives no
inherited capability credit and cannot become an intermediate parent.

Fusion-1 v2 therefore keeps the same 95% scientific inheritance standard but
changes fusion to progressive inheritance.

For each axis independently, a child inherits the axis only if:

    child / strongest_parent >= 0.95
    child / current_axis_best >= 0.95

A child with at least two passed axes is retained as a valid fusion
intermediate. Later rounds may mate that intermediate again to accumulate more
capabilities.

Fusion-1 v2 also replaces disruptive 50/50 crossover with parent-preserving
operators:

- sparse_block_graft
- sparse_block_blend
- near_parent_blend

Functional hidden-neuron blocks are transferred one or two at a time, and
reciprocal A<-B / B<-A children are generated.

No ordinary mutation is enabled in Fusion-1 v2.

The v2 run uses a fresh directory:

    runs/gene_mrta_v117_fusion1/formal_4r20t_seed117_v2_progressive

The unseen-map stage should not start until Fusion-1 v2 is inspected.


## V1.17 Pareto Gene Bank architecture

The post-Stage-A Gene Bank no longer uses manually declared capability
combinations as its survival rule.

The six capability axes remain fixed measurements:

- global_completion_optimality
- global_time_optimality
- global_path_efficiency
- global_priority_optimality
- global_deadline_optimality
- workload_balance

A Gene is represented by the six-dimensional capability vector:

    c(g) = [C, T, E, P, D, B]

Gene Bank admission is now pure Pareto dominance.

Gene A dominates Gene B iff:

    A_k >= B_k for every capability k

and:

    A_k > B_k for at least one capability k

A newly evaluated Gene is retained if it remains nondominated after comparison
with the Bank. Any existing Genes dominated by the new Gene are removed.

The Bank does NOT require:

- a named Time+Priority combination;
- a minimum inherited capability count;
- a 95% parent-union inheritance gate;
- a scalar weighted reward;
- a predefined importance ordering between axes.

The previous inherited_capabilities/archive_capabilities fields are retained
only for backward compatibility with the frozen Stage-A checkpoint and lineage
analysis. They do not control Pareto Bank membership.

### Bounded Pareto retention

High-dimensional Pareto fronts can grow large. Storage is bounded without
adding axis weights:

1. exact Pareto dominance removes dominated Genes;
2. epsilon capability cells remove practically duplicate vectors;
3. if still above the Bank limit, NSGA-II-style crowding distance preserves
   capability-space extremes and spread.

Defaults:

    pareto_epsilon = 0.005
    pareto_max_size = 256

The epsilon rule is storage compression only. Within one epsilon cell the
existing representative is kept; no weighted scalar preference is introduced.

### Pareto mating

Fusion-1 now starts from the frozen Stage-A checkpoint, rebuilds its
nondominated Pareto Bank, and mates capability-space-distant parents.

Parent pairing uses only Euclidean distance between six-dimensional capability
vectors. It does not name or prioritize Time, Priority, Deadline, or any other
axis.

Mating still uses parent-preserving parameter operators:

- sparse_block_graft
- sparse_block_blend
- near_parent_blend

Children are screened on a fixed subset of worlds. The screen-level Pareto
front is promoted to full 32-world evaluation. Fully evaluated children are
then submitted directly to the Pareto Gene Bank.

The relevant log fields are now:

- pareto_size
- pareto_inserted_children
- old_pareto_removed
- best_scores
- analysis_best_by_axis
- maximin_gene

analysis_best_by_axis and maximin_gene are views only; they do not affect
membership.

### Unseen-map evaluation

The unseen audit now evaluates the Pareto Bank itself rather than a manually
selected set of named specialists/hybrids.

For each unseen world it records:

- the world-specific Pareto front across Bank Genes;
- the best maximin Gene on that world;
- the weakest capability of that maximin Gene;
- per-axis Bank frontier maxima.

Hard worlds are ranked first by low Bank maximin capability, then by weak Bank
frontier coverage.

This preserves the SEGB principle:

    define how capabilities are measured;
    let the Gene Bank discover which capability combinations survive.


## V1.18 feasibility-first MRTA protocol

V1.17 is frozen as a partial-service/resource-constrained experiment.

V1.18 changes the primary MRTA semantics:

    completing every task is a hard validity requirement,
    not one Pareto capability among several.

The analogy is survival-first robotics: a locomotion policy that falls cannot
claim useful speed or energy performance; likewise an MRTA allocation that
leaves tasks unfinished is not a valid solution regardless of its speed,
path, priority, deadline, or balance values.

### Success gate

For every training/evaluation world:

    completed_tasks == total_tasks

A Gene may enter the formal Pareto Gene Bank only if it satisfies this gate on
every world in the current training bank.

Before the first successful Gene exists, evolution uses a feasibility
bootstrap ordered lexicographically by:

1. worst-world completion;
2. mean completion;
3. number of fully completed worlds.

Completion is therefore a prerequisite/search signal only. It is not a
post-success Pareto objective.

### World protocol

V1.18 world banks accept only worlds for which an exact MILP proves:

    C* = 1.0

Candidate worlds that cannot complete every task under the fixed time, battery,
routing, and service constraints are rejected before training.

The V1.18 4R/20T base configuration increases resource capacity relative to
V1.17 so full completion is physically possible, while exact filtering remains
the final acceptance test.

### Post-success capability vector

The formal Pareto capability vector is:

    [
      global_time_optimality,
      global_path_efficiency,
      global_priority_service,
      global_deadline_optimality,
      workload_balance
    ]

Completion is deliberately absent.

Priority is also redefined. Since all tasks must eventually be completed,
completed priority mass would be constant and therefore meaningless. V1.18
uses priority-weighted completion earliness:

    sum_j p_j * (1 - finish_j / H) / sum_j p_j

so high-priority tasks receive higher score when completed earlier.

Exact all-complete MILP optima are computed independently for:

- time;
- path efficiency;
- priority service;
- deadline satisfaction.

Workload balance remains Jain fairness after the success gate.

### New implementation

- src/marl2d/gene_mrta_v118/oracle.py
- src/marl2d/gene_mrta_v118/oracle_bank.py
- src/marl2d/gene_mrta_v118/capabilities.py
- src/marl2d/gene_mrta_v118/pareto_bank.py
- src/marl2d/gene_mrta_v118/train.py
- tests/test_gene_mrta_v118.py
- tools/run_gene_mrta_v118_mac.sh


## V1.18-fast constructive feasibility protocol

The exact all-complete MILP implementation is retained for validation, but it
is no longer required for every formal training world.

Reason: a 4R/20T exact all-complete objective solve can take tens of minutes
for a single seed, making a 100-world development bank impractical.

### Constructive feasibility proof

A formal world is now accepted only when the generator constructs and stores an
explicit 20/20 witness route.

The witness:

- assigns every task exactly once;
- respects the episode horizon;
- respects travel-energy limits;
- stores the routes, finish times, travel energy, and resulting robot batteries.

Because completion is upper-bounded by 1 and an explicit feasible 20/20 route
exists, the witness itself proves:

    C* = 1

No completion MILP is needed.

Candidate geometries for which the deterministic witness constructor cannot
produce a valid route are rejected cheaply and the generator advances to the
next seed.

The formal seed family starts at 117100000 so the first worlds share the same
4R/20T geometry seed family as V1.17, while V1.18 still uses different service,
deadline, horizon, and battery semantics.

### Raw Pareto capabilities

Formal V1.18-fast training uses raw [0,1] capabilities:

    [
      time_earliness,
      path_efficiency,
      priority_service,
      deadline_satisfaction,
      workload_balance
    ]

Completion remains a hard feasibility gate and is not a Pareto axis.

These are not called global-optimality ratios. Exact per-world normalization is
removed from the training loop.

This is a deliberate protocol change. Per-world oracle normalization can alter
the relative weighting of worlds after aggregation, so V1.18-fast does not
claim numerical equivalence to the earlier exact-normalized protocol.

The rationale is that each retained capability already has an interpretable
[0,1] scale, while the SEGB Pareto mechanism requires consistent capability
measurements rather than an exact combinatorial optimum for every training
world.

### Exact validation

The exact all-complete solver remains in:

    src/marl2d/gene_mrta_v118/oracle.py
    src/marl2d/gene_mrta_v118/oracle_bank.py

and is used only for:

- small smoke correctness tests;
- a limited validation subset for publication diagnostics;
- optional comparison between raw/reference performance and exact optima.

It is not part of 100-world formal-bank construction.

### Fast implementation

    src/marl2d/gene_mrta_v118/world_bank.py

Default formal bank:

    100 worlds
    4 robots
    20 tasks
    seed family 117100000+
    constructive 20/20 witness

Launcher:

    bash tools/run_gene_mrta_v118_mac.sh world-formal
    bash tools/run_gene_mrta_v118_mac.sh train-formal

The old exact formal-bank file is not reused by the fast protocol.


## V1.18-fast four-axis Pareto revision

Workload balance was removed from formal Pareto survival.

Rationale: in the current homogeneous MRTA problem, all tasks must first be
completed, and task completion earliness already rewards useful parallelism.
A deliberately uneven allocation may be optimal when robots have asymmetric
travel distances to tasks. Therefore Jain workload fairness is not treated as
a required capability unless a future problem statement explicitly requires
fairness.

Formal validity and capabilities are now:

    hard gate:
        completion == 1 on every training world

    Pareto axes:
        time_earliness
        path_efficiency
        priority_service
        deadline_satisfaction

The underlying evaluator may still compute workload/balance diagnostics, but
they do not affect Pareto dominance, epsilon deduplication, crowding, maximin,
parent survival, or Gene Bank membership.

The previous five-axis Gen0 run is retained as calibration evidence and is not
resumed under the new semantics.

New formal run directory:

    runs/gene_mrta_v118/formal_4r20t_100_fast_v3_4axis_seed118

The existing constructive 100-world bank is reused unchanged.


## V1.18-fast three-axis Primary revision

Deadline satisfaction is removed from formal Pareto survival.

Primary V1.18 semantics are now:

    hard gate:
        completion == 1 on every training world

    Pareto axes:
        time_earliness
        path_efficiency
        priority_service

Rationale:

- Time rewards early completion and useful parallel execution.
- Path rewards spatially efficient assignment / shorter obstacle-aware travel.
- Priority rewards serving higher-priority tasks earlier.
- Deadline is not required by the current Primary MRTA problem and therefore
  should not create an additional synthetic trade-off axis.

Task deadlines remain present in the environment/world data for compatibility
with the existing 12D observation architecture and for future deadline-specific
MRTA variants, but deadline satisfaction does not affect:

- Pareto dominance;
- epsilon deduplication;
- crowding;
- maximin;
- parent survival;
- Gene Bank membership.

The prior 4-axis run is frozen as calibration evidence and is not resumed under
the three-axis semantics.

Existing 100-world constructive bank is reused unchanged:

    runs/gene_mrta_v118/world_formal_4r20t_100_fast_v2.json

Fresh three-axis formal run:

    runs/gene_mrta_v118/formal_4r20t_100_fast_v4_3axis_seed118


## V1.19 Public MTRPD benchmark phase

Primary goal:
train one shared 148-parameter RouteTailDirectGene across a public benchmark
with changing robot/task counts and independently published global optima.

External benchmark:
Luo, Qin, Lim MTRPD public set.

Authoritative generation description:
- six TSPLIB families;
- ten subsets for each 30/40/50-vertex scale;
- 180 instances total;
- n = 29/39/49 customers plus one depot;
- public scales use 6/8/10 repairmen for 30/40/50 total vertices;
- route limit L = 2*d_max;
- published proven optima for 179/180 instances.

V1.19 does not hard-code 4 robots or one task count.

Formal per-instance capability:

    retention_i = OPT_i / Cost_i

where Cost_i is total customer latency. Illegal/incomplete solutions receive
zero retention.

Hard success:
all tasks must be legally served and every robot route must be able to return
to depot within L.

V1.19 uses a native route-limit-aware rollout. Every candidate append checks:

    used + segment + return_to_depot <= L

Mutation-only first phase:
- no mating;
- parent/child run on exactly the same evolution instances;
- paired per-instance delta and W/T/L are logged;
- scale Pareto axes are v30/v40/v50 optimum retention;
- overall/worst retention are diagnostics;
- parent sampling uses squared overall-retention pressure after feasibility.

Default public split:
- 4 of 10 replicates/group evolution -> 72 total;
- 3 validation -> 54;
- 3 protected test -> 54.

Mating/fusion is deferred until mutation-only Gene Bank convergence is
characterized.

Implementation:
- src/marl2d/gene_mrta_v119/
- tests/test_gene_mrta_v119.py
- tools/run_gene_mrta_v119_mac.sh
- docs/GENE_HOMOGENEOUS_MRTA_V119_PUBLIC_MTRPD.md

Public raw supplement is not fabricated. The historical supplement endpoint is
kept as an external source and must be inspected before writing the exact raw
importer.


## V1.20 Public MinMax-mTSP primary benchmark

V1.19 MTRPD data acquisition is frozen/aborted for formal use because the
historical raw supplement could not be recovered from the dead original host
or attempted archive paths. Do not report V1.19 synthetic fixtures as public
benchmark results.

V1.20 uses the currently accessible MILS public benchmark from
pengfeihe-angers/mils and mirrors the actual raw assets into this repository.

Raw assets:
- benchmarks/minmax_mtsp_mils/instances.zip
- benchmarks/minmax_mtsp_mils/Certification.zip
- benchmarks/minmax_mtsp_mils/SOURCE_README.md
- benchmarks/minmax_mtsp_mils/PROVENANCE.md

Dataset:
- 77 minmax-mTSP instances;
- 51..5915 vertices;
- robot/salesman counts 3, 5, 10, 20, 30;
- paper set S = 41 small/medium instances;
- paper set L = 36 large instances;
- 72 solution certificates in the source archive;
- five references are read from paper Table A.1;
- 22 instances are marked known exact optimum by the paper;
- remaining references are BKS.

Objective:
    minimize max route length

Per-instance capability:
    reference / Gene objective

Exact reference:
    ratio > 1 beyond tolerance => evaluator mismatch/error.

BKS reference:
    ratio > 1 => legal new-best-known candidate.

Distance definitions were independently checked against public certificates:
- EUC_2D uses continuous Euclidean length;
- ATT uses TSPLIB ATT pseudo-Euclidean.

Training:
- one shared 148-parameter Gene;
- mutation-only primary phase;
- parent/child evaluated on identical public instances;
- paired per-instance delta logged;
- cKDTree nearest-candidate decoder for scalability;
- Set S is development/validation;
- entire Set L is protected from evolution and reserved for large-scale
  generalization evaluation.

Mating/recombination remains disabled until the V1.20 mutation-only Gene Bank
has converged. A later phase will test whether specialists can fuse into one
strong universal Gene.


## V1.21 Resident-World SEGB

V1.21 replaces the V1.20 small-population trainer as the primary SEGB
evolution scheduler.

Formal protocol:
- 1000 candidate Gene worlds per round;
- every Gene runs all 34 fixed evolution benchmark instances;
- 50 rounds;
- 34,000 rollouts per round;
- 1,700,000 rollouts total;
- Round 0: 1000 random Genes;
- Round 1..49: 1000 inherited+mutated children from the previous frozen Gene
  Bank;
- parent sampling probability proportional to equal-axis total score squared;
- Bank is updated only after all 1000 worlds complete the full 34-instance
  evaluation;
- Bank admission uses external capability Pareto selection, not parent-child
  win/loss;
- parent-child delta remains diagnostic only.

Primary round-level convergence evidence:
- population mean/median/min/max/std retention_small;
- population mean/median/min/max/std retention_medium;
- per-axis population mean delta from the previous round;
- population overall retention;
- population worst-instance retention;
- Bank specialists and global best.

V1.20 formal_s_mutation_seed120 is a conventional small-population baseline,
not the primary SEGB result.


## V2.0 zero-shot size generalization

A new evaluation-only OOD protocol is implemented on
`experiment/gene-global-set-mrta-v2`.

Source run:

- `runs/gene_mrta_v20/longrun_1024g_200r_seed200/checkpoint.json`;
- 200 rounds indexed 0..199;
- 1024 Genes/round;
- fixed 100 training worlds;
- training cardinality range 5..20 robots and 10..100 tasks;
- final Round-199 Bank size = 222.

Frozen specialists at the end of the current run:

- Total-Time: `5b448ba2073a90eba8d5`, total_time 58.07686007499695 s;
- On-Time: `ba1efa666ac266786b69`, on_time_completed_tasks 55.09 and total_time 62.88902631759643 s.

New experiment:

- `src/marl2d/gene_mrta_v20/ood_generalization.py`;
- `tests/test_gene_mrta_v20_ood_generalization.py`;
- `docs/GENE_GLOBAL_SET_MRTA_V20_OOD_GENERALIZATION.md`;
- launcher modes `ood-smoke` and `ood-formal`.

Default OOD grid contains in-distribution control, training boundary, robot-only
OOD, task-only OOD, and both-axis OOD up to 60R/300T.

Primary metrics:

- raw Total Time;
- mean per-world TotalTime/BaselineTime;
- raw Priority rank;
- exact mean per-world On-time percentage;
- evaluation runtime.

OOD seeds are deterministic and disjoint from all 100 training seeds.
Both frozen Genes see the same worlds. OOD results never feed back into
training, mutation, parent selection, or the Gene Bank.

Next frozen action:

1. run `bash tools/run_gene_mrta_v20_mac.sh tests`;
2. run `bash tools/run_gene_mrta_v20_mac.sh ood-smoke`;
3. inspect 20-seed/cell results before starting `ood-formal`.


## V2.0 workload-ratio OOD matrix

Implemented after the first zero-shot size-OOD smoke.

Goal:
separate absolute-cardinality effects from task-load-per-robot effects.

Frozen Genes remain unchanged:
- Total-Time champion `5b448ba2073a90eba8d5`;
- On-Time champion `ba1efa666ac266786b69`.

No retraining and no Gene-Bank feedback.

Default workload matrix:
- Robots = 20, 40, 60;
- Tasks/Robot = 2.5, 5, 10, 15;
- cells = 20x50, 20x100, 20x200, 20x300,
  40x100, 40x200, 40x400, 40x600,
  60x150, 60x300, 60x600, 60x900.

Smoke:
- 20 unseen seeds/cell;
- seed namespace 20100001;
- command: `bash tools/run_gene_mrta_v20_mac.sh ood-load-smoke`.

Formal:
- 100 unseen seeds/cell;
- command: `bash tools/run_gene_mrta_v20_mac.sh ood-load-formal`.

Evaluator outputs now include `tasks_per_robot`.

Primary interpretation:
- at fixed Tasks/Robot, compare 20R vs 40R vs 60R for absolute-size degradation;
- at fixed Robot count, compare Tasks/Robot 2.5/5/10/15 for load degradation;
- primary cross-size metrics are TotalTime/BaselineTime and exact On-time %.


## V2.0 density-controlled OOD scaling

Implemented after the workload-ratio OOD smoke.

Purpose:
test cardinality extrapolation while keeping physical robot/task density fixed
within each workload ladder.

Important rollout property:
V2.0 already normalizes task/robot x-y by world_size, pair distance by world
diagonal, deadline/service/finish-time quantities by baseline time scale, and
accumulated distance by a scale derived from time/speed/diagonal. Therefore
larger physical maps do not simply push raw coordinate features outside the
training 0..100 numerical range.

Cell syntax now accepts:
`ROBOTSxTASKS@WORLD_SIZE`.

Density anchor:
- 20R;
- world_size=100;
- robot density = 0.002 / unit^2.

Map scaling:
`world_size = 100 * sqrt(R/20)`.

Stage-1 smoke:
- workload 5 Tasks/Robot:
  20R/100T@100,
  40R/200T@141.421356,
  60R/300T@173.205081,
  80R/400T@200;
- workload 15 Tasks/Robot:
  20R/300T@100,
  40R/600T@141.421356,
  60R/900T@173.205081,
  80R/1200T@200;
- 20 unseen seeds/cell;
- seed namespace 20110001;
- command: `bash tools/run_gene_mrta_v20_mac.sh ood-density-smoke`.

Stage-2 extreme:
- 100R/500T@223.606798;
- 100R/1500T@223.606798;
- 5 unseen seeds/cell;
- seed namespace 20120001;
- command: `bash tools/run_gene_mrta_v20_mac.sh ood-density-extreme`.

Evaluator now records:
- world_size;
- world_area;
- robot_density;
- task_density;
- world_generation_runtime_s;
- Gene cell_runtime_s.

World generation timing is separated from Gene evaluation because feasible
baseline/deadline construction may become the first infrastructure bottleneck
at very large task counts.

Frozen Genes and no-feedback rule remain unchanged.
