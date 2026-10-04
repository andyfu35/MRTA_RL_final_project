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

