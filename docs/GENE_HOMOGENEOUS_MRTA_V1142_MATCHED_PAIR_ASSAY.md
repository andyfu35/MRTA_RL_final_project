# V1.14.2 Frozen-Parent Matched-Pair Recombination Assay

Status as of 2026-10-04:

FORMAL128 COMPLETED (seed=7).

## Why this experiment is needed

V1.14.1 compared two 50-generation evolving systems:

- adaptive Recombination Gene Bank;
- fixed canonical center law.

The seed=7 result did not show a consistent adaptive advantage.

However, once the two Policy Gene Banks begin evolving, their parent pools diverge.
Therefore a later-generation child in the adaptive arm is not necessarily born
from the same two Policy parents as the corresponding child in the center arm.

There is also a unique-child asymmetry: the deterministic center law can produce
four identical children when the same parent pair is repeated four times, while
adaptive mating can produce different children from one pair.

V1.14.2 removes both confounds.

## Primary scientific question

For the exact same frozen Policy-parent pair (A,B), does the frozen V1.14.1
adaptive Recombination Bank produce a better child than the canonical center
law?

This experiment does not evolve the Policy Bank.

It is a controlled offspring assay.

## Frozen inputs

Policy parents:

- completed V1.13 route-tail checkpoint;
- no Policy mutation;
- no Policy admission feedback;
- no generational bank drift.

Adaptive mating rules:

- completed V1.14.1 adaptive checkpoint;
- Recombination Gene statistics are frozen;
- no new rule mutation during the assay.

Environment:

- frozen 95M 100-world exact-oracle development bank.

Common ancestor:

- frozen V1.8 Gene.

Protected data:

- 99M remains untouched.

## Parent-pair manifest

Exactly 128 unique unordered Policy-parent pairs are sampled once.

The pair sampler uses the same historical mating pressure:

P(parent) proportional to Q^10

plus 5 percent uniform exploration.

Second-parent selection prefers capability complementarity and falls back to a
distinct active parent if necessary.

A canonical pair key is:

(min(parent_A,parent_B), max(parent_A,parent_B))

Duplicate pair keys are rejected until the manifest contains 128 unique pairs.

The resulting manifest is written to disk and shared by all mating conditions.

## Conditions

### Center

For each frozen pair (A,B), generate exactly one child using:

theta_C
=
theta_0
+
0.5 Delta_A
+
0.5 Delta_B

### Adaptive

For the same frozen pair (A,B), sample exactly one Recombination Gene from the
completed V1.14.1 adaptive Recombination Bank using its evidence-aware rule
selection protocol.

The sampled rule then produces one child.

Therefore both conditions receive:

- exactly 128 parent pairs;
- exactly 128 generated children;
- exactly one child per pair;
- the same parent identities for every paired comparison;
- the same 100 evaluation worlds.

The only treatment difference is the mating rule.

## Evaluation

Every child is evaluated on all 100 frozen 95M worlds.

No 25-world screening stage is used in the primary assay because the purpose is
to measure offspring quality directly rather than simulate training throughput.

For every child, record the four independent Policy axes:

- mean_time;
- tail10_time;
- continuation_preservation;
- fleet_option_reserve.

For required inherited capabilities, also record:

- parent-relative retention;
- frozen V1.13 ceiling retention;
- dual 0.95 inheritance-gate pass/fail.

A child is four-capability certified when all four Policy axes are at least
95 percent of the frozen V1.13 active ceiling.

## Paired reporting

For every Policy axis a:

Delta_a(pair)
=
S_a(adaptive_child)
-
S_a(center_child)

Report independently:

- mean Delta_a;
- median Delta_a;
- win/tie/loss across the 128 exact parent pairs.

Do not create a weighted aggregate score.

Also report:

- adaptive weakly-dominates-center pair count;
- center weakly-dominates-adaptive pair count;
- neither-dominates count;
- exact-identical-child count;
- dual-gate accepted counts;
- four-capability-certified counts;
- sampled Recombination Gene usage;
- results restricted to pairs where adaptive sampled a non-center rule.

## Interpretation

Possible outcomes:

1. Adaptive wins consistently on one or more axes without losing others:
   evidence that learned mating rules can improve offspring quality.

2. Center matches or dominates adaptive:
   evidence that the simple ancestor midpoint is a strong local recombination
   law for the mature Policy population.

3. Overall tie but non-center subset contains structured wins:
   evidence that adaptive rules may be context-specific and should be selected
   conditionally rather than globally.

This is still 95M development evidence and is not a generalization claim.

## Execution sequence

1. tests
2. smoke with a small pair manifest
3. formal 128-pair assay
4. record results in AI_PROJECT_CONTEXT.md and the experiment ledger
5. only then decide whether multi-seed replication is justified

99M remains protected.


## Implemented files

- src/marl2d/gene_mrta_v1142/__init__.py
- src/marl2d/gene_mrta_v1142/assay.py
- tests/test_gene_mrta_v1142_matched_pair_assay.py
- tools/run_gene_mrta_v1142_matched_pair_mac.sh

The formal assay requires a completed adaptive V1.14.1 checkpoint with at least 50 generations.

Commands:

    bash tools/run_gene_mrta_v1142_matched_pair_mac.sh tests

Then:

    bash tools/run_gene_mrta_v1142_matched_pair_mac.sh smoke

Only if smoke is structurally correct:

    bash tools/run_gene_mrta_v1142_matched_pair_mac.sh formal128

Expected formal outputs:

- pair_manifest.json
- pair_results.jsonl
- summary.json

No V1.14.2 result should be claimed until actual terminal output is provided.


## Completed regression test and smoke result

Date: 2026-10-04

Regression command:

    bash tools/run_gene_mrta_v1142_matched_pair_mac.sh tests

Result:

    14 passed

Smoke command:

    bash tools/run_gene_mrta_v1142_matched_pair_mac.sh smoke

Frozen inputs resolved to:

V1.13 Policy checkpoint:

    runs/gene_mrta_v113_route_tail_evolution/gene_mrta_v113_route_tail_20261003_203157_seed7/checkpoint.json

Completed adaptive V1.14.1 checkpoint:

    runs/gene_mrta_v1141_paired_seed7/adaptive/gene_mrta_v1141_recombination_20261003_231301_seed7/checkpoint.json

Smoke used:

- seed = 7
- 8 unique unordered frozen parent pairs
- one adaptive child per pair
- one center child per pair
- all 100 frozen 95M worlds per child
- no Policy mutation
- no Policy-bank feedback

Smoke output:

- pairs = 8
- adaptive non-center rule pairs = 6
- exact identical adaptive/center children = 2

Per-axis Adaptive - Center:

mean_time:
- mean delta = +0.00027892
- W/T/L = 4/3/1

tail10_time:
- mean delta = -0.00015336
- W/T/L = 2/4/2

continuation_preservation:
- mean delta = +0.00046004
- W/T/L = 4/3/1

fleet_option_reserve:
- mean delta = -0.00052722
- W/T/L = 1/3/4

Axiswise dominance:

- tie = 3
- adaptive weakly dominates = 1
- neither dominates = 4
- center weakly dominates = 0 in this eight-pair smoke

Dual 0.95 inheritance gate:

- adaptive = 5/8
- center = 5/8

Four-capability certification:

- adaptive = 4/8
- center = 4/8

Smoke run directory:

    runs/gene_mrta_v1142_matched_pair_smoke/gene_mrta_v1142_matched_pair_20261004_101048_seed7

Interpretation:

The matched-pair assay is structurally valid.

The smoke is intentionally too small for a scientific comparison. It shows that:

1. the exact same frozen parent pairs are compared;
2. adaptive and center receive one child per pair;
3. both conditions use the same 100 evaluation worlds;
4. non-center adaptive rules are actually being sampled (6/8 pairs);
5. center-equivalent adaptive samples correctly produce identical children (2/8);
6. dual-gate and four-capability counts are computed symmetrically.

The mixed signs across the four axes are expected and must not be scalarized.

The formal128 assay is now authorized as the next step.

99M remains untouched.


## Completed formal128 result

Date: 2026-10-04

Command:

    bash tools/run_gene_mrta_v1142_matched_pair_mac.sh formal128

Regression tests before formal run:

    14 passed

Frozen inputs:

V1.13 Policy checkpoint:

    runs/gene_mrta_v113_route_tail_evolution/gene_mrta_v113_route_tail_20261003_203157_seed7/checkpoint.json

Adaptive V1.14.1 Recombination checkpoint:

    runs/gene_mrta_v1141_paired_seed7/adaptive/gene_mrta_v1141_recombination_20261003_231301_seed7/checkpoint.json

Formal run directory:

    runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7

Formal design:

- 128 unique unordered frozen Policy-parent pairs;
- same parent-pair manifest for adaptive and center;
- one child per pair per condition;
- no Policy mutation;
- no Policy-bank feedback;
- no generational drift;
- all children evaluated on the same 100 frozen 95M worlds;
- no scalarized Policy score.

Observed adaptive rule usage summary:

- adaptive non-center rule pairs = 115/128;
- center-rule adaptive samples = 13/128;
- exact adaptive/center identical Policy children = 19/128.

Because every adaptive center-rule sample must equal the center child, at least 6 additional non-center rule samples collapsed to the same exact Policy child for their specific parent pair. This is a context-specific effective-equivalence observation, not a protocol error.

### Primary per-axis matched result

Adaptive - Center:

mean_time:
- mean delta = +0.00064842
- W/T/L = 47/38/43
- non-tied adaptive win rate = 47/90 = 52.22%

tail10_time:
- mean delta = +0.00104946
- W/T/L = 40/54/34
- non-tied adaptive win rate = 40/74 = 54.05%

continuation_preservation:
- mean delta = +0.00029748
- W/T/L = 62/29/37
- non-tied adaptive win rate = 62/99 = 62.63%

fleet_option_reserve:
- mean delta = -0.00006791
- W/T/L = 51/31/46
- non-tied adaptive win rate = 51/97 = 52.58%

The largest directional advantage is continuation_preservation.

### Exploratory exact sign tests on non-tied pairs

These tests use only W/L counts and ignore ties.

mean_time:
- p = 0.7520

tail10_time:
- p = 0.5614

continuation_preservation:
- p = 0.01543

fleet_option_reserve:
- p = 0.6849

The continuation result is nominally significant at 0.05 as a single exploratory test.

However, four Policy axes were examined. A simple Bonferroni correction would give approximately:

0.01543 * 4 = 0.0617

Therefore this formal128 run does not justify a corrected multi-axis statistical superiority claim yet.

### Axiswise dominance

Across 128 exact parent pairs:

- tie = 29
- adaptive weakly dominates center = 19
- center weakly dominates adaptive = 13
- neither dominates = 67

Among pairs with a strict weak-dominance outcome only:

- adaptive 19
- center 13

This direction favors adaptive but is not statistically strong by itself.

### Inheritance and four-capability fusion

Dual 0.95 inheritance gate:

- adaptive = 98/128
- center = 95/128

Four-capability certification:

- adaptive = 90/128
- center = 87/128

Thus adaptive produced 3 additional dual-gate-passing children and 3 additional four-capability children in this matched-parent development assay.

The difference is small but directionally favorable to adaptive.

### Scientific interpretation

The correct V1.14.2 conclusion is:

1. Adaptive mating is not a universal improvement over center.
2. It shows a small positive average change on mean_time and tail10_time.
3. It shows the clearest directional advantage on continuation_preservation.
4. fleet_option_reserve is essentially neutral/slightly negative on average.
5. Adaptive produces slightly more dual-gate and four-capability children.
6. Most parent pairs remain trade-off cases rather than one method dominating on every axis.

Therefore:

- "Adaptive recombination always beats center" is NOT supported.
- "Adaptive recombination can improve some capability dimensions for the same parents while preserving multi-capability fusion" IS supported as development evidence.
- "A learned non-center rule family is statistically superior overall" is NOT yet supported.

### Next required analysis before a new architecture

Do not change the mating formula family yet.

The next step is to inspect the already-written formal summary/pair_results and isolate the 115 non-center adaptive pairs.

Required questions:

1. Which non-center Recombination Gene IDs produced the continuation wins?
2. Are the wins concentrated in one or more rule phenotypes?
3. What parent capability combinations were present in those wins?
4. Do any non-center rules increase four-capability certification relative to center on the same pair?
5. Which six non-center samples collapsed to an exactly identical center child and why?
6. Are there context patterns that predict when a non-center rule should be used?

This is the first place where context-conditioned rule selection may be justified.

99M remains untouched.
