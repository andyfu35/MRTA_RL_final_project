# V1.14.2 Frozen-Parent Matched-Pair Recombination Assay

Status as of 2026-10-04:

IMPLEMENTED. TESTS / SMOKE / FORMAL128 NOT YET RUN.

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
