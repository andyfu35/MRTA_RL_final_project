# V1.14.1 Phenotype-Canonical, Evidence-Aware Recombination

## Why V1.14.1 exists

V1.14 completed the first system-level Recombination Gene Bank experiment,
but the run exposed two confounds:

1. neutral center-law clones could occupy multiple bank slots because inactive
   coefficients and mutation sigma were part of the old Gene identity;
2. a low-trial rule could appear to be a specialist because a point estimate
   such as 2/2 was compared directly with a mature rule such as 54/295.

V1.14.1 fixes those issues without expanding the formula family.

The scientific question becomes:

Does an adaptive Recombination Gene Bank outperform a fixed center-law
control when both start from the same V1.13 Policy Bank and receive the same
Policy mutation budget?

## 1. Phenotype canonicalization

A disabled coefficient is forced to exactly zero:

beta_i_effective = g_i * beta_i

with g_i in {0,1}.

Bank identity is computed from:

- the seven canonical effective coefficients;
- the seven structural gates.

mutation_sigma is not part of the phenotype ID.

Therefore two genes that produce the same mating equation cannot occupy two
Recombination Bank slots merely because disabled coefficients or mutation
sigma differ.

The genotype fingerprint is still stored separately for diagnostics.

For the all-gates-off center law there can be at most one bank phenotype.

## 2. Correct capability-count equation display

The implementation has always used:

cap_diff = (H_A - H_B) / 4

V1.14.1 reports that normalization explicitly in the printed equation.

## 3. Evidence-aware rule ranking

Raw smoothed means are retained for diagnostics, but mating-rule selection and
mature specialist identification use a lower posterior quantile.

Default:

q = 0.10

For screen yield:

p_screen ~ Beta(1 + selected, 3 + generated - selected)

For full acceptance yield:

p_accept ~ Beta(1 + accepted, 9 + generated - accepted)

For four-capability yield:

p_4cap ~ Beta(0.5 + accepted_4cap,
              9.5 + generated - accepted_4cap)

Retention quality is treated as a bounded fractional-success process using
the accumulated clipped worst-capability retention.

The score used for mature rule ranking is the 10 percent posterior quantile,
not the posterior mean.

This makes 2/2 insufficient to automatically displace a mature 54/295 rule.

## 4. Minimum evidence for specialists

Formal pilot default:

generated >= 32

before a Recombination Gene can be called a mature specialist or join the
formal Pareto front.

Immature rules remain eligible for uniform exploration and can occupy explicit
exploration slots in the bank, but are labeled provisional.

## 5. Exploration reserve

Formal adaptive bank:

- bank limit = 32;
- explicit exploration slots = 8;
- specialist size per recombination axis = 6;
- Pareto slots = 12;
- uniform rule exploration = 25 percent.

This prevents evidence filtering from freezing the bank around early rules.

## 6. Separate random streams

V1.14.1 uses:

policy_rng = seed
rule_rng = seed + 114100003

Normal Policy mutation and Policy-parent selection consume policy_rng.

Recombination-rule mutation and rule selection consume rule_rng.

This prevents the adaptive rule population from changing the normal Policy
mutation random stream merely by consuming additional random numbers.

## 7. Paired control

Two systems start from the exact same V1.13 checkpoint.

Adaptive:

Policy mutation
+
self-evolving Recombination Gene Bank

Center:

Policy mutation
+
fixed all-gates-off center law

Both use:

- the same Policy architecture;
- the same 148 parameters;
- the same 95M development worlds;
- the same Policy mutation schedule;
- the same number of normal children;
- the same number of mating children;
- the same parent-selection policy;
- the same seed.

The paired comparison reports each Policy axis separately:

adaptive - center for

- mean_time;
- tail10_time;
- continuation_preservation;
- fleet_option_reserve.

No weighted aggregate score is created.

## 8. Commands

Tests:

    bash tools/run_gene_mrta_v1141_paired_mac.sh tests

Paired structural smoke:

    bash tools/run_gene_mrta_v1141_paired_mac.sh paired-smoke

Formal single-seed paired 50-generation experiment:

    bash tools/run_gene_mrta_v1141_paired_mac.sh paired50

The paired run writes:

    runs/gene_mrta_v1141_paired_seed7/comparison_50.json

It also prints the exact adaptive and center summary paths.

## 9. Interpretation

A positive adaptive-minus-center delta on one axis means only that the
adaptive system ended higher on that development axis for this paired seed.

It is not yet a statistical generalization claim.

If the single-seed paired result is promising, the next stage is repeated
paired seeds with the same frozen protocol, followed by a frozen 96M
development-validation bank.

The protected 99M benchmark remains untouched.


---

# Current implementation status

Status as of 2026-10-03:

IMPLEMENTED, NOT YET RUN.

The V1.14.1 code, tests, paired comparator, and launcher exist in the repository.
No paired-smoke or paired50 result should be claimed until terminal output is actually produced.

Current implementation files:

- src/marl2d/gene_mrta_v1141/recombination_gene.py
- src/marl2d/gene_mrta_v1141/evolve.py
- src/marl2d/gene_mrta_v1141/compare.py
- tests/test_gene_mrta_v1141_recombination_correction.py
- tools/run_gene_mrta_v1141_paired_mac.sh

Required execution order:

1. bash tools/run_gene_mrta_v1141_paired_mac.sh tests
2. bash tools/run_gene_mrta_v1141_paired_mac.sh paired-smoke
3. inspect structural invariants
4. only then run bash tools/run_gene_mrta_v1141_paired_mac.sh paired50

Adaptive structural invariant:

Rcenter <= 1

Center-control invariant:

Rbank = 1
Rcenter = 1

Expected formal paired output:

runs/gene_mrta_v1141_paired_seed7/comparison_50.json

The paired comparison reports the four Policy-axis deltas separately and does not construct a scalar aggregate.

# Documentation handoff rule

Before a new experimental version is started, all completed results and architecture-changing decisions must be recorded in:

- /AI_PROJECT_CONTEXT.md
- docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md
- the relevant version-specific document

New AI conversations must read the root context file and experiment ledger before continuing development.

The protected 99M benchmark remains untouched.
