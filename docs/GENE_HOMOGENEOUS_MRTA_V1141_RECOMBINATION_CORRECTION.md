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

Status as of 2026-10-04:

COMPLETED SINGLE-SEED PAIRED SMOKE AND PAIRED50 (seed=7).

The V1.14.1 code, tests, paired comparator, and launcher exist in the repository.
Paired-smoke and paired50 have now been executed for seed=7. This is still development evidence only and is not a multi-seed or protected-final claim.

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


---

# V1.14.1 completed results — seed 7

## Regression tests

Command:

    bash tools/run_gene_mrta_v1141_paired_mac.sh tests

Result:

    15 passed

## Paired structural smoke

Command:

    bash tools/run_gene_mrta_v1141_paired_mac.sh paired-smoke

Adaptive smoke:

- Rcenter remained exactly 1;
- Rbank reached 12;
- mature-rule evidence became available;
- non-center formulas survived and entered the small Pareto set;
- Policy remained four-capability certified;
- route-tail multi-task behavior remained active.

Center smoke:

- Rbank = 1;
- Rcenter = 1;
- only the canonical center phenotype existed.

Smoke final Policy scores were identical between adaptive and center on all four axes.

This validates the intended structural invariants of phenotype canonicalization and center-only control.

## Formal paired50

Command:

    bash tools/run_gene_mrta_v1141_paired_mac.sh paired50

Common bootstrap:

    runs/gene_mrta_v113_route_tail_evolution/gene_mrta_v113_route_tail_20261003_203157_seed7/checkpoint.json

Development scenario bank:

    runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json

Adaptive run:

    runs/gene_mrta_v1141_paired_seed7/adaptive/gene_mrta_v1141_recombination_20261003_231301_seed7

Center-only run:

    runs/gene_mrta_v1141_paired_seed7/center/gene_mrta_v1141_recombination_20261004_013416_seed7

Comparison:

    runs/gene_mrta_v1141_paired_seed7/comparison_50.json

### Adaptive final Policy best

- mean_time = 0.9797388961714149
- tail10_time = 0.9273987323265199
- continuation_preservation = 0.8334576354631884
- fleet_option_reserve = 0.8757350433050963

### Center-only final Policy best

- mean_time = 0.9795402913092378
- tail10_time = 0.9273987323265199
- continuation_preservation = 0.8345021458732124
- fleet_option_reserve = 0.8762071604639012

### Adaptive minus Center

- mean_time = +0.0001986048621770431
- tail10_time = 0.0
- continuation_preservation = -0.0010445104100239577
- fleet_option_reserve = -0.0004721171588049078

In percentage-point units:

- mean_time: approximately +0.01986 pp
- tail10_time: 0.00000 pp
- continuation_preservation: approximately -0.10445 pp
- fleet_option_reserve: approximately -0.04721 pp

The four capability axes therefore do not show a consistent adaptive advantage.

Adaptive is slightly higher on mean time, tied on tail10, and lower on continuation and fleet reserve.

Because capability axes are intentionally not scalarized, there is no valid basis to collapse these four outcomes into a single win/loss score.

## Recombination result

The phenotype canonicalization fix worked.

Across the formal adaptive run:

- Rcenter stayed exactly 1;
- the bank did not accumulate duplicate center phenotypes;
- mature Recombination Genes increased from 0 to about 20;
- transient Pareto sizes of 2–3 appeared;
- final mature Pareto size = 1.

At the final generation, all four mature recombination specialists were the same canonical center phenotype:

    c7723fa1e0127975e49e

Center phenotype:

- active_term_count = 0;
- all seven gates = false;
- all effective coefficients = 0;
- mutation_sigma = 0.35.

Within the adaptive arm, the canonical center rule accumulated:

- generated = 884;
- screen_selected = 189;
- accepted = 189;
- four_capability_accepted = 183.

Its final evidence scores were sufficient to dominate the mature adaptive Pareto set on all four recombination axes.

Therefore the V1.14 center-law dominance cannot be explained only by neutral duplicate IDs.

## Scientific interpretation

V1.14.1 successfully removed the two original confounds:

1. duplicate neutral center phenotypes;
2. low-evidence specialist promotion.

However, after those corrections, the adaptive Recombination Gene Bank still did not outperform the fixed center-law control on the four Policy capability axes for seed=7.

Current conclusion:

- phenotype/evidence correction: PASS;
- self-evolving Recombination Bank machinery: PASS;
- evidence that adaptive mating is superior to center law: NOT SUPPORTED for seed=7;
- evidence that a non-center mature mating formula is superior: NOT SUPPORTED for seed=7.

The center formula is now an empirical baseline, not merely an implementation artifact.

## Additional control caveat discovered after paired50

The center-only arm uses one deterministic center rule for four children per parent pair.

With:

32 parent pairs x 4 children

the four children belonging to a single pair are formula-identical and can become identical Policy children.

The adaptive arm can produce different rule phenotypes for the four children of one parent pair.

Thus the two arms have equal nominal generated/evaluation budgets but not necessarily equal unique-child diversity.

This does not create a false adaptive disadvantage; if anything it can handicap the center control's search diversity.

Because center still matched or exceeded adaptive on three of four Policy axes, the current result strengthens the observation that the center law is a strong baseline.

For a publication-grade causal comparison, a future control should match unique-child opportunity as well as nominal offspring count, for example by using one center child per distinct parent pair or by explicitly deduplicating Policy children before screening and equalizing the number of unique candidates evaluated.

## Frozen next action

Do not inspect 99M.

Do not expand to symbolic-expression recombination yet.

The next controlled question should be one of:

1. repeat the corrected comparison with a unique-child-matched center control; or
2. if retaining the present control, run multiple paired seeds and treat center law as the primary baseline.

The preferred next step is unique-child-matched control first, because it removes the remaining experimental asymmetry before spending multi-seed compute.
