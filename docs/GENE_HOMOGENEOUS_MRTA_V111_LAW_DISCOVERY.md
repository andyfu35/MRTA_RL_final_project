# V1.11 Empirical Recombination Law Discovery

## Goal

V1.10 showed that one unchanged 148-parameter policy can accumulate four
simultaneously certified capabilities. V1.11 asks whether successful and
failed recombination outcomes can be used to discover an interpretable
mathematical mixing law instead of choosing a fixed hand-designed crossover.

V1.11 does not change the policy architecture and does not use MILP actions
as labels.

## Frozen inputs

- Policy architecture: V1.8/V1.10, 12D observation, hidden dimension 8,
  148 parameters.
- Parent population: completed V1.10 checkpoint.
- Development scenarios: frozen 95M 100-world exact-oracle bank.
- Common ancestor: frozen V1.8 Gene.
- Protected 99M final benchmark remains untouched.

Only currently ceiling-certified parent capabilities are used.

## Complementary parent pairs

A pair is eligible only if both parents contribute at least one currently
certified capability that the other parent does not carry.

C_A minus C_B must be nonempty, and C_B minus C_A must also be nonempty.

Pair quality is based on the weaker parent's current certified quality. Parent
sampling retains strong quality pressure with a small uniform component.

## Interpretable law family

For each of the eight hidden functional blocks plus the four global
parameters as a ninth group, define ancestor-relative deltas:

Delta_A^(k) = theta_A^(k) - theta_0^(k)

Delta_B^(k) = theta_B^(k) - theta_0^(k)

The experimentally searched parent contribution is:

alpha_k = sigmoid(
    beta_r * r_k
    + beta_q * (Q_A - Q_B)
    + beta_h * (H_A - H_B)
)

where

r_k = log(
    (||Delta_A^(k)|| + epsilon)
    /
    (||Delta_B^(k)|| + epsilon)
)

and H_A, H_B are certified capability counts normalized by four.

The delta scale is:

eta_k = 0.5 + sigmoid(
    gamma_0
    + gamma_c * c_k
    + gamma_s * s_k
    + gamma_m * m_k
)

where:

- c_k is cosine similarity between the two parent deltas;
- s_k in [-1,1] is parameter-sign agreement;
- m_k is group delta magnitude relative to the median group magnitude.

The child is:

theta_C^(k)
=
theta_0^(k)
+
eta_k [
    alpha_k * Delta_A^(k)
    + (1-alpha_k) * Delta_B^(k)
]

Exactly seven coefficients are searched:

(beta_r, beta_q, beta_h, gamma_0, gamma_c, gamma_s, gamma_m)

The law is parent-swap symmetric. Exchanging A and B flips every asymmetric
alpha input, which gives alpha'_k = 1-alpha_k, while eta remains unchanged.
The resulting child is therefore identical.

## Discovery / validation split

The default pilot samples 64 distinct complementary parent pairs.

- 48 pairs are used for formula discovery.
- 16 pairs are held out from formula selection.

Thirty-two coefficient sets are generated using Latin hypercube sampling.
The exact center law is always included:

alpha_k = 0.5

eta_k = 1.0

Every candidate law is evaluated on the same 25 of the frozen 100
development worlds for all 48 discovery parent pairs.

For every required capability a, define:

R_a = min(R_parent,a, R_ceiling,a)

and the child margin is:

M(C) = min_a R_a - 0.95

The pass condition remains:

R_a >= 0.95 for every capability in C_A union C_B.

The winning coefficient set is frozen before any validation parent pair is
used.

## Held-out parent validation

The frozen discovered law is evaluated on all 100 development worlds for the
16 held-out parent pairs.

On exactly those same pairs it is compared against one deterministic,
reproducible draw of:

- Block Blend
- Parameter Blend
- TIES Delta

Reported values include:

- acceptance rate;
- four-capability acceptance rate;
- mean, P10 and median minimum dual retention;
- mean child scores on all four capability axes;
- paired wins, ties and losses against every baseline.

This is held-out-parent validation inside development data. It is not the
untouched 99M final generalization benchmark.

## Dataset outputs

Each run writes:

- pair_manifest.json
- formula_candidates.json
- selected_formula.json
- recombination_dataset.jsonl
- validation_results.jsonl
- summary.json

The discovery dataset contains parent IDs, current certified capabilities,
parent qualities, the seven formula coefficients, nine group-level features,
the resulting alpha and eta values, child capability scores, per-axis
retention, minimum retention margin, and pass/fail.

## Commands

Implementation tests:

    bash tools/run_gene_mrta_v111_law_discovery_mac.sh tests

Small end-to-end smoke:

    bash tools/run_gene_mrta_v111_law_discovery_mac.sh smoke

Default discovery pilot:

    bash tools/run_gene_mrta_v111_law_discovery_mac.sh pilot

## Interpretation guardrail

One pilot cannot establish a universal recombination law. Its purpose is to
test whether a compact equation discovered from one set of parent pairs
transfers to parent pairs not used for formula selection and whether it can
match or outperform the strongest hand-designed V1.10 crossover baselines.

If the result repeats across discovery seeds and later survives an untouched
scenario benchmark, the equation becomes a candidate SEGB recombination law.

The oracle still does not teach an action; it defines the capability ceiling.
