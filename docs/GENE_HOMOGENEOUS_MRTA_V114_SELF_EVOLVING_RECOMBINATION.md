# V1.14 Self-Evolving Recombination Gene Bank

## Purpose

V1.13 established that route-tail multi-task allocation can evolve while
maintaining four certified capability axes. It also showed that fixed
hand-designed crossover operators have substantially different evolutionary
yield.

V1.14 removes the assumption that the researcher should choose one crossover
operator.

The system now evolves two populations:

1. Policy Gene Bank: what task-allocation behavior survives.
2. Recombination Gene Bank: how surviving Policy Genes reproduce.

The target is not a universal crossover equation. The target is a system
that discovers a useful mating rule for the current problem distribution.

## Recombination genotype

Each Recombination Gene contains:

- seven continuous coefficients;
- seven binary structural gates;
- one self-adaptive mutation scale.

The seven terms are:

beta_norm_ratio
beta_quality_diff
beta_capability_diff
gamma_bias
gamma_cosine
gamma_sign_agreement
gamma_magnitude

A gate equal to zero removes that term completely from the equation. This
means evolution changes both numerical coefficients and formula structure.

For functional group k:

alpha_k = sigmoid(
    g_r * beta_r * r_k
    + g_q * beta_q * (Q_A-Q_B)
    + g_h * beta_h * (H_A-H_B)
)

eta_k = 0.5 + sigmoid(
    g_0 * gamma_0
    + g_c * gamma_c * c_k
    + g_s * gamma_s * s_k
    + g_m * gamma_m * m_k
)

and

theta_C^(k)
=
theta_0^(k)
+
eta_k [
    alpha_k * Delta_A^(k)
    + (1-alpha_k) * Delta_B^(k)
]

This law remains parent-swap symmetric.

The all-gates-off genotype is the exact center law:

alpha_k = 0.5
eta_k = 1.0

## Self-adaptive rule mutation

Every Recombination Gene carries its own sigma_R.

For a child rule:

sigma_R' = clip(
    sigma_R * exp(N(0,tau)),
    sigma_min,
    sigma_max
)

Each active coefficient is then mutated by:

beta_i' = clip(
    beta_i + N(0,sigma_R'),
    lower_i,
    upper_i
)

Each structural gate independently flips with probability p_gate.

Therefore the system can learn both:

- how large recombination mutations should be;
- which parent/block descriptors should participate in the mating law.

## Recombination capability axes

No weighted scalar mating reward is used.

Each Recombination Gene is tracked on four independent external axes:

1. screen_yield
   Fraction of generated children surviving the 25-world screening stage.

2. acceptance_yield
   Fraction of generated children passing the full 100-world dual
   inheritance gate.

3. retention_quality
   Mean worst inherited-capability retention measured during screening.

4. four_capability_yield
   Fraction of generated children accepted with all four required
   capabilities.

The Recombination Gene Bank preserves specialists and Pareto-conflicting
rules rather than reducing these axes to a weighted sum.

## Rule selection

For every mating child:

1. one recombination capability axis is sampled uniformly;
2. Recombination Genes are sampled according to that axis raised to a
   selection power;
3. a uniform exploration fraction remains active.

The default is:

selection_power = 2
uniform_exploration = 0.25

This lets successful mating rules receive more reproductive opportunities
without permanently eliminating unexplored rules.

## Recombination Bank lifecycle

Default pilot:

- 24 initial Recombination Genes;
- 8 mutated Recombination Genes spawned every generation;
- bank capacity 32;
- six specialists retained per recombination axis;
- Pareto front retained up to twelve explicit slots;
- remaining capacity filled without constructing a weighted aggregate score.

Newborn rules use conservative Bayesian priors. They do not receive an
artificial 50 percent success rate merely because they are untested.

## Policy evolution

Policy semantics remain V1.13 route-tail multi-task append.

Policy dimensionality remains 148.

The V1.14 run warm-starts from the completed V1.13 Policy Gene Bank, not
from V1.10. Normal mutation continues in parallel with mating so policy
improvements are not forced to depend only on crossover.

The Policy Gene Bank still uses:

- mean_time;
- tail10_time;
- continuation_preservation;
- fleet_option_reserve.

The dual 95 percent inheritance gate is unchanged.

## Interpretation

A V1.14 Recombination Gene is not claimed to be a universal mathematical
law. It is an evolved mating genotype specialized to the current task and
scenario distribution.

The first V1.14 experiment tests whether the system can autonomously shift
reproductive effort toward rules that generate useful four-capability
children while preserving competing recombination specialists.

This is the first self-evolving rule stage. Formula grammar is still the
interpretable adaptive-delta family above. Arbitrary symbolic expression
trees are intentionally deferred until this controlled version is validated.

## Protected evaluation

The frozen 95M bank remains development data.

The protected 99M benchmark remains untouched until V1.14 procedure and
candidate selection are frozen.

The oracle continues to define capability ceilings only. It never teaches
Policy Gene actions or Recombination Gene coefficients.
