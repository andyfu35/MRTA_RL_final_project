# V1.21 Resident-World SEGB Public Benchmark Protocol

Status: PRIMARY TRAINING PROTOCOL

V1.20 successfully validated the public MILS minmax-mTSP dataset, reference
values, distance conventions, variable robot/task Gene interface, and
OPT/BKS scoring. However, its formal trainer used a conventional small
population/child loop. That scheduler is not the intended SEGB training
protocol and must not be reported as the primary SEGB result.

V1.21 keeps the validated V1.20 benchmark/evaluator but restores the intended
round-based Gene Bank evolution.

## Fixed public judges

Evolution uses exactly 34 fixed benchmark instances:

- 21 small instances
- 13 medium instances
- robot counts 3, 5, 10, 20, 30
- 51..1173 vertices

Every candidate Gene in every round is evaluated on all 34 exact same
instances.

Validation (7 instances) and protected large-scale test (36 instances) are not
used during evolution.

## Formal scale

    WORLDS_PER_ROUND = 1000
    FIXED_EVOLUTION_INSTANCES = 34
    ROUNDS = 50

Per round:

    1000 x 34 = 34,000 rollouts

Formal run:

    1000 x 34 x 50 = 1,700,000 rollouts

A "world" is one candidate Gene in the evolutionary population. All 1000
worlds of a round are generated from the same frozen parent Bank state.

## Round 0

Round 0 creates 1000 independent random 148-parameter Genes.

For each Gene:

    Gene -> all 34 fixed instances -> external capability scores

External capability axes:

    retention_small
    retention_medium

Hard feasibility gate:

    every one of the 34 instances must complete

Only after all 1000 worlds finish is the Gene Bank built.

## Round 1..49

At the start of a round the previous Gene Bank is frozen.

For each of the 1000 worlds:

1. sample one parent from the frozen Gene Bank;
2. parent weight = equal-axis total score squared;
3. inherit the parent Gene;
4. apply mutation;
5. evaluate the child on all 34 fixed instances.

No child can modify the Gene Bank while the round is still running.

After all 1000 worlds finish all 34 instances:

    old Bank + successful current-round Genes
        -> Pareto external capability selection
        -> epsilon deduplication
        -> crowding cap
        -> next-round Bank

Parent-child delta is diagnostic only. It does not decide Bank admission.

## Population learning signal

The primary convergence signal is not only the best Gene. Each round records
population statistics over all 1000 candidate worlds:

For each capability axis:

    mean
    median
    min
    max
    standard deviation

Also recorded:

    mean Small delta vs previous round
    mean Medium delta vs previous round
    population overall retention
    population worst-instance retention
    population best Gene by capability
    Bank best Gene by capability
    Bank global best

Therefore the core experimental question becomes:

    Does the mean capability of all 1000 inherited worlds improve from
    round to round?

This directly tests whether the Gene Bank inheritance process shifts the
population distribution upward rather than merely discovering an isolated
elite.

## Parent sampling

Let the two external axes be:

    s = retention_small
    m = retention_medium

Equal-axis total score:

    T = (s + m) / 2

Parent sampling weight:

    w = T^2

Because every parent has the same two axes, using the sum instead of the mean
would differ only by a constant factor and produce identical normalized
sampling probabilities.

## Logs

    world_events.jsonl

One row for every candidate world:

    round
    world index
    Gene ID
    parent ID
    origin
    scores
    overall retention
    worst retention
    exact matches
    BKS improvements
    parent-child delta diagnostic

Formal size:

    50,000 world rows

Round summaries:

    round_history.jsonl

One row per round with:

    1000-world population statistics
    capability-axis mean deltas
    Bank statistics
    parent-child diagnostics
    rollout counts

Bank snapshots:

    bank_snapshots/round_000.json
    ...
    bank_snapshots/round_049.json

Checkpoint:

    checkpoint.json

Resume is allowed only if the fixed instance set, benchmark bytes,
worlds-per-round, candidate-k, random seed, mutation settings, Pareto epsilon,
and Bank size are unchanged.

## V1.20 relationship

V1.20 remains useful for:

- public benchmark parsing;
- OPT/BKS references;
- distance validation;
- variable-size rollout;
- capability scoring;
- Pareto Bank implementation;
- small conventional evolutionary baseline.

V1.20 small-population formal runs must not be labeled as the primary SEGB
experiment.

V1.21 is the primary SEGB scheduler.
