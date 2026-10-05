# V1.20 Public MinMax-mTSP Benchmark

Status: PRIMARY PUBLIC BENCHMARK IMPLEMENTED

## Why V1.20

V1.19 attempted to use the historical MTRPD public supplement. The original
host no longer resolves and the exact historical files were not recoverable
through the attempted archive paths. V1.19 is therefore frozen as an aborted
data-acquisition line; its synthetic structural smoke remains non-formal.

V1.20 switches to a public benchmark whose actual files are currently
available from the authors' repository:

    https://github.com/pengfeihe-angers/mils

Paper:

    Pengfei He, Jin-Kao Hao, Jinhui Xia
    Learning-guided iterated local search for the minmax multiple traveling
    salesman problem
    Computers & Operations Research 185:107255 (2026)
    DOI: 10.1016/j.cor.2025.107255

## Mirrored raw assets

The public raw data are mirrored directly into this repository:

    benchmarks/minmax_mtsp_mils/instances.zip
    benchmarks/minmax_mtsp_mils/Certification.zip
    benchmarks/minmax_mtsp_mils/SOURCE_README.md

Original Git blob IDs:

    instances.zip:
    878aeee12fc4453904ddb86d905baa82203c806d

    Certification.zip:
    f3d65efd832fdb89e88ad2c1fafe1aa5869bb27e

    README.md:
    ee921aa59868c59841f30c43154ff697c2a99bac

Mirror commit:

    ae4e5f5fdc8ac1f01fd822ecb1f90c48f0a4fa0d

Formal V1.20 does not download benchmark data at run time.

## Benchmark structure

instances.zip contains 77 instances.

Robot counts:

    m in {3, 5, 10, 20, 30}

Vertex counts span:

    51 .. 5915

Every instance supplies:

    name
    distance type
    coordinates
    robot/salesman count

Node 0 is the common depot after zero-based conversion. All other nodes are
tasks/cities.

The public paper divides the benchmark into:

    Set S: 41 small/medium instances
    Set L: 36 large instances

V1.20 uses the paper's split structure for generalization:

    Set S -> evolution/validation
    Set L -> protected large-scale test

Set L must not influence mutation selection, Gene Bank admission,
hyperparameters, or stopping decisions.

Within Set S, a deterministic SHA-256 split reserves approximately 20% for
validation. The exact split is printed by the launcher and is fixed by the
instance IDs.

## Reference solutions

Certification.zip contains 72 solution certificates with:

    objective
    routes
    run information

Five instance files do not have a certificate in the archive:

    mtsp51_3
    mtsp51_5
    mtsp51_10
    mtsp150_30
    gtsp150_30

Their reference values are taken directly from Table A.1 of the published
paper:

    mtsp51_3    159.57
    mtsp51_5    118.13
    mtsp51_10   112.07
    mtsp150_30  5246.49
    gtsp150_30  1554.64

The paper marks 22 of the 77 instances with an asterisk, indicating a known
exact optimum. The other references are best-known solutions (BKS), not
mathematically proven optima.

V1.20 preserves this distinction.

For an exact instance:

    retention = OPT / GeneObjective

and retention > 1 beyond tolerance is treated as a parsing/evaluation error.

For a BKS instance:

    retention = BKS / GeneObjective

and retention > 1 is legal and is recorded as a new-BKS candidate.

## Objective

The minmax mTSP objective is:

    minimize max_r RouteLength_r

Every route:

    starts at depot
    visits at least one task
    ends at depot

Every non-depot city is assigned exactly once.

For fixed robot speed this objective is directly proportional to total mission
completion time: the last-finishing robot determines the makespan.

## Distance conventions

The public files contain EUC_2D and ATT instances.

V1.20 verified the formulas against the authors' certificates.

Examples:

    mtsp100_3 certificate: 8509.16
    reconstructed raw Euclidean: 8509.162483...

    kroa200_3 certificate: 10691
    reconstructed raw Euclidean: 10691.0260...

    att532_3 certificate: 9926
    reconstructed ATT distance: 9926

Therefore:

    EUC_2D -> continuous Euclidean distance
    ATT    -> TSPLIB ATT pseudo-Euclidean distance

Do not replace EUC_2D with integer TSPLIB rounding for this benchmark.

## One shared Gene across sizes

V1.20 keeps the same 148-parameter RouteTailDirectGene layout.

There is no model per robot count and no model per task count.

One Gene receives variable candidate tensors and must handle all instances.

Because instances reach 5915 vertices, V1.20 does not score every remaining
task at every decoder step. It uses a nearest-candidate set selected with a
cKDTree.

Default:

    candidate_k = 32 per route tail

The candidate union changes dynamically as routes grow.

The decoder is otherwise parameter-compatible with the previous 148-scalar
Gene. Its step normalization uses the original full task count, not the
candidate subset size.

## Observation adapter

The public minmax benchmark has no priority, deadline, service-time, battery,
or obstacle constraints.

The 12D interface is retained to keep the Gene parameter layout unchanged.

Features represent:

1. geometric tail-to-task distance
2. benchmark edge distance
3. zero service time
4. equal priority / zero priority pressure
5. no deadline pressure
6. remaining normalized route-scale headroom
7. current route workload
8. competition from other robots
9. complete-graph reachability
10. projected route utility
11. opportunity cost to other robots
12. residual projected route headroom

No artificial reward is added.

## Hard legality

The benchmark requires every salesman to receive at least one city.

During the first assignments the decoder restricts legal rows to robots that
have not yet received a task. Once every robot owns one task, all robots are
open.

There is no legal STOP action before all tasks are assigned. Completion is a
problem constraint, not an optimization tradeoff.

## Evolution score

Per instance:

    retention_i = reference_i / objective_i

Gene-level diagnostics:

    overall_reference_retention
    worst_reference_retention
    exact_matches
    bks_improvements

Formal Pareto axes are size-band retention values present in the evolution
split:

    retention_small
    retention_medium

Large instances are protected and therefore cannot become a training axis.

## Paired inheritance

Every child is evaluated on exactly the same evolution instances as its
parent.

For every instance:

    delta_i = retention_i(child) - retention_i(parent)

paired_events.jsonl records:

    parent ID
    child ID
    overall delta
    per-axis delta
    instance wins/ties/losses
    exact-match delta
    BKS-improvement delta

This directly measures whether a mutation inheritance improved the public
benchmark result.

## Phase ordering

V1.20 Primary is mutation-only.

Do not activate mating/recombination during this phase.

After the mutation-only Gene Bank converges:

    freeze Bank
    identify specialists
    start a separate mating/fusion phase
    test whether inherited capabilities can combine
    evaluate the fused Gene on protected Set L

This isolates the causal value of recombination from ordinary mutation.

## Files

    src/marl2d/gene_mrta_v120/benchmark.py
    src/marl2d/gene_mrta_v120/gene.py
    src/marl2d/gene_mrta_v120/rollout.py
    src/marl2d/gene_mrta_v120/capabilities.py
    src/marl2d/gene_mrta_v120/bank.py
    src/marl2d/gene_mrta_v120/train.py
    src/marl2d/gene_mrta_v120/evaluate.py

    tests/test_gene_mrta_v120.py
    tools/run_gene_mrta_v120_mac.sh

## First local validation

    git pull
    bash tools/run_gene_mrta_v120_mac.sh tests
    bash tools/run_gene_mrta_v120_mac.sh describe
    bash tools/run_gene_mrta_v120_mac.sh smoke

Formal mutation-only training:

    caffeinate -dimsu bash tools/run_gene_mrta_v120_mac.sh train-formal

Status:

    bash tools/run_gene_mrta_v120_mac.sh status

Validation:

    bash tools/run_gene_mrta_v120_mac.sh evaluate-validation

Protected large-scale evaluation should be run only after the mutation-only
protocol and stopping rule are frozen:

    bash tools/run_gene_mrta_v120_mac.sh evaluate-protected


## Published objective precision

Public certificates and paper tables report objective values at finite decimal
precision. V1.20 must not interpret a lower full-precision reconstructed value
as a super-optimal solution when both values round to the same published
number.

Example observed during smoke:

    published exact reference: 2299.16
    Gene objective:            2299.157678942173

The Gene objective rounds to 2299.16. Therefore the two values are
indistinguishable at the published two-decimal precision.

V1.20 defines the comparison tolerance as half of one unit in the last
published decimal place, with a small numerical floor.

For a two-decimal reference:

    tolerance = 0.005

For a zero-decimal/integer reference:

    tolerance = 0.5

Exact OPT:
- within the published precision interval -> retention = 1.0, gap = 0;
- below the reference by more than the interval -> evaluator/reference
  inconsistency, abort.

BKS:
- within the published precision interval -> retention = 1.0, not a BKS
  improvement;
- below BKS by more than the interval -> legal BKS improvement candidate.

This prevents publication rounding from injecting artificial retention values
slightly above 1 into the Gene Bank.
