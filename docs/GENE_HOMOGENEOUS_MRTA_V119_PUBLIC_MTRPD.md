# V1.19 Public MTRPD Benchmark Evolution

Status: IMPLEMENTED CORE / PUBLIC RAW IMPORT PENDING INSPECTION

## Goal

Replace self-generated optimum references with a public benchmark whose
instances and exact reference solutions were published independently of SEGB.

Primary source:

Luo, Qin, Lim (2014), Multiple Traveling Repairman Problem with Distance
Constraints (MTRPD).

The original benchmark contains 180 test instances derived from six TSPLIB
families:

- brd14051
- d15112
- d18512
- fnl4461
- nrw1379
- pr1002

For each family, ten subsets are generated at 30, 40 and 50 total vertices
(depot included), giving 6 x 10 x 3 = 180 instances.

The original construction uses n = 29, 39, 49 customers plus one depot.
The number of available repairmen is instance-dependent rather than hard-coded:

    K = max(K_ini, ceil(n / 5))

The route distance limit is:

    L = 2 * d_max

The paper reports proven optimal solutions for 179 / 180 instances.

## Core research question

Can one fixed-size 148-parameter RouteTailDirectGene learn an assignment law
that works across public MTRPD instances with changing:

- robot count K;
- customer/task count;
- geometry;
- route-distance constraints;

while approaching independently published global optima?

## Public objective

Every customer must be visited exactly once.

Every repairman starts and ends at the common depot.

Each route must satisfy:

    route_length <= L

MTRPD minimizes total customer latency:

    Cost(g, i) = sum_j arrival_time_j

For a public instance i with proven optimum OPT_i:

    retention_i(g) = OPT_i / Cost(g, i)

A value of 1 means exact optimum.

If a Gene does not complete every customer legally:

    retention_i(g) = 0

If a computed retention exceeds 1 beyond floating tolerance, training aborts.
This is treated as evidence of a distance/reference import mismatch rather than
as a super-optimal Gene.

## Native MTRPD rollout

V1.19 does not approximate MTRPD using the old battery environment.

At every route-tail append, pair (robot, task) is eligible only when:

    used_distance_r
    + distance(tail_r, task)
    + distance(task, depot)
    <= L

Therefore every accepted append preserves a legal return to depot.

The final route length is rechecked after planning.

Service time is zero and all tasks are equally weighted in the public MTRPD
objective.

## One shared variable-size Gene

The Policy remains the same 148-parameter, hidden_dim=8 route-tail Gene.

No separate model is trained per robot/task count.

The action tensor changes shape with each public instance:

    R_i x T_i

but model parameter count remains fixed.

## Development / validation / protected split

The canonical split is deterministic within each family x vertex-count group.

For the 10 public replicates:

    0..3 -> evolution
    4..6 -> validation
    7..9 -> protected_test

Across all 18 family x size groups this gives:

    Evolution:      72 instances
    Validation:     54 instances
    Protected test: 54 instances
    Total:         180 instances

Only proven-optimal instances may enter the evolution split. If the single
non-proven instance falls in evolution, the canonical importer must move it out
of evolution before formal training.

Protected test instances must never influence mutation selection, Gene Bank
admission, hyperparameters, or stopping decisions.

## V1.19 capability axes

The mutation-only Gene Bank preserves scale specialists:

    opt_retention_v30
    opt_retention_v40
    opt_retention_v50

Each is the mean published-optimum retention across the evolution instances at
that vertex count.

The following are diagnostics, not extra Pareto axes:

- overall optimum retention;
- worst-instance optimum retention;
- exact-optimum match count;
- per-instance retention;
- completion;
- robot count.

This preserves different scale abilities before mating.

## Paired inheritance evaluation

Every mutation child is evaluated on exactly the same evolution instances as
its parent.

For every instance:

    delta_i = retention_i(child) - retention_i(parent)

Every child writes a paired event containing:

- parent ID;
- child ID;
- overall delta;
- per-scale delta;
- instance W/T/L;
- success gains;
- success losses;
- parent and child overall retention.

This directly answers whether an inheritance step improved benchmark
performance.

## Feasibility bootstrap

Before a Gene can enter the formal benchmark Bank it must succeed on every
evolution instance.

Before that point, bootstrap ranking is lexicographic:

1. worst instance completion;
2. mean completion;
3. count of fully successful instances;
4. overall optimum retention.

Completion remains a prerequisite rather than a tradeable objective.

## Parent sampling

V1.19 primary phase is mutation-only.

Parents are sampled from the current formal Pareto Bank when available.
Sampling probability is proportional to:

    overall_retention^2

Before the formal Bank exists, the same squared-pressure principle is applied
to the feasibility bootstrap score.

## Mating is intentionally deferred

V1.19 must first converge under mutation/inheritance.

Only after the mutation-only Bank is characterized do we start the next phase:

    converged scale-specialist Bank
    -> paired mating / recombination
    -> test whether capabilities can be fused
    -> search for one universal near-global-optimum Gene

No mating operator is active in V1.19 primary training.

## Files

Core:

    src/marl2d/gene_mrta_v119/benchmark.py
    src/marl2d/gene_mrta_v119/rollout.py
    src/marl2d/gene_mrta_v119/capabilities.py
    src/marl2d/gene_mrta_v119/bank.py
    src/marl2d/gene_mrta_v119/train.py
    src/marl2d/gene_mrta_v119/dataset_tool.py
    src/marl2d/gene_mrta_v119/fixture.py

Tests:

    tests/test_gene_mrta_v119.py

Launcher:

    tools/run_gene_mrta_v119_mac.sh

## Public data status

The historical supplement is cited by the original work at:

    https://www.computational-logistics.org/orlib/mtrpd/

The supplement is not copied or reconstructed from guessed values in this
repository.

The V1.19 data tool can attempt to download and preserve the raw supplement:

    bash tools/run_gene_mrta_v119_mac.sh download

Then inspect exact raw formats:

    bash tools/run_gene_mrta_v119_mac.sh inspect-raw

Only after inspecting the authoritative raw files should an importer create:

    benchmarks/mtrpd_public/mtrpd_public_manifest.json

Do not fabricate missing optimum values, K values, coordinates, or routes.

## First execution sequence

Structural validation:

    git pull
    bash tools/run_gene_mrta_v119_mac.sh tests
    bash tools/run_gene_mrta_v119_mac.sh smoke

Public data acquisition:

    bash tools/run_gene_mrta_v119_mac.sh download
    bash tools/run_gene_mrta_v119_mac.sh inspect-raw

Formal training becomes eligible after the canonical public manifest is built:

    bash tools/run_gene_mrta_v119_mac.sh describe
    caffeinate -dimsu bash tools/run_gene_mrta_v119_mac.sh train-formal

Status:

    bash tools/run_gene_mrta_v119_mac.sh status
