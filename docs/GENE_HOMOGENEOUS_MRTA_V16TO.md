# V1.6-T-O: MILP Oracle-Guided Gene Evolution

V1.6-T-O preserves the complete V1.6-T environment, observation, Gene
parameterization, and online greedy matcher.

The only new training signal is a seventh capability archive:

```text
global_optimality_retention
```

defined on a cached MILP oracle dataset as:

```text
O = mean_k( T_gene(k) / T_star(k) )
```

where `T_star(k)` is a globally proven optimum from the existing V1.6-T MILP
oracle.

## What does not change

- 8D linear Gene
- greedy online matching
- 4 robots / 20 tasks
- obstacle-aware A* path costs
- variable service time
- priority
- soft deadline
- finite heterogeneous battery
- horizon
- six existing capability archives

MILP is never used online.

## Data separation

The seed spaces are intentionally separated:

```text
V1.6-T regular validation: 96,000,000...
global held-out test:        97,000,000...
MILP oracle train:          110,000,000...
MILP oracle probe:          120,000,000...
fresh evolution worlds:     130,000,000... plus evolution-seed offset
```

The 97M global test worlds are never used for parent selection, archive
selection, or oracle-guided evolution.

## Step 1: build the cached oracle dataset

Smoke:

```bash
bash tools/run_gene_mrta_v16to_mac.sh oracle-smoke
```

Full cached oracle dataset:

```bash
bash tools/run_gene_mrta_v16to_mac.sh oracle-full
```

The full default contains:

```text
256 MILP-optimal training worlds
64 MILP-optimal oracle-probe worlds
```

Only solver results with proven `optimal=True` are retained.

## Step 2: validate training wiring

```bash
bash tools/run_gene_mrta_v16to_mac.sh train-smoke
```

This bootstraps from:

```text
runs/gene_mrta_v16t_suite/multiseed_20260930_104923
```

unless `BOOTSTRAP_SUITE` is overridden.

## Step 3: long evolution

```bash
bash tools/run_gene_mrta_v16to_mac.sh long
```

Default long schedule:

```text
Generation    Fresh worlds/gen    Cached oracle worlds/gen
0-199         16                  16
200-499       32                  32
500-999       64                  64
1000-1999     128                 128
```

Other defaults:

```text
population = 256
archive_per_axis = 16
generations = 2000
oracle HOF = 64
```

The original six axes and the new oracle axis participate independently in
parent selection. There is still no scalar weighted reward.

## Step 4: held-out global-optimum test

After long training:

```bash
bash tools/run_gene_mrta_v16to_global_test_mac.sh
```

The launcher automatically selects the latest V1.6-T-O run and evaluates 20
held-out 97M worlds by default.

The principal paper metric is:

```text
GlobalOptimalityRetention = T_gene / T_star
OptimalityGap = 1 - GlobalOptimalityRetention
```

Hungarian Path-Time is retained only as an intermediate baseline.

## Research interpretation

If retention continues to improve with training scale, the previous gap was
primarily an optimization/data limitation.

If retention plateaus despite larger populations, more worlds, and oracle
guidance, the remaining gap becomes evidence for a representational limit of:

```text
8D linear bidder + greedy matching
```

rather than insufficient training.
