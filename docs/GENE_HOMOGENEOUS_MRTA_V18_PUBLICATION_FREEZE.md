# V1.8 Publication Final Benchmark Freeze

## Status

The V1.8 consequence-aware architecture is frozen before this benchmark.

Model-code freeze commit:

`d58a621c4cf131e0c7a0a08bf64ea0b0a93d8476`

Frozen V1.8 run:

`runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7`

Comparison runs:

- V1.7 Direct: `runs/gene_mrta_v17t_scale/gene_mrta_v17t_scale_20261001_105828_seed7`
- V1.6-T-O: `runs/gene_mrta_v16to/gene_mrta_v16to_20260930_141419_seed7`

The earlier 97M worlds are analysis/development worlds and are not used as the publication-final untouched set.

## Frozen final set

The publication-final seed range is fixed before evaluation:

[
98{,}000{,}000 ldots 98{,}000{,}099
]

Exactly 100 worlds are requested.

The seed base and count are hard-coded in the benchmark module. Changing them
causes the publication runner to reject the run.

## Methods

Every world evaluates:

1. V1.8 consequence-aware Direct Assignment
2. V1.7 8D Direct Assignment
3. V1.6-T-O Bid + Greedy
4. Hungarian path-time
5. exact global MILP T* reference

MILP remains an external evaluation reference only. It does not teach actions.

## Exactness

Primary statistics include only worlds for which the global MILP proves
optimality.

Each world first receives a 300-second MILP limit. If exact optimality is not
proved, it is retried with 900 seconds. A world still not proven optimal is
reported as incomplete and is retried on future invocations.

The final publication benchmark is complete only when:

[
	ext{worlds_proven_optimal}=100
]

## Resumability

Each world is written atomically to:

`runs/gene_mrta_v18_publication_final_100/worlds/world_<seed>.json`

Completed exact worlds are reused on the next invocation. Therefore the same
command resumes after interruption without recomputing completed MILPs.

## Frozen statistics

Per method:

- mean
- standard deviation
- median
- minimum
- 10th percentile
- 25th percentile
- maximum
- deterministic bootstrap 95% CI for the mean

Paired comparisons for V1.8 versus V1.7, V1.6-T-O, and Hungarian:

- mean paired difference
- standard deviation of paired difference
- median paired difference
- deterministic paired-bootstrap 95% CI
- win/tie/loss count
- paired two-sided Wilcoxon signed-rank test
- paired Cohen dz effect size

The bootstrap uses 10,000 resamples with a fixed seed.

## Contamination rule

These 98M worlds are a final untouched evaluation set.

After inspecting their results, they must not be used to modify, select, or
tune V1.8. If the architecture or policy is changed based on these results,
the 98M set becomes development data and a new untouched final seed range must
be reserved before the next final claim.

## Run

```bash
git pull
bash tools/run_gene_mrta_v18_publication_mac.sh
```

If interrupted, run exactly the same command again.

Final outputs:

- `freeze_manifest.json`
- `publication_final_100.json`
- `publication_final_100.csv`
- one JSON file per world under `worlds/`
