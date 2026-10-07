# Gene Global-Set MRTA V2.0 — Zero-Shot Size Generalization

Status: IMPLEMENTED, READY TO RUN  
Branch: `experiment/gene-global-set-mrta-v2`

## Purpose

Freeze the learned V2.0 specialists and test whether the same 139-parameter
set policy generalizes to robot/task cardinalities that were never used during
evolution.

This is an evaluation-only experiment. OOD results must never enter:

- Gene Bank admission;
- parent selection;
- mutation;
- retraining;
- representative selection.

## Frozen source run

Checkpoint:

`runs/gene_mrta_v20/longrun_1024g_200r_seed200/checkpoint.json`

Training distribution:

- fixed 100 training seeds;
- robots: 5..20;
- tasks: 10..100;
- world size: 100 x 100;
- 1024 Genes per round;
- 200 rounds indexed 0..199;
- epsilon-grid Bank resolution:
  - total_time = 0.5 s;
  - priority = 0.10 rank;
  - on_time_completed_tasks = 0.5 task.

At Round 199 the Bank contains 222 Genes.

Current frozen specialists:

### Total-Time specialist

Record ID:

`5b448ba2073a90eba8d5`

Training-suite scores:

- total_time = 58.07686007499695 s;
- priority = 24.155660061836244 rank;
- on_time_completed_tasks = 35.98.

### On-Time specialist

Record ID:

`ba1efa666ac266786b69`

Training-suite scores:

- total_time = 62.88902631759643 s;
- priority = 27.21209596633911 rank;
- on_time_completed_tasks = 55.09.

The launcher does not hard-code these IDs. By default it loads the checkpoint
and freezes the current exact champion on each axis. Optional environment
variables can pin explicit IDs.

## OOD grid

Default cells:

| Robots | Tasks | Regime |
|---:|---:|---|
| 10 | 50 | in-distribution control |
| 20 | 100 | training boundary |
| 25 | 100 | robot OOD |
| 40 | 100 | robot OOD |
| 60 | 100 | robot OOD |
| 20 | 125 | task OOD |
| 20 | 150 | task OOD |
| 20 | 200 | task OOD |
| 20 | 300 | task OOD |
| 25 | 125 | both OOD |
| 40 | 200 | both OOD |
| 60 | 300 | both OOD |

All OOD seeds are deterministic, unique, and explicitly disjoint from the
100 training seeds.

Both frozen Genes are evaluated on the exact same newly generated worlds in
each cell.

## Primary outputs

For each Gene and each size cell:

1. raw Total Time in seconds;
2. mean Baseline Time in seconds;
3. mean per-world TotalTime / BaselineTime;
4. median per-world TotalTime / BaselineTime;
5. raw Priority weighted completion rank;
6. exact mean per-world On-time completion percentage;
7. total distance;
8. shared two-Gene cell evaluation runtime.

The normalized time quantity is:

[
\mathrm{NormalizedTime}
=
\frac{1}{S}
\sum_{s=1}^{S}
\frac{T_{gene,s}}{T_{baseline,s}}
]

This is preferred to directly comparing raw seconds across task scales.

The On-time percentage is the exact mean of per-world deadline completion
rates produced by the evaluator, not a ratio-of-averages approximation.

## Outputs

Each run creates:

- `protocol.json`
- `summary.json`
- `summary.csv`
- `per_seed.jsonl`

The protocol file records the source checkpoint, completed round, selected
frozen Gene IDs, all OOD seeds, all cells, and the no-feedback rule.

## Commands

Regression tests:

```bash
bash tools/run_gene_mrta_v20_mac.sh tests
```

20-seed-per-cell OOD smoke:

```bash
bash tools/run_gene_mrta_v20_mac.sh ood-smoke
```

100-seed-per-cell formal OOD run:

```bash
bash tools/run_gene_mrta_v20_mac.sh ood-formal
```

Optional custom grid:

```bash
V20_OOD_CELLS='20x100,25x125,40x200,60x300,100x500' \
V20_OOD_SEEDS=20 \
bash tools/run_gene_mrta_v20_mac.sh ood-smoke
```

Optional explicit Gene pinning:

```bash
V20_OOD_TOTAL_TIME_GENE_ID=5b448ba2073a90eba8d5 \
V20_OOD_ONTIME_GENE_ID=ba1efa666ac266786b69 \
bash tools/run_gene_mrta_v20_mac.sh ood-smoke
```

## Interpretation rule

A larger raw makespan at larger task counts is not automatically a
generalization failure.

Primary size-generalization evidence is:

- stability of `TotalTime / BaselineTime`;
- stability of exact On-time percentage;
- smooth rather than catastrophic degradation outside 20 robots / 100 tasks.

Only after the default grid is measured should an extreme 100R/500T cell be
added. The extreme test is descriptive unless repeated over enough unseen
seeds.
