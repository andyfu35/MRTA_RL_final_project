# Gene-based Homogeneous MRTA V1.1

V1.1 is a calibration experiment built beside V1. The V1 code is preserved unchanged.

## Purpose

The first V1 run showed a small advantage over a nearest-task heuristic, but also exposed several experimental confounds. V1.1 removes those confounds before any larger training claim is made.

## What changes from V1

1. The gene keeps the same four observations:
   - `distance_norm`
   - `robot_load_norm`
   - `competition_norm`
   - `slack_norm`
2. The ineffective shared bias is removed. The gene has exactly four effective parameters.
3. `competition_norm` considers only robots that are currently free and able to participate in the allocation event.
4. Efficiency is changed to prevent the "do only a few very close tasks" loophole.
5. Per-robot task counts remain floating-point means in reports rather than being rounded independently.
6. Every generation is evaluated on fixed probe worlds for a comparable learning curve.
7. Training still uses new random worlds each generation.
8. Final gene selection uses the fixed probe set. The validation set is untouched until the final report.
9. A five-seed suite compares Gene against the same nearest-task baseline on the same validation worlds.

## Efficiency definition

For each completed assignment:

```text
route_value = 1 - distance / world_diagonal
```

V1.1 capability efficiency is:

```text
efficiency = sum(route_value over completed tasks) / total_number_of_tasks
```

Therefore every unfinished task contributes zero.

This is equivalent to:

```text
efficiency = completion * mean_route_efficiency
```

and guarantees:

```text
efficiency <= completion
```

The raw mean route efficiency is still reported separately as `route_efficiency`.

## Fixed probe and validation

Training worlds change every generation and depend on the training seed.

Probe worlds are fixed:

```text
probe_seed = 71,000,000
probe_worlds = 64
```

They are used only for reporting and final candidate selection. They are never used for parent selection.

Validation worlds are also fixed across training seeds:

```text
validation_seed = 91,000,000
validation_worlds = 128
```

They are not used for training, parent selection, or probe selection.

## Default experiment

Single seed:

```text
80 generations
96 genes
8 random training worlds per generation
64 fixed probe worlds
128 fixed validation worlds
8 genes per capability archive
seed = 7
```

Five-seed suite:

```text
seeds = 7, 17, 27, 37, 47
```

All five seeds use the same probe and validation worlds.

## Mac commands

Smoke test:

```bash
bash tools/run_gene_mrta_v11_mac.sh smoke
```

One complete seed:

```bash
bash tools/run_gene_mrta_v11_mac.sh single
```

Formal five-seed calibration experiment:

```bash
bash tools/run_gene_mrta_v11_mac.sh full
```

Single runs write to:

```text
runs/gene_mrta_v11/
```

The five-seed suite writes:

```text
runs/gene_mrta_v11_suite/
  multiseed_*/
    aggregate_summary.json
    per_seed.csv
    runs/
```

`history.csv` contains both changing-world training metrics and fixed-probe metrics. The fixed-probe columns are the correct columns for plotting learning progress across generations.
