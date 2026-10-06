# Gene MRTA V2.0 — Variable-Set Global Context Direction

This branch intentionally leaves V1.22 unchanged and starts a new architecture line.

## Input schema

Each remaining task is always 5D:

`[x, y, priority, deadline, service_time]`

Each robot is always 4D:

`[x, y, accumulated_distance, estimated_finish_time]`

The entity counts are variable. A world with `R` robots and `N` remaining tasks therefore has `R x 4` robot input and `N x 5` task input. Shared encoders process every row with the same Gene weights. Mean/max set pooling converts variable-size sets into fixed-size context vectors.

The policy scores all currently possible `R x N` robot-task pairs together, selects one pair, updates that robot state, removes the selected task, then re-encodes the complete remaining set. A single rollout continues until all tasks are assigned.

For public benchmarks that do not provide priority/deadline/service time, those three task fields are explicitly set to zero. `deadline=0` means no deadline.

## Capability axes

No fixed score ceiling or 0–1 capability normalization is used.

- `total_time` — minimize raw makespan after all assigned work completes.
- `priority` — minimize priority-weighted completion rank. High-priority tasks executed later produce a larger cost.
- `on_time_completed_tasks` — maximize the number of tasks completed before their own deadline. Total assigned/completed tasks are separately checked as a legality diagnostic.

The Gene Bank uses mixed-direction Pareto dominance on raw values without fixed score ceilings. To avoid preserving hundreds of numerically near-identical Pareto Genes, the archive uses an epsilon grid with default raw resolutions: `total_time=0.5 s`, `priority=0.10`, and `on_time_completed_tasks=0.5 task`. There is still no hard Bank size cap. Exact per-axis champions are always preserved. Parent sampling cannot safely sum raw axes with different scales, so it uses equal-axis percentile ranks and squares the resulting rank score.

## Fixed 100-seed suite

The suite contains 100 committed unique seeds. Each seed deterministically generates:

- 5–20 robots
- 10–100 tasks
- task position
- priority 1–10
- service time
- feasible deadline

Deadline construction first creates one deterministic earliest-finish baseline schedule, records every task completion time, then adds positive random slack. Therefore every generated world has at least one schedule that satisfies every deadline; impossible-by-construction deadline sets are avoided by construction.

## Local commands

```bash
git switch experiment/gene-global-set-mrta-v2
git pull
bash tools/run_gene_mrta_v20_mac.sh tests
bash tools/run_gene_mrta_v20_mac.sh smoke
bash tools/run_gene_mrta_v20_mac.sh train-50
```

`train-50` uses the fixed 100 worlds and defaults to 64 candidate Genes per round for the first local architecture validation. Increase with `V20_GENES_PER_ROUND=...` after correctness/performance is confirmed.

Every round writes `representatives/round_XXX.json`, including all 100 per-seed measurements, so total-time, priority-order, and deadline-completion progress can be checked seed by seed.
