# V1.7 Direct Assignment + MILP Capability Oracles

V1.7 keeps the V1.6-T environment unchanged but removes the external
pairwise-bid matcher from the learned decision path.

## Direct assignment policy

Input at every decision event remains the same 8D robot-task observation:

1. euclidean distance
2. obstacle-aware path cost
3. service time
4. priority
5. deadline remaining
6. battery remaining
7. robot workload
8. competition

The Gene sees the full R x T x 8 tensor. It emits one (robot, task) action,
masks the selected robot and task, recomputes global/robot/task context, then
emits the next action. There is no external Hungarian or greedy matching
optimizer in deployment.

The first architecture has hidden_dim=8 and 106 evolvable parameters.

## MILP role

MILP does not provide target actions. It only computes per-world capability
references.

Exact MILP references:

- completion C*
- efficiency E*
- priority satisfaction P*
- deadline satisfaction D*
- time optimality T*

Historical balance is nonlinear because it uses completion * Jain(workload).
For V1.7 pilot, its denominator is the analytical upper bound C* because
Jain<=1. This is explicitly not claimed as an exact balance optimum.

Capability scoring is performed per world:

    S_a(W) = Metric_a(Gene,W) / OracleReference_a(W)

and then averaged across worlds.

Each capability has its own archive. There is no weighted scalar reward.

## Smoke experiment

    git pull
    bash tools/run_gene_mrta_v17_mac.sh smoke

This runs tests, builds 2 train + 1 probe multi-axis oracle worlds, and evolves
a small direct-assignment population for 20 generations.

## Pilot experiment

After smoke passes:

    bash tools/run_gene_mrta_v17_mac.sh oracle-pilot
    bash tools/run_gene_mrta_v17_mac.sh pilot

Pilot defaults:

- 16 oracle training worlds
- 4 oracle probe worlds
- 100 generations
- population 96
- six independent capability archives
- progressive oracle batch 4 -> 8 -> 16

The pilot is intentionally small. If direct assignment improves capability
scores, the next experiment will scale oracle worlds, population, generations,
and held-out comparison against V1.6-T-O, Hungarian, and MILP.
