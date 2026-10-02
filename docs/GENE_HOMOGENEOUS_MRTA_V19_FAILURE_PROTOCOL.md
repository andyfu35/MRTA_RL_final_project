# V1.9 development protocol: failure-derived capability axes

## Status change of the 98M worlds

The 98M range was a valid untouched publication-final benchmark for the frozen
V1.8 result. After that result was finalized, V1.9 development intentionally
uses the worst V1.8 worlds from the 98M set for failure analysis.

Therefore, from the start of V1.9 development, the 98M set is **development
data**. It must not be reused as the final generalization test for V1.9.

The planned untouched V1.9 final range is:

[
99{,}000{,}000 ldots 99{,}000{,}099
]

and must not be inspected before the V1.9 architecture and training procedure
are frozen.

## Stage A: Bottom-10 mechanism analysis

Run:

```bash
git pull
bash tools/run_gene_mrta_v19_failure_analysis_mac.sh
```

The analyzer selects the 10 proven-optimal 98M worlds with the lowest V1.8
retention directly from `publication_final_100.json`.

For every selected world it independently replays:

- V1.8 consequence-aware Direct Assignment;
- V1.7 8D Direct Assignment;
- V1.6-T-O Bid + Greedy;
- Hungarian path-time.

The analysis does not rerun or alter the frozen MILP optimum. It uses the
publication result as the T* reference and focuses on trajectory mechanisms.

## Event diagnostics

For each V1.8 event the analyzer records:

- all 12 pair features;
- selected policy logit and STOP margin;
- top policy pairs;
- top immediate-time pairs;
- selected pair's immediate-time rank;
- feasible robot-task edge count;
- feasible edge density;
- reachable remaining task ratio;
- per-task owner counts;
- singleton/scarce task count;
- stranded task count;
- per-robot future option counts;
- fleet option mass;
- option-conservation ratio;
- reachable-ratio change;
- newly stranded-task change.

Fleet option mass is the normalized sum, over remaining tasks, of the best
one-step future T utility available from any robot.

For an event, option conservation is measured as:

[
C =
rac{
  U_{realized} + O_{after}
}{
  O_{before}
}
]

where (U_{realized}) is the normalized immediate T utility collected by the
event and (O) is fleet option mass.

This is a diagnostic signal, not a new training objective yet.

## Diagnostic flags

The first analysis pass reports candidate mechanisms:

- `hard_for_all`
- `v18_regression_vs_v17`
- `matcher_or_v16_advantage`
- `continuation_collapse`
- `fleet_reserve_risk`
- `immediate_future_imbalance`

These are heuristic diagnostic flags. They do not by themselves establish a
causal mechanism. V1.9 capability axes are selected only after inspecting the
Bottom-10 aggregate pattern and event traces.

## Candidate axes

The analyzer may recommend the following independent Gene Bank axes when the
corresponding mechanism appears:

1. `continuation_preservation`
2. `fleet_option_reserve`
3. `immediate_time_capture`
4. `hard_world_time_specialist`

They are intended as separate archives / capabilities, not a weighted scalar
sum.

## Outputs

```text
runs/gene_mrta_v19_failure_analysis/
├── bottom10_failure_analysis.json
└── bottom10_failure_analysis.md
```

The JSON contains the full trajectories and machine-readable metrics. The
Markdown contains the Bottom-10 ranking, mechanism flags, candidate axes, and
compact per-world timelines.

## Next stage

Only after Stage A is reviewed do we define the exact V1.9 capability-axis
equations and training/archive logic.

V1.9 should initially retain the V1.8 policy architecture:

- observation: 12D
- hidden dimension: 8
- parameters: 148
- Direct Assignment
- learned STOP/WAIT

This isolates whether failure-derived Gene Bank capability preservation can
raise the lower tail without increasing model capacity.
