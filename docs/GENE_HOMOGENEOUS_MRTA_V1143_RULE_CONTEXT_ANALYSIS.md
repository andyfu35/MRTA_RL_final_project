# V1.14.3 Offline Rule / Parent-Context Analysis

Status as of 2026-10-04:

IMPLEMENTED. NOT YET RUN.

## Purpose

V1.14.2 formal128 showed that adaptive mating is not universally superior to
the canonical center law, but it produced:

- positive mean_time delta;
- positive tail10 delta;
- the clearest directional gain on continuation_preservation;
- +3 dual-gate children;
- +3 four-capability children.

The next question is no longer whether adaptive mating is globally better.

The question is:

Which non-center Recombination Genes help, and for which parent contexts?

V1.14.3 is offline analysis only.

It does not train a new Policy Gene, mutate a Recombination Gene, or touch 99M.

## Source

Default source run:

runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7

Input:

pair_results.jsonl
summary.json

## Analysis

Only the already-completed 95M matched-pair data are reused.

The analysis reports:

1. each non-center Recombination Gene ID;
2. number of matched parent pairs using that rule;
3. per-axis Adaptive - Center mean delta;
4. per-axis W/T/L;
5. adaptive-vs-center dominance counts;
6. dual-gate rescue/loss counts;
7. four-capability rescue/loss counts;
8. exact-identical-child count;
9. parent capability-set context;
10. rule + parent-context combinations;
11. average deviation of alpha from 0.5;
12. average deviation of eta from 1.0;
13. top continuation-improving pairs;
14. four-capability rescue pairs;
15. non-center rules that collapse to the exact center child.

Rules with fewer than five uses are kept in the JSON but are not treated as
reliable rule candidates in the default ranking.

Rule-context combinations require at least three samples before appearing in
the default candidate-context table.

These thresholds are descriptive filters, not validation thresholds.

## Outputs

The analyzer writes into the existing V1.14.2 run directory:

- v1143_rule_context_analysis.json
- v1143_rule_table.csv
- v1143_parent_context_table.csv
- v1143_rule_context_table.csv

## Interpretation guardrail

This analysis reuses the same 95M development outcomes that generated the
hypothesis.

Therefore it can answer:

- what pattern exists in the observed development assay;
- which rules/contexts deserve a future test.

It cannot prove that a context-conditioned mating selector generalizes.

No new context-conditioned selector should be called validated until tested on
independent scenarios or repeated paired seeds.

## Command

    git pull
    bash tools/run_gene_mrta_v1143_analysis_mac.sh

After the output is interpreted and recorded, the V1.14 research series is
frozen for now.

The next major stage is V1.15 scalability / stress testing.
