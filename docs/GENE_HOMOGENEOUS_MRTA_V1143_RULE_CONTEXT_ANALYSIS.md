# V1.14.3 Offline Rule / Parent-Context Analysis

Status as of 2026-10-04:

COMPLETED. V1.14 RECOMBINATION SERIES FROZEN FOR NOW.

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


## Completed analysis result

Date: 2026-10-04

Command:

    bash tools/run_gene_mrta_v1143_analysis_mac.sh

Tests:

    3 passed

Source:

    runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7

Output:

    runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_20261004_110044_seed7/v1143_rule_context_analysis.json

Observed counts:

- source pairs = 128
- non-center adaptive pairs = 115
- four-capability rescues = 3
- four-capability losses = 0
- candidate rule/context groups = 8
- non-center exact-center-equivalent children = 6

Non-center Adaptive - Center:

mean_time:
- mean delta = +0.00072172
- W/T/L = 47/25/43

tail10_time:
- mean delta = +0.00116809
- W/T/L = 40/41/34

continuation_preservation:
- mean delta = +0.00033111
- W/T/L = 62/16/37

fleet_option_reserve:
- mean delta = -0.00007559
- W/T/L = 51/18/46

Most notable reliable rules from the default >=5-use descriptive filter:

1. 6858fcfc2c9540c2f0fd
   - n = 18
   - active terms = 2
   - continuation mean delta = +0.0004531036432003304
   - continuation W/T/L = 11/1/6
   - four-capability rescue-minus-loss = +1

2. 5ef9b51bed957fe3b4b0
   - n = 16
   - active terms = 1
   - continuation mean delta = +0.00004560764417711749
   - continuation W/T/L = 5/6/5
   - four-capability rescue-minus-loss = +1
   - dominance margin = +1

3. e43f02f17b588a421ca9
   - n = 5
   - active terms = 3
   - continuation mean delta = +0.0004215382751630248
   - continuation W/T/L = 4/0/1
   - dominance margin = +2
   - sample count remains small

Other reliable rules showed weaker, neutral, or negative continuation effects.

Interpretation:

The development evidence is consistent with context-specific usefulness of
non-center recombination, not one globally superior non-center law.

The strongest current observation is:

- non-center rules generated 3 four-capability rescues and 0 four-capability losses;
- continuation was the clearest positive capability direction;
- benefits are distributed unevenly across rule phenotypes;
- eight rule/context combinations met the descriptive candidate filter.

This is hypothesis-generating evidence only because the same 95M development
assay is being re-analyzed.

No context-conditioned selector is trained or claimed validated here.

V1.14 series freeze decision:

Do not perform additional formula tuning on 95M now.

Preserve the current results and move to V1.15 scalability / limit stress
testing. A context-conditioned mating selector may be revisited later using
independent validation data.

99M remains untouched.
