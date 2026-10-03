#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V113_CHECKPOINT="${V113_CHECKPOINT:-}"
SEED="${SEED:-7}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v113_route_tail.py \
  tests/test_gene_mrta_v113_evolution.py \
  tests/test_gene_mrta_v1141_recombination_correction.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ ! -f "$SCENARIO_BANK" ]]; then
  echo "Missing frozen 95M scenario bank: $SCENARIO_BANK" >&2
  exit 2
fi

if [[ ! -f "$V18_RUN/summary.json" ]]; then
  echo "Missing V1.8 common ancestor: $V18_RUN/summary.json" >&2
  exit 2
fi

if [[ -z "$V113_CHECKPOINT" ]]; then
  V113_CHECKPOINT="$(ls -1dt runs/gene_mrta_v113_route_tail_evolution/*/checkpoint.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$V113_CHECKPOINT" || ! -f "$V113_CHECKPOINT" ]]; then
  echo "Missing completed V1.13 checkpoint." >&2
  echo "Set V113_CHECKPOINT=/path/to/checkpoint.json" >&2
  exit 2
fi

echo "V113_CHECKPOINT=$V113_CHECKPOINT"
echo "SEED=$SEED"

run_smoke() {
  local recombination_mode="$1"
  local output_dir="$2"

  python -m marl2d.gene_mrta_v1141.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --recombination-mode "$recombination_mode" \
    --generations 3 \
    --normal-offspring 16 \
    --mating-offspring 16 \
    --mating-pairs 4 \
    --children-per-pair 4 \
    --screen-worlds 10 \
    --normal-full-per-axis 1 \
    --mating-full-candidates 4 \
    --archive-size-per-axis 4 \
    --hybrid-bank-limit 16 \
    --mutation-sigma-start 0.02 \
    --mutation-sigma-end 0.01 \
    --initial-recombination-genes 8 \
    --recombination-mutants-per-generation 2 \
    --recombination-bank-limit 12 \
    --recombination-specialist-size 2 \
    --recombination-pareto-limit 4 \
    --recombination-selection-power 2 \
    --recombination-uniform-fraction 0.25 \
    --recombination-gate-flip-rate 0.08 \
    --recombination-min-evidence 8 \
    --recombination-evidence-quantile 0.10 \
    --recombination-exploration-slots 3 \
    --recombination-sigma-start 0.35 \
    --recombination-sigma-tau 0.15 \
    --recombination-sigma-min 0.03 \
    --recombination-sigma-max 0.80 \
    --checkpoint-every 1 \
    --log-every 1 \
    --seed "$SEED" \
    --output-dir "$output_dir"
}

run_pilot50() {
  local recombination_mode="$1"
  local output_dir="$2"

  python -m marl2d.gene_mrta_v1141.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --recombination-mode "$recombination_mode" \
    --generations 50 \
    --normal-offspring 128 \
    --mating-offspring 128 \
    --mating-pairs 32 \
    --children-per-pair 4 \
    --screen-worlds 25 \
    --normal-full-per-axis 4 \
    --mating-full-candidates 16 \
    --archive-size-per-axis 16 \
    --hybrid-bank-limit 128 \
    --mutation-sigma-start 0.04 \
    --mutation-sigma-end 0.01 \
    --initial-recombination-genes 24 \
    --recombination-mutants-per-generation 8 \
    --recombination-bank-limit 32 \
    --recombination-specialist-size 6 \
    --recombination-pareto-limit 12 \
    --recombination-selection-power 2 \
    --recombination-uniform-fraction 0.25 \
    --recombination-gate-flip-rate 0.08 \
    --recombination-min-evidence 32 \
    --recombination-evidence-quantile 0.10 \
    --recombination-exploration-slots 8 \
    --recombination-sigma-start 0.35 \
    --recombination-sigma-tau 0.15 \
    --recombination-sigma-min 0.03 \
    --recombination-sigma-max 0.80 \
    --checkpoint-every 5 \
    --log-every 1 \
    --seed "$SEED" \
    --output-dir "$output_dir"
}

compare_latest() {
  local adaptive_root="$1"
  local center_root="$2"
  local output="$3"

  local adaptive_summary
  local center_summary
  adaptive_summary="$(ls -1dt "$adaptive_root"/*/summary.json | head -n 1)"
  center_summary="$(ls -1dt "$center_root"/*/summary.json | head -n 1)"

  python -m marl2d.gene_mrta_v1141.compare \
    --adaptive "$adaptive_summary" \
    --center "$center_summary" \
    --output "$output"

  echo "ADAPTIVE_SUMMARY=$adaptive_summary"
  echo "CENTER_SUMMARY=$center_summary"
  echo "COMPARISON=$output"
}

PAIR_ROOT="runs/gene_mrta_v1141_paired_seed${SEED}"
ADAPTIVE_ROOT="$PAIR_ROOT/adaptive"
CENTER_ROOT="$PAIR_ROOT/center"

if [[ "$MODE" == "smoke-adaptive" ]]; then
  run_smoke "adaptive" "$ADAPTIVE_ROOT"

elif [[ "$MODE" == "smoke-center" ]]; then
  run_smoke "center" "$CENTER_ROOT"

elif [[ "$MODE" == "paired-smoke" ]]; then
  run_smoke "adaptive" "$ADAPTIVE_ROOT"
  run_smoke "center" "$CENTER_ROOT"
  compare_latest \
    "$ADAPTIVE_ROOT" \
    "$CENTER_ROOT" \
    "$PAIR_ROOT/comparison_smoke.json"

elif [[ "$MODE" == "adaptive50" ]]; then
  run_pilot50 "adaptive" "$ADAPTIVE_ROOT"

elif [[ "$MODE" == "center50" ]]; then
  run_pilot50 "center" "$CENTER_ROOT"

elif [[ "$MODE" == "paired50" ]]; then
  run_pilot50 "adaptive" "$ADAPTIVE_ROOT"
  run_pilot50 "center" "$CENTER_ROOT"
  compare_latest \
    "$ADAPTIVE_ROOT" \
    "$CENTER_ROOT" \
    "$PAIR_ROOT/comparison_50.json"

elif [[ "$MODE" == "resume" ]]; then
  recombination_mode="${2:-}"
  checkpoint="${3:-}"
  if [[ "$recombination_mode" != "adaptive" && "$recombination_mode" != "center" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v1141_paired_mac.sh resume <adaptive|center> <checkpoint.json>" >&2
    exit 2
  fi
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "Missing V1.14.1 checkpoint." >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v1141.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --recombination-mode "$recombination_mode" \
    --resume "$checkpoint" \
    --generations 50 \
    --normal-offspring 128 \
    --mating-offspring 128 \
    --mating-pairs 32 \
    --children-per-pair 4 \
    --screen-worlds 25 \
    --normal-full-per-axis 4 \
    --mating-full-candidates 16 \
    --archive-size-per-axis 16 \
    --hybrid-bank-limit 128 \
    --mutation-sigma-start 0.04 \
    --mutation-sigma-end 0.01 \
    --initial-recombination-genes 24 \
    --recombination-mutants-per-generation 8 \
    --recombination-bank-limit 32 \
    --recombination-specialist-size 6 \
    --recombination-pareto-limit 12 \
    --recombination-selection-power 2 \
    --recombination-uniform-fraction 0.25 \
    --recombination-gate-flip-rate 0.08 \
    --recombination-min-evidence 32 \
    --recombination-evidence-quantile 0.10 \
    --recombination-exploration-slots 8 \
    --recombination-sigma-start 0.35 \
    --recombination-sigma-tau 0.15 \
    --recombination-sigma-min 0.03 \
    --recombination-sigma-max 0.80 \
    --checkpoint-every 5 \
    --log-every 1 \
    --seed "$SEED"

else
  echo "Usage: bash tools/run_gene_mrta_v1141_paired_mac.sh [tests|smoke-adaptive|smoke-center|paired-smoke|adaptive50|center50|paired50|resume <adaptive|center> <checkpoint.json>]" >&2
  exit 2
fi
