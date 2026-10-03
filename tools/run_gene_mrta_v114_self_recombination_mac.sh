#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V113_CHECKPOINT="${V113_CHECKPOINT:-}"

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
  tests/test_gene_mrta_v114_recombination_bank.py

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

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v114.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
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
    --recombination-sigma-start 0.35 \
    --recombination-sigma-tau 0.15 \
    --recombination-sigma-min 0.03 \
    --recombination-sigma-max 0.80 \
    --checkpoint-every 1 \
    --log-every 1 \
    --seed 7

elif [[ "$MODE" == "pilot50" ]]; then
  python -m marl2d.gene_mrta_v114.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
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
    --recombination-sigma-start 0.35 \
    --recombination-sigma-tau 0.15 \
    --recombination-sigma-min 0.03 \
    --recombination-sigma-max 0.80 \
    --checkpoint-every 5 \
    --log-every 1 \
    --seed 7

elif [[ "$MODE" == "resume" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v114_self_recombination_mac.sh resume <checkpoint.json>" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v114.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --resume "$CHECKPOINT" \
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
    --recombination-sigma-start 0.35 \
    --recombination-sigma-tau 0.15 \
    --recombination-sigma-min 0.03 \
    --recombination-sigma-max 0.80 \
    --checkpoint-every 5 \
    --log-every 1 \
    --seed 7

else
  echo "Usage: bash tools/run_gene_mrta_v114_self_recombination_mac.sh [tests|smoke|pilot50|resume <checkpoint.json>]" >&2
  exit 2
fi
