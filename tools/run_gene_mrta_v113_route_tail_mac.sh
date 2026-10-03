#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V110_CHECKPOINT="${V110_CHECKPOINT:-}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v18.py \
  tests/test_gene_mrta_v113_route_tail.py \
  tests/test_gene_mrta_v113_evolution.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ ! -f "$SCENARIO_BANK" ]]; then
  echo "Missing frozen V1.10 scenario bank: $SCENARIO_BANK" >&2
  exit 2
fi

if [[ -z "$V110_CHECKPOINT" ]]; then
  V110_CHECKPOINT="$(ls -1dt runs/gene_mrta_v110_mating/*/checkpoint.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$V110_CHECKPOINT" || ! -f "$V110_CHECKPOINT" ]]; then
  echo "Missing V1.10 checkpoint." >&2
  echo "Set V110_CHECKPOINT=/path/to/checkpoint.json" >&2
  exit 2
fi

echo "V110_CHECKPOINT=$V110_CHECKPOINT"

if [[ "$MODE" != "smoke" && ! -f "$V18_RUN/summary.json" ]]; then
  echo "Missing V1.8 common ancestor: $V18_RUN/summary.json" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v113.smoke \
    --scenario-bank "$SCENARIO_BANK" \
    --v110-checkpoint "$V110_CHECKPOINT" \
    --worlds 5

elif [[ "$MODE" == "evolve-smoke" ]]; then
  python -m marl2d.gene_mrta_v113.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v110-checkpoint "$V110_CHECKPOINT" \
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
    --mating-q-power 10 \
    --inheritance-threshold 0.95 \
    --mutation-sigma-start 0.03 \
    --mutation-sigma-end 0.02 \
    --mating-mutation-sigma 0 \
    --checkpoint-every 1 \
    --log-every 1 \
    --seed 7

elif [[ "$MODE" == "pilot50" ]]; then
  python -m marl2d.gene_mrta_v113.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v110-checkpoint "$V110_CHECKPOINT" \
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
    --mating-q-power 10 \
    --inheritance-threshold 0.95 \
    --mutation-sigma-start 0.08 \
    --mutation-sigma-end 0.02 \
    --mating-mutation-sigma 0 \
    --checkpoint-every 5 \
    --log-every 1 \
    --seed 7

elif [[ "$MODE" == "resume" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v113_route_tail_mac.sh resume <checkpoint.json>" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v113.evolve \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v110-checkpoint "$V110_CHECKPOINT" \
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
    --mating-q-power 10 \
    --inheritance-threshold 0.95 \
    --mutation-sigma-start 0.08 \
    --mutation-sigma-end 0.02 \
    --mating-mutation-sigma 0 \
    --checkpoint-every 5 \
    --log-every 1 \
    --seed 7

else
  echo "Usage: bash tools/run_gene_mrta_v113_route_tail_mac.sh [tests|smoke|evolve-smoke|pilot50|resume <checkpoint.json>]" >&2
  exit 2
fi
