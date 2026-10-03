#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V110_CHECKPOINT="${V110_CHECKPOINT:-}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v18.py \
  tests/test_gene_mrta_v113_route_tail.py

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

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v113.smoke \
    --scenario-bank "$SCENARIO_BANK" \
    --v110-checkpoint "$V110_CHECKPOINT" \
    --worlds 5

else
  echo "Usage: bash tools/run_gene_mrta_v113_route_tail_mac.sh [tests|smoke]" >&2
  exit 2
fi
