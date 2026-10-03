#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V110_CHECKPOINT="${V110_CHECKPOINT:-}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v110_mating.py \
  tests/test_gene_mrta_v111_law_discovery.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ ! -f "$SCENARIO_BANK" ]]; then
  echo "Missing frozen V1.10 scenario bank: $SCENARIO_BANK" >&2
  exit 2
fi

if [[ ! -f "$V18_RUN/summary.json" ]]; then
  echo "Missing V1.8 common ancestor: $V18_RUN/summary.json" >&2
  exit 2
fi

if [[ -z "$V110_CHECKPOINT" ]]; then
  V110_CHECKPOINT="$(ls -1dt runs/gene_mrta_v110_mating/*/checkpoint.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$V110_CHECKPOINT" || ! -f "$V110_CHECKPOINT" ]]; then
  echo "Missing completed V1.10 checkpoint." >&2
  echo "Set V110_CHECKPOINT=/path/to/checkpoint.json" >&2
  exit 2
fi

echo "V110_CHECKPOINT=$V110_CHECKPOINT"

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v111.discover \
    --scenario-bank "$SCENARIO_BANK" \
    --v110-checkpoint "$V110_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --pairs 8 \
    --train-pairs 6 \
    --laws 8 \
    --screen-worlds 10 \
    --batch-size 64 \
    --seed 7

elif [[ "$MODE" == "pilot" ]]; then
  python -m marl2d.gene_mrta_v111.discover \
    --scenario-bank "$SCENARIO_BANK" \
    --v110-checkpoint "$V110_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --pairs 64 \
    --train-pairs 48 \
    --laws 32 \
    --screen-worlds 25 \
    --batch-size 256 \
    --seed 7

else
  echo "Usage: bash tools/run_gene_mrta_v111_law_discovery_mac.sh [tests|smoke|pilot]" >&2
  exit 2
fi
