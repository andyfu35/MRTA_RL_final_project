#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
V113_CHECKPOINT="${V113_CHECKPOINT:-}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v16t_global_optimal.py \
  tests/test_gene_mrta_v116_milp_policy_scaling.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ -z "$V113_CHECKPOINT" ]]; then
  V113_CHECKPOINT="$(ls -1dt runs/gene_mrta_v113_route_tail_evolution/*/checkpoint.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$V113_CHECKPOINT" || ! -f "$V113_CHECKPOINT" ]]; then
  echo "Missing completed V1.13 checkpoint." >&2
  exit 2
fi

echo "V113_CHECKPOINT=$V113_CHECKPOINT"

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "2x10,3x15,4x20" \
    --worlds-per-case 1 \
    --seed-base 116000000 \
    --milp-time-limit 60 \
    --policy-timeout 60

elif [[ "$MODE" == "ladder3" ]]; then
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "2x10,3x15,4x20,5x25,6x30,8x40" \
    --worlds-per-case 3 \
    --seed-base 116010000 \
    --milp-time-limit 300 \
    --policy-timeout 60

elif [[ "$MODE" == "boundary5" ]]; then
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "4x20,5x25,6x30" \
    --worlds-per-case 5 \
    --seed-base 116100000 \
    --milp-time-limit 300 \
    --policy-timeout 60

elif [[ "$MODE" == "small10" ]]; then
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "2x10,3x15,4x20" \
    --worlds-per-case 10 \
    --seed-base 116020000 \
    --milp-time-limit 300 \
    --policy-timeout 60

else
  echo "Usage: bash tools/run_gene_mrta_v116_milp_policy_mac.sh [tests|smoke|ladder3|boundary5|small10]" >&2
  exit 2
fi
