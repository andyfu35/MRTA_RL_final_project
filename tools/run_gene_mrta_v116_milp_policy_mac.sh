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

elif [[ "$MODE" == "exact-regression" ]]; then
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "4x20,5x25" \
    --worlds-per-case 1 \
    --seed-base 116030000 \
    --milp-time-limit unlimited \
    --milp-solver-display \
    --heartbeat-seconds 30 \
    --run-dir "runs/gene_mrta_v116_exact_regression_seed116030000" \
    --policy-timeout 60 \
    --no-stop-after-zero-optimal

elif [[ "$MODE" == "exact-unlimited" ]]; then
  EXACT_WORLDS="${V116_EXACT_WORLDS:-1}"
  EXACT_SEED_BASE="${V116_EXACT_SEED_BASE:-116050000}"
  EXACT_RUN_DIR="${V116_EXACT_RUN_DIR:-runs/gene_mrta_v116_exact_unlimited_6r30_seed${EXACT_SEED_BASE}}"
  echo "V116_EXACT_WORLDS=$EXACT_WORLDS"
  echo "V116_EXACT_SEED_BASE=$EXACT_SEED_BASE"
  echo "V116_EXACT_RUN_DIR=$EXACT_RUN_DIR"
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "6x30" \
    --worlds-per-case "$EXACT_WORLDS" \
    --seed-base "$EXACT_SEED_BASE" \
    --milp-time-limit unlimited \
    --milp-solver-display \
    --heartbeat-seconds 30 \
    --run-dir "$EXACT_RUN_DIR" \
    --policy-timeout 60 \
    --no-stop-after-zero-optimal

elif [[ "$MODE" == "small10" ]]; then
  python -m marl2d.gene_mrta_v116.milp_policy_scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "2x10,3x15,4x20" \
    --worlds-per-case 10 \
    --seed-base 116020000 \
    --milp-time-limit 300 \
    --policy-timeout 60

else
  echo "Usage: bash tools/run_gene_mrta_v116_milp_policy_mac.sh [tests|smoke|ladder3|boundary5|exact-regression|exact-unlimited|small10]" >&2
  exit 2
fi
