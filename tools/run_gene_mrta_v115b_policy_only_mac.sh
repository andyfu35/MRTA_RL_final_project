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
  tests/test_gene_mrta_v115_scaling.py \
  tests/test_gene_mrta_v115b_policy_only.py

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
  python -m marl2d.gene_mrta_v115.policy_only \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "64x320,128x640" \
    --worlds-per-case 1 \
    --seed-base 115100000 \
    --world-timeout 120 \
    --policy-timeout 300

elif [[ "$MODE" == "ladder3" ]]; then
  python -m marl2d.gene_mrta_v115.policy_only \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "64x320,128x640,256x1280" \
    --worlds-per-case 3 \
    --seed-base 115110000 \
    --world-timeout 180 \
    --policy-timeout 300

elif [[ "$MODE" == "probe384" ]]; then
  python -m marl2d.gene_mrta_v115.policy_only \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "384x1920" \
    --worlds-per-case 1 \
    --seed-base 115140000 \
    --world-timeout 300 \
    --policy-timeout 300

elif [[ "$MODE" == "probe388" ]]; then
  python -m marl2d.gene_mrta_v115.policy_only \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "388x1940" \
    --worlds-per-case 1 \
    --seed-base 115150000 \
    --world-timeout 300 \
    --policy-timeout 300

elif [[ "$MODE" == "extreme512" ]]; then
  python -m marl2d.gene_mrta_v115.policy_only \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "512x2560" \
    --worlds-per-case 1 \
    --seed-base 115120000 \
    --world-timeout 300 \
    --policy-timeout 300

elif [[ "$MODE" == "extreme1024" ]]; then
  python -m marl2d.gene_mrta_v115.policy_only \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "1024x5120" \
    --worlds-per-case 1 \
    --seed-base 115130000 \
    --world-timeout 300 \
    --policy-timeout 600

else
  echo "Usage: bash tools/run_gene_mrta_v115b_policy_only_mac.sh [tests|smoke|ladder3|probe384|probe388|extreme512|extreme1024]" >&2
  exit 2
fi
