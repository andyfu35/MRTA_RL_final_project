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
  tests/test_gene_mrta_v115_scaling.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
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
  python -m marl2d.gene_mrta_v115.scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "4x20,8x40,16x80" \
    --worlds-per-case 1 \
    --seed-base 115000000 \
    --geometry-timeout 60 \
    --path-timeout 120 \
    --policy-timeout 120

elif [[ "$MODE" == "ladder3" ]]; then
  python -m marl2d.gene_mrta_v115.scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "4x20,8x40,16x80,32x160,64x320,128x640" \
    --worlds-per-case 3 \
    --seed-base 115010000 \
    --geometry-timeout 120 \
    --path-timeout 300 \
    --policy-timeout 300

elif [[ "$MODE" == "dense-smoke" ]]; then
  python -m marl2d.gene_mrta_v115.scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "8x80,16x160,32x320" \
    --worlds-per-case 1 \
    --seed-base 115020000 \
    --geometry-timeout 120 \
    --path-timeout 300 \
    --policy-timeout 300

elif [[ "$MODE" == "fixed-map-smoke" ]]; then
  python -m marl2d.gene_mrta_v115.scaling \
    --v113-checkpoint "$V113_CHECKPOINT" \
    --cases "4x20,8x40,16x80,32x160" \
    --worlds-per-case 1 \
    --seed-base 115030000 \
    --no-preserve-spatial-density \
    --geometry-timeout 120 \
    --path-timeout 300 \
    --policy-timeout 300

else
  echo "Usage: bash tools/run_gene_mrta_v115_scaling_mac.sh [tests|smoke|ladder3|dense-smoke|fixed-map-smoke]" >&2
  exit 2
fi
