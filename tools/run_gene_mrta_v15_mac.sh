#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v15.py

COMMON_ARGS=(
  --robot-speed 4
  --service-time-min 2
  --service-time-max 35
  --priority-min 0.1
  --priority-max 1.0
  --deadline-min 25
  --deadline-max 50
  --obstacle-count 8
  --obstacle-size-min 10
  --obstacle-size-max 18
  --obstacle-clearance 4
  --grid-resolution 5
)

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v15.train \
    --generations 10 \
    --population 48 \
    --worlds-per-generation 4 \
    --probe-worlds 16 \
    --validation-worlds 32 \
    --archive-per-axis 4 \
    --seed 7 \
    --log-every 1 \
    "${COMMON_ARGS[@]}"
elif [[ "$MODE" == "single" ]]; then
  python -m marl2d.gene_mrta_v15.train \
    --generations 100 \
    --population 128 \
    --worlds-per-generation 8 \
    --probe-worlds 64 \
    --validation-worlds 128 \
    --archive-per-axis 8 \
    --seed 7 \
    --log-every 10 \
    "${COMMON_ARGS[@]}"
elif [[ "$MODE" == "full" ]]; then
  python -m marl2d.gene_mrta_v15.suite \
    --seeds 7 17 27 37 47 \
    --generations 100 \
    --population 128 \
    --worlds-per-generation 8 \
    --probe-worlds 64 \
    --validation-worlds 128 \
    --archive-per-axis 8 \
    --log-every 10 \
    "${COMMON_ARGS[@]}"
else
  echo "Usage: bash tools/run_gene_mrta_v15_mac.sh [smoke|single|full]" >&2
  exit 2
fi
