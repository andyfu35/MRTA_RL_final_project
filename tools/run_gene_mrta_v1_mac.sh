#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v1.py

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v1.train     --generations 5     --population 24     --worlds-per-generation 3     --validation-worlds 12     --archive-per-axis 4     --seed 7     --log-every 1
elif [[ "$MODE" == "full" ]]; then
  python -m marl2d.gene_mrta_v1.train     --generations 80     --population 96     --worlds-per-generation 8     --validation-worlds 64     --archive-per-axis 8     --seed 7     --log-every 5
else
  echo "Usage: bash tools/run_gene_mrta_v1_mac.sh [smoke|full]" >&2
  exit 2
fi
