#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
WORLDS="${2:-20}"
TIME_LIMIT="${TIME_LIMIT:-300}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -dt runs/gene_mrta_v16to/gene_mrta_v16to_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/summary.json" ]]; then
  echo "Could not find an oracle-guided V1.6-T-O run." >&2
  echo "Usage: bash tools/run_gene_mrta_v16to_global_test_mac.sh <run_dir> [worlds]" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
python -m marl2d.gene_mrta_v16t.oracle_global_test \
  --run-dir "$RUN_DIR" \
  --worlds "$WORLDS" \
  --world-seed 97000000 \
  --time-limit "$TIME_LIMIT" \
  --output-dir "runs/gene_mrta_v16to_global_test/$STAMP"
