#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="${1:-}"
WORLDS="${2:-1}"
TIME_LIMIT="${TIME_LIMIT:-300}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v16t_global_optimal.py

if [[ -z "$SUITE_DIR" ]]; then
  SUITE_DIR="$(ls -dt runs/gene_mrta_v16t_suite/multiseed_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$SUITE_DIR" || ! -d "$SUITE_DIR" ]]; then
  echo "Could not find a V1.6-T multiseed suite." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/gene_mrta_v16t_global_optimal/${STAMP}"

python -m marl2d.gene_mrta_v16t.global_optimal_benchmark \
  --suite-dir "$SUITE_DIR" \
  --gene-source time_optimality \
  --worlds "$WORLDS" \
  --world-seed 97000000 \
  --time-limit "$TIME_LIMIT" \
  --output-dir "$OUT_DIR"
