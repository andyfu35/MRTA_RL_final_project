#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v117_schema.py \
  tests/test_gene_mrta_v117_oracle.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

echo "V1.17 Stage-A trainer is intentionally not launched yet." >&2
echo "Freeze the canonical Task/Robot schema and Stage-A base axes first." >&2
exit 2
