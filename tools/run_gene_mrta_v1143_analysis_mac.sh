#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-analyze}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
RUN_DIR="${V1142_RUN_DIR:-}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v1143_rule_context_analysis.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -1dt runs/gene_mrta_v1142_matched_pair/gene_mrta_v1142_matched_pair_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/pair_results.jsonl" ]]; then
  echo "Missing completed V1.14.2 formal run." >&2
  echo "Set V1142_RUN_DIR=/path/to/gene_mrta_v1142_matched_pair_*" >&2
  exit 2
fi

echo "V1142_RUN_DIR=$RUN_DIR"

python -m marl2d.gene_mrta_v1143.analyze \
  --run-dir "$RUN_DIR" \
  --min-rule-count 5 \
  --min-context-count 3
