#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-gene}"
PUBLICATION_RESULT="${PUBLICATION_RESULT:-runs/gene_mrta_v18_publication_final_100/publication_final_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V17_RUN="${V17_RUN:-runs/gene_mrta_v17t_scale/gene_mrta_v17t_scale_20261001_105828_seed7}"
V16TO_RUN="${V16TO_RUN:-runs/gene_mrta_v16to/gene_mrta_v16to_20260930_141419_seed7}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/gene_mrta_v19_failure_analysis}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q   tests/test_gene_mrta_v17.py   tests/test_gene_mrta_v17t_scale.py   tests/test_gene_mrta_v18.py   tests/test_gene_mrta_v18_publication.py   tests/test_gene_mrta_v19_failure_analysis.py

if [[ ! -f "$PUBLICATION_RESULT" ]]; then
  echo "Missing publication result: $PUBLICATION_RESULT" >&2
  exit 2
fi

for RUN in "$V18_RUN" "$V17_RUN" "$V16TO_RUN"; do
  if [[ ! -f "$RUN/summary.json" ]]; then
    echo "Missing model summary: $RUN/summary.json" >&2
    exit 2
  fi
done

python -m marl2d.gene_mrta_v18.bottom10_failure_analysis   --publication-result "$PUBLICATION_RESULT"   --v18-run "$V18_RUN"   --v17-run "$V17_RUN"   --v16to-run "$V16TO_RUN"   --bottom 10   --output-dir "$OUTPUT_DIR"
