#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-gene}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V17_RUN="${V17_RUN:-runs/gene_mrta_v17t_scale/gene_mrta_v17t_scale_20261001_105828_seed7}"
V16TO_RUN="${V16TO_RUN:-runs/gene_mrta_v16to/gene_mrta_v16to_20260930_141419_seed7}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/gene_mrta_v18_publication_final_100}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q   tests/test_gene_mrta_v17.py   tests/test_gene_mrta_v17t_scale.py   tests/test_gene_mrta_v18.py   tests/test_gene_mrta_v18_publication.py

for RUN in "$V18_RUN" "$V17_RUN" "$V16TO_RUN"; do
  if [[ ! -f "$RUN/summary.json" ]]; then
    echo "Missing frozen summary: $RUN/summary.json" >&2
    exit 2
  fi
done

python -m marl2d.gene_mrta_v18.publication_benchmark   --v18-run "$V18_RUN"   --v17-run "$V17_RUN"   --v16to-run "$V16TO_RUN"   --world-seed 98000000   --worlds 100   --time-limit 300   --retry-time-limit 900   --output-dir "$OUTPUT_DIR"
