#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="${1:-}"
GENE_SOURCE="${2:-time_optimality}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v16t.py tests/test_gene_mrta_v16t_hungarian.py

if [[ -z "$SUITE_DIR" ]]; then
  SUITE_DIR="$(ls -dt runs/gene_mrta_v16t_suite/multiseed_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$SUITE_DIR" || ! -d "$SUITE_DIR" ]]; then
  echo "Could not find a V1.6-T multiseed suite directory." >&2
  echo "Usage: bash tools/run_gene_mrta_v16t_hungarian_benchmark_mac.sh <suite_dir> [gene_source]" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/gene_mrta_v16t_hungarian_benchmark/${STAMP}_${GENE_SOURCE}"

python -m marl2d.gene_mrta_v16t.hungarian_benchmark \
  --suite-dir "$SUITE_DIR" \
  --gene-source "$GENE_SOURCE" \
  --worlds 128 \
  --world-seed 97000000 \
  --timing-repeats 3 \
  --scaling-sizes 4 8 16 32 64 100 \
  --task-ratio 5 \
  --scaling-repeats 50 \
  --output-dir "$OUT_DIR"
