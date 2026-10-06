#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-equivalence}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

python - <<'PY'
import torch
print("V122_TORCH", torch.__version__)
print("V122_MPS_BUILT", torch.backends.mps.is_built())
print("V122_MPS_AVAILABLE", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    raise SystemExit("V1.22 requires Apple MPS")
PY

if [[ "$MODE" == "tests" ]]; then
  pytest -q \
    tests/test_gene_mrta_v120.py \
    tests/test_gene_mrta_v121.py \
    tests/test_gene_mrta_v122.py

elif [[ "$MODE" == "equivalence" ]]; then
  V122_EQUIV_GENES="${V122_EQUIV_GENES:-8}"
  python -m marl2d.gene_mrta_v122.validate \
    equivalence \
    --device mps \
    --genes "$V122_EQUIV_GENES" \
    --candidate-k 32

elif [[ "$MODE" == "benchmark" ]]; then
  V122_BENCH_GENES="${V122_BENCH_GENES:-32}"
  python -m marl2d.gene_mrta_v122.validate \
    benchmark \
    --device mps \
    --genes "$V122_BENCH_GENES" \
    --candidate-k 32 \
    --instances mtsp51_3 mtsp100_10 kroa200_20

else
  echo "Usage: bash tools/run_gene_mrta_v122_mac.sh [tests|equivalence|benchmark]" >&2
  exit 2
fi
