#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-equivalence}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

ROOT="runs/gene_mrta_v122"
RUN_DIR="${V122_RUN_DIR:-$ROOT/formal_mps_1000w_34i_50r_seed121}"

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
  pytest -q     tests/test_gene_mrta_v120.py     tests/test_gene_mrta_v121.py     tests/test_gene_mrta_v122.py

elif [[ "$MODE" == "equivalence" ]]; then
  V122_EQUIV_GENES="${V122_EQUIV_GENES:-8}"
  python -m marl2d.gene_mrta_v122.validate     equivalence     --device mps     --genes "$V122_EQUIV_GENES"     --candidate-k 32

elif [[ "$MODE" == "benchmark" ]]; then
  V122_BENCH_GENES="${V122_BENCH_GENES:-32}"
  python -m marl2d.gene_mrta_v122.validate     benchmark     --device mps     --genes "$V122_BENCH_GENES"     --candidate-k 32     --instances mtsp51_3 mtsp100_10 kroa200_20

elif [[ "$MODE" == "train-formal" ]]; then
  V122_WORLDS_PER_ROUND="${V122_WORLDS_PER_ROUND:-1000}"
  V122_ROUNDS="${V122_ROUNDS:-50}"
  V122_CANDIDATE_K="${V122_CANDIDATE_K:-32}"
  V122_MPS_BATCH_SIZE="${V122_MPS_BATCH_SIZE:-32}"
  V122_SEED="${V122_SEED:-121}"

  echo "V122_RUN_DIR=$RUN_DIR"
  echo "V122_DEVICE=mps"
  echo "V122_WORLDS_PER_ROUND=$V122_WORLDS_PER_ROUND"
  echo "V122_FIXED_INSTANCES=34"
  echo "V122_ROUNDS=$V122_ROUNDS"
  echo "V122_ROLLOUTS_PER_ROUND=$((V122_WORLDS_PER_ROUND * 34))"
  echo "V122_PLANNED_TOTAL_ROLLOUTS=$((V122_WORLDS_PER_ROUND * 34 * V122_ROUNDS))"
  echo "V122_CANDIDATE_K=$V122_CANDIDATE_K"
  echo "V122_MPS_BATCH_SIZE=$V122_MPS_BATCH_SIZE"
  echo "V122_SEED=$V122_SEED"

  python -m marl2d.gene_mrta_v122.train     --run-dir "$RUN_DIR"     --worlds-per-round "$V122_WORLDS_PER_ROUND"     --rounds "$V122_ROUNDS"     --expected-instance-count 34     --candidate-k "$V122_CANDIDATE_K"     --mps-batch-size "$V122_MPS_BATCH_SIZE"     --bank-max-size 128     --seed "$V122_SEED"

else
  echo "Usage: bash tools/run_gene_mrta_v122_mac.sh [tests|equivalence|benchmark|train-formal]" >&2
  exit 2
fi
