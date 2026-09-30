#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
ORACLE_DATASET="${ORACLE_DATASET:-runs/gene_mrta_v16to_oracle/oracle_dataset.json}"
BOOTSTRAP_SUITE="${BOOTSTRAP_SUITE:-runs/gene_mrta_v16t_suite/multiseed_20260930_104923}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q   tests/test_gene_mrta_v16t.py   tests/test_gene_mrta_v16t_global_optimal.py   tests/test_gene_mrta_v16to.py

if [[ "$MODE" == "oracle-smoke" ]]; then
  python -m marl2d.gene_mrta_v16t.oracle_dataset     --train-count 8     --probe-count 4     --time-limit 60     --output runs/gene_mrta_v16to_oracle/oracle_dataset_smoke.json
elif [[ "$MODE" == "oracle-full" ]]; then
  python -m marl2d.gene_mrta_v16t.oracle_dataset     --train-count 256     --probe-count 64     --time-limit 300     --output "$ORACLE_DATASET"
elif [[ "$MODE" == "train-smoke" ]]; then
  if [[ ! -f "$ORACLE_DATASET" ]]; then
    echo "Missing oracle dataset: $ORACLE_DATASET" >&2
    echo "Run: bash tools/run_gene_mrta_v16to_mac.sh oracle-full" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v16t.oracle_train     --oracle-dataset "$ORACLE_DATASET"     --bootstrap-suite "$BOOTSTRAP_SUITE"     --generations 10     --population 64     --archive-per-axis 8     --world-schedule 0:8     --oracle-batch-schedule 0:8     --oracle-hof-limit 32     --oracle-probe-every 1     --log-every 1     --seed 7
elif [[ "$MODE" == "long" ]]; then
  if [[ ! -f "$ORACLE_DATASET" ]]; then
    echo "Missing oracle dataset: $ORACLE_DATASET" >&2
    echo "Run: bash tools/run_gene_mrta_v16to_mac.sh oracle-full" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v16t.oracle_train     --oracle-dataset "$ORACLE_DATASET"     --bootstrap-suite "$BOOTSTRAP_SUITE"     --generations 2000     --population 256     --archive-per-axis 16     --world-schedule 0:16,200:32,500:64,1000:128     --oracle-batch-schedule 0:16,200:32,500:64,1000:128     --oracle-hof-limit 64     --oracle-probe-every 10     --log-every 10     --seed 7
else
  echo "Usage: bash tools/run_gene_mrta_v16to_mac.sh [oracle-smoke|oracle-full|train-smoke|long]" >&2
  exit 2
fi
