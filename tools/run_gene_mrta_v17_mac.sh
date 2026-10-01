#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SMOKE_ORACLE="${SMOKE_ORACLE:-runs/gene_mrta_v17_oracle/capability_oracle_smoke.json}"
PILOT_ORACLE="${PILOT_ORACLE:-runs/gene_mrta_v17_oracle/capability_oracle.json}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v17.py

build_smoke_oracle() {
  python -m marl2d.gene_mrta_v17.oracle_dataset     --train-count 2     --probe-count 1     --time-limit 60     --max-attempt-factor 6     --output "$SMOKE_ORACLE"
}

train_smoke() {
  if [[ ! -f "$SMOKE_ORACLE" ]]; then
    build_smoke_oracle
  fi
  python -m marl2d.gene_mrta_v17.train     --oracle-dataset "$SMOKE_ORACLE"     --generations 20     --population 32     --archive-per-axis 3     --hidden-dim 8     --oracle-batch-schedule 0:2     --probe-every 1     --probe-hof-per-axis 2     --log-every 1     --seed 7
}

if [[ "$MODE" == "tests" ]]; then
  exit 0
elif [[ "$MODE" == "oracle-smoke" ]]; then
  build_smoke_oracle
elif [[ "$MODE" == "train-smoke" ]]; then
  train_smoke
elif [[ "$MODE" == "smoke" ]]; then
  build_smoke_oracle
  train_smoke
elif [[ "$MODE" == "oracle-pilot" ]]; then
  python -m marl2d.gene_mrta_v17.oracle_dataset     --train-count 16     --probe-count 4     --time-limit 180     --max-attempt-factor 6     --output "$PILOT_ORACLE"
elif [[ "$MODE" == "pilot" ]]; then
  if [[ ! -f "$PILOT_ORACLE" ]]; then
    echo "Missing pilot oracle dataset: $PILOT_ORACLE" >&2
    echo "Run: bash tools/run_gene_mrta_v17_mac.sh oracle-pilot" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v17.train     --oracle-dataset "$PILOT_ORACLE"     --generations 100     --population 96     --archive-per-axis 6     --hidden-dim 8     --oracle-batch-schedule 0:4,25:8,60:16     --probe-every 5     --probe-hof-per-axis 4     --log-every 5     --seed 7
else
  echo "Usage: bash tools/run_gene_mrta_v17_mac.sh [tests|oracle-smoke|train-smoke|smoke|oracle-pilot|pilot]" >&2
  exit 2
fi
