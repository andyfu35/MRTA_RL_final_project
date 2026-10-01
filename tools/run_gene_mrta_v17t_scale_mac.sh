#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
ORACLE_DATASET="${ORACLE_DATASET:-runs/gene_mrta_v16to_oracle/oracle_dataset.json}"
BOOTSTRAP_RUN="${BOOTSTRAP_RUN:-runs/gene_mrta_v17/gene_mrta_v17_20261001_095420_seed7}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q tests/test_gene_mrta_v17.py tests/test_gene_mrta_v17t_scale.py

if [[ ! -f "$ORACLE_DATASET" ]]; then
  echo "Missing oracle dataset: $ORACLE_DATASET" >&2
  exit 2
fi

if [[ "$MODE" == "tests" ]]; then
  exit 0
elif [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v17.direct_time_scale     --oracle-dataset "$ORACLE_DATASET"     --bootstrap-run "$BOOTSTRAP_RUN"     --generations 10     --population 64     --archive-size 8     --hof-limit 16     --hof-add-per-generation 2     --oracle-batch-schedule 0:8     --probe-every 1     --checkpoint-every 5     --log-every 1     --seed 7
elif [[ "$MODE" == "long" ]]; then
  python -m marl2d.gene_mrta_v17.direct_time_scale     --oracle-dataset "$ORACLE_DATASET"     --bootstrap-run "$BOOTSTRAP_RUN"     --generations 2000     --population 256     --archive-size 32     --hof-limit 64     --hof-add-per-generation 4     --oracle-batch-schedule 0:16,200:32,500:64,1000:128     --probe-every 10     --checkpoint-every 25     --log-every 10     --seed 7
elif [[ "$MODE" == "resume" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v17t_scale_mac.sh resume <checkpoint.json>" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v17.direct_time_scale     --oracle-dataset "$ORACLE_DATASET"     --resume "$CHECKPOINT"     --generations 2000     --population 256     --archive-size 32     --hof-limit 64     --hof-add-per-generation 4     --oracle-batch-schedule 0:16,200:32,500:64,1000:128     --probe-every 10     --checkpoint-every 25     --log-every 10     --seed 7
else
  echo "Usage: bash tools/run_gene_mrta_v17t_scale_mac.sh [tests|smoke|long|resume <checkpoint>]" >&2
  exit 2
fi
