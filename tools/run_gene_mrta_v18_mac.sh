#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
ORACLE_DATASET="${ORACLE_DATASET:-runs/gene_mrta_v16to_oracle/oracle_dataset.json}"
V17_RUN="${V17_RUN:-runs/gene_mrta_v17t_scale/gene_mrta_v17t_scale_20261001_105828_seed7}"
V16TO_RUN="${V16TO_RUN:-runs/gene_mrta_v16to/gene_mrta_v16to_20260930_141419_seed7}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q   tests/test_gene_mrta_v17.py   tests/test_gene_mrta_v17t_scale.py   tests/test_gene_mrta_v18.py

if [[ ! -f "$ORACLE_DATASET" ]]; then
  echo "Missing oracle dataset: $ORACLE_DATASET" >&2
  exit 2
fi

if [[ "$MODE" == "tests" ]]; then
  exit 0

elif [[ "$MODE" == "smoke" ]]; then
  if [[ ! -f "$V17_RUN/summary.json" ]]; then
    echo "Missing V1.7 summary: $V17_RUN/summary.json" >&2
    exit 2
  fi

  python -m marl2d.gene_mrta_v18.direct_time_scale     --oracle-dataset "$ORACLE_DATASET"     --bootstrap-v17-run "$V17_RUN"     --generations 20     --population 64     --archive-size 8     --hof-limit 16     --hof-add-per-generation 2     --oracle-batch-schedule 0:8     --probe-every 1     --mutation-sigma-start 0.25     --mutation-sigma-end 0.2478     --checkpoint-every 5     --log-every 1     --seed 7

elif [[ "$MODE" == "long" ]]; then
  if [[ ! -f "$V17_RUN/summary.json" ]]; then
    echo "Missing V1.7 summary: $V17_RUN/summary.json" >&2
    exit 2
  fi

  python -m marl2d.gene_mrta_v18.direct_time_scale     --oracle-dataset "$ORACLE_DATASET"     --bootstrap-v17-run "$V17_RUN"     --generations 1000     --population 256     --archive-size 32     --hof-limit 64     --hof-add-per-generation 4     --oracle-batch-schedule 0:16,200:32,500:64     --probe-every 10     --mutation-sigma-start 0.25     --mutation-sigma-end 0.1325     --checkpoint-every 25     --log-every 10     --seed 7

elif [[ "$MODE" == "resume" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v18_mac.sh resume <checkpoint.json>" >&2
    exit 2
  fi

  python -m marl2d.gene_mrta_v18.direct_time_scale     --oracle-dataset "$ORACLE_DATASET"     --resume "$CHECKPOINT"     --generations 1000     --population 256     --archive-size 32     --hof-limit 64     --hof-add-per-generation 4     --oracle-batch-schedule 0:16,200:32,500:64     --probe-every 10     --mutation-sigma-start 0.25     --mutation-sigma-end 0.1325     --checkpoint-every 25     --log-every 10     --seed 7

elif [[ "$MODE" == "finalize" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v18_mac.sh finalize <checkpoint.json>" >&2
    exit 2
  fi

  python -m marl2d.gene_mrta_v18.checkpoint_finalize     --checkpoint "$CHECKPOINT"     --oracle-dataset "$ORACLE_DATASET"

elif [[ "$MODE" == "heldout" ]]; then
  V18_RUN="${2:-}"
  if [[ -z "$V18_RUN" || ! -f "$V18_RUN/summary.json" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v18_mac.sh heldout <v18_run_dir>" >&2
    exit 2
  fi

  python -m marl2d.gene_mrta_v18.global_time_test     --v18-run "$V18_RUN"     --v17-run "$V17_RUN"     --v16to-run "$V16TO_RUN"     --worlds 20     --world-seed 97000000     --time-limit 300     --output-dir "$V18_RUN/heldout20"

else
  echo "Usage: bash tools/run_gene_mrta_v18_mac.sh [tests|smoke|long|resume <checkpoint>|finalize <checkpoint>|heldout <v18_run_dir>]" >&2
  exit 2
fi
