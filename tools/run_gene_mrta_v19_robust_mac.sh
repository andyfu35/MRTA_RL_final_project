#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
ORACLE_DATASET="${ORACLE_DATASET:-runs/gene_mrta_v16to_oracle/oracle_dataset.json}"
PUBLICATION_RESULT="${PUBLICATION_RESULT:-runs/gene_mrta_v18_publication_final_100/publication_final_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q   tests/test_gene_mrta_v17.py   tests/test_gene_mrta_v17t_scale.py   tests/test_gene_mrta_v18.py   tests/test_gene_mrta_v18_publication.py   tests/test_gene_mrta_v19_failure_analysis.py   tests/test_gene_mrta_v19_robust_bank.py

if [[ ! -f "$ORACLE_DATASET" ]]; then
  echo "Missing oracle dataset: $ORACLE_DATASET" >&2
  exit 2
fi
if [[ ! -f "$PUBLICATION_RESULT" ]]; then
  echo "Missing publication development result: $PUBLICATION_RESULT" >&2
  exit 2
fi
if [[ ! -f "$V18_RUN/summary.json" ]]; then
  echo "Missing V1.8 summary: $V18_RUN/summary.json" >&2
  exit 2
fi

if [[ "$MODE" == "tests" ]]; then
  exit 0

elif [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v19.train     --oracle-dataset "$ORACLE_DATASET"     --publication-result "$PUBLICATION_RESULT"     --bootstrap-v18-run "$V18_RUN"     --generations 20     --population 64     --archive-size-per-axis 4     --hof-limit-per-axis 4     --oracle-batch-schedule 0:8     --probe-every 1     --bootstrap-sigma 0.03     --mutation-sigma-start 0.06     --mutation-sigma-end 0.04     --checkpoint-every 5     --log-every 1     --seed 7

elif [[ "$MODE" == "long" ]]; then
  python -m marl2d.gene_mrta_v19.train     --oracle-dataset "$ORACLE_DATASET"     --publication-result "$PUBLICATION_RESULT"     --bootstrap-v18-run "$V18_RUN"     --generations 1000     --population 256     --archive-size-per-axis 16     --hof-limit-per-axis 16     --oracle-batch-schedule 0:16,200:32,500:64     --probe-every 10     --bootstrap-sigma 0.06     --mutation-sigma-start 0.12     --mutation-sigma-end 0.015     --checkpoint-every 25     --log-every 10     --seed 7

elif [[ "$MODE" == "resume" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v19_robust_mac.sh resume <checkpoint.json>" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v19.train     --oracle-dataset "$ORACLE_DATASET"     --publication-result "$PUBLICATION_RESULT"     --bootstrap-v18-run "$V18_RUN"     --resume "$CHECKPOINT"     --generations 1000     --population 256     --archive-size-per-axis 16     --hof-limit-per-axis 16     --oracle-batch-schedule 0:16,200:32,500:64     --probe-every 10     --bootstrap-sigma 0.06     --mutation-sigma-start 0.12     --mutation-sigma-end 0.015     --checkpoint-every 25     --log-every 10     --seed 7

else
  echo "Usage: bash tools/run_gene_mrta_v19_robust_mac.sh [tests|smoke|long|resume <checkpoint.json>]" >&2
  exit 2
fi
