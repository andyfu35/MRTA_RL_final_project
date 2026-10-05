#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SMOKE_BANK="${V117_SMOKE_BANK:-runs/gene_mrta_v117_stage_a/oracle_smoke_2r10t.json}"
FORMAL_BANK="${V117_FORMAL_BANK:-runs/gene_mrta_v117_stage_a/oracle_formal_4r20t.json}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v117_schema.py \
  tests/test_gene_mrta_v117_oracle.py \
  tests/test_gene_mrta_v117_stage_a.py

if [[ "$MODE" == "tests" ]]; then
  exit 0

elif [[ "$MODE" == "oracle-smoke" ]]; then
  python -m marl2d.gene_mrta_v117.stage_a_oracle_bank \
    --robots 2 \
    --tasks 10 \
    --world-count 2 \
    --seed-base 117000000 \
    --time-limit 60 \
    --output "$SMOKE_BANK"

elif [[ "$MODE" == "train-smoke" ]]; then
  if [[ ! -f "$SMOKE_BANK" ]]; then
    echo "Missing smoke oracle bank: $SMOKE_BANK" >&2
    echo "Run: bash tools/run_gene_mrta_v117_mac.sh oracle-smoke" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v117.stage_a_train \
    --oracle-bank "$SMOKE_BANK" \
    --generations 5 \
    --population 32 \
    --archive-size 4 \
    --hybrid-limit 16 \
    --normal-children 32 \
    --normal-full-per-axis 2 \
    --mating-pairs 8 \
    --children-per-pair 2 \
    --mating-full-limit 8 \
    --screen-worlds 2 \
    --seed 117

elif [[ "$MODE" == "smoke" ]]; then
  bash "$0" oracle-smoke
  bash "$0" train-smoke

elif [[ "$MODE" == "oracle-formal" ]]; then
  FORMAL_WORLDS="${V117_FORMAL_WORLDS:-32}"
  FORMAL_SEED_BASE="${V117_FORMAL_SEED_BASE:-117100000}"
  echo "V117_FORMAL_WORLDS=$FORMAL_WORLDS"
  echo "V117_FORMAL_SEED_BASE=$FORMAL_SEED_BASE"
  python -m marl2d.gene_mrta_v117.stage_a_oracle_bank \
    --robots 4 \
    --tasks 20 \
    --world-count "$FORMAL_WORLDS" \
    --seed-base "$FORMAL_SEED_BASE" \
    --time-limit unlimited \
    --output "$FORMAL_BANK"

elif [[ "$MODE" == "train-formal" ]]; then
  if [[ ! -f "$FORMAL_BANK" ]]; then
    echo "Missing formal oracle bank: $FORMAL_BANK" >&2
    echo "Run: bash tools/run_gene_mrta_v117_mac.sh oracle-formal" >&2
    exit 2
  fi
  FORMAL_GENERATIONS="${V117_GENERATIONS:-50}"
  python -m marl2d.gene_mrta_v117.stage_a_train \
    --oracle-bank "$FORMAL_BANK" \
    --generations "$FORMAL_GENERATIONS" \
    --population 256 \
    --archive-size 16 \
    --hybrid-limit 128 \
    --normal-children 128 \
    --normal-full-per-axis 4 \
    --mating-pairs 32 \
    --children-per-pair 4 \
    --mating-full-limit 16 \
    --screen-worlds 8 \
    --seed 117

else
  echo "Usage: bash tools/run_gene_mrta_v117_mac.sh [tests|oracle-smoke|train-smoke|smoke|oracle-formal|train-formal]" >&2
  exit 2
fi
