#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

SMOKE_BANK="${V118_SMOKE_BANK:-runs/gene_mrta_v118/oracle_smoke_2r6t.json}"
SMOKE_RUN_DIR="${V118_SMOKE_RUN_DIR:-runs/gene_mrta_v118/smoke_2r6t_seed118}"

FORMAL_BANK="${V118_FORMAL_BANK:-runs/gene_mrta_v118/oracle_formal_4r20t.json}"
FORMAL_RUN_DIR="${V118_FORMAL_RUN_DIR:-runs/gene_mrta_v118/formal_4r20t_seed118}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v118.py

if [[ "$MODE" == "tests" ]]; then
  exit 0

elif [[ "$MODE" == "oracle-smoke" ]]; then
  python -m marl2d.gene_mrta_v118.oracle_bank \
    --robots 2 \
    --tasks 6 \
    --world-count 2 \
    --seed-base 118000000 \
    --time-limit 60 \
    --output "$SMOKE_BANK"

elif [[ "$MODE" == "train-smoke" ]]; then
  if [[ ! -f "$SMOKE_BANK" ]]; then
    echo "Missing V1.18 smoke bank: $SMOKE_BANK" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v118.train \
    --oracle-bank "$SMOKE_BANK" \
    --run-dir "$SMOKE_RUN_DIR" \
    --generations 5 \
    --population 64 \
    --bootstrap-size 32 \
    --mutation-children 48 \
    --mating-children 16 \
    --pareto-max-size 64 \
    --seed 118

elif [[ "$MODE" == "smoke" ]]; then
  if [[ ! -f "$SMOKE_BANK" ]]; then
    python -m marl2d.gene_mrta_v118.oracle_bank \
      --robots 2 \
      --tasks 6 \
      --world-count 2 \
      --seed-base 118000000 \
      --time-limit 60 \
      --output "$SMOKE_BANK"
  fi
  python -m marl2d.gene_mrta_v118.train \
    --oracle-bank "$SMOKE_BANK" \
    --run-dir "$SMOKE_RUN_DIR" \
    --generations 5 \
    --population 64 \
    --bootstrap-size 32 \
    --mutation-children 48 \
    --mating-children 16 \
    --pareto-max-size 64 \
    --seed 118

elif [[ "$MODE" == "oracle-formal" ]]; then
  FORMAL_WORLDS="${V118_FORMAL_WORLDS:-16}"
  FORMAL_SEED_BASE="${V118_FORMAL_SEED_BASE:-118100000}"
  echo "V118_FORMAL_WORLDS=$FORMAL_WORLDS"
  echo "V118_FORMAL_SEED_BASE=$FORMAL_SEED_BASE"
  python -m marl2d.gene_mrta_v118.oracle_bank \
    --robots 4 \
    --tasks 20 \
    --world-count "$FORMAL_WORLDS" \
    --seed-base "$FORMAL_SEED_BASE" \
    --time-limit unlimited \
    --output "$FORMAL_BANK"

elif [[ "$MODE" == "train-formal" ]]; then
  if [[ ! -f "$FORMAL_BANK" ]]; then
    echo "Missing V1.18 formal bank: $FORMAL_BANK" >&2
    echo "Run oracle-formal first." >&2
    exit 2
  fi
  FORMAL_GENERATIONS="${V118_GENERATIONS:-50}"
  echo "V118_FORMAL_RUN_DIR=$FORMAL_RUN_DIR"
  echo "V118_GENERATIONS=$FORMAL_GENERATIONS"
  python -m marl2d.gene_mrta_v118.train \
    --oracle-bank "$FORMAL_BANK" \
    --run-dir "$FORMAL_RUN_DIR" \
    --generations "$FORMAL_GENERATIONS" \
    --population 256 \
    --bootstrap-size 128 \
    --mutation-children 192 \
    --mating-children 64 \
    --pareto-max-size 256 \
    --seed 118

elif [[ "$MODE" == "status-formal" ]]; then
  if [[ ! -f "$FORMAL_RUN_DIR/checkpoint.json" ]]; then
    echo "No V1.18 formal checkpoint: $FORMAL_RUN_DIR/checkpoint.json"
    exit 0
  fi
  python - "$FORMAL_RUN_DIR/checkpoint.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
history = data.get("history", [])
print("V118_FORMAL_STATUS")
print("generation=", data.get("generation"), sep="")
print("pareto_size=", len(data.get("pareto_records", [])), sep="")
if history:
    row = history[-1]
    print("best_worst_completion=", row.get("best_worst_completion"), sep="")
    print("best_mean_completion=", row.get("best_mean_completion"), sep="")
    print("best_success_worlds=", row.get("best_success_worlds"), sep="")
    print("newly_successful_genes=", row.get("newly_successful_genes"), sep="")
    print("maximin_gene=", json.dumps(row.get("maximin_gene"), ensure_ascii=False), sep="")
PY

else
  echo "Usage: bash tools/run_gene_mrta_v118_mac.sh [tests|oracle-smoke|train-smoke|smoke|oracle-formal|train-formal|status-formal]" >&2
  exit 2
fi
