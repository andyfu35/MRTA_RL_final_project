#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

SMOKE_WORLD_BANK="${V118_SMOKE_WORLD_BANK:-runs/gene_mrta_v118/world_smoke_2r6t_fast_v2.json}"
SMOKE_RUN_DIR="${V118_SMOKE_RUN_DIR:-runs/gene_mrta_v118/smoke_2r6t_fast_v3_4axis_seed118}"

FORMAL_WORLD_BANK="${V118_FORMAL_WORLD_BANK:-runs/gene_mrta_v118/world_formal_4r20t_100_fast_v2.json}"
FORMAL_RUN_DIR="${V118_FORMAL_RUN_DIR:-runs/gene_mrta_v118/formal_4r20t_100_fast_v3_4axis_seed118}"

EXACT_SMOKE_BANK="${V118_EXACT_SMOKE_BANK:-runs/gene_mrta_v118/oracle_smoke_2r6t_exact_v1.json}"

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

elif [[ "$MODE" == "world-smoke" ]]; then
  python -m marl2d.gene_mrta_v118.world_bank \
    --robots 2 \
    --tasks 6 \
    --world-count 2 \
    --seed-base 118000000 \
    --output "$SMOKE_WORLD_BANK"

elif [[ "$MODE" == "train-smoke" ]]; then
  if [[ ! -f "$SMOKE_WORLD_BANK" ]]; then
    echo "Missing V1.18 smoke world bank: $SMOKE_WORLD_BANK" >&2
    echo "Run: bash tools/run_gene_mrta_v118_mac.sh world-smoke" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v118.train \
    --world-bank "$SMOKE_WORLD_BANK" \
    --run-dir "$SMOKE_RUN_DIR" \
    --generations 5 \
    --population 64 \
    --bootstrap-size 32 \
    --mutation-children 48 \
    --mating-children 16 \
    --pareto-max-size 64 \
    --seed 118

elif [[ "$MODE" == "smoke" ]]; then
  if [[ ! -f "$SMOKE_WORLD_BANK" ]]; then
    python -m marl2d.gene_mrta_v118.world_bank \
      --robots 2 \
      --tasks 6 \
      --world-count 2 \
      --seed-base 118000000 \
      --output "$SMOKE_WORLD_BANK"
  fi
  python -m marl2d.gene_mrta_v118.train \
    --world-bank "$SMOKE_WORLD_BANK" \
    --run-dir "$SMOKE_RUN_DIR" \
    --generations 5 \
    --population 64 \
    --bootstrap-size 32 \
    --mutation-children 48 \
    --mating-children 16 \
    --pareto-max-size 64 \
    --seed 118

elif [[ "$MODE" == "world-formal" ]]; then
  FORMAL_WORLDS="${V118_FORMAL_WORLDS:-100}"
  FORMAL_SEED_BASE="${V118_FORMAL_SEED_BASE:-117100000}"
  echo "V118_FAST_FORMAL_WORLDS=$FORMAL_WORLDS"
  echo "V118_FAST_FORMAL_SEED_BASE=$FORMAL_SEED_BASE"
  python -m marl2d.gene_mrta_v118.world_bank \
    --robots 4 \
    --tasks 20 \
    --world-count "$FORMAL_WORLDS" \
    --seed-base "$FORMAL_SEED_BASE" \
    --battery-reserve-fraction 0.05 \
    --output "$FORMAL_WORLD_BANK"

elif [[ "$MODE" == "train-formal" ]]; then
  if [[ ! -f "$FORMAL_WORLD_BANK" ]]; then
    echo "Missing V1.18 formal world bank: $FORMAL_WORLD_BANK" >&2
    echo "Run: bash tools/run_gene_mrta_v118_mac.sh world-formal" >&2
    exit 2
  fi
  FORMAL_GENERATIONS="${V118_GENERATIONS:-50}"
  echo "V118_FORMAL_RUN_DIR=$FORMAL_RUN_DIR"
  echo "V118_GENERATIONS=$FORMAL_GENERATIONS"
  python -m marl2d.gene_mrta_v118.train \
    --world-bank "$FORMAL_WORLD_BANK" \
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

elif [[ "$MODE" == "exact-smoke" ]]; then
  python -m marl2d.gene_mrta_v118.oracle_bank \
    --robots 2 \
    --tasks 6 \
    --world-count 2 \
    --seed-base 118000000 \
    --time-limit 60 \
    --output "$EXACT_SMOKE_BANK"

else
  echo "Usage: bash tools/run_gene_mrta_v118_mac.sh [tests|world-smoke|train-smoke|smoke|world-formal|train-formal|status-formal|exact-smoke]" >&2
  exit 2
fi
