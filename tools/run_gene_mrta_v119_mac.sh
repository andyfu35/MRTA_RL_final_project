#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

ROOT="runs/gene_mrta_v119"
SMOKE_MANIFEST="${V119_SMOKE_MANIFEST:-$ROOT/smoke_manifest.json}"
SMOKE_RUN_DIR="${V119_SMOKE_RUN_DIR:-$ROOT/smoke_seed119}"

RAW_DIR="${V119_RAW_DIR:-benchmarks/mtrpd_public/raw}"
FORMAL_MANIFEST="${V119_MANIFEST:-benchmarks/mtrpd_public/mtrpd_public_manifest.json}"
FORMAL_RUN_DIR="${V119_RUN_DIR:-$ROOT/public_mtrpd_mutation_seed119}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

if [[ "$MODE" == "tests" ]]; then
  pytest -q tests/test_gene_mrta_v119.py

elif [[ "$MODE" == "smoke" ]]; then
  pytest -q tests/test_gene_mrta_v119.py
  mkdir -p "$ROOT"
  python -m marl2d.gene_mrta_v119.dataset_tool make-smoke \
    --output "$SMOKE_MANIFEST"
  rm -rf "$SMOKE_RUN_DIR"
  python -m marl2d.gene_mrta_v119.train \
    --manifest "$SMOKE_MANIFEST" \
    --run-dir "$SMOKE_RUN_DIR" \
    --generations 3 \
    --population 16 \
    --bootstrap-size 8 \
    --mutation-children 16 \
    --bank-max-size 16 \
    --progress-every 4 \
    --seed 119

elif [[ "$MODE" == "download" ]]; then
  mkdir -p "$RAW_DIR"
  python -m marl2d.gene_mrta_v119.dataset_tool download \
    --output-dir "$RAW_DIR"

elif [[ "$MODE" == "inspect-raw" ]]; then
  if [[ ! -d "$RAW_DIR" ]]; then
    echo "Missing raw MTRPD directory: $RAW_DIR" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v119.dataset_tool inspect-raw \
    --raw-dir "$RAW_DIR"

elif [[ "$MODE" == "describe" ]]; then
  if [[ ! -f "$FORMAL_MANIFEST" ]]; then
    echo "Missing canonical MTRPD manifest: $FORMAL_MANIFEST" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v119.dataset_tool describe \
    --manifest "$FORMAL_MANIFEST"

elif [[ "$MODE" == "train-formal" ]]; then
  if [[ ! -f "$FORMAL_MANIFEST" ]]; then
    echo "Missing canonical MTRPD manifest: $FORMAL_MANIFEST" >&2
    echo "First acquire the original public supplement and convert it to the canonical V1.19 manifest." >&2
    exit 2
  fi
  V119_GENERATIONS="${V119_GENERATIONS:-50}"
  V119_POPULATION="${V119_POPULATION:-128}"
  V119_CHILDREN="${V119_CHILDREN:-128}"
  echo "V119_MANIFEST=$FORMAL_MANIFEST"
  echo "V119_RUN_DIR=$FORMAL_RUN_DIR"
  echo "V119_GENERATIONS=$V119_GENERATIONS"
  echo "V119_POPULATION=$V119_POPULATION"
  echo "V119_CHILDREN=$V119_CHILDREN"
  python -m marl2d.gene_mrta_v119.train \
    --manifest "$FORMAL_MANIFEST" \
    --run-dir "$FORMAL_RUN_DIR" \
    --generations "$V119_GENERATIONS" \
    --population "$V119_POPULATION" \
    --bootstrap-size 64 \
    --mutation-children "$V119_CHILDREN" \
    --bank-max-size 128 \
    --pareto-epsilon 0.0025 \
    --progress-every 16 \
    --seed 119

elif [[ "$MODE" == "status" ]]; then
  CHECKPOINT="$FORMAL_RUN_DIR/checkpoint.json"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "No V1.19 checkpoint: $CHECKPOINT"
    exit 0
  fi
  python - "$CHECKPOINT" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
history = data.get("history", [])
print("V119_STATUS")
print("generation=", data.get("generation"), sep="")
print("bank_size=", len(data.get("bank_ids", [])), sep="")
print("axes=", json.dumps(data.get("axes", [])), sep="")
if history:
    row = history[-1]
    print("best_worst_completion=", row.get("best_worst_completion"), sep="")
    print("best_success_instances=", row.get("best_success_instances"), sep="")
    print("global_best=", json.dumps(row.get("global_best"), ensure_ascii=False), sep="")
    print("paired_child_overall_wtl=", json.dumps(row.get("paired_child_overall_wtl"), ensure_ascii=False), sep="")
    print("paired_mean_overall_delta=", row.get("paired_mean_overall_delta"), sep="")
PY

else
  echo "Usage: bash tools/run_gene_mrta_v119_mac.sh [tests|smoke|download|inspect-raw|describe|train-formal|status]" >&2
  exit 2
fi
