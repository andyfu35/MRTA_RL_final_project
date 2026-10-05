#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

ROOT="runs/gene_mrta_v121"
RUN_DIR="${V121_RUN_DIR:-$ROOT/formal_1000w_34i_50r_seed121}"
CHECKPOINT="$RUN_DIR/checkpoint.json"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

if [[ "$MODE" == "tests" ]]; then
  pytest -q tests/test_gene_mrta_v120.py tests/test_gene_mrta_v121.py

elif [[ "$MODE" == "smoke" ]]; then
  pytest -q tests/test_gene_mrta_v120.py tests/test_gene_mrta_v121.py
  SMOKE_DIR="$ROOT/smoke_8w_34i_2r_seed121"
  rm -rf "$SMOKE_DIR"
  python -m marl2d.gene_mrta_v121.train \
    --run-dir "$SMOKE_DIR" \
    --worlds-per-round 8 \
    --rounds 2 \
    --expected-instance-count 34 \
    --candidate-k 32 \
    --progress-every 2 \
    --bank-max-size 32 \
    --seed 121

elif [[ "$MODE" == "train-formal" ]]; then
  V121_WORLDS_PER_ROUND="${V121_WORLDS_PER_ROUND:-1000}"
  V121_ROUNDS="${V121_ROUNDS:-50}"
  V121_CANDIDATE_K="${V121_CANDIDATE_K:-32}"
  V121_SEED="${V121_SEED:-121}"

  echo "V121_RUN_DIR=$RUN_DIR"
  echo "V121_WORLDS_PER_ROUND=$V121_WORLDS_PER_ROUND"
  echo "V121_FIXED_INSTANCES=34"
  echo "V121_ROUNDS=$V121_ROUNDS"
  echo "V121_ROLLOUTS_PER_ROUND=$((V121_WORLDS_PER_ROUND * 34))"
  echo "V121_PLANNED_TOTAL_ROLLOUTS=$((V121_WORLDS_PER_ROUND * 34 * V121_ROUNDS))"
  echo "V121_CANDIDATE_K=$V121_CANDIDATE_K"
  echo "V121_SEED=$V121_SEED"

  python -m marl2d.gene_mrta_v121.train \
    --run-dir "$RUN_DIR" \
    --worlds-per-round "$V121_WORLDS_PER_ROUND" \
    --rounds "$V121_ROUNDS" \
    --expected-instance-count 34 \
    --candidate-k "$V121_CANDIDATE_K" \
    --progress-every 25 \
    --bank-max-size 128 \
    --seed "$V121_SEED"

elif [[ "$MODE" == "status" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "No V1.21 checkpoint yet: $CHECKPOINT"
    exit 0
  fi
  python - "$CHECKPOINT" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
history = data.get("history", [])
print("V121_STATUS")
print("completed_round=", data.get("completed_round"), sep="")
print("worlds_per_round=", data.get("worlds_per_round"), sep="")
print("fixed_instances=", len(data.get("instance_ids", [])), sep="")
print("rounds=", data.get("rounds"), sep="")
print("bank_size=", len(data.get("bank_records", [])), sep="")
print("axes=", json.dumps(data.get("axes", [])), sep="")
if history:
    row = history[-1]
    print("rollouts_this_round=", row.get("rollouts_this_round"), sep="")
    print("cumulative_rollouts=", row.get("cumulative_rollouts"), sep="")
    print("successful_worlds=", row.get("successful_worlds"), sep="")
    print("population_axis_stats=", json.dumps(row.get("population_axis_stats"), ensure_ascii=False), sep="")
    print("population_axis_mean_delta_from_previous_round=", json.dumps(row.get("population_axis_mean_delta_from_previous_round"), ensure_ascii=False), sep="")
    print("bank_best_by_axis=", json.dumps(row.get("bank_best_by_axis"), ensure_ascii=False), sep="")
    print("bank_global_best=", json.dumps(row.get("bank_global_best"), ensure_ascii=False), sep="")
    print("parent_child_diagnostic=", json.dumps(row.get("parent_child_diagnostic"), ensure_ascii=False), sep="")
PY

else
  echo "Usage: bash tools/run_gene_mrta_v121_mac.sh [tests|smoke|train-formal|status]" >&2
  exit 2
fi
