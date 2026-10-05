#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"

ROOT="runs/gene_mrta_v120"
RUN_DIR="${V120_RUN_DIR:-$ROOT/public_minmax_mtsp_s_mutation_seed120}"
CHECKPOINT="$RUN_DIR/checkpoint.json"

INSTANCE_ZIP="benchmarks/minmax_mtsp_mils/instances.zip"
CERTIFICATE_ZIP="benchmarks/minmax_mtsp_mils/Certification.zip"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

if [[ "$MODE" == "tests" ]]; then
  pytest -q tests/test_gene_mrta_v120.py

elif [[ "$MODE" == "describe" ]]; then
  python - <<'PY'
import json
from marl2d.gene_mrta_v120.benchmark import load_benchmark, describe_benchmark
rows = load_benchmark()
print("V120_BENCHMARK " + json.dumps(describe_benchmark(rows), ensure_ascii=False))
for split in ("evolution", "validation", "protected_test"):
    selected = [x for x in rows if x.split == split]
    print(
        "V120_SPLIT "
        + json.dumps(
            {
                "split": split,
                "count": len(selected),
                "min_vertices": min(x.vertex_count for x in selected),
                "max_vertices": max(x.vertex_count for x in selected),
                "robot_counts": sorted({x.robot_count for x in selected}),
            },
            ensure_ascii=False,
        )
    )
PY

elif [[ "$MODE" == "smoke" ]]; then
  pytest -q tests/test_gene_mrta_v120.py
  SMOKE_DIR="$ROOT/smoke_seed120"
  rm -rf "$SMOKE_DIR"
  python -m marl2d.gene_mrta_v120.train \
    --run-dir "$SMOKE_DIR" \
    --split evolution \
    --max-vertices 150 \
    --candidate-k 16 \
    --generations 2 \
    --population 8 \
    --bootstrap-size 8 \
    --mutation-children 8 \
    --bank-max-size 16 \
    --progress-every 2 \
    --seed 120

elif [[ "$MODE" == "train-formal" ]]; then
  V120_GENERATIONS="${V120_GENERATIONS:-50}"
  V120_POPULATION="${V120_POPULATION:-32}"
  V120_CHILDREN="${V120_CHILDREN:-32}"
  V120_CANDIDATE_K="${V120_CANDIDATE_K:-32}"

  echo "V120_RUN_DIR=$RUN_DIR"
  echo "V120_GENERATIONS=$V120_GENERATIONS"
  echo "V120_POPULATION=$V120_POPULATION"
  echo "V120_CHILDREN=$V120_CHILDREN"
  echo "V120_CANDIDATE_K=$V120_CANDIDATE_K"

  if [[ -n "${V120_MAX_VERTICES:-}" ]]; then
    echo "V120_MAX_VERTICES=$V120_MAX_VERTICES"
    python -m marl2d.gene_mrta_v120.train \
      --instance-zip "$INSTANCE_ZIP" \
      --certificate-zip "$CERTIFICATE_ZIP" \
      --run-dir "$RUN_DIR" \
      --split evolution \
      --candidate-k "$V120_CANDIDATE_K" \
      --generations "$V120_GENERATIONS" \
      --population "$V120_POPULATION" \
      --bootstrap-size 32 \
      --mutation-children "$V120_CHILDREN" \
      --bank-max-size 64 \
      --pareto-epsilon 0.0025 \
      --progress-every 4 \
      --seed 120 \
      --max-vertices "$V120_MAX_VERTICES"
  else
    python -m marl2d.gene_mrta_v120.train \
      --instance-zip "$INSTANCE_ZIP" \
      --certificate-zip "$CERTIFICATE_ZIP" \
      --run-dir "$RUN_DIR" \
      --split evolution \
      --candidate-k "$V120_CANDIDATE_K" \
      --generations "$V120_GENERATIONS" \
      --population "$V120_POPULATION" \
      --bootstrap-size 32 \
      --mutation-children "$V120_CHILDREN" \
      --bank-max-size 64 \
      --pareto-epsilon 0.0025 \
      --progress-every 4 \
      --seed 120
  fi

elif [[ "$MODE" == "evaluate-validation" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v120.evaluate \
    --checkpoint "$CHECKPOINT" \
    --split validation \
    --output "$RUN_DIR/validation_report.json"

elif [[ "$MODE" == "evaluate-protected" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v120.evaluate \
    --checkpoint "$CHECKPOINT" \
    --split protected_test \
    --output "$RUN_DIR/protected_large_report.json"

elif [[ "$MODE" == "status" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "No V1.20 checkpoint: $CHECKPOINT"
    exit 0
  fi
  python - "$CHECKPOINT" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
history = data.get("history", [])
print("V120_STATUS")
print("generation=", data.get("generation"), sep="")
print("candidate_k=", data.get("candidate_k"), sep="")
print("instance_count=", len(data.get("instance_ids", [])), sep="")
print("bank_size=", len(data.get("bank_ids", [])), sep="")
print("axes=", json.dumps(data.get("axes", [])), sep="")
if history:
    row = history[-1]
    print("global_best=", json.dumps(row.get("global_best"), ensure_ascii=False), sep="")
    print("best_by_size=", json.dumps(row.get("best_by_size"), ensure_ascii=False), sep="")
    print("paired_child_overall_wtl=", json.dumps(row.get("paired_child_overall_wtl"), ensure_ascii=False), sep="")
    print("paired_mean_overall_delta=", row.get("paired_mean_overall_delta"), sep="")
PY

else
  echo "Usage: bash tools/run_gene_mrta_v120_mac.sh [tests|describe|smoke|train-formal|evaluate-validation|evaluate-protected|status]" >&2
  exit 2
fi
