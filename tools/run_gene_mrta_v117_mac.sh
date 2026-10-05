#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SMOKE_BANK="${V117_SMOKE_BANK:-runs/gene_mrta_v117_stage_a/oracle_smoke_2r10t.json}"
FORMAL_BANK="${V117_FORMAL_BANK:-runs/gene_mrta_v117_stage_a/oracle_formal_4r20t.json}"
FORMAL_RUN_DIR="${V117_FORMAL_RUN_DIR:-runs/gene_mrta_v117_stage_a/formal_4r20t_seed117}"
FUSION1_DIR="${V117_FUSION1_DIR:-runs/gene_mrta_v117_fusion1/formal_4r20t_seed117}"
UNSEEN_BANK="${V117_UNSEEN_BANK:-runs/gene_mrta_v117_unseen/oracle_unseen_4r20t_64.json}"
UNSEEN_AUDIT_DIR="${V117_UNSEEN_AUDIT_DIR:-runs/gene_mrta_v117_unseen/audit_4r20t_64}"

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
  python -m marl2d.gene_mrta_v117.stage_a_oracle_bank \
    --robots 2 \
    --tasks 10 \
    --world-count 2 \
    --seed-base 117000000 \
    --time-limit 60 \
    --output "$SMOKE_BANK"

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
  echo "V117_FORMAL_RUN_DIR=$FORMAL_RUN_DIR"
  echo "V117_GENERATIONS=$FORMAL_GENERATIONS"
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
    --seed 117 \
    --run-dir "$FORMAL_RUN_DIR"

elif [[ "$MODE" == "fusion1" ]]; then
  SOURCE_CHECKPOINT="$FORMAL_RUN_DIR/checkpoint.json"
  if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "Missing final Stage-A checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 2
  fi
  FUSION1_ROUNDS="${V117_FUSION1_ROUNDS:-20}"
  echo "V117_FUSION1_DIR=$FUSION1_DIR"
  echo "V117_FUSION1_ROUNDS=$FUSION1_ROUNDS"
  python -m marl2d.gene_mrta_v117.fusion1 \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --output-dir "$FUSION1_DIR" \
    --rounds "$FUSION1_ROUNDS" \
    --mating-pairs 64 \
    --children-per-pair 4 \
    --screen-worlds 8 \
    --full-candidates 32 \
    --archive-size 16 \
    --hybrid-limit 192 \
    --threshold 0.95 \
    --post-mating-sigma 0.0 \
    --seed 11701

elif [[ "$MODE" == "status-fusion1" ]]; then
  if [[ -f "$FUSION1_DIR/checkpoint.json" ]]; then
    python - "$FUSION1_DIR/checkpoint.json" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
history = data.get("history", [])
print("V117_FUSION1_STATUS")
print("run_dir=", path.rsplit("/", 1)[0], sep="")
print("round=", data.get("round"), sep="")
print("active_genes=", len(data.get("records", [])), sep="")
if history:
    row = history[-1]
    print("max_inherited_capabilities=", row.get("max_inherited_capabilities"), sep="")
    print("accepted_full_union_children=", row.get("accepted_full_union_children"), sep="")
    print("best=", json.dumps(row.get("best", {}), ensure_ascii=False), sep="")
    print("best_fusion_gene=", json.dumps(row.get("best_fusion_gene"), ensure_ascii=False), sep="")
PY
  else
    echo "No Fusion-1 checkpoint yet: $FUSION1_DIR/checkpoint.json"
  fi

elif [[ "$MODE" == "oracle-unseen" ]]; then
  UNSEEN_WORLDS="${V117_UNSEEN_WORLDS:-64}"
  UNSEEN_SEED_BASE="${V117_UNSEEN_SEED_BASE:-117200000}"
  echo "V117_UNSEEN_WORLDS=$UNSEEN_WORLDS"
  echo "V117_UNSEEN_SEED_BASE=$UNSEEN_SEED_BASE"
  python -m marl2d.gene_mrta_v117.stage_a_oracle_bank \
    --robots 4 \
    --tasks 20 \
    --world-count "$UNSEEN_WORLDS" \
    --seed-base "$UNSEEN_SEED_BASE" \
    --time-limit unlimited \
    --output "$UNSEEN_BANK"

elif [[ "$MODE" == "audit-unseen" ]]; then
  if [[ ! -f "$FUSION1_DIR/checkpoint.json" ]]; then
    echo "Missing Fusion-1 checkpoint: $FUSION1_DIR/checkpoint.json" >&2
    exit 2
  fi
  if [[ ! -f "$UNSEEN_BANK" ]]; then
    echo "Missing unseen oracle bank: $UNSEEN_BANK" >&2
    echo "Run oracle-unseen first." >&2
    exit 2
  fi
  HARD_COUNT="${V117_HARD_COUNT:-16}"
  python -m marl2d.gene_mrta_v117.unseen_audit \
    --fusion-checkpoint "$FUSION1_DIR/checkpoint.json" \
    --unseen-bank "$UNSEEN_BANK" \
    --output-dir "$UNSEEN_AUDIT_DIR" \
    --archive-size 16 \
    --hybrid-count 12 \
    --hard-count "$HARD_COUNT"

elif [[ "$MODE" == "status-formal" ]]; then
  if [[ -f "$FORMAL_RUN_DIR/checkpoint.json" ]]; then
    python - "$FORMAL_RUN_DIR/checkpoint.json" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
history = data.get("history", [])
print("V117_FORMAL_STATUS")
print("run_dir=", path.rsplit("/", 1)[0], sep="")
print("generation=", data.get("generation"), sep="")
print("active_genes=", len(data.get("records", [])), sep="")
if history:
    row = history[-1]
    print("max_capabilities=", row.get("max_capabilities"), sep="")
    print("max_inherited_capabilities=", row.get("max_inherited_capabilities"), sep="")
    print("best=", json.dumps(row.get("best", {}), ensure_ascii=False), sep="")
    print("best_fusion_gene=", json.dumps(row.get("best_fusion_gene"), ensure_ascii=False), sep="")
PY
  else
    echo "No formal checkpoint yet: $FORMAL_RUN_DIR/checkpoint.json"
  fi

else
  echo "Usage: bash tools/run_gene_mrta_v117_mac.sh [tests|oracle-smoke|train-smoke|smoke|oracle-formal|train-formal|status-formal|fusion1|status-fusion1|oracle-unseen|audit-unseen]" >&2
  exit 2
fi
