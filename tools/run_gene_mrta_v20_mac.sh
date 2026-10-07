#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
ROOT="runs/gene_mrta_v20"
RUN_DIR="${V20_RUN_DIR:-$ROOT/fixed100_50r_seed200_epsgrid_v2}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

python - <<'PY'
import torch
print("V20_TORCH", torch.__version__)
print("V20_MPS_BUILT", torch.backends.mps.is_built())
print("V20_MPS_AVAILABLE", torch.backends.mps.is_available())
PY

if [[ "$MODE" == "tests" ]]; then
  pytest -q tests/test_gene_mrta_v20.py tests/test_gene_mrta_v20_ood_generalization.py
elif [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v20.train \
    --run-dir "${V20_RUN_DIR:-$ROOT/smoke_8g_10s_2r_seed200_epsgrid_v2}" \
    --rounds 2 \
    --genes-per-round 8 \
    --world-count 10 \
    --gene-batch-size 8 \
    --seed 200 \
    --device auto
elif [[ "$MODE" == "analyze-grid" ]]; then
  V20_OLD_CHECKPOINT="${V20_OLD_CHECKPOINT:-$ROOT/fixed100_50r_seed200/checkpoint.json}"
  python -m marl2d.gene_mrta_v20.analyze_bank_grid --checkpoint "$V20_OLD_CHECKPOINT"
elif [[ "$MODE" == "train-50" ]]; then
  V20_ROUNDS="${V20_ROUNDS:-50}"
  V20_GENES_PER_ROUND="${V20_GENES_PER_ROUND:-64}"
  V20_GENE_BATCH_SIZE="${V20_GENE_BATCH_SIZE:-32}"
  V20_SEED="${V20_SEED:-200}"
  V20_TIME_RESOLUTION="${V20_TIME_RESOLUTION:-0.5}"
  V20_PRIORITY_RESOLUTION="${V20_PRIORITY_RESOLUTION:-0.10}"
  V20_ONTIME_RESOLUTION="${V20_ONTIME_RESOLUTION:-0.5}"
  echo "V20_RUN_DIR=$RUN_DIR"
  echo "V20_FIXED_SEEDS=100"
  echo "V20_ROUNDS=$V20_ROUNDS"
  echo "V20_GENES_PER_ROUND=$V20_GENES_PER_ROUND"
  echo "V20_GENE_BATCH_SIZE=$V20_GENE_BATCH_SIZE"
  echo "V20_SEED=$V20_SEED"
  echo "V20_TIME_RESOLUTION=$V20_TIME_RESOLUTION"
  echo "V20_PRIORITY_RESOLUTION=$V20_PRIORITY_RESOLUTION"
  echo "V20_ONTIME_RESOLUTION=$V20_ONTIME_RESOLUTION"
  python -m marl2d.gene_mrta_v20.train \
    --run-dir "$RUN_DIR" \
    --rounds "$V20_ROUNDS" \
    --genes-per-round "$V20_GENES_PER_ROUND" \
    --world-count 100 \
    --gene-batch-size "$V20_GENE_BATCH_SIZE" \
    --seed "$V20_SEED" \
    --total-time-resolution "$V20_TIME_RESOLUTION" \
    --priority-resolution "$V20_PRIORITY_RESOLUTION" \
    --on-time-completed-tasks-resolution "$V20_ONTIME_RESOLUTION" \
    --device auto
elif [[ "$MODE" == "ood-smoke" || "$MODE" == "ood-formal" ]]; then
  V20_OOD_CHECKPOINT="${V20_OOD_CHECKPOINT:-$ROOT/longrun_1024g_200r_seed200/checkpoint.json}"
  V20_OOD_SEED="${V20_OOD_SEED:-20070001}"
  V20_OOD_CELLS="${V20_OOD_CELLS:-10x50,20x100,25x100,40x100,60x100,20x125,20x150,20x200,20x300,25x125,40x200,60x300}"
  if [[ "$MODE" == "ood-formal" ]]; then
    V20_OOD_SEEDS="${V20_OOD_SEEDS:-100}"
  else
    V20_OOD_SEEDS="${V20_OOD_SEEDS:-20}"
  fi
  V20_OOD_STAMP="$(date +%Y%m%d_%H%M%S)"
  V20_OOD_RUN_DIR="${V20_OOD_RUN_DIR:-$ROOT/zero_shot_size_ood_${MODE}_${V20_OOD_STAMP}_seed${V20_OOD_SEED}}"
  echo "V20_OOD_CHECKPOINT=$V20_OOD_CHECKPOINT"
  echo "V20_OOD_RUN_DIR=$V20_OOD_RUN_DIR"
  echo "V20_OOD_SEEDS=$V20_OOD_SEEDS"
  echo "V20_OOD_SEED=$V20_OOD_SEED"
  echo "V20_OOD_CELLS=$V20_OOD_CELLS"
  run_v20_ood() {
    python -m marl2d.gene_mrta_v20.ood_generalization \
      --checkpoint "$V20_OOD_CHECKPOINT" \
      --run-dir "$V20_OOD_RUN_DIR" \
      --seeds "$V20_OOD_SEEDS" \
      --seed "$V20_OOD_SEED" \
      --cells "$V20_OOD_CELLS" \
      --device auto \
      "$@"
  }

  if [[ -n "${V20_OOD_TOTAL_TIME_GENE_ID:-}" && -n "${V20_OOD_ONTIME_GENE_ID:-}" ]]; then
    run_v20_ood \
      --total-time-gene-id "$V20_OOD_TOTAL_TIME_GENE_ID" \
      --on-time-gene-id "$V20_OOD_ONTIME_GENE_ID"
  elif [[ -n "${V20_OOD_TOTAL_TIME_GENE_ID:-}" ]]; then
    run_v20_ood \
      --total-time-gene-id "$V20_OOD_TOTAL_TIME_GENE_ID"
  elif [[ -n "${V20_OOD_ONTIME_GENE_ID:-}" ]]; then
    run_v20_ood \
      --on-time-gene-id "$V20_OOD_ONTIME_GENE_ID"
  else
    run_v20_ood
  fi
else
  echo "Usage: bash tools/run_gene_mrta_v20_mac.sh [tests|smoke|analyze-grid|train-50|ood-smoke|ood-formal]" >&2
  exit 2
fi
