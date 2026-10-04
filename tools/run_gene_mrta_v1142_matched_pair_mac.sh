#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V113_CHECKPOINT="${V113_CHECKPOINT:-}"
V1141_ADAPTIVE_CHECKPOINT="${V1141_ADAPTIVE_CHECKPOINT:-}"
SEED="${SEED:-7}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q \
  tests/test_gene_mrta_v1141_recombination_correction.py \
  tests/test_gene_mrta_v1142_matched_pair_assay.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ ! -f "$SCENARIO_BANK" ]]; then
  echo "Missing frozen 95M scenario bank: $SCENARIO_BANK" >&2
  exit 2
fi

if [[ ! -f "$V18_RUN/summary.json" ]]; then
  echo "Missing V1.8 common ancestor: $V18_RUN/summary.json" >&2
  exit 2
fi

if [[ -z "$V113_CHECKPOINT" ]]; then
  V113_CHECKPOINT="$(ls -1dt runs/gene_mrta_v113_route_tail_evolution/*/checkpoint.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$V1141_ADAPTIVE_CHECKPOINT" ]]; then
  V1141_ADAPTIVE_CHECKPOINT="$(python - <<'PY'
import glob
import json
import os

paths = sorted(
    glob.glob(
        "runs/gene_mrta_v1141_paired_seed*/adaptive/*/checkpoint.json"
    ),
    key=os.path.getmtime,
    reverse=True,
)
for path in paths:
    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except Exception:
        continue
    if (
        data.get("version") == "v1141"
        and data.get("recombination_mode") == "adaptive"
        and int(
            data.get(
                "next_generation",
                0,
            )
        ) >= 50
    ):
        print(path)
        break
PY
)"
fi

if [[ -z "$V113_CHECKPOINT" || ! -f "$V113_CHECKPOINT" ]]; then
  echo "Missing completed V1.13 checkpoint." >&2
  exit 2
fi

if [[ -z "$V1141_ADAPTIVE_CHECKPOINT" || ! -f "$V1141_ADAPTIVE_CHECKPOINT" ]]; then
  echo "Missing completed adaptive V1.14.1 checkpoint." >&2
  echo "Set V1141_ADAPTIVE_CHECKPOINT=/path/to/checkpoint.json" >&2
  exit 2
fi

echo "V113_CHECKPOINT=$V113_CHECKPOINT"
echo "V1141_ADAPTIVE_CHECKPOINT=$V1141_ADAPTIVE_CHECKPOINT"
echo "SEED=$SEED"

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v1142.assay \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --adaptive-v1141-checkpoint "$V1141_ADAPTIVE_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --pairs 8 \
    --seed "$SEED" \
    --output-dir runs/gene_mrta_v1142_matched_pair_smoke

elif [[ "$MODE" == "formal128" ]]; then
  python -m marl2d.gene_mrta_v1142.assay \
    --scenario-bank "$SCENARIO_BANK" \
    --bootstrap-v113-checkpoint "$V113_CHECKPOINT" \
    --adaptive-v1141-checkpoint "$V1141_ADAPTIVE_CHECKPOINT" \
    --anchor-v18-run "$V18_RUN" \
    --pairs 128 \
    --seed "$SEED" \
    --output-dir runs/gene_mrta_v1142_matched_pair

else
  echo "Usage: bash tools/run_gene_mrta_v1142_matched_pair_mac.sh [tests|smoke|formal128]" >&2
  exit 2
fi
