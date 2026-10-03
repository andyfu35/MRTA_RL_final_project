#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tests}"
VENV_DIR="${VENV_DIR:-.venv-gene}"
SCENARIO_BANK="${SCENARIO_BANK:-runs/gene_mrta_v110_scenario_bank/scenario_bank_100.json}"
V18_RUN="${V18_RUN:-runs/gene_mrta_v18t_scale/gene_mrta_v18t_scale_20261001_223157_seed7}"
V19_CHECKPOINT="${V19_CHECKPOINT:-}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"

pytest -q   tests/test_gene_mrta_v17.py   tests/test_gene_mrta_v17t_scale.py   tests/test_gene_mrta_v18.py   tests/test_gene_mrta_v18_publication.py   tests/test_gene_mrta_v19_failure_analysis.py   tests/test_gene_mrta_v19_robust_bank.py   tests/test_gene_mrta_v110_mating.py

if [[ "$MODE" == "tests" ]]; then
  exit 0
fi

if [[ "$MODE" == "build-bank" ]]; then
  python -m marl2d.gene_mrta_v110.scenario_bank     --seed-base 95000000     --candidates 500     --worlds 100     --time-limit 300     --retry-time-limit 900
  exit 0
fi

if [[ ! -f "$SCENARIO_BANK" ]]; then
  echo "Missing V1.10 scenario bank: $SCENARIO_BANK" >&2
  echo "Run: bash tools/run_gene_mrta_v110_mating_mac.sh build-bank" >&2
  exit 2
fi

if [[ -z "$V19_CHECKPOINT" ]]; then
  V19_CHECKPOINT="$(ls -1dt runs/gene_mrta_v19_robust_bank/*/checkpoint.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$V19_CHECKPOINT" || ! -f "$V19_CHECKPOINT" ]]; then
  echo "Missing V1.9 checkpoint." >&2
  echo "Set V19_CHECKPOINT=/path/to/checkpoint.json" >&2
  exit 2
fi

if [[ ! -f "$V18_RUN/summary.json" ]]; then
  echo "Missing V1.8 anchor summary: $V18_RUN/summary.json" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  python -m marl2d.gene_mrta_v110.train     --scenario-bank "$SCENARIO_BANK"     --bootstrap-v19-checkpoint "$V19_CHECKPOINT"     --anchor-v18-run "$V18_RUN"     --generations 5     --normal-offspring 16     --mating-offspring 16     --mating-pairs 4     --children-per-pair 4     --screen-worlds 25     --normal-full-per-axis 1     --mating-full-candidates 4     --archive-size-per-axis 4     --hybrid-bank-limit 16     --mating-q-power 10     --inheritance-threshold 0.95     --mutation-sigma-start 0.03     --mutation-sigma-end 0.02     --checkpoint-every 1     --log-every 1     --seed 7

elif [[ "$MODE" == "long" ]]; then
  python -m marl2d.gene_mrta_v110.train     --scenario-bank "$SCENARIO_BANK"     --bootstrap-v19-checkpoint "$V19_CHECKPOINT"     --anchor-v18-run "$V18_RUN"     --generations 500     --normal-offspring 128     --mating-offspring 128     --mating-pairs 32     --children-per-pair 4     --screen-worlds 25     --normal-full-per-axis 4     --mating-full-candidates 16     --archive-size-per-axis 16     --hybrid-bank-limit 128     --mating-q-power 10     --inheritance-threshold 0.95     --mutation-sigma-start 0.08     --mutation-sigma-end 0.01     --mating-mutation-sigma 0     --checkpoint-every 10     --log-every 5     --seed 7

elif [[ "$MODE" == "resume" ]]; then
  CHECKPOINT="${2:-}"
  if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash tools/run_gene_mrta_v110_mating_mac.sh resume <checkpoint.json>" >&2
    exit 2
  fi
  python -m marl2d.gene_mrta_v110.train     --scenario-bank "$SCENARIO_BANK"     --bootstrap-v19-checkpoint "$V19_CHECKPOINT"     --anchor-v18-run "$V18_RUN"     --resume "$CHECKPOINT"     --generations 500     --normal-offspring 128     --mating-offspring 128     --mating-pairs 32     --children-per-pair 4     --screen-worlds 25     --normal-full-per-axis 4     --mating-full-candidates 16     --archive-size-per-axis 16     --hybrid-bank-limit 128     --mating-q-power 10     --inheritance-threshold 0.95     --mutation-sigma-start 0.08     --mutation-sigma-end 0.01     --mating-mutation-sigma 0     --checkpoint-every 10     --log-every 5     --seed 7

else
  echo "Usage: bash tools/run_gene_mrta_v110_mating_mac.sh [tests|build-bank|smoke|long|resume <checkpoint.json>]" >&2
  exit 2
fi
