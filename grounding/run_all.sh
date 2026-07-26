#!/usr/bin/env bash
# One-shot driver for ALL Phase-1 grounding runs (+ optional Phase-3).
# Launch once (ideally under tmux/nohup); it runs every dataset sequentially,
# logs each to logs/, and writes CSVs to results/.  Only results/ needs to come
# back.
#
# Usage:
#   bash run_all.sh smoke     # ~2 min: tiny runs on every dataset to catch errors
#   bash run_all.sh full      # the real runs (hours)
#   bash run_all.sh dp        # only the DP-SGD runs (if comp already done)
#
# Recommended:  ALWAYS run `smoke` first.  If it finishes with all CSVs present
# and non-trivial accuracies, THEN launch `full` under tmux and walk away.

set -uo pipefail
MODE="${1:-smoke}"
PY="${PY:-python}"                 # override: PY=./.venv-gpu/bin/python bash run_all.sh full
mkdir -p results logs

DATASETS=(mnist cifar10 femnist adult)

if [ "$MODE" = "smoke" ]; then
  # DP needs a few epochs to show it TRAINS (1 epoch can't distinguish fix from
  # the earlier collapse); comp stays short.
  CEPOCHS=3; DPEPOCHS=4; EPS="1 8"; TAG="smoke"
else
  # per-dataset comp epochs; DP epochs shared
  declare -A CE=( [mnist]=30 [cifar10]=80 [femnist]=40 [adult]=40 )
  DPEPOCHS=20; EPS="0.5 1 2 4 8"; TAG="full"
fi

run() {  # run <name> <command...>
  local name="$1"; shift
  echo ">>> [$name] $*" | tee -a "logs/${TAG}.log"
  if "$@" >> "logs/${name}.log" 2>&1; then
    echo "    OK  $name" | tee -a "logs/${TAG}.log"
  else
    echo "    FAIL $name (see logs/${name}.log)" | tee -a "logs/${TAG}.log"
  fi
}

echo "=== run_all.sh MODE=$MODE  start $(date) ===" | tee -a "logs/${TAG}.log"

for ds in "${DATASETS[@]}"; do
  if [ "$MODE" = "smoke" ]; then ce=$CEPOCHS; else ce=${CE[$ds]}; fi
  run "comp_${ds}" $PY comp_cost.py --dataset "$ds" --epochs "$ce" \
      --out "results/comp_${ds}.csv"
  run "dp_${ds}"   $PY dp_accuracy.py --dataset "$ds" --epsilons $EPS \
      --epochs "$DPEPOCHS" --out "results/dp_${ds}.csv"
done

echo "=== done $(date).  CSVs in results/  ===" | tee -a "logs/${TAG}.log"
echo "Fetch just:  results/*.csv"
