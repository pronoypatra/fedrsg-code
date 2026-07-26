#!/usr/bin/env bash
# One-shot driver for ALL Phase-1 grounding runs (+ optional Phase-3).
# Launch once (ideally under tmux/nohup); it runs every dataset sequentially,
# logs each to logs/, and writes CSVs to results/.  Only results/ needs to come
# back.
#
# Usage:
#   bash run_all.sh smoke      # ~5 min: tiny runs on every dataset to catch errors
#   bash run_all.sh full       # the real runs (hours): comp + dp
#   bash run_all.sh full-comp  # only the COMPUTATION-cost runs (ready to go now)
#   bash run_all.sh full-dp    # only the DP-SGD runs
#
# Recommended:  ALWAYS run `smoke` first.  If it finishes with all CSVs present
# and non-trivial accuracies, THEN launch the full runs under tmux and walk away.

set -uo pipefail
MODE="${1:-smoke}"
PY="${PY:-python}"                 # override: PY=./.venv-gpu/bin/python bash run_all.sh full
mkdir -p results logs

DATASETS=(mnist cifar10 femnist adult)

# sub-epoch eval frequency (steps) per dataset -- easy/fast-rising sets need finer
# sampling to resolve the 0.3-0.9 accuracy range before it saturates.
declare -A EV=( [mnist]=20 [cifar10]=100 [femnist]=30 [adult]=10 )

# which stages to run for this mode
DO_COMP=1; DO_DP=1
if [ "$MODE" = "smoke" ]; then
  CEPOCHS=3; DPEPOCHS=4; EPS="0.5 8"; TAG="smoke"
else
  declare -A CE=( [mnist]=30 [cifar10]=80 [femnist]=40 [adult]=40 )
  # eps is spaced GEOMETRICALLY and anchored LOW: in DP-SGD the noise multiplier
  # ~ 1/eps, so accuracy rises steeply at low eps then flattens (concave in eps,
  # matching the A(a,eps) assumption). Linear 1..8 sits on the plateau and looks
  # flat; log-spaced 0.1..16 captures the knee where the variation actually is.
  DPEPOCHS=20; EPS="0.1 0.25 0.5 1 2 4 8 16"; TAG="$MODE"
  case "$MODE" in
    full)      : ;;                         # comp + dp
    full-comp) DO_DP=0 ;;                   # comp only (ready now)
    full-dp)   DO_COMP=0 ;;                 # dp only
    *) echo "unknown mode: $MODE (use smoke|full|full-comp|full-dp)"; exit 1 ;;
  esac
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
  if [ "$DO_COMP" = "1" ]; then
    run "comp_${ds}" $PY comp_cost.py --dataset "$ds" --epochs "$ce" \
        --eval-every "${EV[$ds]}" --out "results/comp_${ds}.csv"
  fi
  if [ "$DO_DP" = "1" ]; then
    run "dp_${ds}"   $PY dp_accuracy.py --dataset "$ds" --epsilons $EPS \
        --epochs "$DPEPOCHS" --out "results/dp_${ds}.csv"
  fi
done

echo "=== done $(date).  CSVs in results/  ===" | tee -a "logs/${TAG}.log"
echo "Fetch just:  results/*.csv"
