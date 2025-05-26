#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

# Outer loop: run 4 iterations
for iter in {1..4}; do
  echo "===== Iteration $iter: starting reward_iteration_1.py in parallel ====="

  # Launch reward_iteration_1.py in parallel with seeds 1,2,3,4
  for seed in 1 2 3 4; do
    OUT="${LOG_DIR}/iter${iter}_reward1_seed${seed}.log"
    echo "[Iteration $iter] nohup reward_iteration_1.py --seed $seed -> ${OUT}"
    nohup python ARCS/src/reward_iteration_1.py --seed "$seed" > "${OUT}" 2>&1 &
  done

  # Wait for all background jobs to finish
  wait
  echo "===== reward_iteration_1.py completed ====="

  # Run reward_iteration_2.py once
  OUT2="${LOG_DIR}/iter${iter}_reward2.log"
  echo "[Iteration $iter] nohup reward_iteration_2.py -> ${OUT2}"
  nohup python ARCS/src/reward_iteration_2.py > "${OUT2}" 2>&1

  echo "===== reward_iteration_2.py completed ====="
done

echo "All iterations completed."
