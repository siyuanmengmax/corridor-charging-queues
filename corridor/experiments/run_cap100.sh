#!/usr/bin/env bash
# Power sensitivity (2026-11-13): effective power capped at
# 100 kW instead of 150 kW on both real corridors. Entry rates scaled by the loss of charger
# capacity (I-95 x0.69 -> 0.7 veh/min, I-15 x0.71 -> 1.3 veh/min), so that the offered load is
# close to the high level of the main runs. theta = 0.30.
# Run from the repository root.
set -euo pipefail
M=(StaticCap "Occupancy-Greedy g=3" "Backlog-Greedy g=3" "Fluid-MPC H=30" \
   "PortStatus-Greedy g=3" "PortStatus+Queue-Greedy g=3")
for c in "i95_nc_cap100 high=0.7" "i15_lv_cap100 high=1.3"; do
  set -- $c
  python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
    --layout $1 --vehicles us --reference StaticCap --thresholds 0.3 --loads $2 \
    --tag cap100_${1%%_*} --methods "${M[@]}"
done
