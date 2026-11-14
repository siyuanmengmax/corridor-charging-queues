#!/usr/bin/env bash
# I-15 case study (Las Vegas to Barstow) with US vehicle parameters (corridor/vehicles.py, profile "us"), 2026-11-13.
# Second real corridor.
# Run from the repository root.
set -euo pipefail
S="python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
   --layout i15_lv --vehicles us --reference StaticCap"
LOADS="low=1.2 moderate=1.5 high=1.8"

# main comparison, gamma sweep point and attribution baselines (main comparison and information tables)
$S --tag i15us_main --thresholds 0.2 0.3 0.4 --loads $LOADS --methods \
   Greedy StaticCap Backlog-Greedy "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=30" \
   Power-Greedy "Occupancy-Greedy g=3" "Occupancy-Greedy g=3 P120"
# NEVI port-status rules
$S --tag i15us_portstatus --thresholds 0.3 --loads moderate=1.5 high=1.8 --methods \
   "Occupancy-Greedy g=3" "PortStatus-Greedy g=3" "PortStatus+Queue-Greedy g=3"
# E10 driver compliance
for c in 0.75 0.5 0.25; do
  $S --tag i15us_e10_c$c --compliance $c --thresholds 0.3 --loads high=1.8 --methods \
     StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=30" "Occupancy-Greedy g=3"
done
