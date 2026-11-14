#!/usr/bin/env bash
# I-95 case study with US vehicle parameters (corridor/vehicles.py, profile "us"), 2026-11-13.
# Replaces i95_case_*, i95_case_m3_*, attr_i95_*, attr_p120_i95_*, portstatus_i95_*, e10_i95_*.
# Run from the repository root.
set -euo pipefail
S="python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
   --layout i95_nc --vehicles us --reference StaticCap"
LOADS="low=0.6 moderate=0.8 high=1.0"

# main comparison, gamma sweep point and attribution baselines (main comparison and information tables)
$S --tag i95us_main --thresholds 0.2 0.3 0.4 --loads $LOADS --methods \
   Greedy StaticCap Backlog-Greedy "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=30" \
   Power-Greedy "Occupancy-Greedy g=3" "Occupancy-Greedy g=3 P120"
# NEVI port-status rules
$S --tag i95us_portstatus --thresholds 0.3 --loads moderate=0.8 high=1.0 --methods \
   "Occupancy-Greedy g=3" "PortStatus-Greedy g=3" "PortStatus+Queue-Greedy g=3"
# E10 driver compliance
for c in 0.75 0.5 0.25; do
  $S --tag i95us_e10_c$c --compliance $c --thresholds 0.3 --loads high=1.0 --methods \
     StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=30" "Occupancy-Greedy g=3"
done
