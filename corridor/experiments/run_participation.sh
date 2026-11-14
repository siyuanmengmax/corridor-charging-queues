#!/usr/bin/env bash
# E11 partial participation (2026-11-13):
# a share p of the EVs use the dispatcher; the others drive to the nearest reachable station and
# are visible to the dispatcher only once they are at a station. The dispatcher knows its own
# vehicles and sees the others through station data of increasing detail
# (OwnFleetGreedyOptimizer: own < port < port_queue < station). theta = 0.30, high demand.
# Run from the repository root.
set -euo pipefail
M=("OwnFleet-own g=3" "OwnFleet-port g=3" "OwnFleet-port_queue g=3" "OwnFleet-station g=3")
for p in 0.75 0.5 0.25; do
  python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
    --reference "OwnFleet-station g=3" --participation $p --thresholds 0.3 \
    --tag part_syn_p$p --loads high --methods "${M[@]}"
  for c in "i95_nc high=1.0" "i15_lv high=1.8"; do
    set -- $c
    python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
      --reference "OwnFleet-station g=3" --participation $p --thresholds 0.3 \
      --layout $1 --vehicles us --tag part_${1%%_*}us_p$p --loads $2 --methods "${M[@]}"
  done
done
