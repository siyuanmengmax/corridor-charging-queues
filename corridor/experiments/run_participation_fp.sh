#!/usr/bin/env bash
# E11b partial participation with informed non-participants (2026-11-14): as run_participation.sh,
# but the EVs outside the dispatcher choose the nearest reachable station that shows a free
# charger at their request (port status, as in a charging app), else the nearest reachable one.
# Run from the repository root.
set -euo pipefail
M=("OwnFleet-own g=3" "OwnFleet-port g=3" "OwnFleet-port_queue g=3" "OwnFleet-station g=3")
for p in 0.75 0.5 0.25; do
  python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
    --reference "OwnFleet-station g=3" --participation $p --other_choice free_port --thresholds 0.3 \
    --tag partfp_syn_p$p --loads high --methods "${M[@]}"
  for c in "i95_nc high=1.0" "i15_lv high=1.8"; do
    set -- $c
    python -m corridor.experiments.run_threshold_sweep --n_seeds 20 --env p1 sumo --workers 30 \
      --reference "OwnFleet-station g=3" --participation $p --other_choice free_port --thresholds 0.3 \
      --layout $1 --vehicles us --tag partfp_${1%%_*}us_p$p --loads $2 --methods "${M[@]}"
  done
done
