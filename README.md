# Station information and request timing for fast-charging queues on intercity highway corridors

Code, data, and results for the paper *Station information and request timing for reducing fast-charging queues on intercity highway corridors* (Siyuan Meng and Chengbo Ai, University of Massachusetts Amherst; under review).

A dispatcher assigns charging requests of electric vehicles to fast-charging stations along a one-directional highway corridor. The repository contains the dispatch methods, two closed-loop simulators (a Python queueing simulator and SUMO via TraCI), the I-95 North Carolina station data, and the result files behind every table and figure.

## Contents

| Path | Content |
|---|---|
| `corridor/models/backlog.py` | Backlog greedy (with the queue-only and uniform-power variants) and fluid-queue MPC |
| `corridor/models/optimizer.py` | Request and station data types, nearest-station dispatch, static-capacity Γ-robust MILP |
| `corridor/env/corridor_sim.py` | Queueing simulator |
| `corridor/env/sumo_corridor.py` | SUMO environment (TraCI) |
| `corridor/layouts.py` | Corridor layouts: `synthetic`, `i95_nc`, `uniform_L<km>_S<stations>_C<chargers>[_P<kW>]` |
| `corridor/sumo/` | Network generator and generated SUMO networks; route files are written on demand |
| `corridor/data/` | I-95 station data (`i95_nc_stations.json`), its build script, and the AADT calibration script |
| `corridor/experiments/` | Experiment drivers, timing, summaries, and figures |
| `corridor/results/` | Result files (one record per episode) |

## Setup

1. Install Eclipse SUMO 1.24 so that `sumo` and `netconvert` are on the `PATH`.
2. `pip install -r requirements.txt`
3. Gurobi: the pip package includes a size-limited license (2,000 variables and constraints), which is enough for corridors of up to about 13 stations. Larger instances need a full or academic license.

All commands are run from the repository root.

## Reproducing the experiments

Every experiment uses `python -m corridor.experiments.run_threshold_sweep` (20 seeds, both simulators by default; `--workers` sets the number of processes). Each run writes `corridor/results/<tag>_<timestamp>.json` and prints a summary with paired 95% t-intervals.

| Experiment | Command (arguments after `run_threshold_sweep`) |
|---|---|
| E1–E3 main factorial | `--methods Greedy StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=15" "Fluid-MPC H=30" "Fluid-MPC H=60" --loads low moderate high --reference StaticCap --tag main_factorial` |
| Attribution baselines | `--methods Power-Greedy "Occupancy-Greedy g=3" "Occupancy-Greedy g=3 P120" --loads low moderate high --reference Power-Greedy --tag attr_synth` (add `--layout i95_nc --thresholds 0.20 0.30 0.40 --loads low=0.6 moderate=0.8 high=1.0` for I-95) |
| Port-status rules (NEVI data) | `--methods "Occupancy-Greedy g=3" "PortStatus-Greedy g=3" "PortStatus+Queue-Greedy g=3" --thresholds 0.30 --loads moderate high --reference "Occupancy-Greedy g=3" --tag portstatus_synth` (add `--layout i95_nc --loads moderate=0.8 high=1.0 --tag portstatus_i95` for I-95) |
| E4 prediction quality | `--methods "Backlog-Greedy g=3" "Fluid-MPC H=30" --thresholds 0.30 --loads high --reference "Backlog-Greedy g=3" --prediction_noise 0.1` (also 0.2, 0.4; or `--prediction_visibility 0.75`, 0.5, 0.25) |
| E5 charging-time variation | `--methods StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=30" --thresholds 0.30 --reference StaticCap --charge_uncertainty 0.0` (also 0.4) |
| E6 waiting weight | `--methods StaticCap Backlog-Greedy "Backlog-Greedy g=3" "Backlog-Greedy g=10" "Fluid-MPC H=30 g=1" "Fluid-MPC H=30" "Fluid-MPC H=30 g=10" --thresholds 0.30 --reference StaticCap` |
| E7 layouts | `--layout uniform_L300_S6_C2 --methods StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=30" --thresholds 0.20 0.30 --loads high=0.32 --reference StaticCap` (also `uniform_L150_S3_C2` at 0.16, `uniform_L450_S9_C2` at 0.48, `uniform_L300_S6_C2_P125` and `uniform_L300_S12_C1_P125` at 0.32) |
| E8 I-95 case study | `--layout i95_nc --methods Greedy StaticCap Backlog-Greedy "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=30" --thresholds 0.20 0.30 0.40 --loads low=0.6 moderate=0.8 high=1.0 --reference StaticCap` |
| E10 driver compliance | `--methods StaticCap "Occupancy-Greedy g=3" "Backlog-Greedy g=3" "Fluid-MPC H=30" --thresholds 0.30 --reference StaticCap --compliance 0.75` (also 0.5, 0.25; with `--layout i95_nc --loads high=1.0` for I-95) |

E9 (computation time) runs serially: `python -m corridor.experiments.run_timing --n_seeds 5`.

Figures: `python -m corridor.experiments.make_figures --out figures` reads the latest result files. Summary tables of the main factorial: `python corridor/experiments/summarize_main.py`.

In the result files, `env` is `phase1` (queueing simulator) or `phase2_sumo` (SUMO); waiting times are in minutes. Method names: `Greedy` is nearest-station dispatch, `StaticCap` the static-capacity MILP, `Power-Greedy` the power-aware greedy rule, `Occupancy-Greedy g=3` the queue greedy rule (`P120`: uniform 120 kW), `PortStatus-Greedy g=3` and `PortStatus+Queue-Greedy g=3` the port-status greedy rules without and with the queue count, `Backlog-Greedy g=3` backlog greedy, and `Fluid-MPC H=<min>` fluid-queue MPC. `QA-R-MPC (current)` in the main factorial is an auxiliary method not reported in the paper. SUMO results are deterministic for a given seed; MILP solutions can vary slightly with the number of solver threads.

## Data

`corridor/data/i95_nc_stations.json` lists the public CCS fast-charging sites within 2 km of I-95 in North Carolina, built by `python -m corridor.data.build_i95_nc --download` from the Alternative Fuels Data Center station API (U.S. Department of Energy, retrieved 14 October 2026) and OpenStreetMap road geometry (© OpenStreetMap contributors, ODbL). `python -m corridor.data.i95_aadt` relates the case-study demand levels to NCDOT AADT counts (2019).

## License

Code: MIT (see `LICENSE`). The station data derive from the Alternative Fuels Data Center (public domain, U.S. government work) and OpenStreetMap (ODbL).
