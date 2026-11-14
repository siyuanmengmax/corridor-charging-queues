# Station information and request timing for fast-charging queues on intercity highway corridors

Code, data, and results for the paper *Station Information and Request Timing for Reducing Fast-Charging Queues on Intercity Highway Corridors* (Siyuan Meng and Chengbo Ai, University of Massachusetts Amherst; under review).

A dispatcher assigns charging requests of electric vehicles to fast-charging stations along a one-directional highway corridor. The repository contains the dispatch methods, two closed-loop simulators (a Python queueing simulator and SUMO via TraCI), the station data of two real corridors (I-95 in North Carolina and I-15 from Las Vegas to Barstow), and the result files behind every table and figure.

## Contents

| Path | Content |
|---|---|
| `corridor/models/backlog.py` | Backlog greedy (with the queue-only, uniform-power, port-status, and own-fleet variants) and fluid-queue MPC |
| `corridor/models/optimizer.py` | Request and station data types, nearest-station dispatch, static-capacity Γ-robust MILP |
| `corridor/env/corridor_sim.py` | Queueing simulator |
| `corridor/env/sumo_corridor.py` | SUMO environment (TraCI) |
| `corridor/layouts.py` | Corridor layouts: `synthetic`, `i95_nc`, `i15_lv`, `uniform_L<km>_S<stations>_C<chargers>[_P<kW>]`; `<layout>_cap<kW>` caps the effective power |
| `corridor/vehicles.py` | Vehicle profiles: `eu` (synthetic corridor) and `us` (real corridors) |
| `corridor/sumo/` | Network generator and generated SUMO networks; route files and capped-power networks are written on demand |
| `corridor/data/` | Station data of both real corridors, their build scripts, and the AADT scripts |
| `corridor/experiments/` | Experiment drivers and run scripts, timing, summaries, and figures |
| `corridor/results/` | Result files (one record per episode) |

## Setup

1. Install Eclipse SUMO 1.24 so that `sumo` and `netconvert` are on the `PATH`.
2. `pip install -r requirements.txt`
3. Gurobi: the pip package includes a size-limited license (2,000 variables and constraints), which is enough for corridors of up to about 13 stations. Larger instances need a full or academic license.

All commands are run from the repository root.

## Reproducing the experiments

Every experiment uses `python -m corridor.experiments.run_threshold_sweep` (20 seeds, both simulators by default; `--workers` sets the number of processes). Each run writes `corridor/results/<tag>_<timestamp>.json` and prints a summary with paired 95% t-intervals.

### Synthetic corridor

| Experiment | Command (arguments after `run_threshold_sweep`) |
|---|---|
| E1–E3 main factorial | `--methods Greedy StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=15" "Fluid-MPC H=30" "Fluid-MPC H=60" --loads low moderate high --reference StaticCap --tag main_factorial` |
| Information steps | `--methods Power-Greedy "Occupancy-Greedy g=3" "Occupancy-Greedy g=3 P120" --loads low moderate high --reference Power-Greedy --tag attr_synth` |
| Port-status rules (NEVI data) | `--methods "Occupancy-Greedy g=3" "PortStatus-Greedy g=3" "PortStatus+Queue-Greedy g=3" --thresholds 0.30 --loads moderate high --reference "Occupancy-Greedy g=3" --tag portstatus_synth` |
| E4 prediction quality | `--methods "Backlog-Greedy g=3" "Fluid-MPC H=30" --thresholds 0.30 --loads high --reference "Backlog-Greedy g=3" --prediction_noise 0.1` (also 0.2, 0.4; or `--prediction_visibility 0.75`, 0.5, 0.25) |
| E5 charging-time variation | `--methods StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=30" --thresholds 0.30 --reference StaticCap --charge_uncertainty 0.0` (also 0.4) |
| E6 waiting weight | `--methods StaticCap Backlog-Greedy "Backlog-Greedy g=3" "Backlog-Greedy g=10" "Fluid-MPC H=30 g=1" "Fluid-MPC H=30" "Fluid-MPC H=30 g=10" --thresholds 0.30 --reference StaticCap` |
| E7 layouts | `--layout uniform_L300_S6_C2 --methods StaticCap "Backlog-Greedy g=3" "Fluid-MPC H=0" "Fluid-MPC H=30" --thresholds 0.20 0.30 --loads high=0.32 --reference StaticCap` (also `uniform_L150_S3_C2` at 0.16, `uniform_L450_S9_C2` at 0.48, `uniform_L300_S6_C2_P125` and `uniform_L300_S12_C1_P125` at 0.32) |
| E10 driver compliance | `--methods StaticCap "Occupancy-Greedy g=3" "Backlog-Greedy g=3" "Fluid-MPC H=30" --thresholds 0.30 --reference StaticCap --compliance 0.75` (also 0.5, 0.25) |

The queueing-simulator records of the synthetic main factorial, information steps, E4, E5, and E6 files were rerun with the SUMO station positions and merged with `python -m corridor.experiments.merge_aligned`.

### Real corridors (US vehicle parameters)

| Experiment | Script |
|---|---|
| E8 I-95 NC: main comparison, information steps, port-status rules, compliance | `bash corridor/experiments/run_i95_us.sh` (tags `i95us_*`) |
| E8 I-15 Las Vegas–Barstow: the same | `bash corridor/experiments/run_i15_us.sh` (tags `i15us_*`) |
| E11 partial participation (non-participants go to the nearest reachable station) | `bash corridor/experiments/run_participation.sh` (tags `part_*`, all three corridors) |
| E11b partial participation (non-participants go to the nearest station showing a free charger) | `bash corridor/experiments/run_participation_fp.sh` (tags `partfp_*`) |
| E12 effective power capped at 100 kW | `bash corridor/experiments/run_cap100.sh` (tags `cap100_*`) |

E9 (computation time) runs serially: `python -m corridor.experiments.run_timing --n_seeds 5` (synthetic corridor; `--i95_us_only` or `--i15_us_only` for the real corridors).

### Summaries and figures

- Synthetic main factorial: `python -m corridor.experiments.summarize_main`
- Real corridors: `python -m corridor.experiments.summarize_i95 i95us` (or `i15us`)
- Partial participation: `python -m corridor.experiments.summarize_participation part` (or `partfp`)
- Figures: `python -m corridor.experiments.make_figures --out paper/figures`, `python -m corridor.experiments.fig_threshold3`, `python -m corridor.experiments.fig_framework`, and `python -m corridor.experiments.fig_corridors --i95_raw <dir> --i15_raw <dir> --states <states-10m.json>` (raw downloads of the build scripts and the us-atlas state outlines). The figure scripts write to `paper/figures/`; create it first.

In the result files, `env` is `phase1` (queueing simulator) or `phase2_sumo` (SUMO); waiting times are in minutes. Method names: `Greedy` is nearest-station dispatch, `StaticCap` the static-capacity MILP, `Power-Greedy` the power-aware greedy rule, `Occupancy-Greedy g=3` the queue greedy rule (`P120`: uniform 120 kW), `PortStatus-Greedy g=3` and `PortStatus+Queue-Greedy g=3` the port-status greedy rules without and with the queue count, `Backlog-Greedy g=3` backlog greedy, `Fluid-MPC H=<min>` fluid-queue MPC, and `OwnFleet-<level> g=3` the partial-participation dispatcher that knows its own vehicles and sees the others through station data (`own`: none, `port`: port status and power, `port_queue`: plus the queue count, `station`: full station data). In partial-participation files, `mean_wait_own_min` and `mean_wait_other_min` are the mean waits of the EVs that use the dispatcher and of the others. `QA-R-MPC (current)` in the main factorial is an auxiliary method not reported in the paper. SUMO results are deterministic for a given seed; MILP solutions can vary slightly with the number of solver threads.

The I-95 results with European vehicle parameters of the first version of this repository (`i95_case_*`, `attr_i95_*`, `attr_p120_i95_*`, `portstatus_i95_*`, `e10_i95_*`, `diag_i95_*`) were replaced by the `i95us_*` runs; they remain in the commit history.

## Data

- `corridor/data/i95_nc_stations.json`: public CCS fast-charging sites within 2 km of I-95 in North Carolina, built by `python -m corridor.data.build_i95_nc --download` from the Alternative Fuels Data Center station API (U.S. Department of Energy, retrieved 14 October 2026) and OpenStreetMap road geometry. `python -m corridor.data.i95_aadt` relates the demand levels to NCDOT AADT counts (2019).
- `corridor/data/i15_lv_stations.json`: the same for I-15 southbound from the I-215 interchange south of Las Vegas to the I-40 junction in Barstow, built by `python -m corridor.data.build_i15 --raw_dir <dir> --download` (AFDC retrieved 13 November 2026). `python -m corridor.data.i15_aadt` relates the demand levels to Caltrans traffic volumes (2024).

## License

Code: MIT (see `LICENSE`). The station data derive from the Alternative Fuels Data Center (public domain, U.S. government work) and OpenStreetMap (© OpenStreetMap contributors, ODbL).
