"""
E9: computation time, measured serially (one process, one Gurobi thread) so that
parallel experiment runs do not distort it.

Part 1 (closed loop): Phase 1 episodes at high demand, threshold 0.30; every solve is
timed (model build + solve, wall clock) together with its batch size.
Part 2 (scaling): batches of random requests on generated corridors with 6 and 12 stations
(larger instances exceed the size-limited Gurobi license installed here).

Run (with the machine otherwise idle):
  python -m corridor.experiments.run_timing [--n_seeds 5]
"""

from __future__ import annotations

import os

os.environ.setdefault("CORRIDOR_GRB_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import gurobipy as gp
import numpy as np

from ..layouts import get_layout
from ..models.optimizer import EVRequest
from .run_threshold_sweep import MEASURE_MIN, RATES, WARMUP_MIN, demand, make_method

N_STATIONS = (6, 12)
METHODS = ["StaticCap", "Backlog-Greedy g=3", "Fluid-MPC H=0", "Fluid-MPC H=30", "Fluid-MPC H=60"]


class Timed:
    """Wraps an optimizer; records (released requests, predicted requests, wall ms) per solve."""

    def __init__(self, opt):
        self.opt = opt
        self.lookahead_min = getattr(opt, "lookahead_min", 0.0)
        self.log = []

    def solve(self, requests, stations, travel_times=None):
        t0 = time.perf_counter()
        res = self.opt.solve(requests, stations, travel_times=travel_times)
        ms = (time.perf_counter() - t0) * 1000.0
        n_pred = sum(1 for r in requests if r.release_min > 0)
        self.log.append((len(requests) - n_pred, n_pred, ms))
        return res


def closed_loop(layout: str, rate: float, n_seeds: int) -> list:
    from .run_experiments import CORRIDOR_PARAMS, run_episode
    out = []
    for m in METHODS:
        for seed in range(n_seeds):
            lay = get_layout(layout)
            stations, length = lay.charging_stations(), lay.length_km
            params = {**CORRIDOR_PARAMS, **demand(rate), "corridor_length_km": length,
                      "sim_duration_min": WARMUP_MIN + MEASURE_MIN, "warmup_min": WARMUP_MIN,
                      "charge_threshold": 0.30, "max_detour_km": None}
            opt = Timed(make_method(m))
            run_episode(m, opt, seed, stations, params)
            out += [{"part": "closed_loop", "layout": layout, "method": m, "seed": seed,
                     "n_released": a, "n_predicted": b, "ms": ms} for a, b, ms in opt.log]
    return out


def scaling(n_reps: int = 20) -> list:
    rng = np.random.default_rng(0)
    out = []
    for n_st in N_STATIONS:
        lay = get_layout(f"uniform_L{50 * n_st}_S{n_st}_C3")
        for n_req in (5, 10, 20, 40, 80):
            for m in METHODS:
                opt = make_method(m)
                for rep in range(n_reps):
                    stations = lay.charging_stations()
                    for st in stations:   # partially busy chargers and a short backlog
                        st.remaining_times = list(rng.uniform(0, 40, size=rng.integers(0, st.n_chargers + 1)))
                        st.committed = [(float(rng.uniform(0, 60)), float(rng.uniform(15, 40)))
                                        for _ in range(rng.integers(0, 4))]
                    reqs = []
                    for i in range(n_req):
                        predicted = getattr(opt, "lookahead_min", 0.0) > 0 and i % 2 == 1
                        reqs.append(EVRequest(id=f"v{i}", position_km=float(rng.uniform(0, lay.length_km * 0.8)),
                                              soc=float(rng.uniform(0.25, 0.4)), battery_kwh=float(rng.normal(70, 10)),
                                              release_min=float(rng.uniform(1, opt.lookahead_min)) if predicted else 0.0))
                    t0 = time.perf_counter()
                    try:
                        opt.solve(reqs, stations, travel_times=None)
                        ms = (time.perf_counter() - t0) * 1000.0
                    except gp.GurobiError:   # size-limited license (2000 variables / constraints)
                        ms = float("nan")
                    out.append({"part": "scaling", "n_stations": n_st, "n_requests": n_req, "method": m,
                                "rep": rep, "ms": ms})
    return out


def summarize(R: list) -> None:
    print("\nClosed loop (ms per solve, released + predicted requests per solve):")
    for lay in dict.fromkeys(r["layout"] for r in R if r["part"] == "closed_loop"):
        for m in METHODS:
            x = [r for r in R if r["part"] == "closed_loop" and r["layout"] == lay and r["method"] == m]
            ms = np.array([r["ms"] for r in x])
            n = np.array([r["n_released"] + r["n_predicted"] for r in x])
            print(f"  {lay:<10} {m:<20} solves={len(x):5d}  batch mean={n.mean():5.1f} max={n.max():3d}  "
                  f"ms mean={ms.mean():7.1f} p95={np.percentile(ms, 95):7.1f} max={ms.max():7.1f}")
    print("\nScaling (mean ms / max ms):")
    for n_st in N_STATIONS:
        for m in METHODS:
            row = f"  {n_st:2d} stations {m:<20}"
            for n_req in (5, 10, 20, 40, 80):
                ms = [r["ms"] for r in R if r["part"] == "scaling" and r["n_stations"] == n_st
                      and r["n_requests"] == n_req and r["method"] == m]
                row += f"  {np.mean(ms):7.1f}/{np.max(ms):<7.1f}" if ms and not np.isnan(ms).any() else "     size limit  "
            print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--output_dir", default="corridor/results")
    args = ap.parse_args()
    R = closed_loop("synthetic", RATES["high"], args.n_seeds) + closed_loop("i95_nc", 1.0, args.n_seeds) + scaling()
    out = Path(args.output_dir) / f"timing_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(R))
    print(f"Saved {len(R)} records to {out}")
    summarize(R)
