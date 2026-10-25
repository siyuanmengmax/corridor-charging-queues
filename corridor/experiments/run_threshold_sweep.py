"""
Decision lead sweep: how does the SOC level at which a vehicle requests
charging change the value of backlog-aware assignment?

Steady-state design: warm-up of 180 min, then 240 min measured, then drain.
Reachable stations are those within the vehicle's remaining range.

Run:
  python -m corridor.experiments.run_threshold_sweep [--n_seeds N] [--env p1 sumo] [--workers W]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.ERROR)
# one Gurobi thread per worker process (must be set before the models are imported)
os.environ.setdefault("CORRIDOR_GRB_THREADS", "1")

WARMUP_MIN = 180.0
MEASURE_MIN = 240.0
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
RATES = {"low": 0.20, "moderate": 0.25, "high": 0.35}     # vehicles entering per minute
SOC_INIT_MEAN = 0.45
REFERENCE = "QA-R-MPC (current)"
METHODS = ["Greedy", REFERENCE, "Backlog-Greedy", "Backlog-MILP g=3"]


def make_method(name: str):
    from ..models.backlog import BacklogAwareOptimizer, BacklogGreedyOptimizer, FluidMPCOptimizer
    from ..models.optimizer import GreedyOptimizer, QueueAwareOptimizer, StaticCapOptimizer
    return {
        "StaticCap":         lambda: StaticCapOptimizer(gamma_c=2.0, delta_t=5.0, speed_kmh=100.0),
        "Greedy":            lambda: GreedyOptimizer(speed_kmh=100.0),
        REFERENCE:           lambda: QueueAwareOptimizer(gamma_c=2.0, delta_t=5.0, speed_kmh=100.0),
        "Backlog-Greedy":    lambda: BacklogGreedyOptimizer(),
        "Backlog-Greedy g=3": lambda: BacklogGreedyOptimizer(gamma_w=3.0),
        "Backlog-MILP g=3":  lambda: BacklogAwareOptimizer(gamma_w=3.0),
        "Fluid-MPC H=0":     lambda: FluidMPCOptimizer(gamma_w=3.0),
        "Fluid-MPC H=15":    lambda: FluidMPCOptimizer(gamma_w=3.0, lookahead_min=15.0),
        "Fluid-MPC H=30":    lambda: FluidMPCOptimizer(gamma_w=3.0, lookahead_min=30.0),
        "Fluid-MPC H=60":    lambda: FluidMPCOptimizer(gamma_w=3.0, lookahead_min=60.0),
        # E6: waiting-time weight
        "Backlog-Greedy g=10": lambda: BacklogGreedyOptimizer(gamma_w=10.0),
        "Fluid-MPC H=30 g=1":  lambda: FluidMPCOptimizer(gamma_w=1.0, lookahead_min=30.0),
        "Fluid-MPC H=30 g=10": lambda: FluidMPCOptimizer(gamma_w=10.0, lookahead_min=30.0),
        # attribution baselines: station power without queue information, and live occupancy only
        "Power-Greedy":      lambda: BacklogGreedyOptimizer(gamma_w=0.0),
        "Occupancy-Greedy g=3": lambda: BacklogGreedyOptimizer(gamma_w=3.0, en_route=False),
        "Occupancy-Greedy g=3 P120": lambda: BacklogGreedyOptimizer(gamma_w=3.0, en_route=False,
                                                                    assumed_power_kw=120.0),
        # NEVI port data: which chargers are busy and their power, with and without the queue count
        "PortStatus-Greedy g=3": lambda: BacklogGreedyOptimizer(gamma_w=3.0, en_route=False,
                                                                remaining_known=False, queue_known=False),
        "PortStatus+Queue-Greedy g=3": lambda: BacklogGreedyOptimizer(gamma_w=3.0, en_route=False,
                                                                      remaining_known=False),
    }[name]()


def demand(rate: float) -> dict:
    return dict(n_vehicles=int(150 * rate), demand_rate=rate, soc_init_mean=SOC_INIT_MEAN)


def run_one(job: tuple) -> dict:
    env_name, load, threshold, method, seed = job[:5]
    extra = dict(job[5]) if len(job) > 5 else {}
    rate = extra.pop("rate", None) or RATES[load]
    layout = extra.pop("layout", "synthetic")
    opt = make_method(method)
    if env_name == "phase1":
        from .run_experiments import CORRIDOR_PARAMS, run_episode
        from ..layouts import get_layout   # same station positions as the SUMO network
        lay = get_layout(layout)
        stations, length = lay.charging_stations(), lay.length_km
        params = {**CORRIDOR_PARAMS, **demand(rate), "corridor_length_km": length,
                  "sim_duration_min": WARMUP_MIN + MEASURE_MIN, "warmup_min": WARMUP_MIN,
                  "charge_threshold": threshold, "max_detour_km": None, **extra}
        r = run_episode(method, opt, seed, stations, params)
    else:
        from .run_sumo_experiments import run_sumo_episode
        r = run_sumo_episode(
            method, opt, seed, demand=demand(rate),
            sim_duration_s=int((WARMUP_MIN + MEASURE_MIN) * 60), warmup_s=int(WARMUP_MIN * 60),
            charge_threshold=threshold, max_detour_km=None, layout=layout, **extra,
        )
    r.update({"env": env_name, "load": load, "rate": rate, "layout": layout, "threshold": threshold,
              "method": method, "seed": seed, **extra})
    return r


def _ci(xs) -> tuple:
    """Mean and half-width of the 95% t-interval (sample standard deviation)."""
    a = np.array(xs, dtype=float)
    return a.mean(), stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


def print_summary(records: List[dict], reference: str = REFERENCE) -> None:
    METHODS = list(dict.fromkeys(r["method"] for r in records))
    THRESHOLDS = sorted({r["threshold"] for r in records})
    REFERENCE = reference
    for env_name, load in dict.fromkeys((r["env"], r["load"]) for r in records):
        rate = next(r.get("rate", RATES.get(load)) for r in records if r["load"] == load)
        print(f"\n=== {env_name} / {load} demand ({rate}/min) ===")
        print(f"{'thr':>5} {'method':<20}{'load':>6}{'Wmax':>13}{'W95':>13}{'Wmean':>12}{'AccMean':>12}"
              f"{'Tstation':>12}{'dWmax vs ref':>16}{'dWmean vs ref':>16}{'dTst vs ref':>16}{'stranded':>9}")
        for thr in THRESHOLDS:
            sel = [r for r in records if (r["env"], r["load"], r["threshold"]) == (env_name, load, thr)]
            ref = {r["seed"]: r for r in sel if r["method"] == REFERENCE}
            for m in METHODS:
                recs = [r for r in sel if r["method"] == m]
                if not recs:
                    continue
                row = f"{thr:5.2f} {m:<20}{np.mean([r['offered_load'] for r in recs]):6.2f}"
                keys = ("max_wait_min", "p95_wait_min", "mean_wait_min", "access_mean_min", "station_mean_min")
                for k in (k for k in keys if k in recs[0]):
                    mu, h = _ci([r[k] for r in recs])
                    row += f"{mu:7.1f}±{h:<5.1f}"
                for k in (k for k in ("max_wait_min", "mean_wait_min", "station_mean_min") if k in recs[0]):
                    mu, h = _ci([r[k] - ref[r["seed"]][k] for r in recs if r["seed"] in ref])
                    row += f"{mu:+9.1f}±{h:<6.1f}"
                row += f"{np.mean([r['n_no_station'] for r in recs]):8.1f}"
                print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--env", nargs="+", default=["p1", "sumo"], choices=["p1", "sumo"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument("--loads", nargs="+", default=["moderate", "high"],
                        help="names from RATES, or name=rate (vehicles/min)")
    parser.add_argument("--layout", default="synthetic", help="see corridor/layouts.py")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--tag", default="threshold_sweep")
    parser.add_argument("--charge_uncertainty", type=float, default=None)
    parser.add_argument("--prediction_noise", type=float, default=None)
    parser.add_argument("--prediction_visibility", type=float, default=None)
    parser.add_argument("--compliance", type=float, default=None)
    parser.add_argument("--output_dir", default="corridor/results")
    args = parser.parse_args()

    envs = [{"p1": "phase1", "sumo": "phase2_sumo"}[e] for e in args.env]
    loads = {}
    for item in args.loads:
        name, _, val = item.partition("=")
        loads[name] = float(val) if val else RATES[name]
    if "phase2_sumo" in envs:   # write network and demand files before the workers start
        from ..sumo.generate_network import ensure_network, write_routes
        ensure_network(args.layout)
        for rate in loads.values():
            for seed in range(args.n_seeds):
                write_routes(**demand(rate), seed=seed,
                             duration_s=int((WARMUP_MIN + MEASURE_MIN) * 60), layout=args.layout)

    extra = {k: getattr(args, k) for k in ("charge_uncertainty", "prediction_noise", "prediction_visibility", "compliance")
             if getattr(args, k) is not None}
    if args.layout != "synthetic":
        extra["layout"] = args.layout
    jobs = [(e, load, thr, m, seed, {**extra, "rate": rate}) for e in envs for load, rate in loads.items()
            for thr in args.thresholds for m in args.methods for seed in range(args.n_seeds)]
    with Pool(args.workers) as pool:
        records = pool.map(run_one, jobs, chunksize=1)

    out = Path(args.output_dir) / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {out}")
    print_summary(records, args.reference)
