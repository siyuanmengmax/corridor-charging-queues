"""
Phase 1 experiment runner for the corridor EV charging scheduling paper.

Compares six methods (revision.md §4.3):
  1. Greedy          - nearest station, no optimisation
  2. Det-SO          - deterministic MILP (Γ_c = 0), no queue model
  3. SAA-20          - Sample Average Approximation, N=20 scenarios
  4. SAA-50          - Sample Average Approximation, N=50 scenarios
  5. R-MPC-StaticCap - Γ-robust MILP, static temporal-capacity proxy (ablation ← original Paper A)
  6. R-MPC-Queue     - Γ-robust MILP + M/M/c PWL epigraph (proposed method)

Outputs a JSON results file under corridor/results/.

Usage:
    python -m corridor.experiments.run_experiments [--n_seeds N] [--gamma_c G]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..env.corridor_sim import CorridorSimulation
from ..models.optimizer import (
    GreedyOptimizer,
    DeterministicOptimizer,
    StaticCapOptimizer,
    QueueAwareOptimizer,
    SAAOptimizer,
    EVRequest,
    ChargingStation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corridor configuration
# ---------------------------------------------------------------------------

def make_stations() -> List[ChargingStation]:
    """
    Six stations along a 300 km corridor with heterogeneous power levels.
    Alternating 150 kW (fast) and 100 kW (standard) — power diversity makes
    station-dependent charging duration (D4) materially affect assignments.
    Charger counts are intentionally tight to create realistic congestion.
    """
    configs = [
        # (position_km, n_chargers, power_kw)
        (0,   2, 150.0),  # cs_0: corridor entry, fast, 2 chargers
        (60,  2, 100.0),  # cs_1: standard
        (120, 3, 150.0),  # cs_2: mid-corridor hub, fast
        (180, 2, 100.0),  # cs_3: standard
        (240, 2, 150.0),  # cs_4: fast
        (300, 2, 100.0),  # cs_5: corridor exit, standard
    ]  # total: 13 chargers
    return [
        ChargingStation(id=f"cs_{i}", position_km=float(p), n_chargers=c, power_kw=pw)
        for i, (p, c, pw) in enumerate(configs)
    ]


CORRIDOR_PARAMS = dict(
    corridor_length_km=300.0,
    n_vehicles=120,
    delta_t=5.0,
    sim_duration_min=240.0,
    demand_rate=0.8,           # higher vehicle entry rate
    speed_kmh=100.0,
    speed_std_kmh=10.0,
    consumption_mean=0.20,
    battery_mean_kwh=70.0,
    battery_std_kwh=10.0,
    soc_init_mean=0.35,        # lower initial SOC → more vehicles need charging early
    soc_init_std=0.15,
    charge_threshold=0.20,
    target_soc=0.80,
    charge_uncertainty=0.20,
)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def make_methods(gamma_c: float = 2.0, gamma_arrival: float = 0.0) -> dict:
    """Return dict of {name: optimizer_instance}."""
    return {
        "Greedy": GreedyOptimizer(speed_kmh=100.0),
        "Det-SO": DeterministicOptimizer(speed_kmh=100.0, delta_t=5.0),
        "SAA-20": SAAOptimizer(n_scenarios=20, speed_kmh=100.0, delta_t=5.0),
        "SAA-50": SAAOptimizer(n_scenarios=50, speed_kmh=100.0, delta_t=5.0),
        "R-MPC-StaticCap": StaticCapOptimizer(
            gamma_c=gamma_c, speed_kmh=100.0, delta_t=5.0
        ),
        "R-MPC-Queue": QueueAwareOptimizer(
            gamma_c=gamma_c,
            gamma_arrival=gamma_arrival,
            speed_kmh=100.0,
            delta_t=5.0,
            n_pwl_breakpoints=6,
        ),
    }


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    method_name: str,
    optimizer,
    seed: int,
    stations: List[ChargingStation],
    params: dict,
) -> dict:
    """Run one full simulation episode and return metrics + solve times."""
    sim = CorridorSimulation(stations=stations, rng_seed=seed, **params)
    sim.reset()

    solve_times_ms: List[float] = []
    n_steps = int(params["sim_duration_min"] / params["delta_t"])

    for step_idx in range(n_steps):
        # Get current station state
        station_snapshot = sim.get_station_snapshot()

        # Get charging requests
        requests = sim.get_charging_requests()

        # Run optimizer
        if requests:
            upcoming = sim.get_upcoming_requests(getattr(optimizer, "lookahead_min", 0.0))
            result = optimizer.solve(
                requests + upcoming, station_snapshot,
                travel_times=sim.travel_times(requests + upcoming),
            )
            solve_times_ms.append(result.solve_time_ms)
            sim.apply_assignments(result)
        else:
            solve_times_ms.append(0.0)

        # Advance simulation
        sim.step()

    # Serve every vehicle that requested during the episode (no censored waits)
    sim.drain()

    ep = sim.episode_metrics()
    ep["method"] = method_name
    ep["seed"] = seed
    if solve_times_ms:
        non_zero = [t for t in solve_times_ms if t > 0]
        ep["solve_time_mean_ms"] = float(np.mean(non_zero)) if non_zero else 0.0
        ep["solve_time_p95_ms"] = float(np.percentile(non_zero, 95)) if non_zero else 0.0
        ep["solve_time_max_ms"] = float(max(non_zero)) if non_zero else 0.0
    else:
        ep["solve_time_mean_ms"] = 0.0
        ep["solve_time_p95_ms"] = 0.0
        ep["solve_time_max_ms"] = 0.0
    return ep


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all(
    n_seeds: int = 10,
    gamma_c: float = 2.0,
    gamma_arrival: float = 0.0,
    output_dir: str = "corridor/results",
    method_filter: List[str] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    stations = make_stations()
    methods = make_methods(gamma_c=gamma_c, gamma_arrival=gamma_arrival)
    if method_filter:
        methods = {k: v for k, v in methods.items() if k in method_filter}

    all_records = []
    seeds = list(range(n_seeds))

    for method_name, optimizer in methods.items():
        logger.info(f"Running {method_name} ({n_seeds} seeds)...")
        for seed in seeds:
            t0 = time.perf_counter()
            try:
                record = run_episode(method_name, optimizer, seed, stations, CORRIDOR_PARAMS)
                wall_s = time.perf_counter() - t0
                record["wall_time_s"] = wall_s
                all_records.append(record)
                logger.info(
                    f"  seed={seed:2d}  mean_wait={record['mean_wait_min']:.2f} min  "
                    f"max_wait={record['max_wait_min']:.2f} min  "
                    f"solve={record['solve_time_mean_ms']:.1f} ms  "
                    f"wall={wall_s:.1f}s"
                )
            except Exception as e:
                logger.error(f"  seed={seed} FAILED: {e}")

    # Save raw records
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"corridor_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"Saved {len(all_records)} records to {out_path}")

    # Print summary table
    _print_summary(all_records, methods.keys())


def _print_summary(records: List[dict], method_names) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY (mean ± 95% CI across seeds)")
    print("=" * 80)
    header = f"{'Method':<22} {'MeanWait(min)':>14} {'MaxWait(min)':>13} {'P95Wait(min)':>13} {'SolveTime(ms)':>14} {'CongRate':>10}"
    print(header)
    print("-" * 80)

    for name in method_names:
        recs = [r for r in records if r["method"] == name]
        if not recs:
            continue

        def ci(vals):
            arr = np.array(vals)
            mean = arr.mean()
            sem = arr.std() / np.sqrt(len(arr)) * 1.96
            return mean, sem

        mw, mw_ci = ci([r["mean_wait_min"] for r in recs])
        xw, xw_ci = ci([r["max_wait_min"] for r in recs])
        pw, pw_ci = ci([r["p95_wait_min"] for r in recs])
        st, st_ci = ci([r["solve_time_mean_ms"] for r in recs])
        cr, _     = ci([r["congestion_rate"] for r in recs])

        print(
            f"{name:<22} "
            f"{mw:7.2f}±{mw_ci:.2f}   "
            f"{xw:7.2f}±{xw_ci:.2f}  "
            f"{pw:7.2f}±{pw_ci:.2f}  "
            f"{st:8.1f}±{st_ci:.1f}   "
            f"{cr:8.3f}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corridor EV charging experiments")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--gamma_c", type=float, default=2.0,
                        help="Gamma_c robustness budget for charging uncertainty")
    parser.add_argument("--gamma_arrival", type=float, default=0.0,
                        help="Gamma_arrival for arrival-time uncertainty buffer (D5)")
    parser.add_argument("--output_dir", type=str, default="corridor/results")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of methods to run (default: all)")
    args = parser.parse_args()

    run_all(
        n_seeds=args.n_seeds,
        gamma_c=args.gamma_c,
        gamma_arrival=args.gamma_arrival,
        output_dir=args.output_dir,
        method_filter=args.methods,
    )