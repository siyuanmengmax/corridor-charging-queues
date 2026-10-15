"""
Phase 2 SUMO closed-loop experiment runner.

Reuses the same optimizers as Phase 1 (run_experiments.py) but replaces
the standalone CorridorSimulation with the SumoCorridorEnv TraCI environment.

Methods compared (same as Phase 1 main comparison):
  - Greedy
  - R-MPC-StaticCap (Gamma_c=2)
  - QA-R-MPC       (Gamma_c=2)

Run:
  python -m corridor.experiments.run_sumo_experiments [--n_seeds N]

Note: each seed launches and closes a separate SUMO instance. SUMO runs headless
(--gui flag adds sumo-gui for debugging). Expect ~30-120 s per seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np

from ..env.sumo_corridor import SumoCorridorEnv, STATION_META
from ..models.optimizer import (
    GreedyOptimizer,
    StaticCapOptimizer,
    QueueAwareOptimizer,
    DeterministicOptimizer,
    SAAOptimizer,
    EVRequest,
    ChargingStation,
    OptimizationResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

MAX_DETOUR_KM = 80.0     # vehicles only consider stations within 80 km ahead
SPEED_KMH     = 100.0   # corridor vehicle speed


def _directed_travel_times(
    requests: List[EVRequest],
    stations: List[ChargingStation],
) -> dict:
    """
    Pre-compute directed travel times for the corridor:
      - Station BEHIND vehicle or > MAX_DETOUR_KM ahead → 1e6 (effectively excluded)
      - Otherwise: directed distance / SPEED_KMH * 60 (minutes)
    """
    tt = {}
    for ev in requests:
        for st in stations:
            delta = st.position_km - ev.position_km
            if delta < -1.0 or delta > MAX_DETOUR_KM:
                tt[(ev.id, st.id)] = 1e6
            else:
                tt[(ev.id, st.id)] = max(0.0, delta) / SPEED_KMH * 60.0
    return tt


def run_sumo_episode(
    method_name: str,
    optimizer,
    seed: int,
    gui: bool = False,
    scenario: str = "heavy",
    **env_kwargs,
) -> dict:
    """Run one SUMO episode and return metrics."""
    env = SumoCorridorEnv(seed=seed, gui=gui, scenario=scenario, **env_kwargs)
    t0 = time.perf_counter()
    solve_times_ms: List[float] = []

    try:
        env.start()
        step = 0
        while not env.done:
            stations = env.get_station_snapshot()
            requests = env.get_charging_requests()

            if requests:
                # Directed travel times: only downstream stations within MAX_DETOUR_KM
                requests = requests + env.get_upcoming_requests(getattr(optimizer, "lookahead_min", 0.0))
                travel_times = env.travel_times(requests)
                t_opt = time.perf_counter()
                result = optimizer.solve(requests, stations, travel_times=travel_times)
                solve_times_ms.append((time.perf_counter() - t_opt) * 1000.0)
                env.apply_assignments(result)

            env.step()
            step += 1
            if step % 12 == 0:  # log every hour of sim time
                logger.debug(
                    f"  {method_name} seed={seed} step={step} "
                    f"sim_time={env.sim_time/60:.0f}min "
                    f"n_served={len(env.completed_waits)}"
                )
        # Serve every vehicle that requested during the episode (no censored waits)
        env.drain()
    finally:
        env.close()

    m = env.episode_metrics()
    wall_s = time.perf_counter() - t0
    avg_ms = float(np.mean(solve_times_ms)) if solve_times_ms else 0.0
    m.update({
        "method": method_name,
        "seed": seed,
        "wall_time_s": wall_s,
        "solve_time_mean_ms": avg_ms,
        "solve_time_p95_ms": float(np.percentile(solve_times_ms, 95)) if solve_times_ms else 0.0,
        "phase": "phase2_sumo",
        "scenario": scenario,
    })
    return m


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all(
    n_seeds: int = 5,
    gamma_c: float = 2.0,
    output_dir: str = "corridor/results",
    gui: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "Greedy":           GreedyOptimizer(speed_kmh=100.0),
        "Det-SO":           DeterministicOptimizer(speed_kmh=100.0, delta_t=5.0),
        "SAA-20":           SAAOptimizer(n_scenarios=20, speed_kmh=100.0, delta_t=5.0),
        "R-MPC-StaticCap":  StaticCapOptimizer(gamma_c=gamma_c, delta_t=5.0, speed_kmh=100.0),
        "QA-R-MPC":         QueueAwareOptimizer(gamma_c=gamma_c, delta_t=5.0, speed_kmh=100.0),
    }

    all_records = []
    for method_name, optimizer in methods.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method_name} ({n_seeds} seeds, SUMO closed-loop)")
        for seed in range(n_seeds):
            logger.info(f"  seed={seed}...")
            try:
                r = run_sumo_episode(method_name, optimizer, seed, gui=gui)
                all_records.append(r)
                logger.info(
                    f"  seed={seed:2d}  served={r['n_served']:3d}  "
                    f"mean_wait={r['mean_wait_min']:.2f}  "
                    f"max_wait={r['max_wait_min']:.2f}  "
                    f"solve={r['solve_time_mean_ms']:.1f}ms  "
                    f"wall={r['wall_time_s']:.0f}s"
                )
            except Exception as e:
                logger.error(f"  seed={seed} FAILED: {e}", exc_info=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / f"sumo_results_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"\nSaved {len(all_records)} records to {out}")
    _print_summary(all_records, methods.keys())


def _print_summary(records: List[dict], method_names) -> None:
    print("\n" + "=" * 75)
    print("PHASE 2 SUMO RESULTS (mean ± 95% CI)")
    print("=" * 75)
    print(f"{'Method':<22} {'MeanWait':>10} {'MaxWait':>10} {'P95Wait':>10} {'Solve(ms)':>12} {'Served':>8}")
    print("-" * 75)
    for name in method_names:
        recs = [r for r in records if r["method"] == name]
        if not recs:
            continue
        def ci(key):
            arr = np.array([r[key] for r in recs])
            return arr.mean(), arr.std() / np.sqrt(len(arr)) * 1.96
        mw, mwci = ci("mean_wait_min")
        xw, xwci = ci("max_wait_min")
        pw, pwci = ci("p95_wait_min")
        st, stci = ci("solve_time_mean_ms")
        sv = np.mean([r["n_served"] for r in recs])
        print(
            f"{name:<22} {mw:6.2f}±{mwci:.2f}  {xw:6.2f}±{xwci:.2f}  "
            f"{pw:6.2f}±{pwci:.2f}  {st:7.1f}±{stci:.1f}  {sv:6.0f}"
        )
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 SUMO closed-loop experiments")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--gamma_c", type=float, default=2.0)
    parser.add_argument("--output_dir", default="corridor/results")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui for debugging")
    args = parser.parse_args()

    run_all(
        n_seeds=args.n_seeds,
        gamma_c=args.gamma_c,
        output_dir=args.output_dir,
        gui=args.gui,
    )
