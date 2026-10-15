"""
Queue-aware Γ-robust MILP optimizer for EV charging scheduling.

Implements the revised Paper A model (post-TRC-rejection overhaul):

  min  Σ_{i,s} (α·t_{i,s} + β·d̄_{i,s})·x_{i,s}   ← station-dependent d̄_{i,s}  (D4)
     + γ·Σ_s W_s                                    ← M/M/c queueing cost        (D1)
     + M·Σ_s σ_s                                    ← soft-feasibility slack

  s.t.
    Σ_s x_{i,s} = 1                            ∀i                   (assignment)
    W_s ≥ m_{s,k}·(Σ_i x_{i,s}) + b_{s,k}    ∀s, ∀k  (PWL epigraph, §3.1)
    Σ_i d̄_{i,s}·x_{i,s} + Γ_c·z_s + Σ_i p_{i,s} ≤ C_s + σ_s   ∀s  (Γ-robust cap)
    p_{i,s} + z_s ≥ d̂_{i,s}·x_{i,s}          ∀i,s
    x_{i,s} ∈ {0,1},  W_s, σ_s, z_s, p_{i,s} ≥ 0

Key changes vs. original Paper A (network/grid/robust_optimizer.py):
  - d̄_{i,s} is now station-dependent (D4): faster stations → shorter charge time.
  - W_s replaces σ_s as the *primary* congestion signal; σ_s is demoted to a
    feasibility backstop (not the main congestion model).
  - PWL tangent-line constraints capture M/M/c expected waiting cost.
  - The old "Rather than explicitly modeling stochastic queues (e.g., M/M/c)..."
    rationale is inverted: we now embrace M/M/c via decoupled precomputation.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .queue_model import StationQueueModel, PWLEpigraph

# Gurobi threads per solve; 0 = all cores. Parallel experiment drivers set 1.
GRB_THREADS = int(os.environ.get("CORRIDOR_GRB_THREADS", "0"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EVRequest:
    """An EV charging request at control step k."""
    id: str
    position_km: float          # Position along corridor (km)
    soc: float                  # Current SOC in [0, 1]
    battery_kwh: float          # Battery capacity (kWh)
    # arrival_time_uncertainty: fraction of travel time that is uncertain (D5)
    arrival_uncertainty: float = 0.10  # ±10% of travel time, default
    # Minutes until the vehicle is expected to request charging (0 = requesting now).
    # Vehicles with release_min > 0 are predictions used for look-ahead only.
    release_min: float = 0.0


@dataclass
class ChargingStation:
    """A charging station on the corridor."""
    id: str
    position_km: float          # Position along corridor (km)
    n_chargers: int             # c = N_s
    power_kw: float             # Rated power per charger (kW)
    # Current occupancy state (for dynamic capacity C_s(k))
    remaining_times: List[float] = field(default_factory=list)  # minutes
    # Vehicles already committed to this station but not yet charging:
    # (minutes until arrival — 0 if already queued, nominal charging minutes)
    committed: List[Tuple[float, float]] = field(default_factory=list)

    def available_capacity(self, delta_t: float) -> float:
        """C_s(k) = Σ_m max(0, ΔT - τ_{s,m}^rem)"""
        idle = self.n_chargers - len(self.remaining_times)
        cap = idle * delta_t
        for rem in self.remaining_times:
            cap += max(0.0, delta_t - rem)
        return cap


@dataclass
class OptimizationResult:
    """Result from one MILP solve."""
    status: str                         # 'OPTIMAL', 'TIME_LIMIT', etc.
    assignments: Dict[str, str]         # ev_id → station_id
    objective: float
    slack_values: Dict[str, float]      # σ_s: capacity overflow
    queue_costs: Dict[str, float]       # W_s: expected queue wait per station
    solve_time_ms: float
    has_congestion: bool                # any σ_s > 0.01


# ---------------------------------------------------------------------------
# Queue-aware Γ-robust optimizer (new method)
# ---------------------------------------------------------------------------

class QueueAwareOptimizer:
    """
    Queue-aware Γ-robust MILP for corridor EV charging scheduling.

    Parameters
    ----------
    gamma_c       : float  Robustness budget Γ_c for charging-duration uncertainty.
    gamma_arrival : float  Conservative buffer Γ_a for arrival-time uncertainty (D5).
    alpha         : float  Weight for travel time.
    beta          : float  Weight for charging time (now meaningful: d̄_{i,s} varies by s).
    gamma_w       : float  Weight for M/M/c queueing cost W_s.
    M             : float  Big-M penalty for capacity overflow σ_s.
    delta_t       : float  Control interval (minutes).
    speed_kmh     : float  Vehicle speed along corridor (km/h).
    time_limit_s  : float  Gurobi time limit per solve (seconds).
    mip_gap       : float  MIP optimality gap.
    n_pwl_breakpoints : int  K tangent lines for PWL epigraph.
    target_soc    : float  Charging target SOC.
    """

    def __init__(
        self,
        gamma_c: float = 2.0,
        gamma_arrival: float = 0.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma_w: float = 2.0,
        M: float = 1e4,
        delta_t: float = 5.0,
        speed_kmh: float = 100.0,
        time_limit_s: float = 60.0,
        mip_gap: float = 0.01,
        n_pwl_breakpoints: int = 6,
        target_soc: float = 0.80,
        station_dep_charging: bool = True,   # D4 ablation flag: False → station-independent d̄_i
    ):
        self.gamma_c = gamma_c
        self.gamma_arrival = gamma_arrival
        self.station_dep_charging = station_dep_charging
        self.alpha = alpha
        self.beta = beta
        self.gamma_w = gamma_w
        self.M = M
        self.delta_t = delta_t
        self.speed_kmh = speed_kmh
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap
        self.n_pwl_breakpoints = n_pwl_breakpoints
        self.target_soc = target_soc

    # -----------------------------------------------------------------------
    # Parameter helpers
    # -----------------------------------------------------------------------

    def _travel_time(self, ev: EVRequest, st: ChargingStation) -> float:
        """Travel time EV → station (minutes)."""
        dist_km = abs(ev.position_km - st.position_km)
        return dist_km / self.speed_kmh * 60.0

    def _nominal_charge(self, ev: EVRequest, st: ChargingStation) -> float:
        """Nominal charge duration (minutes). D4 flag controls station dependence."""
        charge_needed = max(0.0, self.target_soc - ev.soc)
        power = st.power_kw if self.station_dep_charging else 120.0  # 120 kW = fleet mean
        return charge_needed * ev.battery_kwh / power * 60.0

    def _charge_uncertainty(self, nominal: float, eps: float = 0.20) -> float:
        """d̂_{i,s} = ε · d̄_{i,s}."""
        return eps * nominal

    def _build_queue_models(
        self, stations: List[ChargingStation], ev_requests: List[EVRequest]
    ) -> Dict[str, Tuple[StationQueueModel, PWLEpigraph]]:
        """Build per-station batch-arrival queue models and their PWL epigraphs."""
        result = {}
        for st in stations:
            mean_batt = np.mean([ev.battery_kwh for ev in ev_requests]) if ev_requests else 60.0
            qm = StationQueueModel(
                station_id=st.id,
                n_chargers=st.n_chargers,
                power_kw=st.power_kw,
                battery_kwh=mean_batt,
                target_soc=self.target_soc,
                delta_t=self.delta_t,
                n_breakpoints=self.n_pwl_breakpoints,
                gamma_arrival=self.gamma_arrival,
            )
            # Use current free chargers from station state for dynamic PWL
            c_free = st.n_chargers - len(st.remaining_times)
            pwl = qm.build_pwl(c_free=c_free, gamma_arrival=self.gamma_arrival)
            result[st.id] = (qm, pwl)
        return result

    # -----------------------------------------------------------------------
    # Main solve
    # -----------------------------------------------------------------------

    def solve(
        self,
        ev_requests: List[EVRequest],
        stations: List[ChargingStation],
        travel_times: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> OptimizationResult:
        t0 = time.perf_counter()

        if not ev_requests:
            return OptimizationResult(
                status="OPTIMAL", assignments={}, objective=0.0,
                slack_values={s.id: 0.0 for s in stations},
                queue_costs={s.id: 0.0 for s in stations},
                solve_time_ms=0.0, has_congestion=False,
            )

        ev_ids = [ev.id for ev in ev_requests]
        st_ids = [st.id for st in stations]
        ev_map = {ev.id: ev for ev in ev_requests}
        st_map = {st.id: st for st in stations}

        # --- Precompute parameters ---
        t_is = {}
        d_bar = {}
        d_hat = {}
        for ev in ev_requests:
            for st in stations:
                key = (ev.id, st.id)
                t_is[key] = (
                    travel_times[key]
                    if travel_times and key in travel_times
                    else self._travel_time(ev, st)
                )
                d_bar[key] = self._nominal_charge(ev, st)
                d_hat[key] = self._charge_uncertainty(d_bar[key])

        C = {st.id: st.available_capacity(self.delta_t) for st in stations}

        # --- Build per-station PWL queue models ---
        queue_models = self._build_queue_models(stations, ev_requests)

        # --- Build Gurobi model ---
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", GRB_THREADS)
        env.setParam("TimeLimit", self.time_limit_s)
        env.setParam("MIPGap", self.mip_gap)
        env.start()
        m = gp.Model(env=env)

        # Decision variables
        x = m.addVars(ev_ids, st_ids, vtype=GRB.BINARY, name="x")
        W = m.addVars(st_ids, lb=0.0, name="W")          # queueing cost
        sigma = m.addVars(st_ids, lb=0.0, name="sigma")  # capacity overflow
        z = m.addVars(st_ids, lb=0.0, name="z")          # robust dual
        p = m.addVars(ev_ids, st_ids, lb=0.0, name="p")  # robust dual

        # Objective
        obj = (
            gp.quicksum(
                (self.alpha * t_is[(i, s)] + self.beta * d_bar[(i, s)]) * x[i, s]
                for i in ev_ids for s in st_ids
            )
            + self.gamma_w * gp.quicksum(W[s] for s in st_ids)
            + self.M * gp.quicksum(sigma[s] for s in st_ids)
        )
        m.setObjective(obj, GRB.MINIMIZE)

        # Assignment: each EV to exactly one station
        for i in ev_ids:
            m.addConstr(gp.quicksum(x[i, s] for s in st_ids) == 1, name=f"assign_{i}")

        # PWL epigraph constraints: W_s >= m_k * n_s + b_k  for each tangent line k
        for s in st_ids:
            _, pwl = queue_models[s]
            n_s = gp.quicksum(x[i, s] for i in ev_ids)   # integer-valued LinExpr
            for k, (slope, intercept) in enumerate(pwl.tangents):
                m.addConstr(W[s] >= slope * n_s + intercept, name=f"pwl_{s}_{k}")

        # Γ_c-robust capacity constraints (station-dependent d̄_{i,s})
        for s in st_ids:
            # Nominal load + robust buffer <= C_s + σ_s
            m.addConstr(
                gp.quicksum(d_bar[(i, s)] * x[i, s] for i in ev_ids)
                + self.gamma_c * z[s]
                + gp.quicksum(p[i, s] for i in ev_ids)
                <= C[s] + sigma[s],
                name=f"robust_cap_{s}",
            )
            # Dual coupling: p_{i,s} + z_s >= d̂_{i,s} * x_{i,s}
            for i in ev_ids:
                m.addConstr(
                    p[i, s] + z[s] >= d_hat[(i, s)] * x[i, s],
                    name=f"dual_{i}_{s}",
                )

        m.optimize()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract results
        status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
                      GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD"}
        status = status_map.get(m.Status, f"STATUS_{m.Status}")

        if m.SolCount == 0:
            logger.warning(f"No feasible solution found: {status}")
            return OptimizationResult(
                status=status, assignments={}, objective=float("inf"),
                slack_values={s: 0.0 for s in st_ids},
                queue_costs={s: 0.0 for s in st_ids},
                solve_time_ms=solve_ms, has_congestion=True,
            )

        assignments = {}
        for i in ev_ids:
            for s in st_ids:
                if x[i, s].X > 0.5:
                    assignments[i] = s
                    break

        slack_vals = {s: sigma[s].X for s in st_ids}
        queue_costs = {s: W[s].X for s in st_ids}
        has_congestion = any(v > 0.01 for v in slack_vals.values())
        obj_val = m.ObjVal

        m.dispose()
        env.dispose()

        return OptimizationResult(
            status=status,
            assignments=assignments,
            objective=obj_val,
            slack_values=slack_vals,
            queue_costs=queue_costs,
            solve_time_ms=solve_ms,
            has_congestion=has_congestion,
        )


# ---------------------------------------------------------------------------
# Static-capacity optimizer (ablation: original Paper A method = R-MPC-StaticCap)
# ---------------------------------------------------------------------------

class StaticCapOptimizer:
    """
    Original Γ-robust MILP with static temporal-capacity proxy and no queue model.

    This is the pre-overhaul method (equivalent to network/grid/robust_optimizer.py)
    used as the R-MPC-StaticCap ablation baseline in the revised experiments.

    Key differences from QueueAwareOptimizer:
      - No W_s variable or PWL constraints.
      - d̄_i is EV-only (not station-dependent; β*d_bar constant → set β=0).
      - σ_s is the primary (and sole) congestion signal.
    """

    def __init__(
        self,
        gamma_c: float = 2.0,
        alpha: float = 1.0,
        M: float = 1e4,
        delta_t: float = 5.0,
        speed_kmh: float = 100.0,
        time_limit_s: float = 60.0,
        mip_gap: float = 0.01,
        target_soc: float = 0.80,
    ):
        self.gamma_c = gamma_c
        self.alpha = alpha
        self.M = M
        self.delta_t = delta_t
        self.speed_kmh = speed_kmh
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap
        self.target_soc = target_soc

    def _travel_time(self, ev: EVRequest, st: ChargingStation) -> float:
        dist_km = abs(ev.position_km - st.position_km)
        return dist_km / self.speed_kmh * 60.0

    def _nominal_charge(self, ev: EVRequest) -> float:
        """EV-only (not station-dependent): use mean charger power."""
        # Representative mean power: not used in objective (constant per EV)
        # d̄_i = (target - soc) * battery_kwh / mean_power * 60
        charge_needed = max(0.0, self.target_soc - ev.soc)
        mean_power_kw = 120.0  # assumed mean
        return charge_needed * ev.battery_kwh / mean_power_kw * 60.0

    def solve(
        self,
        ev_requests: List[EVRequest],
        stations: List[ChargingStation],
        travel_times: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> OptimizationResult:
        t0 = time.perf_counter()

        if not ev_requests:
            return OptimizationResult(
                status="OPTIMAL", assignments={}, objective=0.0,
                slack_values={s.id: 0.0 for s in stations},
                queue_costs={s.id: 0.0 for s in stations},
                solve_time_ms=0.0, has_congestion=False,
            )

        ev_ids = [ev.id for ev in ev_requests]
        st_ids = [st.id for st in stations]
        ev_map = {ev.id: ev for ev in ev_requests}

        t_is = {}
        d_bar = {}
        d_hat = {}
        for ev in ev_requests:
            d_i = self._nominal_charge(ev)
            d_hat_i = 0.20 * d_i
            for st in stations:
                key = (ev.id, st.id)
                t_is[key] = (
                    travel_times[key]
                    if travel_times and key in travel_times
                    else self._travel_time(ev, st)
                )
                d_bar[key] = d_i
                d_hat[key] = d_hat_i

        C = {st.id: st.available_capacity(self.delta_t) for st in stations}

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", GRB_THREADS)
        env.setParam("TimeLimit", self.time_limit_s)
        env.setParam("MIPGap", self.mip_gap)
        env.start()
        m = gp.Model(env=env)

        x = m.addVars(ev_ids, st_ids, vtype=GRB.BINARY, name="x")
        sigma = m.addVars(st_ids, lb=0.0, name="sigma")
        z = m.addVars(st_ids, lb=0.0, name="z")
        p = m.addVars(ev_ids, st_ids, lb=0.0, name="p")

        obj = (
            gp.quicksum(self.alpha * t_is[(i, s)] * x[i, s]
                        for i in ev_ids for s in st_ids)
            + self.M * gp.quicksum(sigma[s] for s in st_ids)
        )
        m.setObjective(obj, GRB.MINIMIZE)

        for i in ev_ids:
            m.addConstr(gp.quicksum(x[i, s] for s in st_ids) == 1, name=f"assign_{i}")

        for s in st_ids:
            m.addConstr(
                gp.quicksum(d_bar[(i, s)] * x[i, s] for i in ev_ids)
                + self.gamma_c * z[s]
                + gp.quicksum(p[i, s] for i in ev_ids)
                <= C[s] + sigma[s],
                name=f"robust_cap_{s}",
            )
            for i in ev_ids:
                m.addConstr(
                    p[i, s] + z[s] >= d_hat[(i, s)] * x[i, s],
                    name=f"dual_{i}_{s}",
                )

        m.optimize()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
                      GRB.INFEASIBLE: "INFEASIBLE"}
        status = status_map.get(m.Status, f"STATUS_{m.Status}")

        if m.SolCount == 0:
            return OptimizationResult(
                status=status, assignments={}, objective=float("inf"),
                slack_values={s: 0.0 for s in st_ids},
                queue_costs={s: 0.0 for s in st_ids},
                solve_time_ms=solve_ms, has_congestion=True,
            )

        assignments = {}
        for i in ev_ids:
            for s in st_ids:
                if x[i, s].X > 0.5:
                    assignments[i] = s
                    break

        slack_vals = {s: sigma[s].X for s in st_ids}
        has_congestion = any(v > 0.01 for v in slack_vals.values())
        obj_val = m.ObjVal

        m.dispose()
        env.dispose()

        return OptimizationResult(
            status=status, assignments=assignments, objective=obj_val,
            slack_values=slack_vals,
            queue_costs={s: 0.0 for s in st_ids},
            solve_time_ms=solve_ms, has_congestion=has_congestion,
        )


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------

class GreedyOptimizer:
    """Assign each EV to the nearest station (ignores capacity and queues)."""

    def __init__(self, speed_kmh: float = 100.0):
        self.speed_kmh = speed_kmh

    def solve(
        self,
        ev_requests: List[EVRequest],
        stations: List[ChargingStation],
        travel_times: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        assignments = {}
        for ev in ev_requests:
            best_st = min(
                stations,
                key=lambda st: (
                    travel_times[(ev.id, st.id)]
                    if travel_times and (ev.id, st.id) in travel_times
                    else abs(ev.position_km - st.position_km)
                ),
            )
            assignments[ev.id] = best_st.id
        return OptimizationResult(
            status="GREEDY", assignments=assignments, objective=0.0,
            slack_values={s.id: 0.0 for s in stations},
            queue_costs={s.id: 0.0 for s in stations},
            solve_time_ms=(time.perf_counter() - t0) * 1000.0,
            has_congestion=False,
        )


# ---------------------------------------------------------------------------
# Det-SO baseline (Γ_c = 0, no queue)
# ---------------------------------------------------------------------------

class DeterministicOptimizer(StaticCapOptimizer):
    """Det-SO: Γ_c = 0 deterministic, no queue model."""

    def __init__(self, **kwargs):
        kwargs["gamma_c"] = 0.0
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# SAA baseline
# ---------------------------------------------------------------------------

class SAAOptimizer:
    """
    Sample Average Approximation with frequency-based rounding.
    Uses DeterministicOptimizer internally for each scenario.
    """

    def __init__(
        self,
        n_scenarios: int = 50,
        speed_kmh: float = 100.0,
        delta_t: float = 5.0,
        time_limit_s: float = 10.0,
        mip_gap: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(seed)
        self._det = DeterministicOptimizer(
            speed_kmh=speed_kmh,
            delta_t=delta_t,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
        )

    def solve(
        self,
        ev_requests: List[EVRequest],
        stations: List[ChargingStation],
        travel_times: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        if not ev_requests:
            return OptimizationResult(
                status="OPTIMAL", assignments={}, objective=0.0,
                slack_values={s.id: 0.0 for s in stations},
                queue_costs={s.id: 0.0 for s in stations},
                solve_time_ms=0.0, has_congestion=False,
            )

        ev_ids = [ev.id for ev in ev_requests]
        st_ids = [st.id for st in stations]
        freq: Dict[str, Dict[str, float]] = {i: {s: 0.0 for s in st_ids} for i in ev_ids}
        n_solved = 0

        for _ in range(self.n_scenarios):
            # Perturb SOC by ±0.04 to generate different nominal charging demands per scenario.
            # Δsoc ≈ ±0.04 → Δdemand ≈ ±0.04 * 70kWh / 120kW * 60 ≈ ±1.4 min ≈ ±20% of mean.
            scenario = []
            for ev in ev_requests:
                soc_perturb = float(self.rng.uniform(-0.04, 0.04))
                scenario.append(EVRequest(
                    id=ev.id,
                    position_km=ev.position_km,
                    soc=float(np.clip(ev.soc + soc_perturb, 0.05, 0.79)),
                    battery_kwh=ev.battery_kwh,
                ))
            r = self._det.solve(scenario, stations, travel_times)
            if r.assignments:
                for eid, sid in r.assignments.items():
                    freq[eid][sid] += 1.0
                n_solved += 1

        assignments = {}
        for eid in ev_ids:
            best = max(freq[eid], key=lambda s: freq[eid][s])
            assignments[eid] = best

        return OptimizationResult(
            status="OPTIMAL" if n_solved > 0 else "FALLBACK",
            assignments=assignments, objective=float("nan"),
            slack_values={s: 0.0 for s in st_ids},
            queue_costs={s: 0.0 for s in st_ids},
            solve_time_ms=(time.perf_counter() - t0) * 1000.0,
            has_congestion=False,
        )
