"""
Backlog-aware assignment (prototype).

The queue term of QueueAwareOptimizer sees only the chargers that are free now
and the vehicles assigned in the current step. Here the station state also
includes the vehicles already queued or en route, and the predicted wait of a
new vehicle is obtained by replaying the committed vehicles through the
station's chargers (deterministic FCFS with nominal charging times).

Objective (minutes, per assigned vehicle): travel + beta * charging +
gamma * predicted wait, where the predicted wait has two parts:
  - w_is : wait caused by vehicles already committed to station s (linear in x)
  - h_s(n_s): extra wait vehicles of the same batch impose on each other,
              convex in the number n_s assigned to s (exact epigraph at integers)

There is no static capacity constraint and no slack penalty.
"""

from __future__ import annotations

import heapq
import os
import time
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .optimizer import ChargingStation, EVRequest, OptimizationResult

# Gurobi threads per solve; 0 = all cores. Parallel experiment drivers set 1.
GRB_THREADS = int(os.environ.get("CORRIDOR_GRB_THREADS", "0"))

UNREACHABLE = 1e5


def charger_free_times(st: ChargingStation, horizon_min: float) -> List[float]:
    """
    Times (minutes from now) at which each charger becomes free after serving
    the vehicles in service and every committed vehicle arriving within
    horizon_min, FCFS.
    """
    free = sorted(st.remaining_times)[: st.n_chargers]
    free += [0.0] * (st.n_chargers - len(free))
    heapq.heapify(free)
    for eta, service in sorted(st.committed):
        if eta > horizon_min:
            break
        start = max(eta, heapq.heappop(free))
        heapq.heappush(free, start + service)
    return sorted(free)


def nominal_charge(ev: EVRequest, st: ChargingStation, travel_min: float,
                   target_soc: float, speed_kmh: float, consumption: float = 0.20) -> float:
    dist_km = travel_min / 60.0 * speed_kmh
    soc_arr = max(0.0, ev.soc - consumption * dist_km / ev.battery_kwh)
    return max(0.0, target_soc - soc_arr) * ev.battery_kwh / st.power_kw * 60.0


class _BacklogBase:
    def __init__(self, beta: float = 1.0, gamma_w: float = 1.0,
                 speed_kmh: float = 100.0, target_soc: float = 0.80):
        self.beta = beta
        self.gamma_w = gamma_w
        self.speed_kmh = speed_kmh
        self.target_soc = target_soc

    def _travel(self, ev, st, travel_times) -> float:
        if travel_times and (ev.id, st.id) in travel_times:
            return travel_times[(ev.id, st.id)]
        delta = st.position_km - ev.position_km
        return delta / self.speed_kmh * 60.0 if delta >= -1.0 else 1e6

    @staticmethod
    def _empty(stations, status="OPTIMAL") -> OptimizationResult:
        return OptimizationResult(
            status=status, assignments={}, objective=0.0,
            slack_values={s.id: 0.0 for s in stations},
            queue_costs={s.id: 0.0 for s in stations},
            solve_time_ms=0.0, has_congestion=False,
        )


class BacklogGreedyOptimizer(_BacklogBase):
    """Each vehicle in turn joins the station with the earliest predicted completion.

    With en_route=False the rule sees only the vehicles physically at each station
    (in service or queued), as a live-occupancy display would, and not the vehicles
    driving to it, including those it assigned earlier in the same step.
    With assumed_power_kw set, the rule does not know the power of each station: charging
    times, including those of the vehicles queued there, are computed at that power.
    With remaining_known=False the rule sees only which chargers are busy (port status, as
    shared under 23 CFR 680.116(c)): each busy charger is taken to be half-way through, and
    each queued vehicle to need, a reference session of REF_ENERGY_KWH at the station power.
    With queue_known=False it does not see the vehicles queued at the station either.
    """

    REF_ENERGY_KWH = 0.6 * 70.0   # 20-80% SOC of a 70 kWh battery

    def __init__(self, en_route: bool = True, assumed_power_kw: Optional[float] = None,
                 remaining_known: bool = True, queue_known: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.en_route = en_route
        self.assumed_power_kw = assumed_power_kw
        self.remaining_known = remaining_known
        self.queue_known = queue_known
        assert not (en_route and not (remaining_known and queue_known)), \
            "port-status variants are occupancy-only rules"

    def solve(self, ev_requests: List[EVRequest], stations: List[ChargingStation],
              travel_times: Optional[Dict[Tuple[str, str], float]] = None) -> OptimizationResult:
        t0 = time.perf_counter()
        if not ev_requests:
            return self._empty(stations)
        added: Dict[str, List[Tuple[float, float]]] = {st.id: [] for st in stations}
        assignments = {}
        for ev in sorted(ev_requests, key=lambda e: e.soc):
            best, best_cost, best_entry = None, float("inf"), None
            for st in stations:
                tau = self._travel(ev, st, travel_times)
                if tau >= UNREACHABLE:
                    continue
                power = self.assumed_power_kw or st.power_kw
                known = [(eta, svc * st.power_kw / power) for eta, svc in st.committed]
                committed = (known + added[st.id] if self.en_route
                             else [c for c in known if c[0] <= 1e-6])
                remaining = st.remaining_times
                if not self.remaining_known:
                    ref = self.REF_ENERGY_KWH / power * 60.0
                    remaining = [0.5 * ref] * len(st.remaining_times)
                    committed = [(eta, ref) for eta, _ in committed]
                if not self.queue_known:
                    committed = []
                view = ChargingStation(st.id, st.position_km, st.n_chargers, power,
                                       remaining, committed)
                d = nominal_charge(ev, view, tau, self.target_soc, self.speed_kmh)
                wait = max(0.0, charger_free_times(view, tau)[0] - tau)
                cost = tau + self.beta * d + self.gamma_w * wait
                if cost < best_cost:
                    best, best_cost, best_entry = st.id, cost, (tau, d)
            if best is not None:
                assignments[ev.id] = best
                added[best].append(best_entry)
        res = self._empty(stations, status="GREEDY")
        res.assignments = assignments
        res.solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return res


class BacklogAwareOptimizer(_BacklogBase):
    """MILP: linear backlog wait per vehicle plus a convex within-batch term."""

    def __init__(self, time_limit_s: float = 60.0, mip_gap: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap

    def solve(self, ev_requests: List[EVRequest], stations: List[ChargingStation],
              travel_times: Optional[Dict[Tuple[str, str], float]] = None) -> OptimizationResult:
        t0 = time.perf_counter()
        if not ev_requests:
            return self._empty(stations)

        cost: Dict[Tuple[str, str], float] = {}
        d_bar: Dict[Tuple[str, str], float] = {}
        tau_is: Dict[Tuple[str, str], float] = {}
        for ev in ev_requests:
            for st in stations:
                tau = self._travel(ev, st, travel_times)
                if tau >= UNREACHABLE:
                    continue
                d = nominal_charge(ev, st, tau, self.target_soc, self.speed_kmh)
                wait = max(0.0, charger_free_times(st, tau)[0] - tau)
                key = (ev.id, st.id)
                cost[key] = tau + self.beta * d + self.gamma_w * wait
                d_bar[key], tau_is[key] = d, tau
        ev_ids = [ev.id for ev in ev_requests if any((ev.id, st.id) in cost for st in stations)]
        if not ev_ids:
            return self._empty(stations)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", GRB_THREADS)
        env.setParam("TimeLimit", self.time_limit_s)
        env.setParam("MIPGap", self.mip_gap)
        env.start()
        m = gp.Model(env=env)
        x = m.addVars(list(cost.keys()), vtype=GRB.BINARY, name="x")
        W = m.addVars([st.id for st in stations], lb=0.0, name="W")
        m.setObjective(
            gp.quicksum(cost[k] * x[k] for k in cost) + self.gamma_w * W.sum(), GRB.MINIMIZE
        )
        for i in ev_ids:
            m.addConstr(gp.quicksum(x[i, st.id] for st in stations if (i, st.id) in cost) == 1)

        # Within-batch interaction h_s(n): the k-th vehicle added to station s (k = 0, 1, ...)
        # waits an extra max(0, k - (f_s - 1)) * d_s / N_s, with f_s the chargers free on arrival.
        for st in stations:
            cand = [i for i in ev_ids if (i, st.id) in cost]
            if len(cand) < 2:
                continue
            tau_mean = sum(tau_is[i, st.id] for i in cand) / len(cand)
            d_mean = sum(d_bar[i, st.id] for i in cand) / len(cand)
            f_s = sum(1 for t in charger_free_times(st, tau_mean) if t <= tau_mean)
            h = [0.0]
            for k in range(len(cand)):
                h.append(h[-1] + max(0, k - max(f_s - 1, 0)) * d_mean / st.n_chargers)
            n_s = gp.quicksum(x[i, st.id] for i in cand)
            for k in range(len(cand)):
                slope = h[k + 1] - h[k]
                if slope > 0:
                    m.addConstr(W[st.id] >= h[k] + slope * (n_s - k))

        m.optimize()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        res = self._empty(stations, status="OPTIMAL" if m.Status == GRB.OPTIMAL else f"STATUS_{m.Status}")
        res.solve_time_ms = solve_ms
        if m.SolCount > 0:
            res.assignments = {i: s for (i, s) in cost if x[i, s].X > 0.5}
            res.queue_costs = {st.id: W[st.id].X for st in stations}
            res.objective = m.ObjVal
        m.dispose()
        env.dispose()
        return res


class FluidMPCOptimizer(_BacklogBase):
    """
    Rolling-horizon MILP with a time-expanded fluid queue per station.

    Time is cut into slots of delta_t minutes. B[s,t] is the unfinished charging
    work (charger-minutes) at station s at the start of slot t:

        B[s,0]   = remaining work of vehicles in service
        B[s,t+1] >= B[s,t] + work arriving in slot t - N_s * delta_t

    Arriving work comes from vehicles already queued or en route (fixed) and
    from the vehicles being assigned (decision). The number of vehicles waiting
    is approximated by L[s,t] >= (B[s,t] - N_s * d_s) / d_s, and the objective
    is travel + beta * charging + gamma * waiting vehicle-minutes.

    With lookahead_min > 0 the vehicles expected to request within that time
    are assigned tentatively in the same problem; only current requests are
    dispatched, and the problem is solved again at the next step.
    """

    def __init__(self, lookahead_min: float = 0.0, delta_t: float = 5.0, n_slots: int = 40,
                 time_limit_s: float = 60.0, mip_gap: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.lookahead_min = lookahead_min
        self.delta_t = delta_t
        self.n_slots = n_slots
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap

    def solve(self, ev_requests: List[EVRequest], stations: List[ChargingStation],
              travel_times: Optional[Dict[Tuple[str, str], float]] = None) -> OptimizationResult:
        t0 = time.perf_counter()
        if not ev_requests:
            return self._empty(stations)
        T = self.n_slots
        slot = lambda minutes: min(T - 1, int(minutes // self.delta_t))

        base_cost: Dict[Tuple[str, str], float] = {}
        work: Dict[Tuple[str, str], float] = {}
        arr: Dict[Tuple[str, str], int] = {}
        for ev in ev_requests:
            for st in stations:
                tau = self._travel(ev, st, travel_times)
                if tau >= UNREACHABLE:
                    continue
                d = nominal_charge(ev, st, tau, self.target_soc, self.speed_kmh)
                key = (ev.id, st.id)
                base_cost[key] = tau + self.beta * d
                work[key], arr[key] = d, slot(tau)
        ev_ids = [ev.id for ev in ev_requests if any((ev.id, st.id) in work for st in stations)]
        if not ev_ids:
            return self._empty(stations)
        current = {ev.id for ev in ev_requests if ev.release_min <= 0.0}

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", GRB_THREADS)
        env.setParam("TimeLimit", self.time_limit_s)
        env.setParam("MIPGap", self.mip_gap)
        env.start()
        m = gp.Model(env=env)
        x = m.addVars(list(work.keys()), vtype=GRB.BINARY, name="x")
        st_ids = [st.id for st in stations]
        B = m.addVars(st_ids, range(T + 1), lb=0.0, name="B")
        L = m.addVars(st_ids, range(T + 1), lb=0.0, name="L")

        for i in ev_ids:
            m.addConstr(gp.quicksum(x[i, s] for s in st_ids if (i, s) in work) == 1)

        for st in stations:
            fixed = [0.0] * T
            for eta, service in st.committed:
                fixed[slot(eta)] += service
            works = [w for (i, s), w in work.items() if s == st.id] + [c[1] for c in st.committed]
            d_s = sum(works) / len(works) if works else 0.5 * 70.0 / st.power_kw * 60.0
            m.addConstr(B[st.id, 0] == sum(st.remaining_times))
            for t in range(T):
                arriving = gp.quicksum(work[i, st.id] * x[i, st.id]
                                       for i in ev_ids if arr.get((i, st.id)) == t)
                m.addConstr(B[st.id, t + 1] >= B[st.id, t] + fixed[t] + arriving
                            - st.n_chargers * self.delta_t)
            for t in range(T + 1):
                m.addConstr(L[st.id, t] * d_s >= B[st.id, t] - st.n_chargers * d_s)

        m.setObjective(
            gp.quicksum(base_cost[k] * x[k] for k in work)
            + self.gamma_w * self.delta_t * L.sum(),
            GRB.MINIMIZE,
        )
        m.optimize()
        res = self._empty(stations, status="OPTIMAL" if m.Status == GRB.OPTIMAL else f"STATUS_{m.Status}")
        res.solve_time_ms = (time.perf_counter() - t0) * 1000.0
        if m.SolCount > 0:
            res.assignments = {i: s for (i, s) in work if i in current and x[i, s].X > 0.5}
            res.objective = m.ObjVal
        m.dispose()
        env.dispose()
        return res
