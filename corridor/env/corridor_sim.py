"""
Intercity highway corridor simulation (Phase 1 / Route B: standalone, no SUMO).

Step ordering (fixed):
  1. Advance queue state: remove completed vehicles, start waiting ones.
  2. Move driving / en-route vehicles; discharge batteries.
  3. Process newly arrived vehicles: enter station queues.
  4. Spawn new vehicles; remove departed ones.
  5. Collect metrics.

Charging completions are recorded for both:
  - Vehicles served immediately on arrival (wait = 0).
  - Vehicles that waited in queue (wait > 0, tracked when service starts).

The corridor is one-directional: a vehicle can only be sent to a station that
is downstream of it and within max_detour_km (or, if that is None, within the
vehicle's remaining range). Requests made after the warm-up form the
measurement cohort; drain() keeps simulating after the episode until every
vehicle that requested has started charging, so waits are not censored.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..models.optimizer import EVRequest, ChargingStation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------


def _by_station(log) -> dict:
    """Served vehicles, mean wait and mean charging time per station."""
    out = {}
    for sid, w, d in log:
        n, sw, sd, mw = out.get(sid, (0, 0.0, 0.0, 0.0))
        out[sid] = (n + 1, sw + w, sd + d, max(mw, w))
    return {sid: {"n": n, "wait_mean": sw / n, "charge_mean": sd / n, "wait_max": mw}
            for sid, (n, sw, sd, mw) in sorted(out.items())}


@dataclass
class SimVehicle:
    id: str
    position_km: float
    destination_km: float
    soc: float
    battery_kwh: float
    speed_kmh: float
    consumption_kwh_per_km: float

    status: str = "driving"          # driving | en_route | queued | charging | completed
    assigned_station: Optional[str] = None
    queue_entry_time: Optional[float] = None
    charge_end_time: Optional[float] = None   # sim_time when charging ends
    waiting_time: float = 0.0                 # final waiting time (min)
    request_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Station queue
# ---------------------------------------------------------------------------

@dataclass
class QueueState:
    station_id: str
    n_chargers: int
    # (vehicle_id, charge_end_time)
    in_service: List[Tuple[str, float]] = field(default_factory=list)
    # (vehicle_id, queue_entry_time)
    waiting: List[Tuple[str, float]] = field(default_factory=list)

    def n_free(self) -> int:
        return self.n_chargers - len(self.in_service)

    def remaining_times(self, sim_time: float) -> List[float]:
        return [max(0.0, end - sim_time) for _, end in self.in_service]

    def advance(self, sim_time: float) -> Tuple[List[str], List[Tuple[str, float]]]:
        """
        Advance queue to sim_time:
          - Remove vehicles whose charge_end_time <= sim_time → completed.
          - Fill vacated slots from waiting queue (FCFS) → newly_started.

        Returns
        -------
        completed_ids   : vehicle ids that finished charging this step
        newly_started   : (vehicle_id, wait_minutes) for vehicles that just
                          started service (moved from waiting → in_service)
        """
        # 1. Identify completed vehicles
        completed_ids = [vid for vid, end in self.in_service if end <= sim_time]
        self.in_service = [(vid, end) for vid, end in self.in_service if end > sim_time]

        # 2. Start waiting vehicles in freed slots
        newly_started: List[Tuple[str, float]] = []
        while self.waiting and self.n_free() > 0:
            vid, entry_time = self.waiting.pop(0)
            wait = sim_time - entry_time
            newly_started.append((vid, wait))
            # Occupy the charger now so that one freed slot starts one vehicle.
            # The caller replaces the end time via set_service_end().
            self.in_service.append((vid, float("inf")))

        return completed_ids, newly_started

    def set_service_end(self, vehicle_id: str, end_time: float) -> None:
        self.in_service = [
            (vid, end_time if vid == vehicle_id else end) for vid, end in self.in_service
        ]

    def arrive(self, vehicle_id: str, sim_time: float, service_duration: float) -> float:
        """
        Vehicle arrives at station.
        Returns wait_minutes (0.0 if served immediately, −1 if queued).
        """
        if self.n_free() > 0:
            self.in_service.append((vehicle_id, sim_time + service_duration))
            return 0.0
        else:
            self.waiting.append((vehicle_id, sim_time))
            return -1.0  # will be recorded when service starts


# ---------------------------------------------------------------------------
# Corridor simulation
# ---------------------------------------------------------------------------

class CorridorSimulation:

    def __init__(
        self,
        corridor_length_km: float = 300.0,
        stations: Optional[List[ChargingStation]] = None,
        n_vehicles: int = 80,
        delta_t: float = 5.0,
        sim_duration_min: float = 240.0,
        demand_rate: float = 0.4,       # mean new vehicles entering per minute
        speed_kmh: float = 100.0,
        speed_std_kmh: float = 10.0,
        consumption_mean: float = 0.20,
        consumption_std: float = 0.02,
        battery_mean_kwh: float = 70.0,
        battery_std_kwh: float = 10.0,
        soc_init_mean: float = 0.65,
        soc_init_std: float = 0.15,
        charge_threshold: float = 0.20,
        target_soc: float = 0.80,
        charge_uncertainty: float = 0.20,
        max_detour_km: Optional[float] = 80.0,   # None → remaining range of the vehicle
        warmup_min: float = 0.0,                 # requests before this are served but not measured
        prediction_noise: float = 0.0,           # relative error of predicted request time and SOC
        prediction_visibility: float = 1.0,      # share of upcoming vehicles the dispatcher can see
        compliance: float = 1.0,                 # share of drivers who follow the assigned station
        rng_seed: int = 42,
    ):
        self.corridor_length_km = corridor_length_km
        self.n_vehicles = n_vehicles
        self.delta_t = delta_t
        self.sim_duration_min = sim_duration_min
        self.demand_rate = demand_rate
        self.speed_kmh = speed_kmh
        self.speed_std_kmh = speed_std_kmh
        self.consumption_mean = consumption_mean
        self.consumption_std = consumption_std
        self.battery_mean_kwh = battery_mean_kwh
        self.battery_std_kwh = battery_std_kwh
        self.soc_init_mean = soc_init_mean
        self.soc_init_std = soc_init_std
        self.charge_threshold = charge_threshold
        self.target_soc = target_soc
        self.charge_uncertainty_eps = charge_uncertainty
        self.max_detour_km = max_detour_km
        self.warmup_min = warmup_min
        self.prediction_noise = prediction_noise
        self.prediction_visibility = prediction_visibility
        self.compliance = compliance
        self._seed = rng_seed
        # separate stream so that prediction settings do not change the simulated demand
        self.pred_rng = np.random.default_rng(rng_seed + 100_003)
        self.rng = np.random.default_rng(rng_seed)

        self.stations: List[ChargingStation] = stations or self._default_stations()
        self.station_map: Dict[str, ChargingStation] = {s.id: s for s in self.stations}
        self.queue_states: Dict[str, QueueState] = {}

        # Per-vehicle service duration (set when vehicle arrives at station)
        self._service_durations: Dict[str, float] = {}

        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.vehicle_counter: int = 0
        self.vehicles: Dict[str, SimVehicle] = {}

        self.completed_waits: List[float] = []
        self.congestion_events: int = 0
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.access_times: List[float] = []   # request -> start of charging (min)
        self.service_minutes: float = 0.0     # realized charging time of the cohort
        self.station_times: List[float] = []  # wait + charging time per served vehicle
        self.station_log: List[tuple] = []    # (station, wait, charging time) per served vehicle
        self.n_requests: int = 0
        self.n_no_station: int = 0            # needed charging with no reachable station
        self.n_rejected: int = 0              # assignments to an unreachable station
        self.n_noncompliant: int = 0          # dispatches ignored by the driver
        self.draining: bool = False
        self.drain_minutes: float = 0.0

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _default_stations(self) -> List[ChargingStation]:
        L = self.corridor_length_km
        configs = [
            (0,       4, 150.0),
            (L*0.2,   3, 100.0),
            (L*0.4,   6, 150.0),
            (L*0.6,   3, 100.0),
            (L*0.8,   4, 150.0),
            (L,       3, 100.0),
        ]
        return [
            ChargingStation(id=f"cs_{i}", position_km=float(p), n_chargers=c, power_kw=pw)
            for i, (p, c, pw) in enumerate(configs)
        ]

    def _spawn_vehicle(self, position_km: float = 0.0) -> SimVehicle:
        vid = f"ev_{self.vehicle_counter:05d}"
        self.vehicle_counter += 1
        return SimVehicle(
            id=vid,
            position_km=position_km,
            destination_km=self.corridor_length_km + self.rng.uniform(10.0, 100.0),
            soc=float(np.clip(self.rng.normal(self.soc_init_mean, self.soc_init_std), 0.05, 0.99)),
            battery_kwh=float(max(30.0, self.rng.normal(self.battery_mean_kwh, self.battery_std_kwh))),
            speed_kmh=float(max(60.0, self.rng.normal(self.speed_kmh, self.speed_std_kmh))),
            consumption_kwh_per_km=float(max(0.10, self.rng.normal(self.consumption_mean, self.consumption_std))),
        )

    def reset(self) -> None:
        self.sim_time = 0.0
        self.step_count = 0
        self.vehicles = {}
        self.vehicle_counter = 0
        self.queue_states = {s.id: QueueState(s.id, s.n_chargers) for s in self.stations}
        self._service_durations = {}
        self.completed_waits = []
        self.congestion_events = 0
        self._reset_counters()
        # Seed fleet: vehicles at random positions along the corridor
        n_init = max(1, int(self.n_vehicles * 0.6))
        for _ in range(n_init):
            v = self._spawn_vehicle(position_km=float(self.rng.uniform(0, self.corridor_length_km * 0.8)))
            v.soc = float(np.clip(self.rng.normal(self.soc_init_mean, self.soc_init_std), 0.05, 0.99))
            self.vehicles[v.id] = v

    # -----------------------------------------------------------------------
    # Controller interface
    # -----------------------------------------------------------------------

    def get_station_snapshot(self) -> List[ChargingStation]:
        """Stations with current remaining_times for capacity calculation."""
        result = []
        for st in self.stations:
            qs = self.queue_states[st.id]
            result.append(ChargingStation(
                id=st.id,
                position_km=st.position_km,
                n_chargers=st.n_chargers,
                power_kw=st.power_kw,
                remaining_times=qs.remaining_times(self.sim_time),
                committed=self._committed(st, qs),
            ))
        return result

    def _committed(self, st: ChargingStation, qs: QueueState) -> List[Tuple[float, float]]:
        """Queued and en-route vehicles of a station: (minutes to arrival, nominal charge minutes)."""
        out = []
        for vid, _ in qs.waiting:                       # FCFS order
            v = self.vehicles[vid]
            out.append((0.0, max(0.0, self.target_soc - v.soc) * v.battery_kwh / st.power_kw * 60.0))
        for v in self.vehicles.values():
            if v.status == "en_route" and v.assigned_station == st.id:
                dist = max(0.0, st.position_km - v.position_km)
                soc_arr = max(0.0, v.soc - v.consumption_kwh_per_km * dist / v.battery_kwh)
                out.append((dist / v.speed_kmh * 60.0,
                            max(0.0, self.target_soc - soc_arr) * v.battery_kwh / st.power_kw * 60.0))
        return out

    def _reachable(self, position_km: float, soc: float, battery_kwh: float,
                   st: ChargingStation) -> bool:
        """Station is downstream of the vehicle and within the detour limit or range."""
        limit = self.max_detour_km
        if limit is None:
            limit = soc * battery_kwh / self.consumption_mean
        return -1.0 <= st.position_km - position_km <= limit

    def travel_times(self, requests: List[EVRequest]) -> Dict[Tuple[str, str], float]:
        """Directed travel times (min) for the optimizers; unreachable → 1e6."""
        tt = {}
        for ev in requests:
            cross_km = ev.position_km + ev.release_min / 60.0 * self.speed_kmh
            for st in self.stations:
                if (self._reachable(ev.position_km, ev.soc, ev.battery_kwh, st)
                        and st.position_km >= cross_km - 1.0):
                    tt[(ev.id, st.id)] = max(0.0, st.position_km - ev.position_km) / self.speed_kmh * 60.0
                else:
                    tt[(ev.id, st.id)] = 1e6
        return tt

    def get_charging_requests(self) -> List[EVRequest]:
        """Vehicles that crossed the SOC threshold and can reach a station."""
        if self.draining:
            return []
        requests = []
        for v in self.vehicles.values():
            if v.status == "driving" and v.soc <= self.charge_threshold:
                if not any(self._reachable(v.position_km, v.soc, v.battery_kwh, st)
                           for st in self.stations):
                    v.status = "no_station"
                    # stranded only if a station still lies ahead; otherwise it has left the corridor
                    self.n_no_station += v.position_km < self.stations[-1].position_km
                    continue
                requests.append(EVRequest(
                    id=v.id,
                    position_km=v.position_km,
                    soc=v.soc,
                    battery_kwh=v.battery_kwh,
                ))
                v.status = "pending"
                if v.request_time is None:
                    v.request_time = self.sim_time
                    self.n_requests += v.request_time >= self.warmup_min
        return requests

    def get_upcoming_requests(self, horizon_min: float) -> List[EVRequest]:
        """Vehicles on the corridor expected to cross the SOC threshold within horizon_min."""
        out = []
        last_km = self.stations[-1].position_km
        for v in self.vehicles.values():
            if v.status != "driving" or v.soc <= self.charge_threshold:
                continue
            km_to_cross = (v.soc - self.charge_threshold) * v.battery_kwh / self.consumption_mean
            release = km_to_cross / v.speed_kmh * 60.0
            if release <= horizon_min and v.position_km + km_to_cross < last_km:
                if self.pred_rng.random() > self.prediction_visibility:
                    continue
                err = self.prediction_noise
                soc = v.soc * (1.0 + self.pred_rng.uniform(-err, err)) if err else v.soc
                release = release * (1.0 + self.pred_rng.uniform(-err, err)) if err else release
                out.append(EVRequest(id=v.id, position_km=v.position_km,
                                     soc=float(np.clip(soc, self.charge_threshold + 1e-3, 0.99)),
                                     battery_kwh=v.battery_kwh, release_min=max(0.1, release)))
        return out

    def apply_assignments(self, result) -> None:
        """Send assigned vehicles en-route to their station."""
        if result.has_congestion:
            self.congestion_events += 1
        for vid, sid in result.assignments.items():
            v = self.vehicles.get(vid)
            if v is None or v.status != "pending":   # predicted vehicles are not dispatched
                continue
            if not self._complies(vid):   # the driver stops at the nearest reachable station instead
                near = [st for st in self.stations if self._reachable(v.position_km, v.soc, v.battery_kwh, st)]
                if near:
                    sid = min(near, key=lambda st: st.position_km).id
                    self.n_noncompliant += 1
            if not self._reachable(v.position_km, v.soc, v.battery_kwh, self.station_map[sid]):
                self.n_rejected += 1
                continue
            v.status = "en_route"
            v.assigned_station = sid
        # Vehicles left without a valid assignment ask again at the next step
        for v in self.vehicles.values():
            if v.status == "pending":
                v.status = "driving"

    def _complies(self, vid: str) -> bool:
        """Fixed per driver and independent of the dispatch method."""
        if self.compliance >= 1.0:
            return True
        return np.random.default_rng([self._seed, zlib.crc32(vid.encode())]).random() < self.compliance

    def _record_service_start(self, v: SimVehicle, wait: float, duration: float, sid: str = "") -> None:
        if v.request_time is None or v.request_time < self.warmup_min:
            return
        self.completed_waits.append(wait)
        self.service_minutes += duration
        self.station_times.append(wait + duration)
        self.station_log.append((sid, wait, duration))
        # charging starts at the end of the current step
        self.access_times.append(self.sim_time + self.delta_t - v.request_time)

    def drain(self, max_extra_min: float = 1440.0) -> None:
        """
        Keep simulating after the episode, with no new vehicles and no new
        requests, until every vehicle that requested charging has started.
        """
        self.draining = True
        t_end = self.sim_time
        while self.n_unserved() > 0 and self.sim_time - t_end < max_extra_min:
            self.step()
        self.drain_minutes = self.sim_time - t_end

    def n_unserved(self) -> int:
        return sum(1 for v in self.vehicles.values() if v.status in ("en_route", "arrived", "queued"))

    # -----------------------------------------------------------------------
    # Simulation step
    # -----------------------------------------------------------------------

    def step(self) -> dict:
        """Advance simulation by delta_t minutes."""
        dt_h = self.delta_t / 60.0
        next_time = self.sim_time + self.delta_t

        # === 1. Advance queue states ===
        # service_needed: vehicles that just started service (from waiting)
        newly_started_all: List[Tuple[str, str, float]] = []  # (vid, sid, wait)
        completed_all: List[Tuple[str, str]] = []             # (vid, sid)

        for sid, qs in self.queue_states.items():
            completed_ids, newly_started = qs.advance(next_time)
            for vid in completed_ids:
                completed_all.append((vid, sid))
            for vid, wait in newly_started:
                newly_started_all.append((vid, sid, wait))

        # Provide service duration to vehicles that just started from queue
        for vid, sid, wait in newly_started_all:
            v = self.vehicles.get(vid)
            if v is None:
                continue
            # Compute actual service duration now that they start charging
            st = self.station_map[sid]
            nominal = max(0.0, self.target_soc - v.soc) * v.battery_kwh / st.power_kw * 60.0
            noise = self.rng.uniform(-self.charge_uncertainty_eps, self.charge_uncertainty_eps) * nominal
            duration = max(1.0, nominal + noise)
            self.queue_states[sid].set_service_end(vid, next_time + duration)
            v.status = "charging"
            v.waiting_time = wait
            v.charge_end_time = next_time + duration
            self._record_service_start(v, wait, duration, sid)

        # Process completed vehicles
        for vid, sid in completed_all:
            v = self.vehicles.get(vid)
            if v is None:
                continue
            v.status = "driving"
            v.soc = self.target_soc
            v.assigned_station = None
            v.charge_end_time = None
            v.request_time = None
            self._service_durations.pop(vid, None)

        # === 2. Move vehicles ===
        newly_arrived: List[Tuple[str, str]] = []  # (vid, sid)

        for v in list(self.vehicles.values()):
            if v.status in ("driving", "pending", "no_station"):
                dist_km = v.speed_kmh * dt_h
                v.position_km += dist_km
                v.soc -= v.consumption_kwh_per_km * dist_km / v.battery_kwh
                v.soc = max(0.0, v.soc)
                if v.position_km >= v.destination_km:
                    v.status = "completed"

            elif v.status == "en_route" and v.assigned_station:
                st = self.station_map[v.assigned_station]
                dist_to_st = max(0.0, st.position_km - v.position_km)
                travel_min = dist_to_st / v.speed_kmh * 60.0
                if travel_min <= self.delta_t:
                    # Arrives this step
                    travel_km = dist_to_st
                    v.soc -= v.consumption_kwh_per_km * travel_km / v.battery_kwh
                    v.soc = max(0.0, v.soc)
                    v.position_km = st.position_km
                    v.status = "arrived"
                    newly_arrived.append((v.id, v.assigned_station))
                else:
                    dist_km = v.speed_kmh * dt_h
                    v.position_km += dist_km
                    v.soc -= v.consumption_kwh_per_km * dist_km / v.battery_kwh
                    v.soc = max(0.0, v.soc)

        # === 3. Newly arrived vehicles enter queues ===
        for vid, sid in newly_arrived:
            v = self.vehicles.get(vid)
            if v is None:
                continue
            st = self.station_map[sid]
            nominal = max(0.0, self.target_soc - v.soc) * v.battery_kwh / st.power_kw * 60.0
            noise = self.rng.uniform(-self.charge_uncertainty_eps, self.charge_uncertainty_eps) * nominal
            duration = max(1.0, nominal + noise)
            self._service_durations[vid] = duration

            qs = self.queue_states[sid]
            wait = qs.arrive(vid, next_time, duration)
            v.queue_entry_time = next_time

            if wait == 0.0:
                # Served immediately: record wait=0 when charging completes
                v.status = "charging"
                v.waiting_time = 0.0
                v.charge_end_time = next_time + duration
                self._record_service_start(v, 0.0, duration, sid)
            else:
                v.status = "queued"

        # === 4. Spawn new vehicles, remove completed ===
        n_spawn = 0 if self.draining else int(self.rng.poisson(self.demand_rate * self.delta_t))
        for _ in range(n_spawn):
            nv = self._spawn_vehicle(position_km=0.0)
            self.vehicles[nv.id] = nv

        for vid in [v for v, veh in self.vehicles.items() if veh.status == "completed"]:
            self.vehicles.pop(vid)

        self.sim_time = next_time
        self.step_count += 1
        return self._metrics()

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def _metrics(self) -> dict:
        by_status = {}
        for v in self.vehicles.values():
            by_status[v.status] = by_status.get(v.status, 0) + 1
        return {
            "sim_time": self.sim_time,
            "n_vehicles": len(self.vehicles),
            **by_status,
            "total_queued": sum(len(qs.waiting) for qs in self.queue_states.values()),
        }

    def episode_metrics(self) -> dict:
        if not self.completed_waits:
            return {
                "mean_wait_min": 0.0, "max_wait_min": 0.0,
                "p95_wait_min": 0.0, "congestion_rate": 0.0, "n_served": 0,
            }
        w = np.array(self.completed_waits)
        a = np.array(self.access_times) if self.access_times else np.zeros(1)
        st = np.array(self.station_times)
        charger_minutes = sum(s.n_chargers for s in self.stations) * (self.sim_duration_min - self.warmup_min)
        extra = {
            "n_requests": self.n_requests,
            "n_unserved": self.n_unserved(),
            "n_no_station": self.n_no_station,
            "n_rejected": self.n_rejected,
            "n_noncompliant": self.n_noncompliant,
            "access_mean_min": float(a.mean()),
            "access_max_min": float(a.max()),
            "drain_min": self.drain_minutes,
            "station_mean_min": float(st.mean()),
            "station_p95_min": float(np.percentile(st, 95)),
            "charge_mean_min": self.service_minutes / len(w),
            "by_station": _by_station(self.station_log),
            # realized charging time of the cohort / charger-time in the episode
            "offered_load": self.service_minutes / charger_minutes,
        }
        return {
            **extra,
            "mean_wait_min": float(w.mean()),
            "max_wait_min": float(w.max()),
            "p95_wait_min": float(np.percentile(w, 95)),
            "congestion_rate": self.congestion_events / max(1, self.step_count),
            "n_served": int(len(w)),
        }
