"""
SUMO/TraCI closed-loop corridor environment (Phase 2).

Wait time measurement (accurate):
  When vehicle arrives within PROXIMITY_M of its assigned station → record
  station_arrival_sim_s. When SUMO fires parking_starting for that vehicle →
  wait = (parking_start_sim_s - station_arrival_sim_s) / 60.  This isolates
  queueing wait (time waiting for a free charger after arriving) from travel time.

One-directional guard: vehicles may only be dispatched to stations AHEAD of them
  on the corridor (SUMO cannot route vehicles backward on a one-way highway).
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import traci

from ..models.optimizer import EVRequest, ChargingStation, OptimizationResult
from .corridor_sim import _by_station

logger = logging.getLogger(__name__)

SUMO_DIR = Path(__file__).parent.parent / "sumo"
SUMOCFG  = SUMO_DIR / "corridor.sumocfg"



def station_meta(layout: str = "synthetic") -> List[dict]:
    from ..layouts import get_layout
    return [{"id": st.id, "pa_id": f"pa_{st.id}", "pos_km": st.position_km,
             "n_chargers": st.n_chargers, "power_kw": st.power_kw}
            for st in get_layout(layout).charging_stations()]


STATION_META: List[dict] = station_meta("synthetic")

CONSUMPTION_KWH_PER_KM = 0.20
CHARGE_THRESHOLD_SOC   = 0.20
TARGET_SOC             = 0.80
CHARGE_UNCERTAINTY_EPS = 0.20
PROXIMITY_M            = 1_000   # 1 km — "arrived at station" detection radius
MAX_DETOUR_KM          = 80.0    # only stations within 80 km downstream are reachable
SPEED_KMH              = 100.0
HOLD_STOP_S            = 1.0e6   # stop is released by the environment, not by SUMO


@dataclass
class VehicleState:
    id: str
    pos_km: float
    soc: float
    battery_kwh: float
    status: str = "driving"   # driving | pending | en_route | near_station | charging | no_station
    assigned_station: Optional[str] = None
    request_sim_s: Optional[float] = None          # first charging request of the current need
    station_arrival_sim_s: Optional[float] = None  # when vehicle got within PROXIMITY_M
    charge_end_sim_s:      Optional[float] = None  # when charging finishes
    last_dist_m: float = 0.0                       # for SOC Δ computation


class SumoCorridorEnv:
    """SUMO TraCI closed-loop environment on a corridor layout (default: synthetic, 300 km, 6 stations)."""

    def __init__(
        self,
        seed: int = 0,
        gui: bool = False,
        delta_t_s: int = 300,
        sim_duration_s: int = 14_400,
        scenario: str = "heavy",
        demand: Optional[dict] = None,          # n_vehicles, demand_rate, soc_init_mean; overrides scenario
        charge_threshold: float = CHARGE_THRESHOLD_SOC,
        max_detour_km: Optional[float] = MAX_DETOUR_KM,   # None → remaining range of the vehicle
        warmup_s: int = 0,                      # requests before this are served but not measured
        charge_uncertainty: float = CHARGE_UNCERTAINTY_EPS,
        prediction_noise: float = 0.0,          # relative error of predicted request time and SOC
        prediction_visibility: float = 1.0,     # share of upcoming vehicles the dispatcher can see
        compliance: float = 1.0,                # share of drivers who follow the assigned station
        max_drain_s: int = 86_400,
        layout: str = "synthetic",
    ):
        self.seed = seed
        self.gui = gui
        self.delta_t_s = delta_t_s
        self.sim_duration_s = sim_duration_s
        self.scenario = scenario
        self.demand = demand
        self.charge_threshold = charge_threshold
        self.max_detour_km = max_detour_km
        self.warmup_s = warmup_s
        self.charge_uncertainty = charge_uncertainty
        self.prediction_noise = prediction_noise
        self.prediction_visibility = prediction_visibility
        self.compliance = compliance
        # separate stream so that prediction settings do not change the simulated demand
        self.pred_rng = np.random.default_rng(seed + 100_003)
        self.max_drain_s = max_drain_s
        self.layout = layout
        self.meta = station_meta(layout)
        self._by_id = {m["id"]: m for m in self.meta}
        self.rng = np.random.default_rng(seed)

        self.vehicles: Dict[str, VehicleState] = {}
        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.completed_waits: List[float] = []
        self.congestion_events: int = 0
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.access_times: List[float] = []   # request -> start of charging (min)
        self.service_minutes: float = 0.0
        self.station_times: List[float] = []  # wait + charging time per served vehicle
        self.station_log: List[tuple] = []    # (station, wait, charging time) per served vehicle
        self.n_requests: int = 0
        self.n_no_station: int = 0
        self.n_rejected: int = 0
        self.n_noncompliant: int = 0          # dispatches ignored by the driver
        self.n_lost: int = 0                  # left the network without charging
        self.draining: bool = False
        self.drain_s: float = 0.0

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        from ..sumo.generate_network import SCENARIOS, ensure_network, write_routes
        demand = self.demand or SCENARIOS[self.scenario]
        cfg = ensure_network(self.layout)
        routes = write_routes(**demand, seed=self.seed, duration_s=self.sim_duration_s, layout=self.layout)
        cmd = [
            "sumo-gui" if self.gui else "sumo",
            "-c", str(cfg),
            "--route-files", str(routes),
            "--end", str(self.sim_duration_s + self.max_drain_s),
            "--seed", str(self.seed),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--ignore-route-errors", "true",
            "--collision.action", "none",
            # Vehicles queueing for a full station must keep waiting, not be teleported past it
            "--time-to-teleport", "-1",
        ]
        traci.start(cmd)
        self.sim_time = 0.0
        self.step_count = 0
        self.vehicles.clear()
        self.completed_waits.clear()
        self.congestion_events = 0
        self._reset_counters()
        logger.info(f"SUMO started (seed={self.seed})")

    def close(self) -> None:
        try:
            traci.close()
        except Exception:
            pass

    @property
    def done(self) -> bool:
        return self.sim_time >= self.sim_duration_s

    # -----------------------------------------------------------------------
    # Controller interface
    # -----------------------------------------------------------------------

    def get_station_snapshot(self) -> List[ChargingStation]:
        """Read live station occupancy from SUMO parkingArea."""
        result = []
        for m in self.meta:
            try:
                occupants = list(traci.parkingarea.getVehicleIDs(m["pa_id"]))
            except Exception:
                occupants = []
            remaining: List[float] = []
            for vid in occupants:
                v = self.vehicles.get(vid)
                if v and v.charge_end_sim_s is not None:
                    remaining.append(max(0.0, (v.charge_end_sim_s - self.sim_time) / 60.0))
                else:
                    remaining.append(10.0)
            result.append(ChargingStation(
                id=m["id"],
                position_km=m["pos_km"],
                n_chargers=m["n_chargers"],
                power_kw=m["power_kw"],
                remaining_times=remaining[:m["n_chargers"]],
                committed=self._committed(m),
            ))
        return result

    def _committed(self, m: dict) -> List[tuple]:
        """Queued and en-route vehicles of a station: (minutes to arrival, nominal charge minutes)."""
        out = []
        for v in self.vehicles.values():
            if v.assigned_station != m["id"] or v.status not in ("en_route", "near_station"):
                continue
            dist = 0.0 if v.status == "near_station" else max(0.0, m["pos_km"] - v.pos_km)
            soc_arr = max(0.0, v.soc - CONSUMPTION_KWH_PER_KM * dist / v.battery_kwh)
            out.append((dist / SPEED_KMH * 60.0,
                        max(0.0, TARGET_SOC - soc_arr) * v.battery_kwh / m["power_kw"] * 60.0))
        return sorted(out)   # queued vehicles (0 min to arrival) first

    def _complies(self, vid: str) -> bool:
        """Fixed per driver and independent of the dispatch method."""
        if self.compliance >= 1.0:
            return True
        return np.random.default_rng([self.seed, zlib.crc32(vid.encode())]).random() < self.compliance

    def _reachable(self, pos_km: float, soc: float, battery_kwh: float, station_pos_km: float) -> bool:
        limit = self.max_detour_km
        if limit is None:
            limit = soc * battery_kwh / CONSUMPTION_KWH_PER_KM
        return -1.0 <= station_pos_km - pos_km <= limit

    def travel_times(self, requests: List[EVRequest]) -> Dict[tuple, float]:
        """Directed travel times (min) for the optimizers; unreachable → 1e6."""
        tt = {}
        for ev in requests:
            cross_km = ev.position_km + ev.release_min / 60.0 * SPEED_KMH
            for m in self.meta:
                if (self._reachable(ev.position_km, ev.soc, ev.battery_kwh, m["pos_km"])
                        and m["pos_km"] >= cross_km - 1.0):
                    tt[(ev.id, m["id"])] = max(0.0, m["pos_km"] - ev.position_km) / SPEED_KMH * 60.0
                else:
                    tt[(ev.id, m["id"])] = 1e6
        return tt

    def get_upcoming_requests(self, horizon_min: float) -> List[EVRequest]:
        """Vehicles on the corridor expected to cross the SOC threshold within horizon_min."""
        out = []
        last_km = self.meta[-1]["pos_km"]
        for v in self.vehicles.values():
            if v.status != "driving" or v.soc <= self.charge_threshold:
                continue
            km_to_cross = (v.soc - self.charge_threshold) * v.battery_kwh / CONSUMPTION_KWH_PER_KM
            release = km_to_cross / SPEED_KMH * 60.0
            if release <= horizon_min and v.pos_km + km_to_cross < last_km:
                if self.pred_rng.random() > self.prediction_visibility:
                    continue
                err = self.prediction_noise
                soc = v.soc * (1.0 + self.pred_rng.uniform(-err, err)) if err else v.soc
                release = release * (1.0 + self.pred_rng.uniform(-err, err)) if err else release
                out.append(EVRequest(id=v.id, position_km=v.pos_km,
                                     soc=float(np.clip(soc, self.charge_threshold + 1e-3, 0.99)),
                                     battery_kwh=v.battery_kwh, release_min=max(0.1, release)))
        return out

    def get_charging_requests(self) -> List[EVRequest]:
        """Sync states from SUMO, return vehicles needing charging."""
        self._sync_states()
        if self.draining:
            return []
        requests = []
        for v in self.vehicles.values():
            if v.status == "driving" and v.soc <= self.charge_threshold:
                if not any(self._reachable(v.pos_km, v.soc, v.battery_kwh, m["pos_km"])
                           for m in self.meta):
                    v.status = "no_station"
                    # stranded only if a station still lies ahead; otherwise it has left the corridor
                    self.n_no_station += v.pos_km < self.meta[-1]["pos_km"]
                    continue
                requests.append(EVRequest(
                    id=v.id,
                    position_km=v.pos_km,
                    soc=v.soc,
                    battery_kwh=v.battery_kwh,
                ))
                v.status = "pending"
                if v.request_sim_s is None:
                    v.request_sim_s = self.sim_time
                    self.n_requests += v.request_sim_s >= self.warmup_s
        return requests

    def _sync_states(self) -> None:
        """Update SOC and position for all active vehicles."""
        try:
            active = set(traci.vehicle.getIDList())
        except Exception:
            return

        # Register new vehicles
        for vid in active:
            if vid not in self.vehicles:
                try:
                    soc  = float(traci.vehicle.getParameter(vid, "soc")         or "0.35")
                    bkwh = float(traci.vehicle.getParameter(vid, "battery_kwh") or "70.0")
                except Exception:
                    soc  = float(np.clip(self.rng.normal(0.35, 0.15), 0.10, 0.95))
                    bkwh = float(np.clip(self.rng.normal(70.0, 8.0),  40.0, 100.0))
                try:
                    x, _ = traci.vehicle.getPosition(vid)
                    pos_km = x / 1000.0
                except Exception:
                    pos_km = 0.0
                try:
                    dist0 = traci.vehicle.getDistance(vid)
                except Exception:
                    dist0 = 0.0
                self.vehicles[vid] = VehicleState(
                    id=vid, pos_km=pos_km, soc=soc, battery_kwh=bkwh,
                    last_dist_m=dist0,
                )

        # Remove departed
        for vid in [v for v in self.vehicles if v not in active]:
            if self.vehicles[vid].status in ("en_route", "near_station"):
                self.n_lost += 1
            self.vehicles.pop(vid)

        # Update
        for vid, v in self.vehicles.items():
            if vid not in active:
                continue
            try:
                x, _      = traci.vehicle.getPosition(vid)
                new_pos_km = x / 1000.0
                new_dist_m = traci.vehicle.getDistance(vid)
            except Exception:
                continue

            # SOC: distance-based discharge (not while charging)
            if v.status != "charging":
                delta_km = max(0.0, (new_dist_m - v.last_dist_m) / 1000.0)
                v.soc = max(0.0, v.soc - CONSUMPTION_KWH_PER_KM * delta_km / v.battery_kwh)
            v.pos_km     = new_pos_km
            v.last_dist_m = new_dist_m

    # -----------------------------------------------------------------------
    # Apply assignments
    # -----------------------------------------------------------------------

    def apply_assignments(self, result: OptimizationResult) -> None:
        if result.has_congestion:
            self.congestion_events += 1
        try:
            active = set(traci.vehicle.getIDList())
        except Exception:
            return

        for vid, sid in result.assignments.items():
            v = self.vehicles.get(vid)
            if v is None or vid not in active or v.status != "pending":   # predictions are not dispatched
                continue

            if not self._complies(vid):   # the driver stops at the nearest reachable station instead
                near = [mm for mm in self.meta if self._reachable(v.pos_km, v.soc, v.battery_kwh, mm["pos_km"])]
                if near:
                    sid = min(near, key=lambda mm: mm["pos_km"])["id"]
                    self.n_noncompliant += 1
            m = self._by_id[sid]

            # One-directional guard: only dispatch to reachable downstream stations
            if not self._reachable(v.pos_km, v.soc, v.battery_kwh, m["pos_km"]):
                self.n_rejected += 1
                v.status = "driving"
                continue

            try:
                # The charging duration is drawn when charging starts; the stop is
                # held until the environment releases it in step().
                traci.vehicle.setParkingAreaStop(vid, m["pa_id"], duration=HOLD_STOP_S)
                v.status = "en_route"
                v.assigned_station = sid
                v.station_arrival_sim_s = None  # reset; set when proximity detected
                v.charge_end_sim_s = None
            except traci.TraCIException as e:
                logger.debug(f"Dispatch failed {vid}→{sid}: {e}")
                self.n_rejected += 1
                v.status = "driving"

        # Vehicles left without a valid assignment ask again at the next step
        for v in self.vehicles.values():
            if v.status == "pending":
                v.status = "driving"

    # -----------------------------------------------------------------------
    # Simulation step
    # -----------------------------------------------------------------------

    def step(self) -> dict:
        """Advance SUMO by delta_t_s seconds; track arrivals, charging starts and ends."""
        step_end = self.sim_time + self.delta_t_s

        while self.sim_time < step_end:
            traci.simulationStep()
            self.sim_time = traci.simulation.getTime()

            # Arrival near the assigned station (start of the queueing wait)
            for vid, v in self.vehicles.items():
                if v.status != "en_route":
                    continue
                try:
                    x, _ = traci.vehicle.getPosition(vid)
                except traci.TraCIException:
                    continue
                st_pos_km = self._by_id[v.assigned_station]["pos_km"]
                if abs(x / 1000.0 - st_pos_km) * 1000.0 <= PROXIMITY_M:
                    v.station_arrival_sim_s = self.sim_time
                    v.status = "near_station"

            # Vehicles that just entered a charger slot
            for vid in traci.simulation.getParkingStartingVehiclesIDList():
                v = self.vehicles.get(vid)
                if v is None or v.assigned_station is None:
                    continue
                m = self._by_id[v.assigned_station]
                arrival = v.station_arrival_sim_s if v.station_arrival_sim_s is not None else self.sim_time
                wait_min = max(0.0, (self.sim_time - arrival) / 60.0)

                # SOC on arrival: discharge since the last sync
                try:
                    dist_m = traci.vehicle.getDistance(vid)
                    delta_km = max(0.0, (dist_m - v.last_dist_m) / 1000.0)
                    v.soc = max(0.0, v.soc - CONSUMPTION_KWH_PER_KM * delta_km / v.battery_kwh)
                    v.last_dist_m = dist_m
                except traci.TraCIException:
                    pass

                nominal_min = max(0.0, TARGET_SOC - v.soc) * v.battery_kwh / m["power_kw"] * 60.0
                eps = float(self.rng.uniform(-self.charge_uncertainty, self.charge_uncertainty))
                duration_s = max(60.0, nominal_min * (1.0 + eps) * 60.0)

                v.status = "charging"
                v.charge_end_sim_s = self.sim_time + duration_s
                v.station_arrival_sim_s = None
                if v.request_sim_s is not None and v.request_sim_s >= self.warmup_s:
                    self.completed_waits.append(wait_min)
                    self.service_minutes += duration_s / 60.0
                    self.station_times.append(wait_min + duration_s / 60.0)
                    self.station_log.append((v.assigned_station, wait_min, duration_s / 60.0))
                    self.access_times.append((self.sim_time - v.request_sim_s) / 60.0)

            # Charging completions: release the charger slot
            for vid, v in self.vehicles.items():
                if v.status == "charging" and self.sim_time >= v.charge_end_sim_s:
                    try:
                        traci.vehicle.resume(vid)
                    except traci.TraCIException as e:
                        logger.warning(f"resume failed for {vid}: {e}")
                    v.soc = TARGET_SOC
                    v.status = "driving"
                    v.assigned_station = None
                    v.charge_end_sim_s = None
                    v.request_sim_s = None

        self.step_count += 1
        return self._metrics()

    def n_unserved(self) -> int:
        return sum(1 for v in self.vehicles.values() if v.status in ("en_route", "near_station"))

    def drain(self) -> None:
        """
        Keep simulating after the episode, with no new requests, until every
        vehicle that requested charging has started (waits are not censored).
        """
        self.draining = True
        t_end = self.sim_time
        while self.n_unserved() > 0 and self.sim_time - t_end < self.max_drain_s - self.delta_t_s:
            self._sync_states()
            self.step()
        self.drain_s = self.sim_time - t_end

    def _metrics(self) -> dict:
        counts: Dict[str, int] = {}
        for v in self.vehicles.values():
            counts[v.status] = counts.get(v.status, 0) + 1
        return {
            "sim_time_min": self.sim_time / 60.0,
            "n_vehicles": len(self.vehicles),
            "n_served_so_far": len(self.completed_waits),
            **{f"n_{k}": cnt for k, cnt in counts.items()},
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
        charger_minutes = sum(m["n_chargers"] for m in self.meta) * (self.sim_duration_s - self.warmup_s) / 60.0
        extra = {
            "n_requests": self.n_requests,
            "n_unserved": self.n_unserved(),
            "n_no_station": self.n_no_station,
            "n_rejected": self.n_rejected,
            "n_noncompliant": self.n_noncompliant,
            "n_lost": self.n_lost,
            "access_mean_min": float(a.mean()),
            "access_max_min": float(a.max()),
            "drain_min": self.drain_s / 60.0,
            "station_mean_min": float(st.mean()),
            "station_p95_min": float(np.percentile(st, 95)),
            "charge_mean_min": self.service_minutes / len(w),
            "by_station": _by_station(self.station_log),
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
