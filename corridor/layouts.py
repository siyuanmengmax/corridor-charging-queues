"""
Corridor layouts shared by the Phase 1 simulator and the SUMO environment.

  synthetic  300 km, 6 stations, 13 chargers (the controlled experiments; SUMO positions)
  i95_nc     I-95 southbound through North Carolina (corridor/data/i95_nc_stations.json)
  uniform_L<km>_S<n>_C<c>[_P<kw>]  generated: n equally spaced stations with c chargers each,
             alternating 150/100 kW unless a single power P is given (E7)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .models.optimizer import ChargingStation

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Layout:
    name: str
    length_km: float
    stations: Tuple[Tuple[float, int, float], ...]   # (position_km, n_chargers, power_kw)

    @property
    def n_chargers(self) -> int:
        return sum(c for _, c, _ in self.stations)

    def charging_stations(self) -> List[ChargingStation]:
        return [ChargingStation(id=f"cs_{i}", position_km=float(p), n_chargers=c, power_kw=pw)
                for i, (p, c, pw) in enumerate(self.stations)]


SYNTHETIC = Layout("synthetic", 300.0, (
    (15.0, 2, 150.0), (65.0, 2, 100.0), (122.0, 3, 150.0),
    (182.0, 2, 100.0), (242.0, 2, 150.0), (295.0, 2, 100.0),
))


def _from_file(name: str) -> Layout:
    d = json.loads((DATA_DIR / f"{name}_stations.json").read_text())
    stations, last = [], -1.0
    for s in sorted(d["sites"], key=lambda s: s["km"]):
        km = max(s["km"], last + 0.3)    # sites at the same exit: 300 m apart
        stations.append((round(km, 1), s["n_chargers"], s["power_kw"]))
        last = km
    return Layout(name, d["length_km"], tuple(stations))


def _uniform(name: str) -> Layout:
    m = re.fullmatch(r"uniform_L(\d+)_S(\d+)_C(\d+)(?:_P(\d+))?", name)
    if not m:
        raise KeyError(name)
    length, n, c = float(m[1]), int(m[2]), int(m[3])
    spacing = length / n
    power = [float(m[4])] * n if m[4] else [150.0 if i % 2 == 0 else 100.0 for i in range(n)]
    return Layout(name, length, tuple((round(spacing * (i + 0.25), 1), c, power[i]) for i in range(n)))


def get_layout(name: str = "synthetic") -> Layout:
    if name == "synthetic":
        return SYNTHETIC
    if name.startswith("uniform_"):
        return _uniform(name)
    return _from_file(name)
