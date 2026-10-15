"""
Build the real-corridor case study: I-95 southbound through North Carolina.

Sources
  - Road geometry: OpenStreetMap (Overpass API), motorway ways with ref "I 95".
  - Stations: U.S. DOE Alternative Fuels Data Center (AFDC) station API,
    public, open, DC fast, states VA/NC/SC.

Definitions
  - Chainage 0 = VA/NC state line; the corridor ends at the NC/SC state line.
  - A site belongs to the corridor if it lies in NC within 2 km of an I-95 carriageway.
  - Only CCS (J1772COMBO) charging units are counted: the case study models
    vehicles without Supercharger access, so Tesla-only sites are excluded.
  - One charging unit = one charger (serves one vehicle at a time).
  - Records of the same network within 1 km of each other form one site.
  - Effective session power = min(rated unit power, VEHICLE_CAP_KW), averaged over the units.

Run (downloads into --raw_dir, writes corridor/data/i95_nc_stations.json):
  python -m corridor.data.build_i95_nc --raw_dir <dir> [--download]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "i95_nc_stations.json"
VEHICLE_CAP_KW = 150.0
MAX_OFFSET_KM = 2.0
MERGE_KM = 1.0
STATE_LINES_LAT = {"VA/NC": 36.544, "NC/SC": 34.48}

AFDC_URL = ("https://developer.nlr.gov/api/alt-fuel-stations/v1.json?api_key=DEMO_KEY"
            "&fuel_type=ELEC&ev_charging_level=dc_fast&state={state}&status=E&access=public&limit=all")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = ('[out:json][timeout:150];way["highway"="motorway"]["ref"="I 95"]'
                  '(33.9,-80.2,37.7,-77.0);out geom;')


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for st in ("VA", "NC", "SC"):
        subprocess.run(["curl", "-s", "-m", "120", AFDC_URL.format(state=st),
                        "-o", str(raw / f"afdc_{st}.json")], check=True)
    subprocess.run(["curl", "-s", "-m", "180", "--data-urlencode", f"data={OVERPASS_QUERY}",
                    OVERPASS_URL, "-o", str(raw / "ways.json")], check=True)
    (raw / "retrieved.txt").write_text(time.strftime("%Y-%m-%d") + "\n")


def _xy(lat, lon):
    r = 6371.0
    return np.c_[np.radians(lon) * r * np.cos(np.radians(35.5)), np.radians(lat) * r]


def build(raw: Path) -> dict:
    ways = json.loads((raw / "ways.json").read_text())["elements"]
    pts = np.array([(g["lat"], g["lon"]) for w in ways for g in w["geometry"]])
    # Centerline: both carriageways averaged in 0.01-degree latitude bins (I-95 runs N-S here)
    bins = np.round(pts[:, 0] / 0.01).astype(int)
    cl = np.array([pts[bins == b].mean(0) for b in sorted(set(bins), reverse=True)])
    p = _xy(cl[:, 0], cl[:, 1])
    chain = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    lines = {k: chain[np.argmin(abs(cl[:, 0] - lat))] for k, lat in STATE_LINES_LAT.items()}
    origin, length = lines["VA/NC"], lines["NC/SC"] - lines["VA/NC"]

    recs = [s for st in ("VA", "NC", "SC")
            for s in json.loads((raw / f"afdc_{st}.json").read_text())["fuel_stations"]]
    sxy = _xy(np.array([s["latitude"] for s in recs]), np.array([s["longitude"] for s in recs]))
    offset, _ = cKDTree(_xy(pts[:, 0], pts[:, 1])).query(sxy)
    _, near = cKDTree(p).query(sxy)

    units = []   # (chainage_km, network, rated_kw, afdc_id)
    for s, off, i in zip(recs, offset, near):
        if s["state"] != "NC" or off > MAX_OFFSET_KM:
            continue
        for u in s.get("ev_charging_units") or []:
            ccs = u["connectors"].get("J1772COMBO") or {}
            if ccs.get("port_count"):
                units.append((float(chain[i] - origin), s["ev_network"], ccs["power_kw"], s["id"]))

    sites = []
    for km, net, kw, sid in sorted(units):
        site = next((x for x in sites if x["network"] == net and abs(x["km"] - km) <= MERGE_KM), None)
        if site is None:
            site = {"km": km, "network": net, "rated_kw": [], "afdc_ids": []}
            sites.append(site)
        site["rated_kw"].append(kw)
        if sid not in site["afdc_ids"]:
            site["afdc_ids"].append(sid)
    for x in sites:
        x["n_chargers"] = len(x["rated_kw"])
        x["power_kw"] = float(np.mean([min(k, VEHICLE_CAP_KW) for k in x["rated_kw"]]))
        x["km"] = round(x["km"], 1)

    retrieved = (raw / "retrieved.txt").read_text().strip() if (raw / "retrieved.txt").exists() else None
    return {
        "name": "i95_nc",
        "description": "I-95 southbound, VA/NC to NC/SC state line; public CCS DC fast charging",
        "length_km": round(float(length), 1),
        "retrieved": retrieved,
        "sources": {"stations": "AFDC station API (developer.nlr.gov)", "geometry": "OpenStreetMap via Overpass"},
        "vehicle_cap_kw": VEHICLE_CAP_KW,
        "sites": sites,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    if args.download:
        download(raw)
    data = build(raw)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{data['name']}: {data['length_km']} km, {len(data['sites'])} sites, "
          f"{sum(s['n_chargers'] for s in data['sites'])} chargers -> {OUT}")
    for s in data["sites"]:
        print(f"  {s['km']:6.1f} km  {s['network']:<22} {s['n_chargers']} x {s['power_kw']:.0f} kW  "
              f"(rated {s['rated_kw']})")
