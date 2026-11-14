"""
Build the second real-corridor case study: I-15 southbound from Las Vegas to Barstow.

Sources
  - Road geometry: OpenStreetMap (Overpass API), motorway ways with ref "I 15", plus I-215
    and I-40 to locate the two end interchanges.
  - Stations: U.S. DOE Alternative Fuels Data Center (AFDC) station API,
    public, open, DC fast, states NV/CA.

Definitions (as in build_i95_nc.py)
  - Chainage 0 = the I-15/I-215 interchange south of Las Vegas; the corridor ends at the
    I-15/I-40 junction in Barstow.
  - A site belongs to the corridor if it lies within 2 km of an I-15 carriageway between the two ends.
  - Only CCS (J1772COMBO) charging units of at least 20 kW are counted; Tesla-only sites are excluded.
  - One charging unit = one charger; records of the same network within 1 km form one site.
  - Effective session power = min(rated unit power, VEHICLE_CAP_KW), averaged over the units.

Run (downloads into --raw_dir, writes corridor/data/i15_lv_stations.json):
  python -m corridor.data.build_i15 --raw_dir <dir> [--download]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "i15_lv_stations.json"
VEHICLE_CAP_KW = 150.0
MAX_OFFSET_KM = 2.0
MERGE_KM = 1.0
BIN_KM = 0.5
MIN_KW = 20.0          # units listed as DC fast below this power are not fast chargers
STATES = ("NV", "CA")

AFDC_URL = ("https://developer.nlr.gov/api/alt-fuel-stations/v1.json?api_key=DEMO_KEY"
            "&fuel_type=ELEC&ev_charging_level=dc_fast&state={state}&status=E&access=public&limit=all")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = ('[out:json][timeout:180];('
                  'way["highway"="motorway"]["ref"~"I 15"](34.7,-117.3,36.2,-114.9);'
                  'way["highway"="motorway"]["ref"~"I 40"](34.7,-117.3,35.0,-116.6);'
                  'way["highway"="motorway"]["ref"~"I 215"](35.95,-115.35,36.15,-115.0););out geom;')


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for st in STATES:
        subprocess.run(["curl", "-s", "-m", "180", AFDC_URL.format(state=st),
                        "-o", str(raw / f"afdc_{st}.json")], check=True)
    subprocess.run(["curl", "-s", "-m", "280", "-A", "corridor-charging-research/1.0",
                    "--data-urlencode", f"data={OVERPASS_QUERY}", OVERPASS_URL,
                    "-o", str(raw / "ways.json")], check=True)
    (raw / "retrieved.txt").write_text(time.strftime("%Y-%m-%d") + "\n")


def _xy(lat, lon):
    r = 6371.0
    return np.c_[np.radians(lon) * r * np.cos(np.radians(35.5)), np.radians(lat) * r]


def _refs(way) -> set:
    return {r.strip() for r in way["tags"].get("ref", "").split(";")}


def build(raw: Path) -> dict:
    ways = json.loads((raw / "ways.json").read_text())["elements"]
    geo = lambda sel: np.array([(g["lat"], g["lon"]) for w in ways if sel(w) for g in w["geometry"]])
    pts = geo(lambda w: "I 15" in _refs(w))        # excludes "I 15 Truck"
    i215, i40 = geo(lambda w: "I 215" in _refs(w)), geo(lambda w: "I 40" in _refs(w))
    xy = _xy(pts[:, 0], pts[:, 1])
    tree = cKDTree(xy)
    def junction(other):   # the I-15 point closest to another motorway
        d, idx = tree.query(_xy(other[:, 0], other[:, 1]))
        return xy[idx[d.argmin()]]
    start, end = junction(i215), junction(i40)
    # Centerline: both carriageways averaged in 0.5 km bins along the start-end axis
    axis = (end - start) / np.linalg.norm(end - start)
    s = (xy - start) @ axis
    keep = (s >= 0) & (s <= (end - start) @ axis)
    bins = np.floor(s[keep] / BIN_KM).astype(int)
    p = np.array([xy[keep][bins == b].mean(0) for b in sorted(set(bins))])
    p = np.vstack([start, p, end])
    chain = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    length = chain[-1]

    recs = [x for st in STATES for x in json.loads((raw / f"afdc_{st}.json").read_text())["fuel_stations"]]
    sxy = _xy(np.array([x["latitude"] for x in recs]), np.array([x["longitude"] for x in recs]))
    offset, _ = tree.query(sxy)
    _, near = cKDTree(p).query(sxy)

    units = []   # (chainage_km, network, rated_kw, afdc_id)
    for x, off, i in zip(recs, offset, near):
        if off > MAX_OFFSET_KM or not (0 < i < len(p) - 1):   # outside the two end interchanges
            continue
        for u in x.get("ev_charging_units") or []:
            ccs = u["connectors"].get("J1772COMBO") or {}
            if ccs.get("port_count") and (ccs.get("power_kw") or 0) >= MIN_KW:
                units.append((float(chain[i]), x["ev_network"], ccs["power_kw"], x["id"]))

    sites = []
    for km, net, kw, sid in sorted(units):
        site = next((y for y in sites if y["network"] == net and abs(y["km"] - km) <= MERGE_KM), None)
        if site is None:
            site = {"km": km, "network": net, "rated_kw": [], "afdc_ids": []}
            sites.append(site)
        site["rated_kw"].append(kw)
        if sid not in site["afdc_ids"]:
            site["afdc_ids"].append(sid)
    for y in sites:
        y["n_chargers"] = len(y["rated_kw"])
        y["power_kw"] = float(np.mean([min(k, VEHICLE_CAP_KW) for k in y["rated_kw"]]))
        y["km"] = round(y["km"], 1)

    retrieved = (raw / "retrieved.txt").read_text().strip() if (raw / "retrieved.txt").exists() else None
    return {
        "name": "i15_lv",
        "description": "I-15 southbound, I-215 interchange (Las Vegas) to I-40 junction (Barstow); "
                       "public CCS DC fast charging",
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
