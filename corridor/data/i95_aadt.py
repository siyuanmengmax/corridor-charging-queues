"""
I-95 mainline traffic in North Carolina, used to relate the case-study demand levels to
traffic volumes (NCDOT AADT stations, ArcGIS feature service of the Traffic Survey Group).

  python -m corridor.data.i95_aadt [--year 2019]

2019 is the latest year with counts on every mainline segment (2020 is depressed by the
pandemic; 2021 and 2022 cover only a few segments).
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

import numpy as np

URL = ("https://services.arcgis.com/NuWFvHYDMVmmxMeM/ArcGIS/rest/services/"
       "NCDOT_AADT_Traffic_Segmentation/FeatureServer/0/query")
RATES = {"low": 0.6, "moderate": 0.8, "high": 1.0}   # EVs entering southbound per minute
K_FACTOR = 0.09                                       # design-hour share of AADT (assumption)


def mainline(year: int) -> list:
    q = urllib.parse.urlencode({
        "where": "ROUTE='I-95' AND (LOCATION LIKE '%EXIT%' OR LOCATION LIKE '%LINE%')",
        "outFields": f"LOCATION,COUNTY,AADT_{year}", "outSR": 4326, "f": "json"})
    with urllib.request.urlopen(f"{URL}?{q}", timeout=60) as r:
        feats = json.load(r)["features"]
    rows = [(f["geometry"]["y"], f["attributes"]["LOCATION"], f["attributes"][f"AADT_{year}"])
            for f in feats]
    return sorted((r for r in rows if str(r[2]).strip()), key=lambda r: -r[0])   # north to south


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    rows = mainline(args.year)
    aadt = np.array([float(r[2]) for r in rows])
    entry = next(float(r[2]) for r in rows if "VA LINE" in r[1])
    print(f"{len(rows)} mainline segments, AADT {args.year}: entry (VA line) {entry:.0f}, "
          f"mean {aadt.mean():.0f}, min {aadt.min():.0f}, max {aadt.max():.0f} (both directions)")
    daily = entry / 2 / 1440                 # southbound vehicles per minute, daily average
    peak = K_FACTOR * entry / 2 / 60         # southbound vehicles per minute, design hour, 50/50 split
    print(f"southbound at entry: {daily:.1f} veh/min daily average, {peak:.1f} veh/min design hour")
    for name, lam in RATES.items():
        print(f"  {name:9s} {lam:.1f} EV/min = {60 * lam:.0f} EV/h: {100 * lam / daily:.1f}% of daily-average "
              f"flow, {100 * lam / peak:.1f}% of design-hour flow")
