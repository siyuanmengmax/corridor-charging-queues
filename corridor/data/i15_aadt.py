"""
Traffic counts for the I-15 case: Caltrans Traffic Census AADT on I-15 in San Bernardino County
(ArcGIS service CHhighway/Traffic_AADT), from the Nevada state line to Barstow.
The layer matched the 2024 Traffic Volumes workbook on 2026-11-13 (e.g., East Baker peak hour 4,400).

  python -m corridor.data.i15_aadt
"""
import json
import subprocess

URL = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/Traffic_AADT/FeatureServer/0/query"

if __name__ == "__main__":
    out = subprocess.run(["curl", "-s", "-m", "90", "-A", "corridor-charging-research/1.0", "-G", URL,
                          "--data-urlencode", "where=RTE='015'", "--data-urlencode", "outFields=*",
                          "--data-urlencode", "returnGeometry=false", "--data-urlencode", "f=json"],
                         capture_output=True, text=True, check=True).stdout
    rows = sorted({(float(a["PM"]), a["DESCRIPTION"], a["BACK_AADT"], a["AHEAD_AADT"], a["BACK_PEAK_HOUR"])
                   for a in (f["attributes"] for f in json.loads(out)["features"]) if a["CNTY"] == "SBD"})
    for pm, desc, back, ahead, peak in rows:
        if pm >= 70:   # Barstow (I-40 junction, PM ~ 70) to the state line (PM 186.2)
            print(f"PM {pm:7.3f}  {desc:<32} AADT back {back:>7} ahead {ahead:>7}  peak hour {peak}")
