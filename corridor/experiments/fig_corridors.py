"""
Figure of the three corridors (paper: fig_corridors.pdf).

  (a), (b)  maps of the two real corridors: OSM motorway geometry, state outlines (us-atlas,
            Census cartographic boundaries), and the CCS sites at their AFDC coordinates
  (c)       the three corridors as lines along the chainage used in the simulations

Marker area is proportional to the number of chargers; marker shape and color give the
effective power class. Inputs are the raw downloads of the build scripts:

  python -m corridor.experiments.fig_corridors --i95_raw <dir> --i15_raw <dir> --states <states-10m.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from corridor.layouts import get_layout

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
})
OUT = Path("paper/figures/fig_corridors.pdf")

# power classes: (upper bound kW, label, marker, color) -- Okabe-Ito
CLASSES = [(65, "30–60 kW", "v", "#D55E00"), (130, "100–125 kW", "s", "#E69F00"),
           (999, "140–150 kW", "o", "#0072B2")]


def pclass(kw):
    return next(c for c in CLASSES if kw <= c[0])


def size(n):
    return 9.0 * n          # marker area (pt^2) proportional to chargers


def topo_states(path, fips):
    """Polygons (lon, lat arrays) of the states with the given FIPS codes from a us-atlas TopoJSON."""
    t = json.loads(Path(path).read_text())
    sx, sy = t["transform"]["scale"]
    tx, ty = t["transform"]["translate"]
    arcs = []
    for a in t["arcs"]:
        q = np.cumsum(np.array(a, float), axis=0)
        arcs.append(np.c_[q[:, 0] * sx + tx, q[:, 1] * sy + ty])
    def ring(idx):
        pts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in idx]
        return np.vstack(pts)
    out = []
    for g in t["objects"]["states"]["geometries"]:
        if g["id"] not in fips:
            continue
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        for p in polys:
            for r in p:
                out.append(ring(r))
    return out


def motorway(raw, refs):
    ways = json.loads((Path(raw) / "ways.json").read_text())["elements"]
    lines = []
    for w in ways:
        wr = {r.strip() for r in w["tags"].get("ref", "").split(";")}
        if wr & refs:
            lines.append(np.array([(g["lon"], g["lat"]) for g in w["geometry"]]))
    return lines


def site_coords(raw, data_file):
    recs = {}
    for f in Path(raw).glob("afdc_*.json"):
        for x in json.loads(f.read_text())["fuel_stations"]:
            recs[x["id"]] = (x["longitude"], x["latitude"])
    d = json.loads(Path(data_file).read_text())
    out = []
    for s in d["sites"]:
        xy = [recs[i] for i in s["afdc_ids"] if i in recs]
        if len(xy) != len(s["afdc_ids"]):
            raise SystemExit(f"AFDC ids of the site at {s['km']} km missing in {raw}")
        out.append((*np.mean(xy, axis=0), s["n_chargers"], s["power_kw"]))
    return out


def draw_sites(ax, sites, zorder=3):
    for x, y, n, kw in sites:
        _, _, m, c = pclass(kw)
        ax.scatter([x], [y], s=size(n), marker=m, facecolor=c, edgecolor="black",
                   linewidth=0.4, alpha=0.9, zorder=zorder)


def map_panel(ax, states, lines, sites, extent, labels, title):
    for r in states:
        ax.fill(r[:, 0], r[:, 1], facecolor="#F2F2F2", edgecolor="#9A9A9A", linewidth=0.5, zorder=0)
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color="#444444", linewidth=0.9, zorder=1)
    # large sites drawn first so that small ones stay visible
    draw_sites(ax, sorted(sites, key=lambda s: -s[2]))
    for lon, lat, text, ha in labels:
        ax.text(lon, lat, text, ha=ha, va="center", fontsize=6.5, color="#333333")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(extent[2:]))))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.set_title(title, loc="left")


def strip_panel(ax):
    rows = [("Synthetic", get_layout("synthetic")), ("I-95 NC", get_layout("i95_nc")),
            ("I-15 LV–Barstow", get_layout("i15_lv"))]
    for k, (name, L) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([0, L.length_km], [y, y], color="#444444", linewidth=0.9, zorder=1)
        draw_sites(ax, [(p, y, c, pw) for p, c, pw in sorted(L.stations, key=lambda s: -s[1])])
        ax.text(-6, y, name, ha="right", va="center", fontsize=7)
        ax.text(306, y, f"{L.length_km:.0f} km, {len(L.stations)} sites, {L.n_chargers} chargers",
                ha="left", va="center", fontsize=6.5, color="#333333", clip_on=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 300)
    ax.set_yticks([])
    ax.set_xlabel("Position along the corridor (km)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("(c) Station layouts used in the simulations", loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i95_raw", required=True)
    ap.add_argument("--i15_raw", required=True)
    ap.add_argument("--states", required=True)
    a = ap.parse_args()

    fig = plt.figure(figsize=(7.16, 3.9), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.55, 1])
    axa, axb = top.subplots(1, 2, width_ratios=[1.25, 1])
    map_panel(axa, topo_states(a.states, {"37", "51", "45"}), motorway(a.i95_raw, {"I 95"}),
              site_coords(a.i95_raw, "corridor/data/i95_nc_stations.json"),
              (-80.6, -76.4, 34.2, 36.9),
              [(-78.0, 36.72, "VA", "center"), (-79.2, 34.45, "SC", "center"), (-79.4, 35.75, "NC", "center")],
              "(a) I-95 southbound, North Carolina (293 km)")
    map_panel(axb, topo_states(a.states, {"06", "32"}), motorway(a.i15_raw, {"I 15"}),
              site_coords(a.i15_raw, "corridor/data/i15_lv_stations.json"),
              (-117.5, -114.7, 34.7, 36.4),
              [(-115.05, 36.17, "Las Vegas", "left"), (-117.05, 34.80, "Barstow", "center"),
               (-115.4, 35.2, "CA", "center"), (-115.0, 35.75, "NV", "center")],
              "(b) I-15 southbound, Las Vegas–Barstow (236 km)")
    strip_panel(bottom.subplots())

    handles = [Line2D([], [], marker=m, linestyle="", markerfacecolor=c, markeredgecolor="black",
                      markeredgewidth=0.4, markersize=6, label=lab) for _, lab, m, c in CLASSES]
    handles += [plt.scatter([], [], s=size(n), marker="o", facecolor="white", edgecolor="black",
                            linewidth=0.4, label=f"{n} charger{'s' if n > 1 else ''}") for n in (1, 4, 12)]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False,
               title="Effective power (shape, color) and number of chargers (area)", title_fontsize=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
