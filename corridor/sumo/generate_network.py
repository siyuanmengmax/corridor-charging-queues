"""
Generate SUMO network files for the intercity corridor Phase 2 validation.

Network design:
  - One-directional two-lane highway, split into sections of up to 60 km
  - Stations from a corridor layout (corridor/layouts.py), modeled as
    parkingAreas on the outer lane
  - Vehicles enter at km 0 and leave beyond the corridor end

Files generated (synthetic layout: in this directory; other layouts: net/<name>/):
  corridor.nod.xml  - nodes
  corridor.edg.xml  - edges
  corridor.net.xml  - compiled SUMO network (via netconvert)
  corridor.add.xml  - parkingArea + chargingStation definitions
  corridor.rou.xml  - vehicle types and routes
  corridor.sumocfg  - simulation config

Run:
  python -m corridor.sumo.generate_network
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

OUT_DIR = Path(__file__).parent
HIGHWAY_SPEED_MS = 27.78    # 100 km/h
N_HIGHWAY_LANES = 2
SECTION_KM = 60.0

SIM_DURATION_S = 14_400  # 240 minutes — matches Phase 1
DELTA_T_S = 300          # 5-minute control step
STEP_LENGTH_S = 1        # SUMO step granularity

# EV fleet parameters — matches Phase 1 heavy scenario
BATTERY_KWH = 70.0
CONSUMPTION_KWH_PER_KM = 0.20
VEHICLE_SPEED_MS = 27.78   # 100 km/h

# Demand scenarios — same parameters as Phase 1 (run_gamma_sweep.DEMAND_SCENARIOS)
SCENARIOS = {
    "light":  dict(n_vehicles=60,  demand_rate=0.4, soc_init_mean=0.55),
    "medium": dict(n_vehicles=90,  demand_rate=0.6, soc_init_mean=0.45),
    "heavy":  dict(n_vehicles=120, demand_rate=0.8, soc_init_mean=0.35),
}

ROUTE_DIR = OUT_DIR / "routes"


def _layout(layout):
    from ..layouts import get_layout
    return get_layout(layout) if isinstance(layout, str) else layout


def net_dir(layout="synthetic") -> Path:
    name = _layout(layout).name
    return OUT_DIR if name == "synthetic" else OUT_DIR / "net" / name


def sumocfg_path(layout="synthetic") -> Path:
    return net_dir(layout) / "corridor.sumocfg"


def section_boundaries_m(layout="synthetic") -> list:
    """Section ends every SECTION_KM, moved off any station so each station lies inside an edge."""
    lay = _layout(layout)
    length_m = lay.length_km * 1000.0
    n = max(1, int(round(lay.length_km / SECTION_KM)))
    bounds = [length_m * i / n for i in range(n + 1)]
    st_m = [p * 1000.0 for p, _, _ in lay.stations]
    for i in range(1, n + 1):
        while any(-200.0 <= x - bounds[i] <= 200.0 for x in st_m):
            bounds[i] += 500.0
    return [round(b) for b in bounds]


def station_list(layout="synthetic") -> list:
    """(id, position_m, n_chargers, power_kw)"""
    return [(f"cs_{i}", round(p * 1000.0), c, pw) for i, (p, c, pw) in enumerate(_layout(layout).stations)]


def route_file(n_vehicles: int, demand_rate: float, soc_init_mean: float,
               seed: int, duration_s: int, layout="synthetic") -> Path:
    name = f"n{n_vehicles}_r{demand_rate:g}_soc{soc_init_mean:g}_T{duration_s}_seed{seed}.rou.xml"
    lay = _layout(layout).name
    return ROUTE_DIR / name if lay == "synthetic" else ROUTE_DIR / lay / name


def write_nodes(layout="synthetic") -> Path:
    out = net_dir(layout) / "corridor.nod.xml"
    bounds = section_boundaries_m(layout)
    nodes = []
    for i, x_m in enumerate(bounds):
        nodes.append(f'    <node id="n{i}" x="{x_m:.0f}" y="0" type="priority"/>')
    # Source and sink nodes (slightly before/after corridor)
    nodes.insert(0, '    <node id="n_source" x="-500" y="0" type="priority"/>')
    nodes.append(f'    <node id="n_sink" x="{bounds[-1] + 500}" y="0" type="priority"/>')

    content = "<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n"
    out.write_text(content)
    print(f"  Written: {out}")
    return out


def write_edges(layout="synthetic") -> Path:
    out = net_dir(layout) / "corridor.edg.xml"
    edges = []

    # Entry edge (source → n0)
    edges.append(
        f'    <edge id="e_entry" from="n_source" to="n0" '
        f'numLanes="{N_HIGHWAY_LANES}" speed="{HIGHWAY_SPEED_MS:.2f}" priority="1"/>'
    )

    # Highway sections between station junctions
    boundaries = section_boundaries_m(layout)
    for i in range(len(boundaries) - 1):
        edges.append(
            f'    <edge id="e_{i}" from="n{i}" to="n{i+1}" '
            f'numLanes="{N_HIGHWAY_LANES}" speed="{HIGHWAY_SPEED_MS:.2f}" priority="1"/>'
        )

    # Exit edge (last node → sink)
    edges.append(
        f'    <edge id="e_exit" from="n{len(boundaries)-1}" to="n_sink" '
        f'numLanes="{N_HIGHWAY_LANES}" speed="{HIGHWAY_SPEED_MS:.2f}" priority="1"/>'
    )

    content = "<edges>\n" + "\n".join(edges) + "\n</edges>\n"
    out.write_text(content)
    print(f"  Written: {out}")
    return out


def build_net(nod: Path, edg: Path) -> Path:
    out = nod.parent / "corridor.net.xml"
    cmd = [
        "netconvert",
        "--node-files", str(nod),
        "--edge-files", str(edg),
        "--output-file", str(out),
        "--no-internal-links",
        "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("netconvert ERROR:", result.stderr[:500])
        raise RuntimeError("netconvert failed")
    print(f"  Built:   {out}")
    return out


def write_additional(layout="synthetic") -> Path:
    out = net_dir(layout) / "corridor.add.xml"
    lines = ["<additional>"]

    # Map station position → which edge it falls on
    bounds = section_boundaries_m(layout)
    section_starts = bounds[:-1]
    section_ends = bounds[1:]

    for sid, pos_m, n_chargers, power_kw in station_list(layout):
        # Find which section contains this position
        edge_id = None
        for i, (s, e) in enumerate(zip(section_starts, section_ends)):
            if s < pos_m <= e:
                edge_id = f"e_{i}"
                local_pos = pos_m - s
                break
        if edge_id is None:
            raise ValueError(f"Station {sid} at {pos_m}m not on any edge")

        # parkingArea on the shoulder (lane 1 = outer lane)
        pa_start = local_pos - 50
        pa_end = local_pos + 50
        lines.append(
            f'    <!-- Station {sid}: {pos_m/1000:.0f} km, '
            f'{n_chargers} chargers @ {power_kw:.0f} kW -->'
        )
        lines.append(
            f'    <parkingArea id="pa_{sid}" '
            f'lane="{edge_id}_1" '
            f'startPos="{pa_start:.0f}" endPos="{pa_end:.0f}" '
            f'roadsideCapacity="{n_chargers}" '
            f'name="Charging Station {sid}"/>'
        )
        # ChargingStation collocated with parkingArea
        lines.append(
            f'    <chargingStation id="{sid}" '
            f'lane="{edge_id}_1" '
            f'startPos="{pa_start:.0f}" endPos="{pa_end:.0f}" '
            f'power="{int(power_kw * 1000)}" '  # SUMO uses Watts
            f'efficiency="0.90" chargeInTransit="0"/>'
        )

    lines.append("</additional>")
    out.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out}")
    return out


def write_routes(n_vehicles: int, demand_rate: float, soc_init_mean: float,
                 seed: int = 0, duration_s: int = SIM_DURATION_S, layout="synthetic") -> Path:
    """
    Write the route file of one demand realization (cached by parameters and seed).
    Same demand process as Phase 1: 60% of n_vehicles pre-placed over the first
    80% of the corridor, then Poisson entries at km 0 at demand_rate per minute.
    """
    out = route_file(n_vehicles, demand_rate, soc_init_mean, seed, duration_s, layout)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    bounds = section_boundaries_m(layout)
    length_m = _layout(layout).length_km * 1000.0
    section_edges = [f"e_{i}" for i in range(len(bounds) - 1)]
    full_route_edges = " ".join(["e_entry"] + section_edges + ["e_exit"])

    import random
    rng = random.Random(seed)

    def attrs():
        soc = max(0.05, min(0.99, rng.gauss(soc_init_mean, 0.15)))
        battery_kwh = max(30.0, rng.gauss(BATTERY_KWH, 10.0))
        speed = max(20.0, min(VEHICLE_SPEED_MS, rng.gauss(VEHICLE_SPEED_MS, 2.5)))
        return soc, battery_kwh, speed

    routes = [f'    <route id="r_full" edges="{full_route_edges}"/>']
    for k in range(len(section_edges)):
        routes.append(f'    <route id="r_from_{k}" edges="{" ".join(section_edges[k:] + ["e_exit"])}"/>')

    def vehicle(idx, route, depart, pos, soc, battery_kwh, speed):
        return (
            f'    <vehicle id="ev_{idx:04d}" type="ev" route="{route}" '
            f'depart="{depart}" departPos="{pos:.0f}" departSpeed="{speed:.2f}" departLane="best">'
            f'<param key="soc" value="{soc:.3f}"/>'
            f'<param key="battery_kwh" value="{battery_kwh:.1f}"/>'
            f'</vehicle>'
        )

    vehicles = []
    n_initial = max(1, int(n_vehicles * 0.6))
    for i in range(n_initial):
        pos_m = rng.uniform(0, length_m * 0.8)
        k = max(i for i, b in enumerate(bounds[:-1]) if b <= pos_m)
        vehicles.append(vehicle(i, f"r_from_{k}", 0, pos_m - bounds[k], *attrs()))

    t, idx = 0.0, n_initial
    rate_per_s = demand_rate / 60.0
    while True:
        t += rng.expovariate(rate_per_s)
        if t >= duration_s:
            break
        vehicles.append(vehicle(idx, "r_full", int(t), 0, *attrs()))
        idx += 1

    lines = [
        "<routes>",
        f'    <vType id="ev" accel="2.0" decel="4.5" sigma="0.5" '
        f'length="4.5" maxSpeed="{VEHICLE_SPEED_MS:.2f}" color="0,0.8,0"/>',
        "",
        *routes,
        "",
        *vehicles,
        "</routes>",
    ]
    out.write_text("\n".join(lines) + "\n")
    return out


def write_sumocfg(net: Path, add: Path, rou: Path) -> Path:
    out = net.parent / "corridor.sumocfg"
    content = textwrap.dedent(f"""\
        <configuration>
            <input>
                <net-file value="{net.name}"/>
                <route-files value="{os.path.relpath(rou, net.parent)}"/>
                <additional-files value="{add.name}"/>
            </input>
            <time>
                <begin value="0"/>
                <step-length value="{STEP_LENGTH_S}"/>
            </time>
            <processing>
                <ignore-route-errors value="true"/>
                <collision.action value="none"/>
            </processing>
            <report>
                <no-warnings value="true"/>
                <no-step-log value="true"/>
            </report>
        </configuration>
    """)
    out.write_text(content)
    print(f"  Written: {out}")
    return out


def generate_all(layout="synthetic") -> dict:
    lay = _layout(layout)
    print(f"Generating SUMO corridor network files for layout {lay.name}...")
    net_dir(lay).mkdir(parents=True, exist_ok=True)
    nod = write_nodes(lay)
    edg = write_edges(lay)
    net = build_net(nod, edg)
    add = write_additional(lay)
    rou = write_routes(**SCENARIOS["heavy"], seed=0, layout=lay)
    cfg = write_sumocfg(net, add, rou)

    stations = station_list(lay)
    print(f"\nNetwork summary:")
    print(f"  Corridor:   {lay.length_km:.1f} km, sections at {section_boundaries_m(lay)} m")
    print(f"  Stations:   {len(stations)} ({', '.join(f'{p/1000:.1f}' for _, p, _, _ in stations)} km)")
    print(f"  Chargers:   {lay.n_chargers} total")
    print(f"\nRun simulation with:")
    print(f"  sumo -c {cfg}")
    return {"net": net, "add": add, "rou": rou, "cfg": cfg, "stations": stations}


def ensure_network(layout="synthetic") -> Path:
    """Generate the network of a layout once; returns its sumocfg."""
    cfg = sumocfg_path(layout)
    if not cfg.exists():
        generate_all(layout)
    return cfg


if __name__ == "__main__":
    import sys
    generate_all(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
