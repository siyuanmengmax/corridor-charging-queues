"""
Numbers of the real-corridor case studies (I-95 and I-15: main comparison, information steps,
request timing, look-ahead, compliance), from the runs with US vehicle parameters
(run_i95_us.sh, run_i15_us.sh). Run from the repository root:

  python corridor/experiments/summarize_i95.py [i95us|i15us]
"""
import glob
import sys
import json

import numpy as np
from scipy import stats

from corridor.layouts import get_layout


def latest(pat):
    f = sorted(glob.glob(f"corridor/results/{pat}"))[-1]
    print("#", f)
    return json.load(open(f))


TAG = sys.argv[1] if len(sys.argv) > 1 else "i95us"
MAIN = latest(f"{TAG}_main_*.json")
PORT = latest(f"{TAG}_portstatus_*.json")
E10 = {c: latest(f"{TAG}_e10_c{c}_*.json") for c in ("0.75", "0.5", "0.25")}
SUMO, P1 = "phase2_sumo", "phase1"
RATES = tuple(sorted({r["rate"] for r in MAIN}))
THR = (0.2, 0.3, 0.4)
LO, MID, HI = RATES
KM = {f"cs_{i}": (p, c, pw) for i, (p, c, pw) in enumerate(get_layout(MAIN[0]["layout"]).stations)}


def sel(R, **k):
    return [r for r in R if all(r[a] == b for a, b in k.items())]


def ci(x):
    x = np.array(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def mean(R, key="mean_wait_min", **k):
    return np.mean([r[key] for r in sel(R, **k)])


def paired(R, m, ref, key="mean_wait_min", R2=None, **k):
    a = {r["seed"]: r[key] for r in sel(R, method=m, **k)}
    b = {r["seed"]: r[key] for r in sel(R2 or R, method=ref, **k)}
    mu, h = ci([a[s] - b[s] for s in a if s in b])
    return mu, h, abs(mu) > h


def fmt(t):
    return f"{t[0]:+.1f} ± {t[1]:.1f}{' *' if t[2] else ''}"


F30, F0, BG1, BG3, SC, GR = "Fluid-MPC H=30", "Fluid-MPC H=0", "Backlog-Greedy", "Backlog-Greedy g=3", "StaticCap", "Greedy"
OCC, P120, PW = "Occupancy-Greedy g=3", "Occupancy-Greedy g=3 P120", "Power-Greedy"
PS, PSQ = "PortStatus-Greedy g=3", "PortStatus+Queue-Greedy g=3"

print("\n## Demand: requests per entering vehicle, offered load (SUMO, theta 0.30)")
for rate in RATES:
    x = sel(MAIN, env=SUMO, rate=rate, threshold=0.3, method=F30)
    print(f"  {rate}: requests/episode {np.mean([r['n_requests'] for r in x]):.0f}  "
          f"requests per entering veh (measured 240 min) {np.mean([r['n_requests'] for r in x]) / (rate * 240):.2f}  "
          f"load F30 {mean(MAIN, 'offered_load', env=SUMO, rate=rate, threshold=0.3, method=F30):.2f}  "
          f"SC {mean(MAIN, 'offered_load', env=SUMO, rate=rate, threshold=0.3, method=SC):.2f}")

print("\n## Table i95res (SUMO, theta 0.30)")
rows = [("mean_wait_min", [GR, SC, BG1, BG3, F0, F30]), ("p95_wait_min", [SC, BG3, F30]),
        ("max_wait_min", [SC, BG3, F30]), ("station_mean_min", [SC, BG3, F30])]
for key, ms in rows:
    for m in ms:
        print(f"  {key:<17}{m:<20}" + "".join(f"{mean(MAIN, key, env=SUMO, rate=r, threshold=0.3, method=m):9.1f}"
                                              for r in RATES))
    if key in ("mean_wait_min", "station_mean_min"):
        print(f"  {key:<17}{'diff F30-SC':<20}" + "".join(
            f"  {fmt(paired(MAIN, F30, SC, key, env=SUMO, rate=r, threshold=0.3)):>14}" for r in RATES))

print("\n## Time at station F30-SC: significant in how many of 18 settings (2 sims x 3 rates x 3 thr)")
n = sum(paired(MAIN, F30, SC, "station_mean_min", env=e, rate=r, threshold=t)[2] for e in (P1, SUMO) for r in RATES for t in THR)
print("  ", n, "of 18;  ranges:", [fmt(paired(MAIN, F30, SC, "station_mean_min", env=e, rate=r, threshold=t))
                                   for e in (P1, SUMO) for r in RATES for t in THR])

print("\n## Nearest station: mean wait per rate (SUMO / queue sim), theta 0.30; worst station mean wait in the queue sim at 1.0")
for rate in RATES:
    print(f"  {rate}: SUMO {mean(MAIN, env=SUMO, rate=rate, threshold=0.3, method=GR):.1f}  "
          f"P1 {mean(MAIN, env=P1, rate=rate, threshold=0.3, method=GR):.1f}  "
          f"min over thr (SUMO) {min(mean(MAIN, env=SUMO, rate=rate, threshold=t, method=GR) for t in THR):.1f}")
for r in sel(MAIN, env=P1, rate=HI, threshold=0.3, method=GR):
    bad = [(KM[s][0], KM[s][2], round(v["wait_mean"])) for s, v in r.get("by_station", {}).items() if v["wait_mean"] > 800]
    if bad:
        print("   seed", r["seed"], bad)

print("\n## Per-station breakdown, SUMO, 1.0 veh/min, theta 0.30, 20 seeds: vehicles per episode / mean wait")
for m in (SC, BG3, F30):
    x = sel(MAIN, env=SUMO, rate=HI, threshold=0.3, method=m)
    for s, (km, c, pw) in KM.items():
        if pw <= 60:
            ns = [r["by_station"].get(s, {"n": 0})["n"] for r in x]
            ws = [r["by_station"][s]["wait_mean"] for r in x if s in r["by_station"] and r["by_station"][s]["n"]]
            print(f"  {m:<20} {km:6.1f} km {pw:4.0f} kW: {np.mean(ns):5.1f} veh/ep, "
                  f"mean wait {np.mean(ws) if ws else float('nan'):6.1f} (over {len(ws)} episodes with use)")

print("\n## Gamma and look-ahead")
print("  BG1 vs BG3 SUMO 1.0 th.3:", round(mean(MAIN, env=SUMO, rate=HI, threshold=0.3, method=BG1), 1),
      round(mean(MAIN, env=SUMO, rate=HI, threshold=0.3, method=BG3), 1))
for e in (SUMO, P1):
    for r in RATES:
        print(f"  {e:<12}{r}: F30-BG3 " + "  ".join(f"th{t}: {fmt(paired(MAIN, F30, BG3, env=e, rate=r, threshold=t))}" for t in THR)
              + "   F0-BG3 " + "  ".join(f"th{t}: {fmt(paired(MAIN, F0, BG3, env=e, rate=r, threshold=t))}" for t in THR))

print("\n## Request threshold: F30 mean wait (SUMO) and offered load by theta")
for r in RATES:
    print(f"  {r}: " + "  ".join(f"th{t}: {mean(MAIN, env=SUMO, rate=r, threshold=t, method=F30):.1f} "
                                 f"(load {mean(MAIN, 'offered_load', env=SUMO, rate=r, threshold=t, method=F30):.2f})" for t in THR))

print("\n## Table attr, I-95 columns (theta 0.30): SUMO 0.8 | SUMO 1.0 | queue sim 1.0")
cols = [(SUMO, MID), (SUMO, HI), (P1, HI)]
for m in (SC, PW, PS, PSQ, P120, OCC, BG3, F30):
    R = PORT if m in (PS, PSQ) else MAIN
    print(f"  {m:<28}" + "".join(f"{mean(R, env=e, rate=r, threshold=0.3, method=m):8.1f}" for e, r in cols))
for lab, m, ref in [("queue information", P120, SC), ("station power", OCC, P120),
                    ("en-route vehicles", BG3, OCC), ("joint + look-ahead", F30, BG3)]:
    print(f"  {lab:<28}" + "".join(f"  {fmt(paired(MAIN, m, ref, env=e, rate=r, threshold=0.3)):>15}" for e, r in cols))
for e, r in cols:
    q = paired(MAIN, P120, SC, env=e, rate=r, threshold=0.3)[0]
    tot = paired(MAIN, F30, SC, env=e, rate=r, threshold=0.3)[0]
    print(f"  share of queue step in total ({e},{r}): {q / tot:.0%}  (total {tot:+.1f})")
print("  PortStatus+Queue minus queue greedy (PORT file, same run):")
for e, r in cols:
    print(f"    {e},{r}: {fmt(paired(PORT, PSQ, OCC, env=e, rate=r, threshold=0.3))}   "
          f"PortStatus minus StaticCap: {mean(PORT, env=e, rate=r, threshold=0.3, method=PS) - mean(MAIN, env=e, rate=r, threshold=0.3, method=SC):+.1f}")

print("\n## En-route step (BG3 - OCC) and power step (OCC - P120) over 18 settings")
for lab, m, ref in [("en-route", BG3, OCC), ("power", OCC, P120)]:
    t = [paired(MAIN, m, ref, env=e, rate=r, threshold=th) for e in (P1, SUMO) for r in RATES for th in THR]
    print(f"  {lab}: significant {sum(x[2] for x in t)} of 18, range {min(x[0] for x in t):+.1f} to {max(x[0] for x in t):+.1f}")

print("\n## E10 compliance (theta 0.30, 1.0 veh/min): mean wait SUMO; F30 - SC paired (SUMO, queue sim)")
print(f"  c=1.00 (main run): F30 {mean(MAIN, env=SUMO, rate=HI, threshold=0.3, method=F30):.1f}  "
      f"SC {mean(MAIN, env=SUMO, rate=HI, threshold=0.3, method=SC):.1f}")
for c, R in E10.items():
    print(f"  c={c}: " + "  ".join(f"{m} {mean(R, env=SUMO, method=m):.1f}" for m in (SC, OCC, BG3, F30))
          + f"   F30-SC SUMO {fmt(paired(R, F30, SC, env=SUMO))}  P1 {fmt(paired(R, F30, SC, env=P1))}")
