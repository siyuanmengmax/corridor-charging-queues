"""
E11 partial participation (run_participation.sh): mean wait of all vehicles and of participants,
paired differences between the station-data levels. Run from the repository root:

  python corridor/experiments/summarize_participation.py [part|partfp]

part: non-participants go to the nearest reachable station; partfp (run_participation_fp.sh):
they go to the nearest one showing a free charger.
"""
import glob
import json
import sys

import numpy as np
from scipy import stats

LV = ["OwnFleet-own g=3", "OwnFleet-port g=3", "OwnFleet-port_queue g=3", "OwnFleet-station g=3"]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "part"
NAME = {LV[0]: "own only", LV[1]: "+ port", LV[2]: "+ queue count", LV[3]: "+ full station"}


def ci(x):
    x = np.array(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def paired(R, m, ref, key):
    a = {r["seed"]: r[key] for r in R if r["method"] == m}
    b = {r["seed"]: r[key] for r in R if r["method"] == ref}
    mu, h = ci([a[s] - b[s] for s in a if s in b])
    return f"{mu:+6.1f}±{h:4.1f}{'*' if abs(mu) > h else ' '}"


for cor in ("syn", "i95us", "i15us"):
    for p in ("0.75", "0.5", "0.25"):
        R = json.load(open(sorted(glob.glob(f"corridor/results/{PREFIX}_{cor}_p{p}_*.json"))[-1]))
        for env in ("phase2_sumo", "phase1"):
            E = [r for r in R if r["env"] == env]
            print(f"\n{cor} p={p} {env}")
            for key in ("mean_wait_min", "mean_wait_own_min", "mean_wait_other_min"):
                print(f"  {key:<20}" + "".join(f"{NAME[m]:>16}: {np.mean([r[key] for r in E if r['method'] == m]):6.1f}" for m in LV))
            for key in ("mean_wait_min", "mean_wait_own_min"):
                print(f"  {key:<20} port-own {paired(E, LV[1], LV[0], key)}  queue-port {paired(E, LV[2], LV[1], key)}"
                      f"  station-queue {paired(E, LV[3], LV[2], key)}  station-own {paired(E, LV[3], LV[0], key)}")
