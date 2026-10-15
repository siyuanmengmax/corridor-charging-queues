"""
Figures of the TR-D manuscript from the formal result files (run from the repository root).

  python -m corridor.experiments.make_figures [--out paper/figures]

  fig_main.pdf        E1: mean and p95 wait by method and demand, threshold 0.30
  fig_threshold.pdf   E2: mean and p95 wait vs request threshold (SUMO)
  fig_threshold_p1.pdf  E2 in the Phase 1 simulator
  fig_lookahead.pdf   E3: paired change vs backlog greedy by look-ahead length (SUMO, high demand)
  fig_i95.pdf         I-95 case study (when its result file exists)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
SINGLE, DOUBLE = 3.5, 7.2   # inches (Elsevier 90 / 190 mm)

# Display names and colours (Okabe-Ito)
METHODS = {
    "Greedy":             ("Nearest station", "#999999", "o"),
    "StaticCap":          ("Static capacity MILP", "#E69F00", "s"),
    "Backlog-Greedy g=3": ("Backlog greedy", "#56B4E9", "^"),
    "Fluid-MPC H=0":      ("Fluid MPC, H = 0", "#009E73", "v"),
    "Fluid-MPC H=30":     ("Fluid MPC, H = 30 min", "#D55E00", "D"),
}
LOADS = {"low": "Low demand", "moderate": "Moderate demand", "high": "High demand"}
ENVS = {"phase1": "Queue simulator", "phase2_sumo": "SUMO"}


def latest(pattern: str):
    files = sorted(glob.glob(f"corridor/results/{pattern}"))
    return json.load(open(files[-1])) if files else None


def sel(R, **kw):
    return [r for r in R if all(r.get(k) == v for k, v in kw.items())]


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def paired(R, m, ref, key, **kw):
    a = {r["seed"]: r[key] for r in sel(R, method=m, **kw)}
    b = {r["seed"]: r[key] for r in sel(R, method=ref, **kw)}
    return ci([a[s] - b[s] for s in a if s in b])


def fig_main(R, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.6), sharex=True)
    width = 0.16
    for col, env in enumerate(ENVS):
        for row, (key, lab) in enumerate([("mean_wait_min", "Mean wait (min)"), ("p95_wait_min", "95th pct. wait (min)")]):
            ax = axes[row, col]
            for j, (m, (name, c, _)) in enumerate(METHODS.items()):
                mu, h = zip(*[ci([r[key] for r in sel(R, env=env, load=l, threshold=0.30, method=m)]) for l in LOADS])
                ax.bar(np.arange(3) + (j - 2) * width, mu, width, yerr=h, color=c, label=name,
                       error_kw=dict(lw=0.6, capsize=1.5))
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(ENVS[env])
            ax.set_xticks(range(3), ["Low", "Moderate", "High"])
        axes[1, col].set_xlabel("Demand")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_main.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_threshold(R, out: Path, env: str, name: str) -> None:
    thr = sorted({r["threshold"] for r in R})
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.8), sharex=True)
    for col, load in enumerate(LOADS):
        for row, (key, lab) in enumerate([("mean_wait_min", "Mean wait (min)"), ("p95_wait_min", "95th pct. wait (min)")]):
            ax = axes[row, col]
            for m, (mname, c, mk) in METHODS.items():
                if m == "Fluid-MPC H=0":
                    continue
                mu, h = map(np.array, zip(*[ci([r[key] for r in sel(R, env=env, load=load, threshold=t, method=m)])
                                           for t in thr]))
                ax.plot(thr, mu, color=c, marker=mk, ms=3, lw=1, label=mname)
                ax.fill_between(thr, mu - h, mu + h, color=c, alpha=0.15, lw=0)
            if row == 0:
                ax.set_title(LOADS[load])
            else:
                ax.set_xlabel("Request threshold (SOC)")
            if col == 0:
                ax.set_ylabel(lab)
            ax.set_ylim(bottom=0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def fig_lookahead(R, out: Path) -> None:
    ref = "Backlog-Greedy g=3"
    hs = [0, 15, 30, 60]
    thr = [0.25, 0.30, 0.35]
    colors = ["#56B4E9", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    for ax, (key, lab) in zip(axes, [("mean_wait_min", "Change in mean wait (min)"),
                                     ("p95_wait_min", "Change in 95th pct. wait (min)")]):
        for k, (t, c) in enumerate(zip(thr, colors)):
            mu, h = zip(*[paired(R, f"Fluid-MPC H={H}", ref, key, env="phase2_sumo", load="high", threshold=t)
                          for H in hs])
            ax.errorbar(np.arange(4) + (k - 1) * 0.12, mu, yerr=h, color=c, marker="o", ms=3, lw=1,
                        capsize=2, label=f"Threshold {t:.2f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(4), [str(H) for H in hs])
        ax.set_xlabel("Look-ahead length (min)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_lookahead.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_i95(R, out: Path) -> None:
    thr = sorted({r["threshold"] for r in R})
    loads = list(dict.fromkeys(r["load"] for r in R))
    fig, axes = plt.subplots(2, len(loads), figsize=(DOUBLE, 3.8), sharex=True)
    for col, load in enumerate(loads):
        for row, (key, lab) in enumerate([("mean_wait_min", "Mean wait (min)"),
                                          ("station_mean_min", "Mean time at station (min)")]):
            ax = axes[row, col]
            for m, (mname, c, mk) in METHODS.items():
                if m == "Greedy" or not sel(R, method=m):
                    continue
                mu, h = map(np.array, zip(*[ci([r[key] for r in sel(R, env="phase2_sumo", load=load, threshold=t,
                                                                     method=m)]) for t in thr]))
                ax.plot(thr, mu, color=c, marker=mk, ms=3, lw=1, label=mname)
                ax.fill_between(thr, mu - h, mu + h, color=c, alpha=0.15, lw=0)
            if row == 0:
                rate = sel(R, load=load)[0]["rate"]
                ax.set_title(f"{rate:g} veh/min")
            else:
                ax.set_xlabel("Request threshold (SOC)")
            if col == 0:
                ax.set_ylabel(lab)
            ax.set_ylim(bottom=0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_i95.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = latest("main_factorial_*.json")
    fig_main(R, out)
    fig_threshold(R, out, "phase2_sumo", "fig_threshold.pdf")
    fig_threshold(R, out, "phase1", "fig_threshold_p1.pdf")
    fig_lookahead(R, out)
    case = (latest("i95_case_2*.json") or []) + (latest("i95_case_m3_*.json") or [])
    if case:
        fig_i95(case, out)
    print(f"figures written to {out}")
