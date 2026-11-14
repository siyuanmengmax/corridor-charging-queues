"""
Research framework figure (paper: fig_framework.pdf): data and corridors -> dispatch methods
(ablation chain and port-data rules) -> closed-loop evaluation -> experiments -> findings.

Run from the repository root:  python -m corridor.experiments.fig_framework
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 6.5, "pdf.fonttype": 42})
OUT = Path("paper/figures/fig_framework.pdf")

BLUE, LBLUE, GREY, LGREY, ORANGE = "#0072B2", "#E3EEF7", "#555555", "#F2F2F2", "#FBEBD9"


def box(ax, x, y, w, h, text, fc="white", ec="black", fs=6.3, bold=False, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black",
                                 shrinkA=0, shrinkB=0))


def stage(ax, y, h, label):
    ax.add_patch(FancyBboxPatch((0.0, y), 0.55, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=BLUE, ec="none"))
    ax.text(0.275, y + h / 2, label, ha="center", va="center", rotation=90, color="white",
            fontsize=5.6, fontweight="bold")


def main():
    fig = plt.figure(figsize=(3.5, 4.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.7)
    ax.axis("off")
    X0, W = 0.75, 9.2          # content area right of the stage labels

    # 1. data and corridors
    y = 9.85
    stage(ax, y, 1.8, "Data")
    box(ax, X0, y + 0.95, 2.95, 0.85, "AFDC CCS sites\nOSM geometry", fc=LGREY)
    box(ax, X0 + 3.12, y + 0.95, 2.95, 0.85, "AADT counts\n(NCDOT, Caltrans)", fc=LGREY)
    box(ax, X0 + 6.24, y + 0.95, 2.96, 0.85, "Vehicle parameters\n(IEA, NREL)", fc=LGREY)
    box(ax, X0, y, W, 0.75, "Corridors: synthetic 300 km, I-95 NC 293 km, I-15 236 km;\ndemand levels, battery, consumption, entry SOC")
    for cx in (X0 + 1.47, X0 + 4.6, X0 + 7.72):
        arrow(ax, (cx, y + 0.95), (cx, y + 0.75))

    # 2. dispatch methods
    y2 = 5.55
    arrow(ax, (X0 + W / 2, y), (X0 + W / 2, y2 + 3.95))
    stage(ax, y2, 3.95, "Dispatch methods")
    ax.text(X0 + 0.1, y2 + 3.72, "Ablation chain: one element of station information added per step",
            fontsize=6.0, va="center", style="italic", color=GREY)
    steps = [("Static-capacity MILP", "free charger time, next interval"),
             ("+ queue at the station", "queue greedy, uniform power"),
             ("+ station power", "queue greedy"),
             ("+ vehicles en route", "backlog greedy"),
             ("+ joint optimization, look-ahead", "fluid-queue MPC")]
    bw, bh, gap = 5.85, 0.56, 0.12
    for k, (a, b) in enumerate(steps):
        yy = y2 + 3.42 - (k + 1) * bh - k * gap
        box(ax, X0, yy, bw, bh, f"{a}\n({b})", fc=LBLUE if k else "white", fs=6.0)
        if k:
            arrow(ax, (X0 + bw / 2, yy + bh + gap), (X0 + bw / 2, yy + bh))
    px = X0 + bw + 0.25
    pw = W - bw - 0.25
    box(ax, px, y2 + 1.15, pw, 2.27, "Port-data rules\n(NEVI rule)\n\nport status\nand power\n\n+ queue count",
        fc=ORANGE, fs=6.0)
    ax.text(px + pw / 2, y2 + 0.68, "partial participation:\nown vehicles +\nstation data", ha="center",
            va="center", fontsize=5.7, color=GREY, linespacing=1.1)

    # 3. closed-loop evaluation
    y3 = 3.55
    arrow(ax, (X0 + W / 2, y2), (X0 + W / 2, y3 + 1.6))
    stage(ax, y3, 1.6, "Evaluation")
    box(ax, X0, y3 + 0.5, 3.6, 1.1, "Queueing simulator\n(FCFS stations)", fc=LGREY)
    box(ax, X0 + 5.6, y3 + 0.5, 3.6, 1.1, "SUMO via TraCI\n(parking-area stations)", fc=LGREY)
    ax.text(X0 + W / 2, y3 + 1.05, "same demand,\nsame interface", ha="center", va="center", fontsize=5.7,
            color=GREY)
    ax.text(X0 + W / 2, y3 + 0.2, "dispatch every 5 min; 20 paired seeds; paired 95% CIs", ha="center",
            va="center", fontsize=6.0)

    # 4. experiments
    y4 = 1.5
    arrow(ax, (X0 + W / 2, y3), (X0 + W / 2, y4 + 1.75))
    stage(ax, y4, 1.75, "Experiments")
    exps = ["Main\ncomparison", "Information\nsteps", "Partial\nparticipation", "Request\nthreshold $\\theta$",
            "Robustness,\ncompliance"]
    ew = (W - 4 * 0.12) / 5
    for k, e in enumerate(exps):
        box(ax, X0 + k * (ew + 0.12), y4 + 0.2, ew, 1.35, e, fs=5.9)

    # 5. findings
    arrow(ax, (X0 + W / 2, y4), (X0 + W / 2, 1.2))
    stage(ax, 0.05, 1.15, "Findings")
    box(ax, X0, 0.05, W, 1.15, "Which station data are needed (queue count beyond port status)\n"
                                "When to request charging (spacing of the large sites)", fs=6.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
