"""
Request threshold on the three corridors (paper: fig_threshold3.pdf): mean wait in SUMO at the
high demand level of each corridor, mean and 95% CI over 20 seeds.

  python -m corridor.experiments.fig_threshold3
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from corridor.experiments.make_figures import DOUBLE, METHODS, ci, latest, sel

PANELS = [("(a) Synthetic, 0.35 veh/min", "main_factorial_*.json", dict(load="high")),
          ("(b) I-95 NC, 1.0 veh/min", "i95us_main_*.json", dict(rate=1.0)),
          ("(c) I-15 LV–Barstow, 1.8 veh/min", "i15us_main_*.json", dict(rate=1.8))]
SHOW = ("StaticCap", "Backlog-Greedy g=3", "Fluid-MPC H=30")


def main(out=Path("paper/figures/fig_threshold3.pdf")):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), layout="constrained")
    for ax, (title, pat, k) in zip(axes, PANELS):
        R = sel(latest(pat), env="phase2_sumo", **k)
        thr = sorted({r["threshold"] for r in R})
        for m in SHOW:
            name, c, mk = METHODS[m]
            mu, h = map(np.array, zip(*[ci([r["mean_wait_min"] for r in sel(R, threshold=t, method=m)]) for t in thr]))
            ax.plot(thr, mu, color=c, marker=mk, ms=3, lw=1, label=name)
            ax.fill_between(thr, mu - h, mu + h, color=c, alpha=0.15, lw=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Request threshold (SOC)")
        ax.set_xticks(thr, [f"{t:.2f}" for t in thr])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean wait (min)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="outside upper center", frameon=False)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()
