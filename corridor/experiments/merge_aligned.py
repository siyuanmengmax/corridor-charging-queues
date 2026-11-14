"""
Replace the queueing-simulator records of the synthetic-corridor result files with the
reruns that use the SUMO station positions (tags p1aligned_*), keeping the SUMO records.

  python -m corridor.experiments.merge_aligned
"""

import glob
import json
import time
from pathlib import Path

RES = Path("corridor/results")
PAIRS = {   # original file pattern -> rerun tag(s)
    "main_factorial_*.json": ["p1aligned_main"],
    "attr_synth_*.json": ["p1aligned_attr"],
    "attr_p120_synth_*.json": ["p1aligned_attr"],
    "e6_weights_*.json": ["p1aligned_e6"],
    **{f"e4_noise_{x}_*.json": [f"p1aligned_e4_noise_{x}"] for x in ("0.1", "0.2", "0.4")},
    **{f"e4_visibility_{x}_*.json": [f"p1aligned_e4_visibility_{x}"] for x in ("0.75", "0.5", "0.25")},
    **{f"e5_uncertainty_{x}_*.json": [f"p1aligned_e5_uncertainty_{x}"] for x in ("0.0", "0.4")},
}


def latest(pattern):
    files = sorted(glob.glob(str(RES / pattern)))
    return files[-1] if files else None


if __name__ == "__main__":
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for pattern, tags in PAIRS.items():
        src = latest(pattern)
        old = json.load(open(src))
        methods = {r["method"] for r in old}
        new = [r for t in tags for r in json.load(open(latest(f"{t}_2*.json"))) if r["method"] in methods]
        keep = [r for r in old if r["env"] != "phase1"]
        n_old = sum(r["env"] == "phase1" for r in old)
        assert len(new) == n_old, (pattern, len(new), n_old)
        out = RES / (pattern.split("*")[0].rstrip("_") + f"_{stamp}.json")
        json.dump(keep + new, open(out, "w"), indent=1)
        print(f"{Path(src).name} -> {out.name}: {len(keep)} SUMO + {len(new)} queueing-simulator records")
