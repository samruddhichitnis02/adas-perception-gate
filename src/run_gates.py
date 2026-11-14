"""Stage 7 driver: run the gate for every candidate against the baseline and save the outputs.

baseline = baseline_s0.  Candidates: strong_aug (2 seeds), hires (2 seeds), baseline_s1 (seed-noise
control: same recipe, different seed - shows what 'no real change' looks like), degraded_s0 (1 epoch;
must FAIL to prove the gate can reject).
Writes results/gate/<cand>_vs_baseline_s0.json and results/gate/summary.json.
"""
import json, sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import evalcore as E
import gate as G

BASELINE = "baseline_s0"
CANDIDATES = ["baseline_s1", "strong_aug_s0", "strong_aug_s1", "hires_s0", "hires_s1", "degraded_s0"]


def main(only=None):
    """Run the gate for `only` (default: all CANDIDATES) and merge into results/gate/summary.json."""
    cfg = yaml.safe_load(open(ROOT / "gate_config.yaml"))
    gt = E.GT()
    out = RESULTS / "gate"; out.mkdir(parents=True, exist_ok=True)
    sp = out / "summary.json"
    done = {r["candidate"]: r for r in json.load(open(sp))["results"]} if sp.exists() else {}
    for c in (only or CANDIDATES):
        res = G.run_gate(BASELINE, c, cfg, gt)
        (out / f"{c}_vs_{BASELINE}.json").write_text(json.dumps(res, indent=2))
        done[c] = {"candidate": c, "baseline": BASELINE, "verdict": res["verdict"], "n_fail": res["n_fail"],
                   "n_warn": res["n_warn"], "n_checks": res["n_checks"], "reasons": res["reasons"]}
        ordered = [done[k] for k in CANDIDATES if k in done]
        sp.write_text(json.dumps({"config": cfg, "results": ordered}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
