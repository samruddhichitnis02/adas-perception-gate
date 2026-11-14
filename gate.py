"""Validation gate: decide whether a candidate model is safe to promote over a baseline.

usage:
  python gate.py --baseline baseline_s0 --candidate hires_s0 [--config gate_config.yaml] [--out results/gate]

Needs results/preds/<run>.npz for both runs (produced by src/predict.py). Prints PASS or FAIL with
reasons, and writes results/gate/<candidate>_vs_<baseline>.json. Exit code 0 = PASS, 1 = FAIL.

Rules (thresholds in gate_config.yaml), each judged with a PAIRED bootstrap over val images:
  1. overall mAP50-95 (and mAP50) must not drop
  2. no slice mAP50 may drop by more than max_drop (slices with >= min_gt_boxes GT boxes)
  3. no error type (classification / localization / duplicate / background FP / missed) may grow by
     more than its threshold, per 100 GT boxes
Verdict per check: FAIL if the CI bound shows a regression beyond the threshold; WARN if only the point
estimate is beyond the threshold (inconclusive); else PASS.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import *
import evalcore as E
import taxonomy_ref as T

ERR_NAMES = ["classification_error", "localization_error", "duplicate", "background_fp", "missed"]
ERR_COL = {"classification_error": 1, "localization_error": 2, "duplicate": 3, "background_fp": 4, "missed": 5}


def load_model(gt, run, min_conf=0.0):
    pr = E.Preds(gt, run, min_conf=min_conf)
    matches = E.match_all(gt, pr)
    perimg = RESULTS / "taxonomy" / f"{run}_perimg.npy"
    if not perimg.exists():  # compute the taxonomy on the fly (C++ if built, else the Python reference)
        import run_taxonomy as RT
        RT.per_model(run, gt)
    return {"pr": pr, "matches": matches, "err": np.load(perimg)}


def verdict(delta_point, lo, hi, thr, higher_is_better=True):
    """Return PASS/WARN/FAIL for a delta (cand - base). thr>=0 is the allowed regression magnitude.
    higher_is_better: regression = delta < -thr ; else regression = delta > +thr."""
    if higher_is_better:
        if hi < -thr: return "FAIL"            # even the optimistic CI end is worse than allowed
        if delta_point < -thr: return "WARN"
    else:
        if lo > thr: return "FAIL"
        if delta_point > thr: return "WARN"
    return "PASS"


def run_gate(baseline, candidate, cfg, gt=None, quiet=False):
    gt = gt or E.GT()
    t0 = time.time()
    mc = cfg["bootstrap"].get("min_conf_for_ap", 0.0)
    B, C = load_model(gt, baseline, mc), load_model(gt, candidate, mc)
    nb = cfg["bootstrap"]["n_resamples"]; a = (1 - cfg["bootstrap"]["ci_level"]) / 2
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])

    # ---- metric definitions: name -> (list of APPrep per IoU thr) for each model
    metrics = {}
    all_t = list(range(len(E.IOU_THRS)))
    metrics["overall:mAP50-95"] = (None, all_t)
    metrics["overall:mAP50"] = (None, [0])
    slice_min = cfg["slices"]["min_gt_boxes"]
    gated_slices = [s for s in gt.slice_names if gt.gt_mask[s].sum() >= slice_min]
    for s in gated_slices:
        metrics[f"slice:{s}"] = (s, [0])

    def prep(M):
        return {k: [E.APPrep(gt, M["pr"], M["matches"][t], s) for t in ts] for k, (s, ts) in metrics.items()}
    PB, PC = prep(B), prep(C)

    def val(P, k, w):
        return float(np.nanmean([p.ap(gt, B["pr"] if P is PB else C["pr"], w) for p in P[k]]))

    ones = np.ones(gt.n_img)
    point = {k: (val(PB, k, ones), val(PC, k, ones)) for k in metrics}
    errs_b, errs_c = B["err"], C["err"]
    ngt_img = np.bincount(gt.img, minlength=gt.n_img).astype(float)

    boot = {k: np.empty(nb) for k in metrics}
    boot_err = {e: np.empty(nb) for e in ERR_NAMES}
    for i in range(nb):
        w = rng.multinomial(gt.n_img, np.ones(gt.n_img) / gt.n_img).astype(float)
        for k in metrics:
            boot[k][i] = val(PC, k, w) - val(PB, k, w)
        den = max((w * ngt_img).sum(), 1.0)
        for e in ERR_NAMES:
            c = ERR_COL[e]
            boot_err[e][i] = 100 * ((w * errs_c[:, c]).sum() - (w * errs_b[:, c]).sum()) / den

    checks = []

    def add(name, kind, base, cand, delta, lo, hi, thr, hib):
        checks.append({"check": name, "kind": kind, "baseline": base, "candidate": cand, "delta": delta,
                       "ci_low": lo, "ci_high": hi, "threshold": thr,
                       "verdict": verdict(delta, lo, hi, thr, hib)})

    for k in ("overall:mAP50-95", "overall:mAP50"):
        d = point[k][1] - point[k][0]
        lo, hi = np.nanpercentile(boot[k], [100 * a, 100 * (1 - a)])
        add(k, "overall", point[k][0], point[k][1], d, float(lo), float(hi), cfg["overall"]["max_drop"], True)
    for s in gated_slices:
        k = f"slice:{s}"
        d = point[k][1] - point[k][0]
        lo, hi = np.nanpercentile(boot[k], [100 * a, 100 * (1 - a)])
        add(k, "slice", point[k][0], point[k][1], d, float(lo), float(hi), cfg["slices"]["max_drop"], True)
    ngt_total = ngt_img.sum()
    for e in ERR_NAMES:
        c = ERR_COL[e]
        rb, rc = 100 * errs_b[:, c].sum() / ngt_total, 100 * errs_c[:, c].sum() / ngt_total
        lo, hi = np.percentile(boot_err[e], [100 * a, 100 * (1 - a)])
        add(f"error:{e}", "error_rate_per_100_gt", float(rb), float(rc), float(rc - rb), float(lo), float(hi),
            cfg["errors"]["max_growth_per_100_gt"][e], False)

    fails = [c for c in checks if c["verdict"] == "FAIL"]
    warns = [c for c in checks if c["verdict"] == "WARN"]
    final = "FAIL" if fails or (warns and cfg.get("fail_on_warn")) else "PASS"
    reasons = [f"{c['verdict']}: {c['check']} delta={c['delta']:+.4f} (95% CI {c['ci_low']:+.4f}..{c['ci_high']:+.4f}, "
               f"threshold {'' if c['threshold'] == 0 else ('-' if c['kind'] != 'error_rate_per_100_gt' else '+')}{c['threshold']:g})"
               for c in fails + warns]
    result = {"baseline": baseline, "candidate": candidate, "verdict": final, "reasons": reasons,
              "n_fail": len(fails), "n_warn": len(warns), "n_checks": len(checks), "checks": checks,
              "gated_slices": gated_slices,
              "slices_not_gated": [s for s in gt.slice_names if s not in gated_slices],
              "slices_excluded_lt30_boxes": gt.excluded_slices,
              "bootstrap": cfg["bootstrap"], "seconds": round(time.time() - t0, 1)}
    if not quiet:
        print(f"{final}: {candidate} vs {baseline}  ({len(fails)} fail, {len(warns)} warn, {len(checks)} checks)")
        for r in reasons:
            print("  " + r)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True); ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", default=str(ROOT / "gate_config.yaml")); ap.add_argument("--out", default=str(RESULTS / "gate"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    res = run_gate(args.baseline, args.candidate, cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.candidate}_vs_{args.baseline}.json").write_text(json.dumps(res, indent=2))
    sys.exit(0 if res["verdict"] == "PASS" else 1)
