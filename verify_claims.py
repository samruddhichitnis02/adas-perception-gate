"""Stage 9: verify the claims made in README.md / report.md against the raw artifacts.

Checks (each prints PASS/FAIL; exit code 1 if any FAIL; details in results/verify_report.json):
  1. DATA      val image/box totals + per-class box counts recomputed from the RAW nuImages metadata
               (independent of src/) equal results/slices.json and the YOLO label files on disk.
  2. TRAINING  every run has the expected number of epochs, finite losses, train loss went down.
  3. MAP       our mAP equals pycocotools for every run (stored cross-check) and a fresh pycocotools
               recomputation for one run from the saved predictions.
  4. C++       taxonomy json says C++ == Python for every run; one run is re-run live (C++ and Python)
               and compared; benchmark speed-ups quoted are re-derived from the stored timings.
  5. GATE      every stored verdict re-derives from its checks + gate_config.yaml; the degraded model
               returned FAIL; no PASS verdict hides a FAIL check.
  6. TRIAGE    every triage pattern's counts are recomputed from taxonomy labels + GT; its figure exists.
  7. DOCS      every number in README.md and report.md (outside code blocks) matches some value in
               results/ (to the precision shown, as a value or a percentage) or an allow-listed constant.
usage: python verify_claims.py [--skip-live]
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from common import *

OK, BAD = [], []
LIVE = "--skip-live" not in sys.argv


def check(name, cond, detail=""):
    (OK if cond else BAD).append({"check": name, "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def J(p):
    return json.load(open(p, encoding="utf-8"))


# ------------------------------------------------------------------ 1. data
def verify_data():
    m = META_DIR / "v1.0-val"
    cats = {c["token"]: c["name"] for c in J(m / "category.json")}
    sd = {d["token"] for d in J(m / "sample_data.json") if d["is_key_frame"]}
    cnt, n_img = Counter(), len(sd)
    for a in J(m / "object_ann.json"):
        name = cats[a["category_token"]]
        if name in CLASS_MAP and a["sample_data_token"] in sd:
            x1, y1, x2, y2 = a["bbox"]
            if min(x2, 1600) - max(x1, 0) >= 1 and min(y2, 900) - max(y1, 0) >= 1:
                cnt[CLASS_MAP[name]] += 1
    sl = J(RESULTS / "slices.json")["splits"]["val"]
    check("data: val image count from raw metadata == slices.json", n_img == sl["images"], f"{n_img} vs {sl['images']}")
    check("data: val box count from raw metadata == slices.json", sum(cnt.values()) == sl["boxes"], f"{sum(cnt.values())} vs {sl['boxes']}")
    lbl = sum(len(p.read_text().splitlines()) for p in (YOLO_DIR / "labels" / "val").glob("*.txt") if p.read_text().strip())
    check("data: val YOLO label lines on disk == slices.json", lbl == sl["boxes"], f"{lbl} vs {sl['boxes']}")
    gt = J(META_OUT / "val.json")["images"]
    per = Counter(CLASSES[b["cls"]] for im in gt for b in im["boxes"])
    check("data: per-class val box counts match raw metadata", per == cnt)
    tr = J(RESULTS / "slices.json")["splits"]["train"]
    check("data: train subset size == config", tr["images"] == TRAIN_SUBSET_IMAGES, f"{tr['images']}")
    tl = {i["log"] for i in J(META_OUT / "train.json")["images"]}; vl = {i["log"] for i in gt}
    check("data: no log (drive) shared between train subset and val", not (tl & vl), f"overlap={len(tl & vl)}")


# ------------------------------------------------------------------ 2. training
def verify_training():
    import csv
    for meta in sorted((RESULTS / "training").glob("*_meta.json")):
        m = J(meta); name = f"{m['config']}_s{m['seed']}"
        rows = list(csv.DictReader(open(RESULTS / "training" / f"{name}_results.csv")))
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in rows]
        check(f"training: {name} ran {m['hyperparams']['epochs']} epochs", len(rows) == m["hyperparams"]["epochs"], f"{len(rows)}")
        fin = all(math.isfinite(v) for r in rows for v in r.values())
        check(f"training: {name} logged values are finite", fin)
        if len(rows) > 2:
            check(f"training: {name} train box loss decreased", rows[-1]["train/box_loss"] < rows[0]["train/box_loss"])


# ------------------------------------------------------------------ 3. mAP
def verify_map():
    for ev in sorted((RESULTS / "eval").glob("*.json")):
        e = J(ev)
        pc = e.get("pycocotools")
        if pc:
            d1 = abs(pc["AP"] - e["overall"]["mAP50-95"]); d2 = abs(pc["AP50"] - e["overall"]["mAP50"])
            check(f"map: {e['run']} ours == pycocotools (|d|<1e-3)", d1 < 1e-3 and d2 < 1e-3, f"{d1:.5f} {d2:.5f}")
        else:
            check(f"map: {e['run']} has pycocotools cross-check", False)
    if LIVE:
        import evalcore as E, evaluate as EV
        gt = E.GT(); run = "baseline_s0"
        fresh = EV.coco_crosscheck(gt, E.Preds(gt, run))
        stored = J(RESULTS / "eval" / f"{run}.json")["overall"]
        check(f"map: fresh pycocotools run on saved predictions of {run} matches stored mAP",
              abs(fresh["AP"] - stored["mAP50-95"]) < 1e-3 and abs(fresh["AP50"] - stored["mAP50"]) < 1e-3,
              f"{fresh} vs {stored}")


# ------------------------------------------------------------------ 4. C++
def verify_cpp():
    for t in sorted((RESULTS / "taxonomy").glob("*.json")):
        d = J(t)
        check(f"cpp: {d['run']} C++ output identical to Python reference", d["cpp_python_identical"] and d["n_differing_entries"] == 0)
    b = RESULTS / "cpp_benchmark_repeat.json"
    if b.exists():
        d = J(b)
        check("cpp: repeated benchmark reports identical outputs", d["taxonomy"]["identical"] and d["match_iou0.5"]["identical_python_cpp_numba"])
        sp = d["taxonomy"]["python_median_s"] / d["taxonomy"]["cpp_median_s"]
        check("cpp: quoted taxonomy speed-up re-derives from stored timings", abs(sp - d["taxonomy"]["speedup_cpp_vs_python"]) < 0.2, f"{sp:.1f}")
    else:
        check("cpp: repeated benchmark file exists", False, "run src/benchmark_cpp.py")
    if LIVE:
        import evalcore as E, taxonomy_ref as T, tempfile
        gt = E.GT(); pr = E.Preds(gt, "baseline_s0")
        keep, off = T.filtered_preds(gt, pr)
        tmp = Path(tempfile.mkdtemp(dir=DATA)); T.write_bin(tmp / "i.bin", gt, off, pr.xyxy[keep], pr.cls[keep])
        py = T.classify_python(gt.off, gt.xyxy, gt.cls, off, pr.xyxy[keep], pr.cls[keep])
        cp, _ = T.run_cpp("taxonomy", tmp / "i.bin", tmp / "o.bin")
        check("cpp: LIVE re-run on baseline_s0 val set, C++ == Python (labels, refs, missed)", all(np.array_equal(a, b) for a, b in zip(py, cp)))
        for p in tmp.iterdir(): p.unlink()
        tmp.rmdir()


# ------------------------------------------------------------------ 5. gate
def verify_gate():
    import gate as G
    cfg = yaml.safe_load(open(ROOT / "gate_config.yaml"))
    s = J(RESULTS / "gate" / "summary.json")
    verdicts = {}
    for r in s["results"]:
        full = J(RESULTS / "gate" / f"{r['candidate']}_vs_{r['baseline']}.json")
        re_ver = []
        for c in full["checks"]:
            hib = c["kind"] != "error_rate_per_100_gt"
            re_ver.append(G.verdict(c["delta"], c["ci_low"], c["ci_high"], c["threshold"], hib))
        any_fail = "FAIL" in re_ver; any_warn = "WARN" in re_ver
        final = "FAIL" if any_fail or (any_warn and cfg.get("fail_on_warn")) else "PASS"
        check(f"gate: {r['candidate']} verdict {r['verdict']} re-derives from its checks", final == r["verdict"] == full["verdict"])
        check(f"gate: {r['candidate']} stored per-check verdicts consistent", re_ver == [c["verdict"] for c in full["checks"]])
        verdicts[r["candidate"]] = r["verdict"]
    check("gate: degraded model returns FAIL", verdicts.get("degraded_s0") == "FAIL")


# ------------------------------------------------------------------ 6. triage
def verify_triage():
    import evalcore as E
    gt = E.GT()
    for t in sorted((RESULTS / "triage").glob("*.json")):
        d = J(t); run = d["run"]
        z = np.load(RESULTS / "taxonomy" / f"{run}_labels.npz"); missed = z["missed"].astype(bool)
        check(f"triage: {run} total missed == taxonomy missed", int(missed.sum()) == d["n_missed"] == J(RESULTS / "taxonomy" / f"{run}.json")["overall"]["missed"])
        for p in d["missed_gt_patterns"]:
            msk = np.ones(len(gt.cls), bool)
            for tg in p["tags"]:
                msk &= gt.gt_mask.get(tg, np.array([tg in x for x in gt.tags]))
            if p["class"]:
                msk &= gt.cls == CLASSES.index(p["class"])
            ok = int(msk.sum()) == p["n_gt"] and int((msk & missed).sum()) == p["missed"]
            check(f"triage: {run} pattern '{p['description'][:60]}' counts recomputed", ok)
            if "figure" in p:
                check(f"triage: figure exists {p['figure']}", (ROOT / p["figure"]).exists())


# ------------------------------------------------------------------ 7. docs
ALLOW = {"3.12", "12.8", "16.2", "12", "1600", "900", "640", "960", "5090", "2000", "16000", "0.01", "0.001",
         "0.25", "0.5", "0.1", "0.75", "0.95", "32", "96", "300", "100", "30", "200", "1.0", "0.95", "0.05"}


def pool_numbers():
    nums = []

    def walk(x):
        if isinstance(x, bool): return
        if isinstance(x, (int, float)):
            if math.isfinite(x): nums.append(float(x))
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    for p in list(RESULTS.rglob("*.json")):
        try: walk(J(p))
        except Exception: pass
    walk(yaml.safe_load(open(ROOT / "gate_config.yaml")))
    return np.array(nums)


def verify_docs():
    pool = pool_numbers()
    tok = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*%?")
    for doc in ("README.md", "report.md"):
        p = ROOT / doc
        if not p.exists():
            check(f"docs: {doc} exists", False); continue
        txt = re.sub(r"```.*?```", "", p.read_text(encoding="utf-8"), flags=re.S)
        txt = re.sub(r"`[^`]*`", "", txt); txt = re.sub(r"\(https?://[^)]*\)|https?://\S+|\([^)]*\.(?:md|json|py|jpg|png)\)", "", txt)
        bad = []
        for m in tok.finditer(txt):
            s = m.group(0); pct = s.endswith("%"); raw = s.rstrip("%").replace(",", "").lstrip("+")
            if raw.endswith("."): raw = raw[:-1]
            if not raw: continue
            if "." not in raw and not pct and len(raw.replace("-", "")) < 4:
                continue  # small integers (counts like "3 configs") are not checked
            if raw.lstrip("-") in ALLOW: continue
            d = len(raw.split(".")[1]) if "." in raw else 0
            val = float(raw); a = abs(val)
            cands = [pool, pool * 100, np.abs(pool), np.abs(pool) * 100]
            if d > 0:  # unit conversions (seconds->minutes etc.) only for decimals; integers must match exactly
                cands += [pool / 60, pool / 1e6, pool * 1000]
            if not any(np.any(np.round(c, d) == (a if c is not pool and c is not pool * 100 else val)) or np.any(np.round(np.abs(c), d) == a) for c in cands):
                bad.append(s)
        check(f"docs: every number in {doc} traces to results/ ({len(bad)} unmatched)", not bad, ", ".join(bad[:25]))


if __name__ == "__main__":
    for f in (verify_data, verify_training, verify_map, verify_cpp, verify_gate, verify_triage, verify_docs):
        try:
            f()
        except Exception as ex:  # a verification that crashes is a failed verification
            check(f"{f.__name__} ran without error", False, repr(ex))
    (RESULTS / "verify_report.json").write_text(json.dumps({"passed": len(OK), "failed": len(BAD), "failures": BAD}, indent=2))
    print(f"\n{len(OK)} passed, {len(BAD)} failed")
    sys.exit(1 if BAD else 0)
