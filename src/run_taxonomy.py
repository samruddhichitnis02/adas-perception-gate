"""Stages 4+5: error taxonomy per model, computed by BOTH the pure-Python reference and the C++ CLI.
The two must be bit-identical on the full val set; timings of both are recorded.

usage: python src/run_taxonomy.py <run> [<run> ...]
Writes results/taxonomy/<run>.json  and appends timings to results/cpp_benchmark.json
Also (first model only unless --all-match) checks the C++ `match` mode vs pure Python and the numba
matcher used for AP.
"""
import json, sys, tempfile, time
from pathlib import Path

import numpy as np

from common import *
import evalcore as E
import taxonomy_ref as T


def save_perimg(run, gt, pr, keep, label, ref, missed):
    """Per-image error counts (n_img, 6): correct, cls, loc, dup, bg, missed. Used by the gate bootstrap."""
    cnt = np.zeros((gt.n_img, 6), np.int32)
    np.add.at(cnt, (pr.img[keep], label), 1)
    np.add.at(cnt, (gt.img[missed.astype(bool)], 5), 1)
    np.save(RESULTS / "taxonomy" / f"{run}_perimg.npy", cnt)
    np.savez_compressed(RESULTS / "taxonomy" / f"{run}_labels.npz", keep=np.nonzero(keep)[0].astype(np.int32),
                        label=label, ref=ref, missed=missed)
    return cnt


def per_model(run, gt):
    pr = E.Preds(gt, run)
    keep, p_off = T.filtered_preds(gt, pr)
    p_box, p_cls = pr.xyxy[keep], pr.cls[keep]
    tmp = Path(tempfile.mkdtemp(dir=DATA))
    in_bin, out_bin = tmp / "in.bin", tmp / "out.bin"
    T.write_bin(in_bin, gt, p_off, p_box, p_cls)

    t0 = time.perf_counter()
    py = T.classify_python(gt.off, gt.xyxy, gt.cls, p_off, p_box, p_cls)
    py_sec = time.perf_counter() - t0
    (cl, cpp_sec) = T.run_cpp("taxonomy", in_bin, out_bin)
    same = all(np.array_equal(a, b) for a, b in zip(py, cl))
    n_diff = int(sum((a != b).sum() for a, b in zip(py, cl)))
    res = T.aggregate(gt, pr, keep, *py)
    res.update({"run": run, "cpp_python_identical": bool(same), "n_differing_entries": n_diff,
                "python_seconds": round(py_sec, 3), "cpp_compute_seconds": round(cpp_sec, 6),
                "n_pred_classified": int(len(p_cls)), "n_gt": int(len(gt.cls)), "n_images": gt.n_img})
    (RESULTS / "taxonomy").mkdir(parents=True, exist_ok=True)
    (RESULTS / "taxonomy" / f"{run}.json").write_text(json.dumps(res, indent=2))
    save_perimg(run, gt, pr, keep, *py)
    print(f"{run}: identical={same} py={py_sec:.2f}s cpp={cpp_sec*1000:.1f}ms  {res['overall']}")
    for p in (in_bin, out_bin):
        p.unlink()
    tmp.rmdir()
    return res


def match_mode_check(run, gt):
    """C++ `match` mode vs pure-Python vs numba matcher, on ALL predictions of one model."""
    pr = E.Preds(gt, run)
    tmp = Path(tempfile.mkdtemp(dir=DATA))
    in_bin, out_bin = tmp / "in.bin", tmp / "out.bin"
    T.write_bin(in_bin, gt, pr.off, pr.xyxy, pr.cls)
    out = {"run": run, "n_predictions": int(len(pr.cls)), "thresholds": {}}
    for thr in (0.5, 0.75):
        t0 = time.perf_counter(); py = T.match_python(gt.off, gt.xyxy, gt.cls, pr.off, pr.xyxy, pr.cls, thr)
        py_s = time.perf_counter() - t0
        cpp, cpp_s = T.run_cpp("match", in_bin, out_bin, thr)
        E._match(pr.off, pr.xyxy.astype(np.float64), pr.cls, gt.off, gt.xyxy.astype(np.float64), gt.cls, thr)  # JIT warmup
        t0 = time.perf_counter()
        nb = E._match(pr.off, pr.xyxy.astype(np.float64), pr.cls, gt.off, gt.xyxy.astype(np.float64), gt.cls, thr)
        nb_s = time.perf_counter() - t0
        out["thresholds"][str(thr)] = {
            "cpp_equals_python": bool(np.array_equal(py, cpp)),
            "cpp_equals_numba": bool(np.array_equal(nb, cpp)),
            "python_seconds": round(py_s, 3), "cpp_compute_seconds": round(cpp_s, 6),
            "numba_seconds": round(nb_s, 4)}
        print(thr, out["thresholds"][str(thr)])
    for p in (in_bin, out_bin):
        p.unlink()
    tmp.rmdir()
    return out


if __name__ == "__main__":
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    gt = E.GT()
    bench = RESULTS / "cpp_benchmark.json"
    data = json.loads(bench.read_text()) if bench.exists() else {"taxonomy": {}, "match": None}
    for r in runs:
        res = per_model(r, gt)
        data["taxonomy"][r] = {k: res[k] for k in ("cpp_python_identical", "n_differing_entries", "python_seconds",
                                                    "cpp_compute_seconds", "n_pred_classified", "n_gt", "n_images")}
    if data.get("match") is None and runs:
        data["match"] = match_mode_check(runs[0], gt)
    bench.write_text(json.dumps(data, indent=2))
