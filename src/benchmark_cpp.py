"""Stage 5 benchmark: median-of-N timings of Python reference vs C++ vs numba, on the full val set.

usage: python src/benchmark_cpp.py <run> [N=5]
Writes results/cpp_benchmark_repeat.json. Compute time only (file I/O to/from the C++ CLI excluded;
the C++ CLI reports its own internal compute timer). Run on an otherwise idle machine for clean numbers.
"""
import json, platform, statistics, sys, tempfile, time
from pathlib import Path

import numpy as np

from common import *
import evalcore as E
import taxonomy_ref as T


def med(xs):
    return round(statistics.median(xs), 6)


def main(run, n=5):
    gt = E.GT(); pr = E.Preds(gt, run)
    tmp = Path(tempfile.mkdtemp(dir=DATA))
    out = {"run": run, "repeats": n, "machine": platform.processor(), "taxonomy": {}, "match_iou0.5": {}}

    keep, p_off = T.filtered_preds(gt, pr)
    p_box, p_cls = pr.xyxy[keep], pr.cls[keep]
    T.write_bin(tmp / "tax.bin", gt, p_off, p_box, p_cls)
    py_t, cpp_t, identical = [], [], True
    for _ in range(n):
        t0 = time.perf_counter(); ref = T.classify_python(gt.off, gt.xyxy, gt.cls, p_off, p_box, p_cls)
        py_t.append(time.perf_counter() - t0)
        res, sec = T.run_cpp("taxonomy", tmp / "tax.bin", tmp / "o.bin"); cpp_t.append(sec)
        identical &= all(np.array_equal(a, b) for a, b in zip(ref, res))
    out["taxonomy"] = {"n_pred": int(len(p_cls)), "n_gt": int(len(gt.cls)), "identical": bool(identical),
                       "python_median_s": med(py_t), "cpp_median_s": med(cpp_t),
                       "speedup_cpp_vs_python": round(statistics.median(py_t) / statistics.median(cpp_t), 1)}

    T.write_bin(tmp / "m.bin", gt, pr.off, pr.xyxy, pr.cls)
    p64, g64 = pr.xyxy.astype(np.float64), gt.xyxy.astype(np.float64)
    E._match(pr.off, p64, pr.cls, gt.off, g64, gt.cls, 0.5)  # JIT warm-up
    py_t, cpp_t, nb_t, ok = [], [], [], True
    for _ in range(n):
        t0 = time.perf_counter(); ref = T.match_python(gt.off, gt.xyxy, gt.cls, pr.off, pr.xyxy, pr.cls, 0.5)
        py_t.append(time.perf_counter() - t0)
        res, sec = T.run_cpp("match", tmp / "m.bin", tmp / "o.bin", 0.5); cpp_t.append(sec)
        t0 = time.perf_counter(); nb = E._match(pr.off, p64, pr.cls, gt.off, g64, gt.cls, 0.5); nb_t.append(time.perf_counter() - t0)
        ok &= bool(np.array_equal(ref, res) and np.array_equal(nb, res))
    out["match_iou0.5"] = {"n_pred": int(len(pr.cls)), "identical_python_cpp_numba": ok,
                           "python_median_s": med(py_t), "cpp_median_s": med(cpp_t), "numba_median_s": med(nb_t),
                           "speedup_cpp_vs_python": round(statistics.median(py_t) / statistics.median(cpp_t), 1)}
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    (RESULTS / "cpp_benchmark_repeat.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
