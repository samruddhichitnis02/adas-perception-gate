"""Stage 3: overall mAP50 / mAP50-95, per-class AP, per-slice AP for each model.

usage: python src/evaluate.py <run> [<run> ...] [--crosscheck]
Writes results/eval/<run>.json. --crosscheck also runs pycocotools COCOeval (maxDets=300) on the
same predictions and stores its overall AP/AP50 next to ours (they should agree closely).
"""
import json, sys, time

import numpy as np

from common import *
import evalcore as E


def evaluate(run, crosscheck=False, gt=None):
    gt = gt or E.GT()
    pr = E.Preds(gt, run)
    t0 = time.time()
    matches = E.match_all(gt, pr)
    all_t = list(range(len(E.IOU_THRS)))
    out = {"run": run, "n_predictions": int(len(pr.cls)), "iou_thresholds": E.IOU_THRS.tolist()}
    m50, per50, ngt = E.mean_ap(gt, pr, matches, None, (0,))
    m5095, per5095, _ = E.mean_ap(gt, pr, matches, None, all_t)
    out["overall"] = {"mAP50": m50, "mAP50-95": m5095}
    out["per_class"] = {CLASSES[c]: {"AP50": float(per50[0, c]), "AP50-95": float(np.nanmean(per5095[:, c])),
                                     "n_gt": int(ngt[c])} for c in range(len(CLASSES))}
    out["per_slice"] = {}
    for s in gt.slice_names:
        a50, _, n = E.mean_ap(gt, pr, matches, s, (0,))
        a5095, _, _ = E.mean_ap(gt, pr, matches, s, all_t)
        out["per_slice"][s] = {"mAP50": a50, "mAP50-95": a5095, "n_gt": int(gt.gt_mask[s].sum()),
                               "kind": "native" if S_is_native(s) else "derived",
                               "fp_counted": s in pr.mask}
    out["excluded_slices_lt30_boxes"] = gt.excluded_slices
    out["eval_seconds"] = round(time.time() - t0, 1)
    if crosscheck:
        out["pycocotools"] = coco_crosscheck(gt, pr)
    (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval" / f"{run}.json").write_text(json.dumps(out, indent=2))
    print(f"{run}: mAP50={m50:.4f} mAP50-95={m5095:.4f}" + (f"  coco AP50={out['pycocotools']['AP50']:.4f} AP={out['pycocotools']['AP']:.4f}" if crosscheck else ""))
    return out


def S_is_native(s):
    import slices as S
    return S.is_native(s)


def coco_crosscheck(gt, pr):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    images = [{"id": int(i), "width": 1600, "height": 900} for i in range(gt.n_img)]
    anns = []
    for k in range(len(gt.cls)):
        x1, y1, x2, y2 = map(float, gt.xyxy[k])
        anns.append({"id": k + 1, "image_id": int(gt.img[k]), "category_id": int(gt.cls[k]),
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    ds = {"images": images, "annotations": anns, "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = ds; cg.createIndex()
        dets = [[int(pr.img[k]), float(pr.xyxy[k, 0]), float(pr.xyxy[k, 1]), float(pr.xyxy[k, 2] - pr.xyxy[k, 0]),
                 float(pr.xyxy[k, 3] - pr.xyxy[k, 1]), float(pr.conf[k]), int(pr.cls[k])] for k in range(len(pr.cls))]
        cd = cg.loadRes(np.array(dets))
        ev = COCOeval(cg, cd, "bbox"); ev.params.maxDets = [1, 10, 300]
        ev.evaluate(); ev.accumulate()
    # precision[T, R, K, A, M]: A=0 (all areas), M=2 (maxDets=300). summarize() hardcodes maxDets=100.
    p = ev.eval["precision"][:, :, :, 0, 2]
    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean())
    return {"AP": mean_valid(p), "AP50": mean_valid(p[0])}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cc = "--crosscheck" in sys.argv
    g = E.GT()
    for r in args:
        evaluate(r, cc, g)
