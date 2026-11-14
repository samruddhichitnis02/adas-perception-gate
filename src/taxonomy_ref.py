"""Error taxonomy: pure-Python REFERENCE implementation + C++ runner + aggregation.

Definitions (operating point: predictions with conf >= CONF_THR; IoU thresholds documented here):
  Walk predictions per image in descending confidence.
  correct         same-class GT with IoU >= TP_IOU (0.5) that is not yet matched -> match it (best IoU).
  duplicate       otherwise: same-class GT with IoU >= 0.5 exists but was already matched earlier.
  classification  otherwise: a DIFFERENT-class GT with IoU >= 0.5 (right place, wrong class).
  localization    otherwise: same-class GT with 0.1 <= IoU < 0.5 (right class, box too off).
  background      otherwise (IoU < 0.1 with everything of the same class, or only a weak
                  other-class overlap): false positive on background.
  missed          every GT that no 'correct' prediction matched.
Slice counts: correct/missed are attributed through the GT box's tags (so correct+missed == n_gt in
the slice); classification/localization/duplicate/background through the PREDICTION's own tags (only
slices a prediction can be assigned to; attr:* slices have no prediction-side counts).
"""
import json, math, os, subprocess, time
from pathlib import Path

import numpy as np

from common import *

CONF_THR = 0.25
TP_IOU, LOC_IOU = 0.5, 0.1
LABELS = ["correct", "classification_error", "localization_error", "duplicate", "background_fp"]
CPP_EXE = ROOT / "cpp" / "build" / "taxonomy.exe"


def _iou(p, g):
    px1, py1, px2, py2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    gx1, gy1, gx2, gy2 = float(g[0]), float(g[1]), float(g[2]), float(g[3])
    iw = min(px2, gx2) - max(px1, gx1)
    ih = min(py2, gy2) - max(py1, gy1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    pa = (px2 - px1) * (py2 - py1)
    ga = (gx2 - gx1) * (gy2 - gy1)
    return inter / (pa + ga - inter)


def classify_python(gt_off, gt_box, gt_cls, p_off, p_box, p_cls, tp_iou=TP_IOU, loc_iou=LOC_IOU):
    """Pure-Python reference. Returns (label[n_pred], ref_gt[n_pred], missed[n_gt])."""
    gt_box = gt_box.tolist(); p_box = p_box.tolist(); gt_cls = gt_cls.tolist(); p_cls = p_cls.tolist()
    n_gt, n_pred = len(gt_cls), len(p_cls)
    label = [-1] * n_pred; ref = [-1] * n_pred; taken = [False] * n_gt
    for im in range(len(gt_off) - 1):
        g0, g1 = int(gt_off[im]), int(gt_off[im + 1])
        for p in range(int(p_off[im]), int(p_off[im + 1])):
            pb = p_box[p]
            best_tp = tp_iou; m = -1
            best_dup = best_cls = best_loc = -1.0
            g_dup = g_cls = g_loc = -1
            for g in range(g0, g1):
                v = _iou(pb, gt_box[g])
                same = gt_cls[g] == p_cls[p]
                if same and not taken[g]:
                    if v >= best_tp:
                        best_tp = v; m = g
                if same and taken[g] and v >= tp_iou and v > best_dup:
                    best_dup = v; g_dup = g
                if (not same) and v >= tp_iou and v > best_cls:
                    best_cls = v; g_cls = g
                if same and v >= loc_iou and v < tp_iou and v > best_loc:
                    best_loc = v; g_loc = g
            if m >= 0:
                taken[m] = True; label[p] = 0; ref[p] = m
            elif g_dup >= 0:
                label[p] = 3; ref[p] = g_dup
            elif g_cls >= 0:
                label[p] = 1; ref[p] = g_cls
            elif g_loc >= 0:
                label[p] = 2; ref[p] = g_loc
            else:
                label[p] = 4
    missed = [0 if t else 1 for t in taken]
    return np.array(label, np.int32), np.array(ref, np.int32), np.array(missed, np.int32)


def match_python(gt_off, gt_box, gt_cls, p_off, p_box, p_cls, thr):
    """Pure-Python COCO-style greedy matching (reference for the C++ `match` mode)."""
    gt_box = gt_box.tolist(); p_box = p_box.tolist(); gt_cls = gt_cls.tolist(); p_cls = p_cls.tolist()
    out = [-1] * len(p_cls); taken = [False] * len(gt_cls)
    for im in range(len(gt_off) - 1):
        for p in range(int(p_off[im]), int(p_off[im + 1])):
            best = min(thr, 1 - 1e-10); m = -1
            for g in range(int(gt_off[im]), int(gt_off[im + 1])):
                if taken[g] or gt_cls[g] != p_cls[p]:
                    continue
                v = _iou(p_box[p], gt_box[g])
                if v <= 0 or v < best:
                    continue
                best = v; m = g
            if m >= 0:
                taken[m] = True; out[p] = m
    return np.array(out, np.int32)


# ------------------------------------------------------------------ C++ bridge
def write_bin(path, gt, p_off, p_box, p_cls):
    n_pred = len(p_cls)
    with open(path, "wb") as f:
        np.array([gt.n_img, len(gt.cls), n_pred], np.int32).tofile(f)
        gt.off.astype(np.int32).tofile(f); np.asarray(p_off).astype(np.int32).tofile(f)
        gt.xyxy.astype(np.float32).tofile(f); gt.cls.astype(np.int32).tofile(f)
        np.asarray(p_box).astype(np.float32).tofile(f); np.asarray(p_cls).astype(np.int32).tofile(f)


def run_cpp(mode, in_bin, out_bin, *args):
    r = subprocess.run([str(CPP_EXE), mode, str(in_bin), str(out_bin), *map(str, args)],
                       capture_output=True, text=True, check=True)
    sec = float(r.stdout.strip().split()[-1])
    with open(out_bin, "rb") as f:
        n_pred, n_gt = np.fromfile(f, np.int32, 2)
        if mode == "taxonomy":
            return (np.fromfile(f, np.int32, n_pred), np.fromfile(f, np.int32, n_pred),
                    np.fromfile(f, np.int32, n_gt)), sec
        return np.fromfile(f, np.int32, n_pred), sec


def filtered_preds(gt, pr, conf_thr=CONF_THR):
    """Predictions with conf >= thr (still sorted by image, conf desc) + offsets."""
    keep = pr.conf >= conf_thr
    off = np.zeros(gt.n_img + 1, np.int64)
    np.add.at(off, pr.img[keep] + 1, 1)
    return keep, np.cumsum(off)


# ------------------------------------------------------------------ aggregation
def aggregate(gt, pr, keep, label, ref, missed):
    """Counts per class and per slice. Returns dict."""
    p_cls = pr.cls[keep]
    idx = np.nonzero(keep)[0]
    n_c = len(CLASSES)
    per_class = {c: {l: 0 for l in LABELS + ["missed"]} for c in CLASSES}
    for k, lab in enumerate(label):
        # a correct/dup/loc/cls error is attributed to the predicted class
        per_class[CLASSES[p_cls[k]]][LABELS[lab]] += 1
    for g in np.nonzero(missed)[0]:
        per_class[CLASSES[gt.cls[g]]]["missed"] += 1
    overall = {l: sum(per_class[c][l] for c in CLASSES) for l in LABELS + ["missed"]}

    per_slice = {}
    for s in gt.slice_names:
        gm = gt.gt_mask[s]
        n_gt = int(gm.sum()); n_missed = int((missed.astype(bool) & gm).sum())
        entry = {"n_gt": n_gt, "missed": n_missed, "correct": n_gt - n_missed,
                 "recall": (n_gt - n_missed) / n_gt if n_gt else float("nan")}
        pm = pr.mask.get(s)
        if pm is not None:
            pmk = pm[idx]
            for li, l in enumerate(LABELS[1:], start=1):
                entry[l] = int(((label == li) & pmk).sum())
            entry["n_pred_in_slice"] = int(pmk.sum())
            entry["precision_proxy"] = float(((label == 0) & pmk).sum() / max(pmk.sum(), 1))
        per_slice[s] = entry
    return {"conf_threshold": CONF_THR, "tp_iou": TP_IOU, "loc_iou": LOC_IOU,
            "n_predictions_kept": int(keep.sum()), "overall": overall,
            "per_class": per_class, "per_slice": per_slice}
