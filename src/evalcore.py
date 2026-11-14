"""Evaluation core: COCO-style AP (101-point interpolation), greedy IoU matching (numba),
per-slice AP, and per-image bootstrap weights.

Slice AP rule (documented in README):
  * GT boxes outside the slice are ignored.
  * A prediction matched (at the evaluation IoU) to an out-of-slice GT is ignored.
  * An UNMATCHED prediction counts as a false positive only if the prediction itself can be
    assigned to the slice (image-level slices, box size, border); for slices that depend on GT-only
    attributes (attr:*) unmatched predictions are ignored, so those slices measure recall-weighted AP.
  * Slice mAP averages only classes with >= MIN_CLASS_GT GT boxes in the slice.
"""
import json

import numpy as np
from numba import njit

from common import *
import slices as S

IOU_THRS = np.round(np.linspace(0.5, 0.95, 10), 2)
MIN_CLASS_GT = 5
REC_THRS = np.linspace(0, 1, 101)


# ------------------------------------------------------------------ data
class GT:
    """Val ground truth flattened to arrays, sorted by image."""

    def __init__(self):
        self.images = json.load(open(META_OUT / "val.json"))["images"]
        img, xyxy, cls, tags = [], [], [], []
        for im in self.images:
            for b in im["boxes"]:
                img.append(im["id"]); xyxy.append(b["xyxy"]); cls.append(b["cls"])
                tags.append(S.box_tags(im, b["xyxy"], b["attrs"]))
        self.img = np.array(img, np.int32)
        self.xyxy = np.array(xyxy, np.float32).reshape(-1, 4)
        self.cls = np.array(cls, np.int16)
        self.n_img = len(self.images)
        self.off = np.zeros(self.n_img + 1, np.int64)
        np.add.at(self.off, self.img + 1, 1)
        self.off = np.cumsum(self.off)
        sl = json.load(open(RESULTS / "slices.json"))["splits"]["val"]["slices"]
        self.slice_names = [k for k, v in sl.items() if v["sufficient_for_claims"]]
        self.excluded_slices = [k for k, v in sl.items() if not v["sufficient_for_claims"]]
        self.gt_mask = {s: np.array([s in t for t in tags]) for s in self.slice_names}
        self.tags = tags


class Preds:
    """Predictions sorted by (image, conf desc) with per-slice masks for taggable slices."""

    def __init__(self, gt, run=None, path=None, min_conf=0.0):
        z = dict(np.load(path or RESULTS / "preds" / f"{run}.npz"))
        if min_conf > 0:  # optional confidence floor (the gate uses 0.01 for speed; see gate_config.yaml)
            k = z["conf"] >= min_conf
            z = {a: v[k] for a, v in z.items()}
        order = np.lexsort((-z["conf"], z["img"]))
        self.img = z["img"][order]; self.xyxy = z["xyxy"][order]
        self.conf = z["conf"][order]; self.cls = z["cls"][order].astype(np.int16)
        self.off = np.zeros(gt.n_img + 1, np.int64)
        np.add.at(self.off, self.img + 1, 1)
        self.off = np.cumsum(self.off)
        im = gt.images
        loc = np.array([x["location"] for x in im]); cam = np.array([x["camera"] for x in im])
        dark = np.array([x["brightness"] < S.DARK_THRESH for x in im])
        w = np.array([x["w"] for x in im]); h = np.array([x["h"] for x in im])
        b = self.xyxy; area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        pi = self.img
        edge = (b[:, 0] <= S.EDGE_PX) | (b[:, 1] <= S.EDGE_PX) | (b[:, 2] >= w[pi] - S.EDGE_PX) | (b[:, 3] >= h[pi] - S.EDGE_PX)
        self.mask = {}
        for s in gt.slice_names:
            if s.startswith("loc:"): self.mask[s] = loc[pi] == s[4:]
            elif s.startswith("cam:"): self.mask[s] = cam[pi] == s[4:]
            elif s == "bright:dark": self.mask[s] = dark[pi]
            elif s == "bright:bright": self.mask[s] = ~dark[pi]
            elif s == "size:small": self.mask[s] = area < S.SMALL
            elif s == "size:large": self.mask[s] = area > S.LARGE
            elif s == "size:medium": self.mask[s] = (area >= S.SMALL) & (area <= S.LARGE)
            elif s == "edge:touches_border": self.mask[s] = edge
            # attr:* -> no mask (not assignable to predictions)
        self.order_by_class = [np.argsort(-self.conf[self.cls == c], kind="stable") for c in range(len(CLASSES))]
        self.idx_by_class = [np.nonzero(self.cls == c)[0] for c in range(len(CLASSES))]


# ------------------------------------------------------------------ matching
@njit(cache=True)
def _match(p_off, p_xyxy, p_cls, g_off, g_xyxy, g_cls, thr):
    """COCO-style greedy matching. Predictions must be sorted by (image, conf desc).
    Returns for each prediction the matched GT index (global) or -1."""
    n = p_xyxy.shape[0]
    out = np.full(n, -1, np.int64)
    taken = np.zeros(g_xyxy.shape[0], np.bool_)
    for im in range(len(p_off) - 1):
        for p in range(p_off[im], p_off[im + 1]):
            best = min(thr, 1 - 1e-10)
            m = -1
            px1, py1, px2, py2 = p_xyxy[p, 0], p_xyxy[p, 1], p_xyxy[p, 2], p_xyxy[p, 3]
            pa = (px2 - px1) * (py2 - py1)
            for g in range(g_off[im], g_off[im + 1]):
                if taken[g] or g_cls[g] != p_cls[p]:
                    continue
                iw = min(px2, g_xyxy[g, 2]) - max(px1, g_xyxy[g, 0])
                ih = min(py2, g_xyxy[g, 3]) - max(py1, g_xyxy[g, 1])
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                ga = (g_xyxy[g, 2] - g_xyxy[g, 0]) * (g_xyxy[g, 3] - g_xyxy[g, 1])
                iou = inter / (pa + ga - inter)
                if iou < best:
                    continue
                best = iou
                m = g
            if m >= 0:
                taken[m] = True
                out[p] = m
    return out


def match_all(gt, pr):
    """matches[t] = array (n_pred,) of matched GT index or -1, for each IoU threshold."""
    return np.stack([_match(pr.off, pr.xyxy.astype(np.float64), pr.cls, gt.off,
                            gt.xyxy.astype(np.float64), gt.cls, float(t)) for t in IOU_THRS])


# ------------------------------------------------------------------ AP
def _ap101(tp, fp, w, ngt):
    """tp/fp/w sorted by conf desc (only valid entries). Returns COCO 101-pt AP."""
    if ngt <= 0:
        return np.nan
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(w * tp); cfp = np.cumsum(w * fp)
    rec = ctp / ngt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, REC_THRS, side="left")
    q = np.zeros(len(REC_THRS))
    ok = idx < len(prec)
    q[ok] = prec[idx[ok]]
    return q.mean()


def class_aps(gt, pr, match_t, slice_name=None, w_img=None):
    """Per-class AP at one IoU threshold for a slice (None = whole val set).
    Returns (aps[8] with nan where < MIN_CLASS_GT GT, ngt[8])."""
    w_gt = np.ones(len(gt.cls)) if w_img is None else w_img[gt.img].astype(float)
    w_p = np.ones(len(pr.cls)) if w_img is None else w_img[pr.img].astype(float)
    if slice_name is None:
        gmask = np.ones(len(gt.cls), bool); pmask = np.ones(len(pr.cls), bool)
    else:
        gmask = gt.gt_mask[slice_name]; pmask = pr.mask.get(slice_name)
    matched = match_t >= 0
    safe = np.where(matched, match_t, 0)
    tp_all = matched & gmask[safe]
    if pmask is None:
        fp_all = np.zeros(len(pr.cls), bool)
    else:
        fp_all = (~matched) & pmask
    valid = tp_all | fp_all
    aps = np.full(len(CLASSES), np.nan); ngts = np.zeros(len(CLASSES))
    for c in range(len(CLASSES)):
        gsel = gmask & (gt.cls == c)
        ngt = w_gt[gsel].sum(); ngts[c] = (gsel.sum() if w_img is None else ngt)
        if gsel.sum() < MIN_CLASS_GT:
            continue
        idx = pr.idx_by_class[c][pr.order_by_class[c]]
        idx = idx[valid[idx]]
        aps[c] = _ap101(tp_all[idx].astype(float), fp_all[idx].astype(float), w_p[idx], ngt)
    return aps, ngts


class APPrep:
    """Weight-independent pieces of AP for one (slice, IoU threshold): per class the conf-sorted valid
    predictions with their tp/fp flags, so a bootstrap resample only needs weighted cumsums."""

    def __init__(self, gt, pr, match_t, slice_name=None):
        gmask = np.ones(len(gt.cls), bool) if slice_name is None else gt.gt_mask[slice_name]
        pmask = np.ones(len(pr.cls), bool) if slice_name is None else pr.mask.get(slice_name)
        matched = match_t >= 0
        tp_all = matched & gmask[np.where(matched, match_t, 0)]
        fp_all = np.zeros(len(pr.cls), bool) if pmask is None else ((~matched) & pmask)
        valid = tp_all | fp_all
        self.cls = []
        for c in range(len(CLASSES)):
            gidx = np.nonzero(gmask & (gt.cls == c))[0]
            if len(gidx) < MIN_CLASS_GT:
                self.cls.append(None); continue
            idx = pr.idx_by_class[c][pr.order_by_class[c]]
            idx = idx[valid[idx]]
            self.cls.append((idx, tp_all[idx].astype(np.float64), fp_all[idx].astype(np.float64), gidx))

    def ap(self, gt, pr, w_img):
        w_gt = w_img[gt.img]; w_p = w_img[pr.img]
        aps = [_ap101(tp, fp, w_p[idx], w_gt[gidx].sum())
               for (idx, tp, fp, gidx) in (x for x in self.cls if x is not None)]
        return float(np.nanmean(aps)) if aps else float("nan")


def mean_ap(gt, pr, matches, slice_name=None, thr_idx=(0,), w_img=None):
    """Mean over classes and the given IoU thresholds (indices into IOU_THRS)."""
    per = []
    for t in thr_idx:
        a, n = class_aps(gt, pr, matches[t], slice_name, w_img)
        per.append(a)
    per = np.array(per)  # (T, C)
    return float(np.nanmean(per)) if not np.all(np.isnan(per)) else float("nan"), per, n
