"""Stage 6: triage report - top failure patterns per model, each backed by numbers + example grids.

usage: python src/triage.py <run> [<run> ...]
Reads results/taxonomy/<run>_labels.npz (so run src/run_taxonomy.py first).
Writes results/triage/<run>.json and figures/<run>/pattern_XX_<name>.jpg

Pattern = conjunction of up to two slice tags from DIFFERENT groups (loc, cam, bright, size, edge, attr),
optionally restricted to one class. Missed-GT patterns are ranked by EXCESS misses
 (missed - overall_miss_rate * n_gt); patterns need >= MIN_SUPPORT GT boxes. Derived tags (size, bright,
 edge) are proxies, not ground truth, and are labelled as such in the output.
"""
import itertools, json, sys
from collections import Counter, defaultdict

import cv2
import numpy as np

from common import *
import evalcore as E
import slices as S
import taxonomy_ref as T

MIN_SUPPORT = 100
TOP_K = 8
FIG_PATTERNS = 5
GRID = (3, 4)
TILE = 224


def group_of(tag):
    return tag.split(":")[0]


def pretty(tags, cls):
    names = {"size:small": "small objects (<32^2 px)", "size:medium": "medium objects", "size:large": "large objects",
             "bright:dark": "dark images (brightness proxy for night)", "bright:bright": "bright images",
             "edge:touches_border": "boxes touching the image border (truncation proxy)"}
    parts = [names.get(t, t) for t in tags]
    s = " + ".join(parts) if parts else "all"
    return (f"{cls} " if cls else "") + s


def gt_patterns(gt, missed, near_cls, near_loc):
    """Rank (tag-pair x optional class) groups of GT boxes by excess missed count."""
    n = len(gt.cls)
    base = missed.mean()
    tag_sets = gt.tags
    # index boxes by tag
    by_tag = defaultdict(list)
    for i, ts in enumerate(tag_sets):
        for t in ts:
            by_tag[t].append(i)
    by_tag = {t: np.array(v) for t, v in by_tag.items()}
    # 'bright:bright' is the near-universal default (>99% of boxes): never informative on its own
    tags = sorted(t for t in by_tag if t != "bright:bright")
    cands = [((t,), by_tag[t]) for t in tags]
    for a, b in itertools.combinations(tags, 2):
        if group_of(a) == group_of(b):
            continue
        inter = np.intersect1d(by_tag[a], by_tag[b], assume_unique=True)
        if len(inter) >= MIN_SUPPORT:
            cands.append(((a, b), inter))
    # pass 1: miss rate of every candidate (component set = tags + optional class)
    table = {}
    for tg, idx in cands:
        for c in [None] + list(range(len(CLASSES))):
            sel = idx if c is None else idx[gt.cls[idx] == c]
            if len(sel) >= MIN_SUPPORT:
                comp = frozenset(tg) | (frozenset() if c is None else frozenset([f"class:{CLASSES[c]}"]))
                table[comp] = (tg, c, sel)
    for c in range(len(CLASSES)):  # class alone is also a sub-pattern
        sel = np.nonzero(gt.cls == c)[0]
        table.setdefault(frozenset([f"class:{CLASSES[c]}"]), ((), c, sel))
    rate = {k: missed[v[2]].mean() for k, v in table.items()}

    def informative(comp):
        """Every extra condition must raise the miss rate by >= 3 points over each simpler sub-pattern."""
        if len(comp) == 1:
            return True
        for r in range(1, len(comp)):
            for sub in itertools.combinations(comp, r):
                s = frozenset(sub)
                if s in rate and rate[comp] < rate[s] + 0.03:
                    return False
        return True

    out = []
    for comp, (tg, c, sel) in table.items():
        if not tg and c is not None:
            continue  # class-only groups are not 'slice' patterns
        m = int(missed[sel].sum())
        excess = m - base * len(sel)
        if excess <= 0 or not informative(comp):
            continue
        miss_sel = sel[missed[sel] == 1]
        out.append({"tags": list(tg), "class": None if c is None else CLASSES[c], "n_gt": int(len(sel)),
                    "missed": m, "miss_rate": m / len(sel), "overall_miss_rate": float(base),
                    "excess_missed": float(excess),
                    "missed_with_wrong_class_pred": int(near_cls[miss_sel].sum()),
                    "missed_with_poorly_localized_pred": int((near_loc[miss_sel] & ~near_cls[miss_sel]).sum()),
                    "missed_with_no_nearby_pred": int((~near_cls[miss_sel] & ~near_loc[miss_sel]).sum()),
                    "_idx": miss_sel})
    out.sort(key=lambda d: -d["excess_missed"])
    # drop near-duplicate patterns: keep a pattern only if <70% of its missed boxes are already covered
    kept, covered = [], set()
    for d in out:
        ids = set(d["_idx"].tolist())
        if len(ids & covered) / max(len(ids), 1) < 0.7:
            kept.append(d); covered |= ids
        if len(kept) == TOP_K:
            break
    return kept


def fp_patterns(gt, pr, keep, label):
    """For each FP type, which pred-side groups (class x size / bright / loc / cam) are over-represented."""
    idx = np.nonzero(keep)[0]
    out = {}
    names = ["classification_error", "localization_error", "duplicate", "background_fp"]
    for li, name in enumerate(names, start=1):
        is_t = label == li
        base = is_t.mean()
        rows = []
        for s, pm in pr.mask.items():
            for c in [None] + list(range(len(CLASSES))):
                sel = pm[idx]
                if c is not None:
                    sel = sel & (pr.cls[idx] == c)
                n = int(sel.sum())
                if n < MIN_SUPPORT:
                    continue
                k = int((is_t & sel).sum())
                rows.append({"slice": s, "class": None if c is None else CLASSES[c], "n_pred": n, "count": k,
                             "rate": k / n, "overall_rate": float(base), "excess": float(k - base * n)})
        rows.sort(key=lambda d: -d["excess"])
        out[name] = rows[:5]
    return out


def make_grid(gt, idxs, path, rng, preds=None):
    idxs = list(idxs)
    rng.shuffle(idxs)
    idxs = idxs[: GRID[0] * GRID[1]]
    tiles = []
    for gi in idxs:
        im = gt.images[gt.img[gi]]
        img = cv2.imread(str(YOLO_DIR / "images" / "val" / im["name"]))
        x1, y1, x2, y2 = gt.xyxy[gi]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 1.5 + 16
        a, b = int(max(0, cx - half)), int(max(0, cy - half))
        c, d = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        crop = img[b:d, a:c].copy()
        cv2.rectangle(crop, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (0, 0, 255), 2)
        sc = TILE / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))))
        t = np.zeros((TILE, TILE, 3), np.uint8)
        t[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(t, CLASSES[gt.cls[gi]], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) < GRID[0] * GRID[1]:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    rows = [np.hstack(tiles[r * GRID[1]:(r + 1) * GRID[1]]) for r in range(GRID[0])]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def triage(run, gt):
    pr = E.Preds(gt, run)
    z = np.load(RESULTS / "taxonomy" / f"{run}_labels.npz")
    keep_idx, label, ref, missed = z["keep"], z["label"], z["ref"], z["missed"].astype(bool)
    keep = np.zeros(len(pr.cls), bool); keep[keep_idx] = True
    near_cls = np.zeros(len(gt.cls), bool); near_loc = np.zeros(len(gt.cls), bool)
    near_cls[ref[(label == 1) & (ref >= 0)]] = True
    near_loc[ref[(label == 2) & (ref >= 0)]] = True
    pats = gt_patterns(gt, missed.astype(np.int8), near_cls, near_loc)
    rng = np.random.default_rng(SEED)
    for k, p in enumerate(pats[:FIG_PATTERNS]):
        name = "_".join(t.replace(":", "-") for t in p["tags"]) + (f"_{p['class']}" if p["class"] else "")
        fig = f"figures/{run}/pattern_{k+1:02d}_{name}.jpg"
        make_grid(gt, p["_idx"], ROOT / fig, rng)
        p["figure"] = fig
    for p in pats:
        p["description"] = (f"{pretty(p['tags'], p['class'])}: {p['missed']}/{p['n_gt']} GT boxes missed "
                            f"({100*p['miss_rate']:.1f}% vs {100*p['overall_miss_rate']:.1f}% overall)")
        p["derived_tags_are_proxies"] = any(not S.is_native(t) for t in p["tags"])
        p.pop("_idx")
    res = {"run": run, "operating_point": {"conf": T.CONF_THR, "tp_iou": T.TP_IOU, "loc_iou": T.LOC_IOU},
           "overall_miss_rate": float(missed.mean()), "n_gt": int(len(gt.cls)), "n_missed": int(missed.sum()),
           "missed_gt_patterns": pats, "false_positive_patterns": fp_patterns(gt, pr, keep, label),
           "min_support": MIN_SUPPORT}
    (RESULTS / "triage").mkdir(parents=True, exist_ok=True)
    (RESULTS / "triage" / f"{run}.json").write_text(json.dumps(res, indent=2))
    print(f"== {run}  overall miss rate {100*missed.mean():.1f}%")
    for p in pats[:5]:
        print("  ", p["description"])
    return res


if __name__ == "__main__":
    gt = E.GT()
    for r in sys.argv[1:]:
        triage(r, gt)
