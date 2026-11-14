"""Stage 1b: slice definitions. Writes results/slices.json.

NATIVE slices come straight from nuImages metadata:
  loc:*   log location            cam:*  camera channel
  attr:*  object attribute (moving / parked / stopped / pedestrian pose / cycle rider)
DERIVED slices are computed by us and are NOT ground truth:
  bright:dark|bright  mean grayscale brightness < DARK_THRESH  (day/night PROXY only)
  size:small|medium|large  box area in original pixels (COCO 32^2 / 96^2 cut-offs)
  edge:touches_border  box within 2px of an image edge (truncation PROXY; nuImages has
                       no truncation or occlusion field)
"""
import json
from collections import Counter, defaultdict

from common import *

DARK_THRESH = 80.0
MIN_BOXES = 30  # slices below this are excluded from significance claims
SMALL, LARGE = 32 ** 2, 96 ** 2
EDGE_PX = 2

NATIVE_PREFIXES = ("loc:", "cam:", "attr:")
# slices for which a PREDICTION can be assigned (needs only image props + its own box)
PRED_TAGGABLE_PREFIXES = ("loc:", "cam:", "bright:", "size:", "edge:")


def size_tag(x1, y1, x2, y2):
    a = (x2 - x1) * (y2 - y1)
    return "size:small" if a < SMALL else ("size:large" if a > LARGE else "size:medium")


def edge_tag(x1, y1, x2, y2, w, h):
    return x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= w - EDGE_PX or y2 >= h - EDGE_PX


def image_tags(im):
    return ["loc:" + im["location"], "cam:" + im["camera"],
            "bright:dark" if im["brightness"] < DARK_THRESH else "bright:bright"]


def box_tags(im, xyxy, attrs=()):
    """All slice tags for one box (GT: pass attrs; predictions: attrs=())."""
    t = image_tags(im) + [size_tag(*xyxy)]
    if edge_tag(*xyxy, im["w"], im["h"]):
        t.append("edge:touches_border")
    t += ["attr:" + a for a in attrs
          if a.startswith(("vehicle.moving", "vehicle.parked", "vehicle.stopped", "pedestrian.", "cycle."))]
    return t


def is_native(slice_name):
    return slice_name.startswith(NATIVE_PREFIXES)


def pred_taggable(slice_name):
    return slice_name.startswith(PRED_TAGGABLE_PREFIXES)


def main():
    out = {"dark_threshold_mean_gray": DARK_THRESH, "min_boxes_for_claims": MIN_BOXES,
           "box_size_cutoffs_px2": [SMALL, LARGE], "edge_px": EDGE_PX, "splits": {}}
    for split in ("train", "val"):
        d = json.load(open(META_OUT / f"{split}.json"))["images"]
        img_cnt, box_cnt, cls_cnt = Counter(), Counter(), defaultdict(Counter)
        for im in d:
            seen = set()
            for b in im["boxes"]:
                for t in box_tags(im, b["xyxy"], b["attrs"]):
                    box_cnt[t] += 1
                    cls_cnt[t][CLASSES[b["cls"]]] += 1
                    seen.add(t)
            for t in set(image_tags(im)) | seen:
                img_cnt[t] += 1
        sl = {}
        for t in sorted(set(box_cnt) | set(img_cnt)):
            sl[t] = {"images": img_cnt[t], "boxes": box_cnt[t],
                     "kind": "native" if is_native(t) else "derived",
                     "sufficient_for_claims": box_cnt[t] >= MIN_BOXES,
                     "boxes_per_class": dict(cls_cnt[t])}
        out["splits"][split] = {"images": len(d), "boxes": sum(len(i["boxes"]) for i in d), "slices": sl}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "slices.json").write_text(json.dumps(out, indent=2))
    for t, v in out["splits"]["val"]["slices"].items():
        print(f"{t:28s} {v['kind']:8s} img={v['images']:6d} box={v['boxes']:7d} {'' if v['sufficient_for_claims'] else 'INSUFFICIENT (<30)'}")


if __name__ == "__main__":
    main()
