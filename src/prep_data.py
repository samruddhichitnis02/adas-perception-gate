"""Stage 1a: convert nuImages (train subset + full val) into YOLO format and a
rich ground-truth metadata file used for slicing / evaluation / error taxonomy.

Outputs
  data/yolo/images/{train,val}/*.jpg   (hard links to the original files, no copy)
  data/yolo/labels/{train,val}/*.txt
  data/yolo/dataset.yaml
  data/meta/{train,val}.json           per-image meta + GT boxes (xyxy px, class, native attrs)
  data/meta/attribute_audit.json       which native attributes exist (written before slicing)
"""
import json, os, random, shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2

from common import *


def load(split, name):
    with open(META_DIR / f"v1.0-{split}" / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def brightness(path):
    # decode at 1/4 resolution, grayscale: fast and good enough for a day/night proxy
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return float(img.mean()) if img is not None else float("nan")


def build_split(split, subset_n=None):
    cats = {c["token"]: c["name"] for c in load(split, "category")}
    attrs = {a["token"]: a["name"] for a in load(split, "attribute")}
    logs = {l["token"]: l for l in load(split, "log")}
    samples = {s["token"]: s for s in load(split, "sample")}
    sensors = {s["token"]: s["channel"] for s in load(split, "sensor")}
    calib = {c["token"]: c["sensor_token"] for c in load(split, "calibrated_sensor")}
    sd = {d["token"]: d for d in load(split, "sample_data") if d["is_key_frame"]}

    anns_by_sd = {}
    n_ann_total = 0
    for a in load(split, "object_ann"):
        n_ann_total += 1
        anns_by_sd.setdefault(a["sample_data_token"], []).append(a)

    keys = sorted(sd)  # deterministic order
    if subset_n and subset_n < len(keys):
        rng = random.Random(SEED)
        keys = sorted(rng.sample(keys, subset_n))

    images, dropped_cat = [], Counter()
    for k in keys:
        d = sd[k]
        s = samples[d["sample_token"]]
        log = logs[s["log_token"]]
        w, h = d["width"], d["height"]
        boxes = []
        for a in anns_by_sd.get(k, []):
            cname = cats[a["category_token"]]
            if cname not in CLASS_MAP:
                dropped_cat[cname] += 1
                continue
            x1, y1, x2, y2 = a["bbox"]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append({
                "xyxy": [x1, y1, x2, y2],
                "cls": CLASS_ID[CLASS_MAP[cname]],
                "attrs": sorted(attrs[t] for t in a["attribute_tokens"]),
            })
        images.append({
            "id": len(images),
            "file": d["filename"],  # e.g. samples/CAM_FRONT/xxx.jpg
            "name": os.path.basename(d["filename"]),
            "w": w, "h": h,
            "camera": sensors[calib[d["calibrated_sensor_token"]]],
            "location": log["location"],
            "log": log["logfile"],
            "date": log["date_captured"],
            "boxes": boxes,
        })
    return images, {"annotations_in_split": n_ann_total, "dropped_unmapped_categories": dict(dropped_cat)}


def materialize(split, images):
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split
    for p in (img_dir, lbl_dir):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    def one(im):
        src = SAMPLES_DIR / im["file"]
        dst = img_dir / im["name"]
        try:
            os.link(src, dst)  # same volume -> zero extra disk
        except OSError:
            shutil.copy2(src, dst)
        lines = []
        for b in im["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            lines.append(f"{b['cls']} {(x1+x2)/2/im['w']:.6f} {(y1+y2)/2/im['h']:.6f} "
                         f"{(x2-x1)/im['w']:.6f} {(y2-y1)/im['h']:.6f}")
        (lbl_dir / (im["name"][:-4] + ".txt")).write_text("\n".join(lines))
        im["brightness"] = brightness(src)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, images))


def main():
    META_OUT.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", TRAIN_SUBSET_IMAGES), ("val", None)):
        print(f"[{split}] building ...")
        images, info = build_split(split, n)
        materialize(split, images)
        nb = sum(len(i["boxes"]) for i in images)
        print(f"[{split}] images={len(images)} boxes={nb} {info}")
        (META_OUT / f"{split}.json").write_text(json.dumps({"info": info, "images": images}))

    (YOLO_DIR / "dataset.yaml").write_text(
        f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    # Attribute audit: what native metadata actually exists (written BEFORE slices are chosen)
    attr_names = [a["name"] for a in load("val", "attribute")]
    log_keys = sorted(load("val", "log")[0].keys())
    sample_keys = sorted(load("val", "sample")[0].keys())
    ann_keys = sorted(load("val", "object_ann")[0].keys())
    audit = {
        "log_fields": log_keys, "sample_fields": sample_keys, "object_ann_fields": ann_keys,
        "object_attributes": attr_names,
        "native_available": ["location (log)", "camera channel (sensor)", "date_captured (log)",
                             "object attributes (moving/parked/stopped, pedestrian pose, cycle rider)"],
        "native_missing": ["time of day", "weather", "truncation", "occlusion"],
    }
    (META_OUT / "attribute_audit.json").write_text(json.dumps(audit, indent=2))
    print("audit written")


if __name__ == "__main__":
    main()
