"""Stage 2: fine-tune a small pretrained detector (YOLO11s) under named configurations.

Every config differs from `baseline` in exactly ONE named factor:
  baseline    imgsz=640, default ultralytics augmentation, EPOCHS epochs
  hires       factor = input resolution          (imgsz=960)
  strong_aug  factor = augmentation strength     (mixup, bigger scale/hsv jitter, small rotation)
  degraded    factor = training length           (1 epoch)  -> deliberately weak model for the gate demo

usage: python src/train.py <config> <seed>
Per-epoch loss/val curves use a FIXED 2000-image val subset (val_curve.txt) to keep epochs fast;
the final numbers in the report come from our own evaluation on the FULL val set.
"""
import json, shutil, sys, time

from ultralytics import YOLO

from common import *

EPOCHS = 12
BATCH = 32
CONFIGS = {
    "baseline": dict(imgsz=640, epochs=EPOCHS),
    "hires": dict(imgsz=960, epochs=EPOCHS),
    "strong_aug": dict(imgsz=640, epochs=EPOCHS, mixup=0.2, scale=0.9, hsv_s=0.9, hsv_v=0.6, degrees=5.0),
    "degraded": dict(imgsz=640, epochs=1),
}
BASE_WEIGHTS = "yolo11s.pt"


def make_curve_yaml():
    imgs = sorted(p.name for p in (YOLO_DIR / "images" / "val").glob("*.jpg"))
    import random
    sub = random.Random(SEED).sample(imgs, 2000)
    lst = YOLO_DIR / "val_curve.txt"
    lst.write_text("\n".join((YOLO_DIR / "images" / "val" / n).as_posix() for n in sorted(sub)))
    y = YOLO_DIR / "dataset_curve.yaml"
    y.write_text(f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: val_curve.txt\nnames:\n"
                 + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return y


def main(cfg_name, seed):
    cfg = CONFIGS[cfg_name]
    name = f"{cfg_name}_s{seed}"
    yaml = make_curve_yaml()
    model = YOLO(BASE_WEIGHTS)
    t0 = time.time()
    model.train(data=str(yaml), batch=BATCH, seed=seed, workers=8, device=0, project=str(RUNS),
                name=name, exist_ok=True, plots=False, patience=100, cache=False, verbose=False, **cfg)
    wall = time.time() - t0
    out = RESULTS / "training"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUNS / name / "results.csv", out / f"{name}_results.csv")
    shutil.copy(RUNS / name / "weights" / "last.pt", RUNS / name / "final.pt")
    (out / f"{name}_meta.json").write_text(json.dumps(
        {"config": cfg_name, "seed": seed, "hyperparams": cfg, "base_weights": BASE_WEIGHTS,
         "batch": BATCH, "train_seconds": round(wall, 1), "weights": f"runs/{name}/final.pt"}, indent=2))
    print(f"done {name} in {wall/60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
