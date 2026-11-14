"""Collect results into results/summary.json + results/experiment_table.md and plot training curves.

Reads: results/training/*_results.csv + *_meta.json, results/eval/*.json, results/taxonomy/*.json
Writes: results/summary.json, results/experiment_table.md, figures/training_curves.png
Every number quoted in README.md / report.md must be traceable to summary.json (verify_claims.py checks).
"""
import csv, json, statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import *

ORDER = ["baseline", "strong_aug", "hires", "degraded"]
FACTOR = {"baseline": "reference (imgsz 640, default aug, 12 epochs)",
          "strong_aug": "augmentation strength (mixup .2, scale .9, hsv_s .9, hsv_v .6, rot 5deg)",
          "hires": "input resolution (imgsz 960)",
          "degraded": "training length (1 epoch)"}


def read_csv(p):
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): float(v) for k, v in r.items()} for r in rows]


def write_config():
    import platform, torch, ultralytics, numpy, importlib.metadata as md
    import train as TR
    sl = json.load(open(RESULTS / "slices.json"))["splits"]
    cfg = {"dataset": "nuImages v1.0 (all), keyframes", "seed": SEED,
           "train_subset": {"images_used": sl["train"]["images"], "boxes": sl["train"]["boxes"],
                            "selection": f"random.Random({SEED}).sample of the 67k train keyframes",
                            "images_in_full_train_split": 67279},
           "val": {"images": sl["val"]["images"], "boxes": sl["val"]["boxes"], "selection": "FULL val split (no subsetting)"},
           "classes": CLASSES, "class_map": CLASS_MAP,
           "model": TR.BASE_WEIGHTS, "epochs_main": TR.EPOCHS, "batch": TR.BATCH, "configs": TR.CONFIGS,
           "curve_val_subset_images": 2000,
           "versions": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                        "numpy": numpy.__version__, "pycocotools": md.version("pycocotools"), "numba": md.version("numba")},
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (RESULTS / "config.json").write_text(json.dumps(cfg, indent=2))


def main():
    write_config()
    runs = {}
    for meta in sorted((RESULTS / "training").glob("*_meta.json")):
        m = json.load(open(meta)); name = f"{m['config']}_s{m['seed']}"
        ev = RESULTS / "eval" / f"{name}.json"
        if not ev.exists():
            continue
        e = json.load(open(ev))
        tx = RESULTS / "taxonomy" / f"{name}.json"
        t = json.load(open(tx)) if tx.exists() else None
        curve = read_csv(RESULTS / "training" / f"{name}_results.csv")
        runs[name] = {"config": m["config"], "seed": m["seed"], "train_minutes": round(m["train_seconds"] / 60, 1),
                      "epochs": len(curve), "imgsz": m["hyperparams"]["imgsz"],
                      "mAP50": e["overall"]["mAP50"], "mAP50-95": e["overall"]["mAP50-95"],
                      "per_class_AP50": {c: v["AP50"] for c, v in e["per_class"].items()},
                      "final_train_box_loss": curve[-1]["train/box_loss"], "final_train_cls_loss": curve[-1]["train/cls_loss"],
                      "taxonomy_overall": t["overall"] if t else None,
                      "cpp_python_identical": t["cpp_python_identical"] if t else None,
                      "pycocotools": e.get("pycocotools")}
    cfgs = defaultdict(list)
    for n, r in runs.items():
        cfgs[r["config"]].append(r)
    configs = {}
    for c in ORDER:
        if c not in cfgs:
            continue
        rs = cfgs[c]
        configs[c] = {"factor": FACTOR[c], "seeds": sorted(r["seed"] for r in rs), "n_seeds": len(rs)}
        for k in ("mAP50", "mAP50-95"):
            v = [r[k] for r in rs]
            configs[c][k] = {"mean": statistics.mean(v), "min": min(v), "max": max(v), "values": v}
    summary = {"runs": runs, "configs": configs,
               "note": "Seed variation is the range over 2 seeds; with n=2 it is a crude noise indicator, not a CI."}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["| Config | Varied factor | Seeds | mAP50 (mean, range) | mAP50-95 (mean, range) | Train min/run |",
             "|---|---|---|---|---|---|"]
    for c, v in configs.items():
        tm = statistics.mean(r["train_minutes"] for r in cfgs[c])
        lines.append(f"| {c} | {v['factor']} | {v['n_seeds']} | {v['mAP50']['mean']:.4f} ({v['mAP50']['min']:.4f}-{v['mAP50']['max']:.4f}) "
                     f"| {v['mAP50-95']['mean']:.4f} ({v['mAP50-95']['min']:.4f}-{v['mAP50-95']['max']:.4f}) | {tm:.1f} |")
    (RESULTS / "experiment_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # training curves from the real per-epoch CSVs (val metrics on the fixed 2000-image subset)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"baseline": "#1f77b4", "strong_aug": "#ff7f0e", "hires": "#2ca02c", "degraded": "#d62728"}
    for n, r in runs.items():
        cv = read_csv(RESULTS / "training" / f"{n}_results.csv")
        ep = [c["epoch"] for c in cv]
        ls = "-" if r["seed"] == 0 else "--"
        kw = dict(color=colors[r["config"]], linestyle=ls, marker="o" if len(ep) == 1 else None, label=n)
        ax[0].plot(ep, [c["train/box_loss"] for c in cv], **kw)
        ax[1].plot(ep, [c["train/cls_loss"] for c in cv], **kw)
        ax[2].plot(ep, [c["metrics/mAP50(B)"] for c in cv], **kw)
    for a, t in zip(ax, ["train box loss", "train cls loss", "val mAP50 (2000-img curve subset)"]):
        a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[2].legend(fontsize=7)
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(FIGURES / "training_curves.png", dpi=130)


if __name__ == "__main__":
    main()
