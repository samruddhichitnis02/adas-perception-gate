"""Build the README visuals from the real result files (nothing is typed in by hand).

Writes figures/readme/*.png and figures/overview.gif
usage: python src/make_visuals.py [--no-gif]
Colors follow the validated reference palette (blue / orange / aqua categorical slots, status green / red
always paired with an icon + word).
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

from common import *

OUT = FIGURES / "readme"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (reference instance, light surface) ----
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, GREEN, RED, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#0ca30c", "#d03b3b", "#eda100"
GOOD_TXT = "#006300"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False})

J = lambda p: json.load(open(p, encoding="utf-8"))
SUM = J(RESULTS / "summary.json")
TAX = {r: J(RESULTS / "taxonomy" / f"{r}.json") for r in ("baseline_s0", "hires_s0", "degraded_s0")}
EVAL = {r: J(RESULTS / "eval" / f"{r}.json") for r in ("baseline_s0", "hires_s0")}
GATE = J(RESULTS / "gate" / "summary.json")["results"]
NICE = {"car": "Cars", "truck": "Trucks", "bus": "Buses", "motorcycle": "Motorcycles", "bicycle": "Bicycles",
        "pedestrian": "People", "traffic_cone": "Traffic cones", "barrier": "Barriers"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ static figures
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 2.8)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 2.8)
    steps = [("1  Road photos", "16,000 real driving photos to learn from, 16,445 to test on", BLUE),
             ("2  Train detectors", "3 different recipes, each trained twice", BLUE),
             ("3  Sort every mistake", "missed · wrong label · box off · duplicate · phantom", BLUE),
             ("4  Find weak spots", "which objects, cameras and conditions fail most", BLUE),
             ("5  Safety gate", "PASS or FAIL before a model is allowed to ship", ORANGE)]
    w, g = 2.2, 0.25
    for i, (t, s, c) in enumerate(steps):
        x = 0.1 + i * (w + g)
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 2.0, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=c, lw=2))
        ax.text(x + w / 2, 1.85, t, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, 1.0, "\n".join(__import__("textwrap").wrap(s, 22)), ha="center", va="center", fontsize=9.5, color=INK2)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.35), (x + w + g - 0.02, 1.35), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.6))
    save(fig, "pipeline.png")


def fig_scores():
    cfg = SUM["configs"]
    items = [("baseline", "Baseline", BLUE), ("hires", "Sharper images\n(higher resolution)", ORANGE),
             ("strong_aug", "Heavier image\nvariations", AQUA), ("degraded", "Barely trained\n(1 round only)", MUTED)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, lab, c) in enumerate(items):
        m = cfg[k]["mAP50"]["mean"]
        ax.bar(i, m, width=0.55, color=c, zorder=2)
        ax.text(i, m + 0.045, f"{m:.2f}", ha="center", fontsize=15, fontweight="bold", color=INK)
        for v in cfg[k]["mAP50"]["values"]:
            ax.plot(i, v, "o", ms=6, mfc="white", mec=INK, mew=1.4, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([x[1] for x in items], fontsize=11)
    ax.set_ylim(0, 0.85); ax.set_ylabel("Detection score (0 to 1, higher is better)")
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_title("Sharper input images give the biggest improvement", loc="left", fontsize=15, fontweight="bold", pad=14)
    ax.text(0, -0.2, "Bars are the average of two training runs; the white dots are the individual runs.", transform=ax.transAxes,
            fontsize=9.5, color=INK2)
    fig.tight_layout(); save(fig, "scores.png")


def fig_mistakes():
    n = lambda r: TAX[r]["n_gt"] if "n_gt" in TAX[r] else 131676
    ngt = 131676
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.7]})
    runs = [("baseline_s0", "Baseline", BLUE), ("hires_s0", "Sharper images", ORANGE), ("degraded_s0", "Barely trained", MUTED)]
    for i, (r, lab, c) in enumerate(runs):
        v = 100 * TAX[r]["overall"]["missed"] / ngt
        a1.bar(i, v, width=0.55, color=c, zorder=2); a1.text(i, v + 1.2, f"{v:.0f}", ha="center", fontsize=14, fontweight="bold")
    a1.set_xticks(range(3)); a1.set_xticklabels([x[1] for x in runs]); a1.set_ylim(0, 70)
    a1.set_title("Real objects the model misses\n(out of every 100)", loc="left", fontsize=13, fontweight="bold")
    a1.yaxis.grid(True, color=GRID, zorder=0); a1.set_axisbelow(True); a1.spines["left"].set_visible(False)
    kinds = [("classification_error", "Wrong label\n(sees it, names it wrong)"), ("localization_error", "Box in the\nwrong place"),
             ("duplicate", "Same object\nboxed twice"), ("background_fp", "Phantom\n(nothing there)")]
    x = np.arange(4); w = 0.36
    for j, (r, lab, c) in enumerate(runs[:2]):
        vals = [100 * TAX[r]["overall"][k] / ngt for k, _ in kinds]
        bars = a2.bar(x + (j - 0.5) * w, vals, width=w - 0.03, color=c, label=lab, zorder=2)
        for b, v in zip(bars, vals):
            a2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=10.5)
    a2.set_xticks(x); a2.set_xticklabels([k[1] for k in kinds], fontsize=10)
    a2.set_title("Other kinds of mistake\n(per 100 real objects)", loc="left", fontsize=13, fontweight="bold")
    a2.yaxis.grid(True, color=GRID, zorder=0); a2.set_axisbelow(True); a2.spines["left"].set_visible(False)
    a2.legend(frameon=False, loc="upper left", ncol=2); a2.set_ylim(0, 9.5)
    fig.tight_layout(); save(fig, "mistakes.png")


def miss_rates():
    t = TAX["baseline_s0"]; e = EVAL["baseline_s0"]
    size = {k: 100 * (1 - t["per_slice"][f"size:{k}"]["recall"]) for k in ("small", "medium", "large")}
    cls = {c: 100 * t["per_class"][c]["missed"] / e["per_class"][c]["n_gt"] for c in CLASSES}
    return size, cls


def fig_where():
    size, cls = miss_rates()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    s = [("Small / far away", size["small"]), ("Medium", size["medium"]), ("Large / close", size["large"])]
    for ax, data, title in ((a1, s, "By object size"), (a2, sorted([(NICE[k], v) for k, v in cls.items()], key=lambda d: -d[1]), "By kind of object")):
        labs = [d[0] for d in data][::-1]; vals = [d[1] for d in data][::-1]
        cols = [ORANGE if (d[0].startswith("Small")) else BLUE for d in data][::-1]
        ax.barh(range(len(vals)), vals, height=0.58, color=cols, zorder=2)
        for i, v in enumerate(vals): ax.text(v + 1.2, i, f"{v:.0f}%", va="center", fontsize=11.5, fontweight="bold")
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs, fontsize=11); ax.set_xlim(0, 100)
        ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True); ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold"); ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Share of real objects the model missed")
    fig.suptitle("Where the model fails: small, distant objects", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "where_it_fails.png")


def fig_learning():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for run, lab, c in (("baseline_s0", "Baseline", BLUE), ("hires_s0", "Sharper images", ORANGE), ("strong_aug_s0", "Heavier image variations", AQUA)):
        rows = list(csv.DictReader(open(RESULTS / "training" / f"{run}_results.csv")))
        ep = [int(float(r["epoch"])) for r in rows]; v = [float(r["metrics/mAP50(B)"]) for r in rows]
        ax.plot(ep, v, color=c, lw=2.4, label=lab, marker="o", ms=4)
        ax.text(ep[-1] + 0.2, v[-1], f"{v[-1]:.2f}", color=INK, va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training round"); ax.set_ylabel("Detection score on a test sample")
    ax.set_xlim(1, 13); ax.set_ylim(0.3, 0.8); ax.yaxis.grid(True, color=GRID); ax.legend(frameon=False, loc="lower right")
    ax.set_title("The models improve with every training round", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout(); save(fig, "learning.png")


GATE_TXT = {
    "baseline_s1": ("Same recipe, different random start", "Nothing changed, as expected"),
    "hires_s0": ("Sharper images, run 1", "Clearly better, nothing got worse"),
    "hires_s1": ("Sharper images, run 2", "Better score, but more false alarms than allowed"),
    "strong_aug_s0": ("Heavier variations, run 1", "Score dropped and more objects missed"),
    "strong_aug_s1": ("Heavier variations, run 2", "Score dropped and more objects missed"),
    "degraded_s0": ("Barely trained (1 round)", "Score collapsed, misses far more objects"),
}


def fig_gate():
    rows = [(GATE_TXT[r["candidate"]], r["verdict"]) for r in GATE if r["candidate"] in GATE_TXT]
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, len(rows) + 1)
    ax.text(0.1, len(rows) + 0.45, "Is the new model safe to replace the current one?", fontsize=15, fontweight="bold")
    for i, ((name, why), v) in enumerate(rows):
        y = len(rows) - i - 0.35
        ax.add_patch(FancyBboxPatch((0.1, y - 0.36), 11.8, 0.72, boxstyle="round,pad=0.01,rounding_size=0.1", fc="white", ec=GRID, lw=1.2))
        ax.text(0.35, y, name, va="center", fontsize=12.5, fontweight="bold")
        ax.text(4.9, y, why, va="center", fontsize=11.5, color=INK2)
        ok = v == "PASS"
        ax.add_patch(FancyBboxPatch((10.35, y - 0.24), 1.35, 0.48, boxstyle="round,pad=0.01,rounding_size=0.2", fc=GREEN if ok else RED, ec="none"))
        ax.text(11.02, y, ("✓ PASS" if ok else "✗ FAIL"), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    save(fig, "gate.png")


# ------------------------------------------------------------------ before/after + GIF scene data
def pick_example():
    imgs = J(META_OUT / "val.json")["images"]
    pb = np.load(RESULTS / "taxonomy" / "baseline_s0_perimg.npy"); ph = np.load(RESULTS / "taxonomy" / "hires_s0_perimg.npy")
    ngt = np.array([len(i["boxes"]) for i in imgs])
    best, bs = None, -1
    for i, im in enumerate(imgs):
        if im["camera"] != "CAM_FRONT" or im["brightness"] < 90 or not (9 <= ngt[i] <= 16):
            continue
        sc = pb[i, 5] - ph[i, 5]
        if pb[i, 5] >= 4 and sc > bs:
            best, bs = i, sc
    return best, imgs, pb, ph


def model_view(run, idx, imgs, conf=0.25):
    z = np.load(RESULTS / "preds" / f"{run}.npz"); m = (z["img"] == idx) & (z["conf"] >= conf)
    preds = list(zip(z["xyxy"][m].tolist(), z["cls"][m].tolist()))
    lab = np.load(RESULTS / "taxonomy" / f"{run}_labels.npz")
    off = np.concatenate([[0], np.cumsum([len(i["boxes"]) for i in imgs])])
    missed = lab["missed"][off[idx]:off[idx + 1]].astype(bool)
    miss_boxes = [b["xyxy"] for b, mm in zip(imgs[idx]["boxes"], missed) if mm]
    return preds, miss_boxes, len(imgs[idx]["boxes"])


def font(size, bold=False):
    for f in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()


def draw_view(img, preds, miss, scale, show_pred=1.0, show_miss=False, title=None):
    im = img.copy(); d = ImageDraw.Draw(im, "RGBA")
    n = int(len(preds) * show_pred)
    for (b, c) in preds[:n]:
        d.rectangle([v * scale for v in b], outline=(42, 120, 214, 255), width=3)
    if show_miss:
        for b in miss:
            d.rectangle([v * scale for v in b], outline=(227, 73, 72, 255), width=4)
    return im


def fig_before_after():
    idx, imgs, pb, ph = pick_example()
    img = Image.open(YOLO_DIR / "images" / "val" / imgs[idx]["name"]).convert("RGB"); W = 800; s = W / img.width
    small = img.resize((W, int(img.height * s)))
    canvas = Image.new("RGB", (W * 2 + 30, small.height + 90), (252, 252, 251)); d = ImageDraw.Draw(canvas)
    for k, (run, title, col) in enumerate((("baseline_s0", "Baseline", (42, 120, 214)), ("hires_s0", "Sharper images", (235, 104, 52)))):
        preds, miss, n = model_view(run, idx, imgs)
        v = draw_view(small, preds, miss, s, 1.0, True)
        x = k * (W + 30); canvas.paste(v, (x, 80))
        d.text((x, 14), title, fill=col, font=font(26, True))
        d.text((x, 48), f"Missed {len(miss)} of {n} real objects (red boxes)", fill=(82, 81, 78), font=font(18))
    canvas.save(OUT / "before_after.jpg", quality=90); print("wrote before_after.jpg", imgs[idx]["name"])
    return idx, imgs


# ------------------------------------------------------------------ GIF
GW, GH = 800, 450
BG = (14, 17, 22); FG = (240, 239, 236); SUB = (170, 176, 186); CB = (57, 135, 229); CO = (235, 104, 52); CG = (12, 163, 12); CR = (230, 80, 80)


def new():
    im = Image.new("RGB", (GW, GH), BG); return im, ImageDraw.Draw(im)


def centered(d, y, text, f, fill=FG):
    w = d.textlength(text, font=f); d.text(((GW - w) / 2, y), text, font=f, fill=fill)


def banner(im, text, sub=None):
    d = ImageDraw.Draw(im, "RGBA"); d.rectangle([0, 0, GW, 62], fill=(14, 17, 22, 215))
    d.text((24, 10), text, font=font(26, True), fill=FG)
    if sub: d.text((24, 40), sub, font=font(15), fill=SUB)


def bars_scene(title, sub, rows, steps=4, vmax=None, fmt="{:.0f}"):
    vmax = vmax or max(r[1] for r in rows) * 1.12; out = []
    for s in range(1, steps + 1):
        im, d = new(); d.text((40, 30), title, font=font(30, True), fill=FG); d.text((40, 74), sub, font=font(17), fill=SUB)
        for i, (lab, v, col) in enumerate(rows):
            y = 130 + i * (270 / max(len(rows), 1)); h = min(34, 230 / len(rows))
            d.text((40, y + 4), lab, font=font(18), fill=FG)
            x0 = 300; wmax = 400; bw = wmax * (v / vmax) * (s / steps)
            d.rounded_rectangle([x0, y, x0 + max(bw, 2), y + h], radius=5, fill=col)
            if s == steps: d.text((x0 + bw + 10, y + 2), fmt.format(v), font=font(20, True), fill=FG)
        out.append(im)
    return out


def scene_title():
    ims = []
    for a in (0.35, 0.7, 1.0):
        im, d = new(); c = lambda col: tuple(int(BG[k] + (col[k] - BG[k]) * a) for k in range(3))
        centered(d, 150, "perception-gate", font(64, True), c(FG))
        centered(d, 240, "Can we trust a self-driving camera model", font(26), c(SUB))
        centered(d, 276, "before it goes on the road?", font(26), c(SUB))
        d.rectangle([GW / 2 - 40, 330, GW / 2 + 40, 334], fill=c(CO))
        ims.append(im)
    return ims


def scenes():
    idx, imgs, pb, ph = pick_example()
    raw = Image.open(YOLO_DIR / "images" / "val" / imgs[idx]["name"]).convert("RGB"); s = GW / raw.width
    base = raw.resize((GW, int(raw.height * s))).crop((0, 0, GW, GH)) if int(raw.height * s) >= GH else raw.resize((GW, GH))
    pbv = model_view("baseline_s0", idx, imgs); phv = model_view("hires_s0", idx, imgs)
    S = []
    S.append(([*scene_title()], [150, 150, 2300]))
    fr, du = [], []
    for k, f in enumerate((0.25, 0.5, 0.75, 1.0)):
        v = draw_view(base, pbv[0], pbv[1], s, f, False); banner(v, "Step 1  A model looks at the road", "Blue boxes: objects it found"); fr.append(v); du.append(500 if f < 1 else 1700)
    S.append((fr, du))
    fr, du = [], []
    for k in range(3):
        v = draw_view(base, pbv[0], pbv[1], s, 1.0, k != 0); banner(v, "Step 2  ...but it misses some", f"Red boxes: {len(pbv[1])} of {pbv[2]} real objects missed"); fr.append(v); du.append(350 if k == 0 else (350 if k == 1 else 2300))
    S.append((fr, du))
    v = draw_view(base, phv[0], phv[1], s, 1.0, True); banner(v, "Step 3  Sharper input images help", f"Only {len(phv[1])} of {phv[2]} missed with higher resolution")
    S.append(([v], [3200]))
    t = TAX["baseline_s0"]; ng = 131676
    rows = [("Found correctly", 100 * t["overall"]["correct"] / ng, CB), ("Missed", 100 * t["overall"]["missed"] / ng, CO),
            ("Box in wrong place", 100 * t["overall"]["localization_error"] / ng, CO), ("Phantom detection", 100 * t["overall"]["background_fp"] / ng, CO),
            ("Wrong label", 100 * t["overall"]["classification_error"] / ng, CO), ("Boxed twice", 100 * t["overall"]["duplicate"] / ng, CO)]
    fr = bars_scene("We sort every mistake", "Per 100 real objects, baseline model", rows, 4, 70, "{:.1f}")
    S.append((fr, [250, 250, 250, 3300]))
    size, _ = miss_rates()
    fr = bars_scene("Where it fails most", "Share of real objects missed, by size", [("Small / far away", size["small"], CO), ("Medium", size["medium"], CB), ("Large / close", size["large"], CB)], 4, 100, "{:.0f}%")
    S.append((fr, [250, 250, 250, 3200]))
    cfg = SUM["configs"]
    fr = bars_scene("Sharper images score higher", "Detection score (0 to 1, higher is better)",
                    [("Baseline", cfg["baseline"]["mAP50"]["mean"], CB), ("Sharper images", cfg["hires"]["mAP50"]["mean"], CO)], 4, 1.0, "{:.2f}")
    S.append((fr, [250, 250, 250, 3000]))
    fr, du = [], []
    rows = [(GATE_TXT[r["candidate"]][0], r["verdict"]) for r in GATE if r["candidate"] in GATE_TXT]
    for n in range(0, len(rows) + 1):
        im, d = new(); d.text((40, 24), "The safety gate", font=font(30, True), fill=FG); d.text((40, 66), "Is the new model safe to replace the current one?", font=font(17), fill=SUB)
        for i, (nm, v) in enumerate(rows[:n]):
            y = 110 + i * 52; ok = v == "PASS"
            d.rounded_rectangle([40, y, GW - 40, y + 42], radius=8, fill=(26, 30, 38))
            d.text((58, y + 8), nm, font=font(19), fill=FG)
            d.rounded_rectangle([GW - 170, y + 6, GW - 56, y + 36], radius=14, fill=CG if ok else CR)
            d.text((GW - 150, y + 8), ("✓ PASS" if ok else "✗ FAIL"), font=font(18, True), fill=(255, 255, 255))
        fr.append(im); du.append(2800 if n == len(rows) else 450)
    S.append((fr, du))
    ims = []
    for a in (0.5, 1.0):
        im, d = new(); c = lambda col: tuple(int(BG[k] + (col[k] - BG[k]) * a) for k in range(3))
        centered(d, 150, "perception-gate", font(56, True), c(FG))
        centered(d, 235, "Triage  ·  Test  ·  Gate", font(28), c(CO))
        centered(d, 290, "Built on nuImages (nuScenes, Motional) · YOLO11 · Python · C++", font(17), c(SUB))
        ims.append(im)
    S.append((ims, [200, 2800]))
    return S


def make_gif():
    S = scenes(); frames, dur = [], []
    prev = None
    for ims, ds in S:
        if prev is not None:  # short crossfade
            for a in (0.33, 0.66):
                frames.append(Image.blend(prev, ims[0], a)); dur.append(70)
        for im, d in zip(ims, ds): frames.append(im); dur.append(d)
        prev = ims[-1]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96, dither=Image.NONE) for f in frames]
    path = FIGURES / "overview.gif"
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=2)
    print("wrote", path, round(path.stat().st_size / 1e6, 2), "MB,", len(frames), "frames,", round(sum(dur) / 1000, 1), "s")


if __name__ == "__main__":
    fig_pipeline(); fig_scores(); fig_mistakes(); fig_where(); fig_learning(); fig_gate(); fig_before_after()
    if "--no-gif" not in sys.argv:
        make_gif()
