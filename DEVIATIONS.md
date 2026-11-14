# Deviations from the project brief, and other things to know

Everything here is a place where what was built differs from, narrows, or adds to the brief.

## Data

1. **Dataset.** nuImages v1.0 ("all": metadata + samples) was used as the primary dataset. No fallback to KITTI or
   COCO was needed. BDD100K was not used.
2. **Class set (24 nuImages categories reduced to 8).** Detection classes: car, truck, bus (bendy + rigid),
   motorcycle, bicycle, pedestrian (adult, child, construction worker, police officer), traffic_cone, barrier.
   These nuImages categories are **not labelled** in our data: trailer, construction vehicle, police/ambulance,
   stroller, wheelchair, personal-mobility, animal, debris, bicycle rack, pushable/pullable, ego vehicle.
   Consequence: a model that correctly detects, say, a trailer is scored as a *background false positive*.
   The counts of dropped boxes are in `data/meta/train.json` and `data/meta/val.json` (`info.dropped_unmapped_categories`).
3. **Train subset.** 16,000 of the 67,279 train keyframes, chosen with `random.Random(0).sample` (fixed seed).
   The **validation set is the full val split** (16,445 images, 131,676 boxes), not a subset. Sizes are in `results/config.json`.
   Reason: time; the GPU could have handled more, but 3 configs x 2 seeds + a degraded model had to fit in one session.
4. **No leakage check beyond logs.** Train and val share no log (drive); this is verified in `verify_claims.py`. Nearby
   frames inside the train subset are of course correlated.
5. **Native vs derived slices.** nuImages has no time-of-day, weather, truncation or occlusion field (see
   `data/meta/attribute_audit.json`). Therefore:
   - native: location, camera channel, object attributes (moving/parked/stopped, pedestrian pose, cycle with/without rider)
   - derived (NOT ground truth): `bright:dark|bright` (mean grayscale < 80, a night proxy), `size:*` (box area),
     `edge:touches_border` (a truncation proxy).
   Only 196 of 16,445 val images (505 boxes) are "dark", so night conclusions rest on a small sample.

## Training

6. **Short schedule: 12 epochs** per run, YOLO11s fine-tuned from the COCO-pretrained `yolo11s.pt`. Models are
   not converged to their ceiling. In particular the *strong augmentation* configuration is likely handicapped by the
   short schedule: strong augmentation normally needs longer training, so its result says "worse at 12 epochs", not
   "augmentation hurts".
7. **Two seeds only.** The spread between the two seeds is a crude noise indicator (n = 2), not a confidence interval.
8. **Per-epoch curves use a fixed 2,000-image val subset** to keep epochs fast. All reported results come from the
   full val set via our own evaluation.
9. **Degraded model** = the baseline recipe trained for 1 epoch (seed 0).
10. **Two runs were retrained from scratch after interruptions** (a PC crash and then my own mistake of launching jobs in
    windows that were later closed). `strong_aug_s1` and `hires_s1` each completed a full 12 epochs; no partial run was used.

## Evaluation

11. **AP implementation.** Our own COCO-style AP (101-point interpolation, greedy matching, per-class mean). It matches
    `pycocotools` to 4 decimals for every model (stored in `results/eval/*.json` as `pycocotools`).
12. **Slice AP rule.** GT outside a slice is ignored; predictions matched to out-of-slice GT are ignored; unmatched
    predictions count as false positives only if the prediction can itself be assigned to the slice (image-level slices,
    size, border). For `attr:*` slices a prediction cannot be assigned, so unmatched predictions are ignored and those
    slices measure **recall-weighted AP**: their numbers are inflated relative to other slices and are not comparable across slice families.
13. **Slice mAP averages only classes with >= 5 GT boxes in the slice.** Slices with < 30 boxes are excluded from all
    claims (none were below 30 for val; the gate additionally requires >= 100 boxes).

## Error taxonomy

14. **Operating point.** Taxonomy and triage use predictions with confidence >= 0.25, IoU 0.5 (correct/duplicate/class
    error) and 0.1 (localization/background boundary). This is not the same as AP, which uses all confidences. A
    prediction that overlaps weakly (0.1-0.5) with a *different-class* box is counted as background FP (the "both wrong" case).

## C++ stage

15. **MSVC could not be installed** (no administrator rights; installer exit code 1602). A portable MinGW-w64 g++ 16.2
    (WinLibs, user-level winget install) was used instead. The binary is statically linked.
16. **CLI instead of pybind11.** The brief allowed either. The C++ tool reads/writes flat binary files and is driven from Python.
17. **Timings.** `results/cpp_benchmark.json` is a single shot taken while training used the machine;
    `results/cpp_benchmark_repeat.json` is a median of repeated runs on an idle machine and is the one quoted in the README.
    C++ times are the CLI's internal compute timer (file I/O excluded); Python times are wall-clock compute only.

## Gate

18. **Confidence floor for gate AP.** The gate computes AP on predictions with confidence >= 0.01 (not 0.001) so the
    bootstrap is fast enough. Measured cost: -0.0016 mAP50-95 (baseline), -0.0008 (degraded). It is applied equally to
    baseline and candidate. Numbers in `results/gate/*.json` therefore differ slightly from `results/eval/*.json`.
19. **200 bootstrap resamples** (not 1000+), seed 0, paired over images. CI endpoints have some Monte-Carlo noise.
20. **Threshold values are my choices**, not derived from risk analysis; see `gate_config.yaml` and the README for rationale.

## Environment and process

21. **Python.** The Microsoft Store Python 3.12 on the machine was not runnable from this user account, and the existing
    virtual environments belonged to other user profiles. A per-user python.org Python 3.12.10 was installed with winget
    and a venv created at `.venv/` inside the project. No other project's environment was modified.
22. **`run_all.sh` was not executed end to end.** The machine has no bash (no Git Bash/WSL). Every stage was run by hand with
    the same commands, in the same order, and `run_all.sh` mirrors them. Treat `run_all.sh` as untested as a script.
23. **No git or GitHub operations** were performed; nothing was uploaded.
