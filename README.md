# perception-gate: detection failure triage and a validation gate

Fine-tune a small detector on a driving dataset (nuImages), explain *where and how* it fails, and decide
automatically whether a candidate model is safe to promote over a baseline.

Everything quoted below comes from files in `results/`. `verify_claims.py` re-checks the key facts and that every
number in this README and in `report.md` traces to those files. Deviations from the brief are in
[DEVIATIONS.md](DEVIATIONS.md).

## Results at a glance

| Config (one varied factor) | Seeds | mAP50 (mean, range) | mAP50-95 (mean, range) |
|---|---|---|---|
| baseline (imgsz 640, default aug, 12 epochs) | 2 | 0.6105 (0.6094-0.6116) | 0.3791 (0.3783-0.3799) |
| strong_aug (augmentation strength) | 2 | 0.5855 (0.5847-0.5863) | 0.3547 (0.3546-0.3548) |
| hires (input resolution 960) | 2 | 0.7022 (0.6999-0.7045) | 0.4574 (0.4559-0.4590) |
| degraded (1 epoch, gate demo only) | 1 | 0.3271 | 0.1700 |

- Higher input resolution is the one change that clearly helps (+0.09 mAP50), mostly through small objects.
- Strong augmentation is worse at this 12-epoch schedule (not a general statement about augmentation).
- The gate **passes** hires_s0, **fails** hires_s1 (on one error-growth rule), **fails** both strong_aug seeds,
  **passes** the same-recipe control (baseline_s1) and **fails** the deliberately degraded model.
- The C++ matcher/classifier is bit-identical to the Python reference on the full val set and about 65x faster than pure Python.

## Setup

Python 3.12, a CUDA GPU, and a C++17 compiler. Tested on Windows 11, RTX 5090, MinGW-w64 g++ 16.2 (no admin needed).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Data: register at nuscenes.org, download nuImages v1.0 "All" **Metadata** and **Samples** and extract them to
`nuimages-v1.0-all-metadata/` and `nuimages-v1.0-all-samples/` in the repo root (sweeps are not needed).

C++ build (static, no DLLs needed):

```
g++ -O2 -std=c++17 -Wall -Wextra -static cpp/taxonomy.cpp -o cpp/build/taxonomy.exe
```

`run_all.sh` lists every stage in order (see "Limitations" about its test status). Individual stages:
`src/prep_data.py`, `src/slices.py`, `src/train.py <cfg> <seed>`, `src/predict.py <run>`, `src/evaluate.py <run> --crosscheck`,
`src/run_taxonomy.py <run>`, `src/triage.py <run>`, `src/benchmark_cpp.py <run>`, `src/run_gates.py`, `src/summarize.py`, `verify_claims.py`.
Gate on any pair: `python gate.py --baseline baseline_s0 --candidate hires_s0`.

## Data and slices

nuImages v1.0, keyframes. 24 categories reduced to 8 detection classes (car, truck, bus, motorcycle, bicycle, pedestrian,
traffic_cone, barrier). Unmapped categories (trailer, construction vehicle, emergency vehicles, stroller, ...) are
unlabelled, so correctly detecting one is scored as a background false positive (see DEVIATIONS.md).

- Train: a fixed-seed subset of 16,000 of the 67,279 train keyframes (128,753 boxes).
- Val: the **full** val split, 16,445 images and 131,676 boxes. No log (drive) appears in both train subset and val.
- Sizes and settings: `results/config.json`. Slice counts: `results/slices.json`.

**Attribute audit** (`data/meta/attribute_audit.json`, written before slices were chosen): nuImages has log location,
camera channel, capture date and object attributes. It has **no** time-of-day, weather, truncation or occlusion field.

| Slice family | Slices | Kind |
|---|---|---|
| `loc:` | boston-seaport, singapore-onenorth | **native** |
| `cam:` | 6 cameras | **native** |
| `attr:` | vehicle moving/parked/stopped, pedestrian moving/standing/sitting-lying, cycle with/without rider | **native** |
| `bright:` | dark / bright (mean gray < 80) | **derived**, a day/night *proxy*, not ground truth |
| `size:` | small (<32^2 px), medium, large (>96^2 px) | **derived** |
| `edge:` | box touches image border | **derived**, a truncation *proxy* |

Every slice has at least 30 boxes in val, so none had to be dropped; the thinnest is `bright:dark` with 505 boxes in 196 images.
Night-time conclusions therefore rest on a small sample.

## Training

YOLO11s (COCO-pretrained), batch 32, 12 epochs, 2 seeds per config. Each config changes exactly one factor from baseline
(`src/train.py`). Per-epoch loss and val metrics are in `results/training/*_results.csv` and plotted in
`figures/training_curves.png` (val curves use a fixed 2,000-image subset; all reported numbers use the full val set).

## Evaluation

Own COCO-style AP implementation (`src/evalcore.py`), cross-checked against `pycocotools` for every model: they agree to 4
decimals. Per-class AP, per-slice AP and the pycocotools check are in `results/eval/<run>.json`.

Slice AP rule: GT outside the slice is ignored; predictions matched to out-of-slice GT are ignored; an unmatched prediction
is a false positive only if the prediction itself can be assigned to the slice. For the `attr:` slices a prediction cannot
be assigned, so those slices measure recall-weighted AP and read higher than other slices (for baseline_s0 `attr:vehicle.parked` is
0.700 vs 0.610 overall). Do not compare `attr:` numbers with other families.

Per-class AP50 (hires_s0 vs baseline_s0): car 0.841 vs 0.772, truck 0.573 vs 0.472, bus 0.505 vs 0.424, pedestrian 0.713 vs 0.589.

## Error taxonomy

Operating point: confidence >= 0.25; IoU 0.5 for a match, 0.1 for the localization/background boundary. Predictions are
walked in descending confidence per image:

| Label | Rule |
|---|---|
| correct | same-class GT, IoU >= 0.5, not yet matched |
| duplicate | same-class GT with IoU >= 0.5 already matched by a higher-confidence prediction |
| classification error | different-class GT with IoU >= 0.5 |
| localization error | same-class GT with IoU between 0.1 and 0.5 |
| background false positive | anything else (includes weak overlap with another class) |
| missed | a GT no correct prediction matched |

Counts per class and per slice: `results/taxonomy/<run>.json`. Overall (baseline_s0): 79,994 correct, 3,534 classification
errors, 8,407 localization errors, 2,731 duplicates, 8,759 background false positives, 51,682 missed.

## C++ matching and error classification

`cpp/taxonomy.cpp` is a small CLI (the brief allowed this instead of pybind11) with two modes: `taxonomy` (the full error
classification) and `match` (the COCO-style greedy matching used for AP). `src/taxonomy_ref.py` is a pure-Python reference with the
same arithmetic order and tie-breaks.

- **Identical output:** on the full val set, all 7 models, C++ labels, references and missed flags equal the Python reference with
  0 differing entries. `match` mode is also identical to Python and to the numba matcher used by the evaluator.
- **Timings** (`results/cpp_benchmark_repeat.json`, median of 7 runs, idle machine, baseline_s0, compute only):

| Task | Python reference | C++ | numba |
|---|---|---|---|
| Taxonomy on 103,425 predictions | 1.286 s | 0.0197 s | n/a |
| IoU 0.5 matching on 1,477,818 predictions | 2.556 s | 0.0386 s | 0.0324 s |

C++ is about 65x faster than pure Python. The numba matcher used by the evaluator is as fast as C++ (slightly faster here),
so C++ is not needed for speed in the AP path; its value here is the verified, dependency-free standalone tool.

## Triage findings

Patterns are conjunctions of up to two slice tags (plus an optional class) ranked by excess misses over the model's overall
miss rate. A condition is kept only if it raises the miss rate by at least 3 points over every simpler sub-pattern. Full lists
with counts: `results/triage/<run>.json`; example grids: `figures/<run>/` (missed GT boxes in red).

**baseline_s0** (overall miss rate 39.2%):

1. **Small objects are mostly missed:** 25,913 of 39,445 small GT boxes missed (65.7% vs 39.2% overall). For 23,603 of those misses
   there was no nearby prediction at all, i.e. the detector produced nothing, not a wrong box.
   Small predictions are also the most over-represented source of background false positives (3,245 of 20,333 small predictions, 16.0% vs 8.5%)
   and of localization errors (2,809 of 20,333, 13.8% vs 8.1%).
2. **Medium trucks:** 2,215 of 3,263 missed (67.9%). Here 759 of the misses had a wrong-class prediction on the box,
   so this is partly class confusion, not only non-detection. Truck predictions are also the largest source of classification errors
   (817 of 4,497 truck predictions, 18.2% vs 3.4% overall).
3. **Pedestrians:** standing pedestrians 4,877 of 9,406 missed (51.8%); moving pedestrians 8,896 of 19,571 (45.5%).
4. **Front camera:** CAM_FRONT misses 15,830 of 37,462 (42.3%), mild but the largest camera.

**Replication and the effect of resolution.** The top patterns reproduce on the second baseline seed (small objects 64.6% missed
for baseline_s1) so they are not seed accidents. Raising resolution cuts the small-object miss rate to 50.1% (hires_s0) and 48.8% (hires_s1),
but small objects stay the top failure for every model, and small background false positives grow with resolution
(5,163 of 29,189 small predictions in hires_s1).

The derived tags (size, brightness, border) are proxies; patterns using them are flagged `derived_tags_are_proxies` in the JSON.
A "dark images" pattern does not appear in the top lists: with only 505 dark-image boxes it never accumulates enough excess misses.

## The validation gate

`gate.py` takes a baseline and a candidate and prints PASS or FAIL with reasons (exit code 0 or 1). Thresholds are in
`gate_config.yaml`.

Checks (29 for the current slice set):
1. Overall mAP50-95 and mAP50 must not drop.
2. No slice mAP50 may drop by more than 0.03. Only slices with at least 100 GT boxes are gated (stricter than the 30-box floor).
3. No error type (classification, localization, duplicate, background FP, missed) may grow by more than its threshold,
   per 100 GT boxes: 0.5, 0.5, 0.5, 1.0, 1.0.

Statistics: a paired bootstrap over val images (200 resamples, same resample for baseline and candidate, 95% CI).
**FAIL** when the CI shows the regression is beyond the threshold; **WARN** when only the point estimate is beyond it (inconclusive, does not
fail unless `fail_on_warn` is set); otherwise PASS. For speed the gate computes AP on predictions with confidence >= 0.01, which costs
0.0016 mAP50-95 for the baseline (applied to both sides).

**Gate runs** (baseline = baseline_s0; `results/gate/`):

| Candidate | Verdict | Fails / warns | Why |
|---|---|---|---|
| baseline_s1 (same recipe, other seed) | PASS | 0 / 0 | noise control |
| hires_s0 | PASS | 0 / 0 | mAP50-95 +0.0769 (CI 0.0747 to 0.0787) |
| hires_s1 | **FAIL** | 1 / 0 | background FP +1.4824 per 100 GT (limit 1.0) although mAP rose |
| strong_aug_s0 | **FAIL** | 3 / 3 | mAP50-95 -0.0234; missed +2.0383 per 100 GT |
| strong_aug_s1 | **FAIL** | 3 / 2 | mAP50-95 -0.0236; missed +1.3207 per 100 GT |
| degraded_s0 (1 epoch) | **FAIL** | 27 / 1 | mAP50-95 -0.2075; missed +21.3752 per 100 GT |

The degraded model shows the gate can reject. The hires result deserves attention: two seeds of the same candidate got opposite verdicts, and
the failing rule (background FP growth) is also noisy for an unchanged recipe, where baseline_s1 grew by 0.68 per 100 GT against a limit of 1.0. That
threshold sits close to seed-to-seed noise. I did not retune it after seeing the results.

## What the gate cannot catch (and other limitations)

- **Anything outside this val set.** Other cities, weather, sensors, night (only 196 dark images), rare objects. A model can pass and still fail in the field.
- **Seed noise.** The bootstrap resamples *images*, not training runs. The same-recipe control moved the `bright:dark` slice by -0.027 against a 0.03
  limit and background FPs by +0.68 against 1.0. With one candidate run per decision, a gate verdict can flip with the seed (as hires did).
- **Correlated frames.** Images from one drive are correlated, but the bootstrap treats them as independent, so intervals are likely too narrow. Resampling by drive (82 val logs) would be more honest.
- **Many checks, no correction.** 29 checks per candidate with no multiple-comparison adjustment.
- **Slices that do not exist.** Only the slices I defined are checked. Derived slices are proxies (brightness is not time of day; border contact is not truncation). Rare safety-critical groups (children, wheelchairs) are merged or unlabelled.
- **Label gaps.** Unlabelled categories turn real detections into "false positives", and the gate treats them as errors.
- **`attr:` slices are blind to false positives** (recall-weighted AP).
- **Operating point.** Error-type rules use a single confidence cut-off (0.25); calibration and threshold choice are not tested.
- **Everything beyond per-frame accuracy:** latency, memory, temporal stability, tracking, planning impact, robustness to perturbations, and behaviour on safety-critical edge cases.
- **The baseline defines "good".** A weak baseline lets weak candidates pass.

Other limitations: 12-epoch schedules, a 16,000-image train subset, only 2 seeds per config (the range is a noise hint, not a CI),
and `run_all.sh` was **not executed end to end** (no bash on this machine). The same commands were run by hand in the same order.

## Data attribution and what is not in this repository

The dataset is nuImages from nuScenes (Motional), used under its non-commercial terms (see nuscenes.org/terms-of-use). The dataset is not
included here: download it yourself as described in Setup. The example images in `figures/` are small crops from nuImages, shown for
non-commercial research illustration with attribution to the nuScenes/nuImages dataset.

Also not included because they are large and regenerable: model weights (`runs/`), raw predictions (`results/preds/`, rebuild with
`src/predict.py`), and the prepared dataset (`data/`, rebuild with `src/prep_data.py`). `verify_claims.py` needs these rebuilt first.

## Repo layout

```
src/            data prep, slices, training, prediction, evaluation, taxonomy (+Python reference), triage, summaries
cpp/            taxonomy.cpp (build instructions above)
gate.py         the validation gate        gate_config.yaml   thresholds
results/        config, slices, training logs, eval, taxonomy, triage, gate outputs, C++ benchmarks, summary
figures/        training curves and triage example grids
verify_claims.py   DEVIATIONS.md   report.md   run_all.sh   requirements.txt
```

## How this maps to AV / ADAS perception roles

The project is deliberately company-neutral. Typical perception-validation work breaks down as below; the table maps each
activity to the stage that demonstrates it. (It describes this project, not any employer's process.)

| Activity common in AV/ADAS perception roles | Where it shows up here |
|---|---|
| Preparing datasets, checking data quality and labels | Stage 1: attribute audit, class mapping, dropped-category counts, leakage check |
| Running and comparing training experiments | Stage 2: three one-factor configs, 2 seeds, real loss curves |
| Slicing performance by operating condition | Slices: native vs derived, with proxies labelled as such |
| Triaging model failures into actionable categories | Error taxonomy and triage with example grids |
| Systems-level code in C++ with verified correctness | C++ matcher/classifier, bit-identical to a reference, benchmarked |
| Release gating with statistical evidence | `gate.py` with paired-bootstrap CIs and a documented list of what it misses |
