# Report: can we trust a new detection model before it ships?

## What I did

I took a public driving-camera dataset (nuImages, about 16,000 training pictures and the full 16,445-picture validation set),
trained a small object detector in three different ways (a baseline, a higher-resolution version and a version with stronger
data augmentation), and built two tools:

1. a **triage tool** that sorts every mistake a model makes into types (missed an object, wrong label, badly placed box, duplicate, phantom detection)
   and shows where the mistakes cluster (small objects, particular cameras, and so on);
2. a **gate** that compares a new ("candidate") model with the current ("baseline") one and says PASS or FAIL, with reasons.

## What I found

**Resolution mattered most.** The overall detection score (mAP50) was 0.6105 for the baseline and 0.7022 for the higher-resolution
model, averaged over two training runs each. Stronger augmentation was worse (0.5855). That says only that it was worse
in this short 12-epoch training, not that augmentation is bad in general.

**The model's biggest weakness is small, distant objects.** For the baseline, 25,913 of 39,445 small objects (65.7%) were missed,
against 39.2% of all objects. In 23,603 of those cases the model produced nothing near the object at all. Higher resolution helped
(50.1% missed for one run, 48.8% for the other) but small objects stayed the worst category for every model.

**Medium-sized trucks are often confused with something else.** 2,215 of 3,263 were missed (67.9%), and for 759 of them the model drew a box with the wrong label.

**Standing and moving pedestrians are also missed often** (51.8% and 45.5% of them for the baseline).

These patterns appeared in both baseline training runs, so they are not an accident of one random start.

## What the gate said

| Candidate model | Verdict | Main reason |
|---|---|---|
| Same recipe as baseline, different random seed | PASS | no meaningful change, as it should be |
| Higher resolution, run 1 | PASS | clearly better |
| Higher resolution, run 2 | **FAIL** | more phantom detections than allowed, although the score improved |
| Stronger augmentation (both runs) | **FAIL** | the overall score dropped and more objects were missed |
| Trained for one epoch only (deliberately bad) | **FAIL** | the score fell by 0.2075 and 21.3752 more objects per 100 were missed |

The one-epoch model proves the gate can say no. The most important result is the higher-resolution pair: **two runs of the same
change got opposite verdicts**. The rule that failed run 2 (phantom detections) also moves around between two runs of the unchanged
recipe (0.68 per 100 objects against a limit of 1.0). So that rule is too close to normal run-to-run noise to be trusted as a hard
rule. I left the threshold as I first set it instead of adjusting it after seeing the results.

## About the C++ part

The core matching step (deciding which predicted box corresponds to which real object, and labelling each error) is implemented in
both Python and C++. On the whole validation set the two gave identical answers for every model. The C++ version was about 65 times faster than
plain Python (0.0197 seconds against 1.286 seconds for the error classification). A compiled Python helper (numba) was as fast as C++ for the matching step used in scoring.


