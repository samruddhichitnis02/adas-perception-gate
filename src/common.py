"""Shared paths and constants for perception-gate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "nuimages-v1.0-all-metadata"
SAMPLES_DIR = ROOT / "nuimages-v1.0-all-samples"
DATA = ROOT / "data"
YOLO_DIR = DATA / "yolo"
META_OUT = DATA / "meta"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RUNS = ROOT / "runs"

SEED = 0
TRAIN_SUBSET_IMAGES = 16000  # fixed-seed subset of the 67k train keyframes (see README)

# 8 detection classes. nuImages categories not listed here are NOT labelled in our
# data (trailer, construction vehicle, emergency vehicles, stroller, wheelchair,
# personal mobility, animal, debris, ...). Documented in README/DEVIATIONS.
CLASS_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}
CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "traffic_cone", "barrier"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
