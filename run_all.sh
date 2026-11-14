#!/usr/bin/env bash
# Full reproduction of perception-gate, start to finish.
# Written for Git Bash / WSL / Linux. On Windows PowerShell run the same python commands with
# .\.venv\Scripts\python.exe (see README "Setup"). Takes roughly 2.5-3 hours on an RTX 5090.
#
# Prereqs: python venv with requirements.txt installed; nuImages "all" metadata + samples extracted to
#   nuimages-v1.0-all-metadata/  and  nuimages-v1.0-all-samples/  in the repo root; g++ (C++17) on PATH.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
[ -x ".venv/Scripts/python.exe" ] && PY=".venv/Scripts/python.exe"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

mkdir -p logs results figures cpp/build

echo "== 1. data prep + slices =="
$PY src/prep_data.py
$PY src/slices.py

echo "== 2. C++ build =="
g++ -O2 -std=c++17 -Wall -Wextra -static cpp/taxonomy.cpp -o cpp/build/taxonomy.exe

echo "== 3. training (3 configs x 2 seeds + degraded) =="
for cfg in baseline strong_aug hires; do
  for seed in 0 1; do $PY src/train.py "$cfg" "$seed"; done
done
$PY src/train.py degraded 0

RUNS="baseline_s0 baseline_s1 strong_aug_s0 strong_aug_s1 hires_s0 hires_s1 degraded_s0"

echo "== 4. predict / evaluate / taxonomy (+C++ verification) / triage =="
$PY src/predict.py $RUNS
$PY src/evaluate.py $RUNS --crosscheck
$PY src/run_taxonomy.py $RUNS
$PY src/benchmark_cpp.py baseline_s0
$PY src/triage.py $RUNS

echo "== 5. gate =="
$PY src/run_gates.py

echo "== 6. summary + verification =="
$PY src/summarize.py
$PY verify_claims.py
