#!/usr/bin/env python
"""
run_mana_baseline.py — Run MANA framework on all 43,320 MARSIM pcap files.

Produces a CSV with per-file predictions and per-method trigger flags.
Must be run BEFORE train_classifiers.py.

Usage:
    conda activate marsim_env
    python run_mana_baseline.py

Output:
    D:\IDMAN_Downloads\ZIP\8202936\dataset\dataset\mana_predictions.csv

Notes:
    - MANA must be installed in the environment (python setup.py install).
    - methods.json must exist at D:\IDMAN_Downloads\ZIP\8202936\methods.json
    - Two methods excluded (OrbitPositionsMethod, PhysicalEnvironmentLimitMethod)
      because required data files (gps.tle, water_map.png) were not bundled
      during setup.py install.
    - Expected runtime: ~30-90 minutes for 43K files depending on hardware.
"""

import json
import os
import time
import pandas as pd

from mana.feeder import PcapFeeder
from mana.handler import DetectionHandler
from mana.method import load_methods_json

# ============================================================
# Paths
# ============================================================
METHODS_JSON_PATH = r"D:\IDMAN_Downloads\ZIP\8202936\methods.json"
BASE_PATH = r"D:\IDMAN_Downloads\ZIP\8202936\dataset\dataset"
OUTPUT_PATH = r"D:\IDMAN_Downloads\ZIP\8202936\dataset\dataset\mana_predictions.csv"

# ============================================================
# Load MANA configuration
# ============================================================
# load_methods_json returns (device_ids, method_classes, method_options)
device_ids, method_classes, method_options = load_methods_json(METHODS_JSON_PATH)

# Load dataset manifest
with open(os.path.join(BASE_PATH, "dataset.json")) as f:
    dataset = json.load(f)

print("=" * 60)
print("MANA Baseline Runner")
print("=" * 60)
print(f"Total files to process: {len(dataset)}")
print(f"Active MANA methods ({len(method_classes)}):")
for mc in method_classes:
    print(f"  - {mc.__name__}")
print()
print("NOTE: OrbitPositionsMethod (EDV) and PhysicalEnvironmentLimitMethod")
print("      (PCC_env) are EXCLUDED — required data files (gps.tle,")
print("      water_map.png) were not copied during setup.py install.")
print("=" * 60)

# ============================================================
# Run MANA on every pcap file
# ============================================================
results = []
start_time = time.time()

for i, entry in enumerate(dataset):
    filename = entry["filename"]
    filepath = os.path.join(BASE_PATH, filename)

    # ---- Per-file trigger list (MUST be inside loop to reset) ----
    triggered_methods = []

    def on_spoofing_attack(device_id, spoofing_indicator, method, state):
        """Callback fired when MANA detects a spoofing indicator above threshold."""
        method_name = type(method).__name__
        if method_name not in triggered_methods:
            triggered_methods.append(method_name)

    # ---- Create a NEW handler per file (handler accumulates NMEA state) ----
    handler = DetectionHandler(
        device_ids=device_ids,
        method_classes=method_classes,
        method_options=method_options,
        detection_threshold=0.1,  # matches the original MANA paper
        on_spoofing_attack=on_spoofing_attack,
    )

    feeder = PcapFeeder(handler, filepath)

    try:
        feeder.run()
    except Exception as e:
        print(f"  ERROR on {filename}: {e}")
        results.append({"filename": filename, "mana_pred": -1, "error": str(e)})
        continue

    # Any method triggered → predict spoofed
    mana_pred = 1 if len(triggered_methods) > 0 else 0

    result = {
        "filename": filename,
        "mana_pred": mana_pred,
        "mana_triggered_methods": ",".join(triggered_methods) if triggered_methods else "",
        "pdm": 1 if "MultipleReceiversMethod" in triggered_methods else 0,
        "pcc_sog": 1 if "PhysicalSpeedLimitMethod" in triggered_methods else 0,
        "pcc_rot": 1 if "PhysicalRateOfTurnLimitMethod" in triggered_methods else 0,
        "pcc_height": 1 if "PhysicalHeightLimitMethod" in triggered_methods else 0,
        "cdm": 1 if "TimeDriftMethod" in triggered_methods else 0,
        "cnm": 1 if "CarrierToNoiseDensityMethod" in triggered_methods else 0,
    }
    results.append(result)

    # Progress reporting every 1000 files
    if (i + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (len(dataset) - (i + 1)) / rate
        print(
            f"  [{i+1:>5}/{len(dataset)}]  "
            f"{rate:.1f} files/s, ~{remaining/60:.0f} min remaining"
        )

# ============================================================
# Save results
# ============================================================
mana_df = pd.DataFrame(results)
mana_df.to_csv(OUTPUT_PATH, index=False)

elapsed = time.time() - start_time
print()
print("=" * 60)
print(f"Done in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
print(f"Saved to {OUTPUT_PATH}")
print(f"Shape: {mana_df.shape}")
print(
    f"Predictions: spoofed={int((mana_df['mana_pred']==1).sum())}, "
    f"unspoofed={int((mana_df['mana_pred']==0).sum())}, "
    f"errors={int((mana_df['mana_pred']==-1).sum())}"
)
print()

# Per-method trigger summary
print("Per-method trigger counts:")
for col in ["pdm", "pcc_sog", "pcc_rot", "pcc_height", "cdm", "cnm"]:
    if col in mana_df.columns:
        count = int((mana_df[col] == 1).sum())
        print(f"  {col:>12s}: triggered on {count} files")

print("=" * 60)
