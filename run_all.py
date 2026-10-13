"""Run everything, end to end.

    python run_all.py                 # features (if missing) -> MANA (if missing) -> all configs -> windows -> figures -> tables
    python run_all.py --no-windows    # skip the pcap-dependent window experiment
    python run_all.py --no-tune       # skip randomized search (the slowest step)
    python run_all.py --only plots tables

Steps are idempotent: existing feature/MANA CSVs are reused, results are
overwritten.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from marsim_ml.settings import SETTINGS  # noqa: E402

STEPS = ["extract", "mana", "experiments", "windows", "plots", "tables"]


def sh(*args):
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=STEPS, default=None)
    ap.add_argument("--no-windows", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--skip-svm", action="store_true")
    a = ap.parse_args()
    print(f"project dir: {HERE}\npcap dir:    {SETTINGS['pcap_dir']}\nworkers:     {SETTINGS['n_workers']}")
    steps = a.only or STEPS
    if a.no_windows and "windows" in steps:
        steps.remove("windows")
    skip = ["--skip", "SVM"] if a.skip_svm else []
    t0 = time.time()

    if "extract" in steps:
        sh("marsim_ml.extract")
    if "mana" in steps:
        sh("marsim_ml.mana_baseline")
    if "experiments" in steps:
        from marsim_ml.run import CONFIGS
        for name in CONFIGS:
            if a.no_tune and name == "tuned":
                continue
            sh("marsim_ml.run", "--config", name, *skip)
    if "windows" in steps:
        sh("marsim_ml.extract", "--all-windows")
        sh("marsim_ml.window_experiment", *skip)
    if "plots" in steps:
        sh("marsim_ml.plots")
    if "tables" in steps:
        sh("marsim_ml.tables")
    print(f"\nall done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
