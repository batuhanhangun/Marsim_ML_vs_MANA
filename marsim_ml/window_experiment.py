"""Detection latency: performance as a function of observation window length.

For each prefix window W in settings.window_lengths, features are computed
from only the first W seconds of each recording (see extract.py), the
baseline pipeline is trained and tested on the repetition split, and we
record overall / A3 / low-speed-A3 metrics. MANA is shown as a full-file
reference since it was not rerun per window.

    python -m marsim_ml.window_experiment          # uses data/windows/features_wXXX.csv
    python -m marsim_ml.window_experiment --extract   # extract missing windows first
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load
from .run import PP_CODE, run_config
from .settings import SETTINGS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="extract missing window features first")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args(argv)

    from .extract import window_csv
    start = float(SETTINGS.get("window_start") or 0.0)
    out_root = SETTINGS["results_dir"] / "window_sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    lengths = [w for w in SETTINGS["window_lengths"] if start + w < 120] + [120]
    for w in lengths:
        csv = SETTINGS["data_dir"] / "marsim_features.csv" if w >= 120 else window_csv(w, start)
        if not csv.exists():
            if a.extract and w < 120:
                from .extract import extract
                extract(w, csv, SETTINGS["pcap_dir"], SETTINGS["n_workers"], window_start=start)
            else:
                print(f"[window] missing {csv}; run `python -m marsim_ml.extract --window {w} --start {start:g}`")
                continue
        print(f"=== window {w}s" + (" (full recording)" if w >= 120 else f" starting at {start:g}s"))
        ds = load(SETTINGS["data_dir"], features_csv=csv)
        cfg = dict(split=dict(kind="repetition"), preprocess=PP_CODE, tune=False)
        res = run_config(ds, cfg, out_root / f"w{w:03d}", skip=a.skip, verbose=False)
        ps = pd.read_csv(out_root / f"w{w:03d}" / "per_scenario.csv")
        curve = pd.read_csv(out_root / f"w{w:03d}" / "a3_speed_curve.csv")
        low = curve[curve.shift_speed <= 12]
        for clf, m in res.items():
            a3 = ps[(ps.classifier == clf) & (ps.scenario == "A3")].iloc[0]
            rows.append({"window_s": w, "window_start": 0.0 if w >= 120 else start, "classifier": clf, "f1": m["f1"], "recall": m["recall"],
                         "fpr": m["fpr"], "a3_f1": a3["f1"], "a3_recall": a3["recall"],
                         "a3_recall_speed_le12": float(np.nanmean(low[f"{clf}_recall"]))})
            print(f"  {clf:<5} F1={m['f1']:.4f}  A3 F1={a3['f1']:.4f}  A3 recall(speed<=12)={rows[-1]['a3_recall_speed_le12']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False)
    print(df.pivot(index="window_s", columns="classifier", values="f1").round(4))


if __name__ == "__main__":
    main()
