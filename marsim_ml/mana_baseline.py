"""Run the MANA framework (six methods, default thresholds) on every pcap file.

    python -m marsim_ml.mana_baseline            # -> data/mana_predictions.csv
    python -m marsim_ml.mana_baseline --time     # also record per-file wall time

Requires the `mana` package installed from https://github.com/fkie-cad/mana.
Output schema matches the frozen mana_predictions.csv exactly, plus an optional
`mana_ms` column when --time is given (used for Table: end-to-end timing).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .settings import SETTINGS

METHOD_SHORT = {
    "MultipleReceiversMethod": "pdm",
    "PhysicalSpeedLimitMethod": "pcc_sog",
    "PhysicalRateOfTurnLimitMethod": "pcc_rot",
    "PhysicalHeightLimitMethod": "pcc_height",
    "TimeDriftMethod": "cdm",
    "CarrierToNoiseDensityMethod": "cnm",
}


def run(pcap_dir: Path, methods_json: Path, out_csv: Path, record_time: bool = False,
        sample: int | None = None, seed: int = 42):
    from mana.feeder import PcapFeeder
    from mana.handler import DetectionHandler
    from mana.method import load_methods_json

    device_ids, method_classes, method_options = load_methods_json(str(methods_json))
    dataset = json.loads((pcap_dir / "dataset.json").read_text())
    if sample:
        import random
        rng = random.Random(seed)
        by = {}
        for e in dataset:
            by.setdefault((e["scenario"], e["label"]), []).append(e)
        per = max(1, sample // len(by))
        dataset = [e for grp in by.values() for e in rng.sample(grp, min(per, len(grp)))]
    print(f"[mana] {len(dataset)} files, methods: {[m.__name__ for m in method_classes]}")

    rows, t_start = [], time.time()
    for i, entry in enumerate(dataset, 1):
        triggered: list[str] = []

        def on_attack(*args, _t=triggered, **kwargs):
            # MANA versions differ in how they call this hook (positional vs.
            # keyword arguments such as spoofing_indicator=...). Find the
            # method object wherever it is.
            method = kwargs.get("method")
            if method is None:
                for a in list(args) + list(kwargs.values()):
                    if type(a).__name__ in METHOD_SHORT:
                        method = a
                        break
            name = type(method).__name__ if method is not None else "Unknown"
            if name not in _t:
                _t.append(name)

        handler = DetectionHandler(device_ids=device_ids, method_classes=method_classes,
                                   method_options=method_options, detection_threshold=0.1,
                                   on_spoofing_attack=on_attack)
        t0 = time.perf_counter()
        try:
            PcapFeeder(handler, str(pcap_dir / entry["filename"])).run()
            err = None
        except Exception as exc:
            err = str(exc)
        ms = (time.perf_counter() - t0) * 1000
        row = {"filename": entry["filename"],
               "mana_pred": -1 if err else int(bool(triggered)),
               "mana_triggered_methods": ",".join(triggered)}
        for full, short in METHOD_SHORT.items():
            row[short] = int(full in triggered)
        if record_time:
            row["mana_ms"] = ms
        if err:
            row["error"] = err
        rows.append(row)
        if i % 1000 == 0:
            el = time.time() - t_start
            print(f"  {i:>6}/{len(dataset)}  {i/el:.1f} files/s  ETA {(len(dataset)-i)/(i/el)/60:.0f} min", flush=True)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_err = int((df["mana_pred"] == -1).sum())
    print(f"[mana] done in {(time.time()-t_start)/60:.1f} min; errors={n_err}; saved {out_csv}")
    if n_err:
        print("  NOTE: rows with mana_pred=-1 are reported, not silently coerced. Decide how to treat them.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SETTINGS["data_dir"] / "mana_predictions.csv"))
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample", type=int, default=None,
                    help="run on a stratified random sample of N files (scenario x label)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"[mana] {out} exists, skipping (use --force to redo)")
        return
    run(SETTINGS["pcap_dir"], SETTINGS["mana_methods_json"], out, a.time, a.sample)
    if a.sample:
        frozen = SETTINGS["data_dir"] / "mana_predictions.csv"
        if frozen.exists():
            new = pd.read_csv(out)
            old = pd.read_csv(frozen)
            m = old.merge(new, on="filename", suffixes=("_frozen", "_new"))
            ok = m[m.mana_pred_new != -1]
            print(f"[mana] sample check: {len(ok)}/{len(m)} completed, "
                  f"agreement with frozen predictions {(ok.mana_pred_frozen == ok.mana_pred_new).mean():.4f}")
            if "mana_ms" in new:
                print(f"[mana] timing ms: median {new.mana_ms.median():.0f}, mean {new.mana_ms.mean():.0f}, "
                      f"IQR {new.mana_ms.quantile(.25):.0f}-{new.mana_ms.quantile(.75):.0f}")


if __name__ == "__main__":
    main()
