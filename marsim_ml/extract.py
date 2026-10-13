"""Feature extraction from MARSIM pcap files.

Same 138 feature definitions as the frozen marsim_features.csv (per-epoch raw
statistics from receiver 1, consecutive-epoch derived features, multi-receiver
features, clock regularity, epoch counts), with two additions:

* `window_s`  : only the first `window_s` seconds of each recording are used
                (prefix window). window_s=None reproduces the full-file features.
* parallel    : multiprocessing over files.

    python -m marsim_ml.extract                      # full 120 s -> data/marsim_features.csv
    python -m marsim_ml.extract --window 30          # -> data/windows/features_w030.csv
    python -m marsim_ml.extract --all-windows        # every length in settings

Requires scapy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import META_COLS
from .settings import SETTINGS

warnings.filterwarnings("ignore")
os.environ["SCAPY_USE_LIBPCAP"] = "0"
import logging
for _n in ("scapy.runtime", "scapy.loading", "scapy"):
    logging.getLogger(_n).setLevel(logging.ERROR)

RX1 = ("192.168.0.10", 62996)
RX2 = ("192.168.0.11", 62997)
EPOCH_MATCH_TOL = 0.5

RAW_KEYS = ["sog_knots", "cog_deg", "num_satellites", "hdop", "altitude", "pdop", "hdop_gsa",
            "vdop", "fix_mode", "mean_snr", "std_snr", "min_snr", "max_snr", "num_sats_with_snr",
            "sog_vtg_knots", "sog_vtg_kmh", "cog_true", "cog_magnetic"]
DERIVED_KEYS = ["position_jump_m", "computed_sog_knots", "sog_discrepancy",
                "cog_change_rate", "cog_heading_discrepancy", "time_delta"]
INTER_KEYS = ["inter_rx_distance_m", "inter_rx_sog_diff", "inter_rx_cog_diff"]
NAN = float("nan")


# ------------------------------------------------------------------ helpers
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def angular_diff(a, b):
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return NAN
    d = abs(a - b) % 360
    return min(d, 360 - d)


def nmea_time(s):
    try:
        t = float(s)
    except (TypeError, ValueError):
        return NAN
    return int(t / 10000) * 3600 + int((t % 10000) / 100) * 60 + t % 100


def nmea_coord(s, hemi):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return NAN
    deg = int(v / 100)
    dec = deg + (v - deg * 100) / 60.0
    return -dec if hemi in ("S", "W") else dec


def f(v):
    try:
        return float(v) if v not in (None, "") else NAN
    except (TypeError, ValueError):
        return NAN


# ------------------------------------------------------------------ parsers
def parse_rmc(fl):
    if len(fl) < 10:
        return {}
    return dict(utc_time=nmea_time(fl[1]), lat_deg=nmea_coord(fl[3], fl[4]),
                lon_deg=nmea_coord(fl[5], fl[6]), sog_knots=f(fl[7]), cog_deg=f(fl[8]))


def parse_gga(fl):
    if len(fl) < 10:
        return {}
    return dict(utc_time=nmea_time(fl[1]), lat_deg=nmea_coord(fl[2], fl[3]),
                lon_deg=nmea_coord(fl[4], fl[5]), fix_quality=f(fl[6]),
                num_satellites=f(fl[7]), hdop=f(fl[8]), altitude=f(fl[9]))


def parse_gsa(fl):
    if len(fl) < 18:
        return {}
    return dict(fix_mode=f(fl[2]), pdop=f(fl[15]), hdop_gsa=f(fl[16]), vdop=f(fl[17].split("*")[0]))


def parse_gsv(fl):
    out, i = [], 4
    while i + 3 < len(fl):
        v = f(fl[i + 3].split("*")[0])
        if not math.isnan(v):
            out.append(v)
        i += 4
    return out


def parse_vtg(fl):
    if len(fl) < 8:
        return {}
    return dict(cog_true=f(fl[1]), cog_magnetic=f(fl[3]), sog_vtg_knots=f(fl[5]), sog_vtg_kmh=f(fl[7].split("*")[0]))


# ------------------------------------------------------------ pcap reading
def read_pcap(path: Path):
    from scapy.all import IP, UDP, PcapReader, Raw  # local import: slow, and workers each need it

    sent = {1: defaultdict(dict), 2: defaultdict(dict)}
    gsv = {1: defaultdict(list), 2: defaultdict(list)}
    cur = {1: None, 2: None}
    try:
        reader = PcapReader(str(path))
    except Exception:
        return sent, gsv
    try:
        for pkt in reader:
            if not (pkt.haslayer(IP) and pkt.haslayer(UDP) and pkt.haslayer(Raw)):
                continue
            key = (pkt[IP].src, pkt[UDP].sport)
            rx = 1 if key == RX1 else 2 if key == RX2 else None
            if rx is None:
                continue
            payload = pkt[Raw].load.decode("ascii", errors="ignore").strip()
            if not payload.startswith("$GP"):
                continue
            fl = payload.split(",")
            typ = fl[0]
            if typ in ("$GPRMC", "$GPGGA"):
                p = parse_rmc(fl) if typ == "$GPRMC" else parse_gga(fl)
                t = p.get("utc_time", NAN)
                if not math.isnan(t):
                    cur[rx] = t
                    sent[rx][t].update(p)
            elif cur[rx] is not None:
                if typ == "$GPGSA":
                    sent[rx][cur[rx]].update(parse_gsa(fl))
                elif typ == "$GPGSV":
                    gsv[rx][cur[rx]].extend(parse_gsv(fl))
                elif typ == "$GPVTG":
                    sent[rx][cur[rx]].update(parse_vtg(fl))
    except Exception:
        pass
    finally:
        reader.close()
    return sent, gsv


def build_epochs(sentences, gsv, window_s: float | None, window_start: float = 0.0):
    """Epochs with (window_start <= t - t_first < window_start + window_s)."""
    times = sorted(sentences)
    if times and (window_s is not None or window_start):
        t0 = times[0]
        end = window_start + window_s if window_s is not None else float("inf")
        times = [t for t in times if window_start <= t - t0 < end]
    epochs = []
    for t in times:
        e = dict(sentences[t])
        e["utc_time"] = t
        s = gsv.get(t, [])
        if s:
            e.update(mean_snr=float(np.mean(s)), std_snr=float(np.std(s)), min_snr=float(np.min(s)),
                     max_snr=float(np.max(s)), num_sats_with_snr=len(s))
        else:
            e.update(mean_snr=NAN, std_snr=NAN, min_snr=NAN, max_snr=NAN, num_sats_with_snr=0)
        epochs.append(e)
    return epochs


# ------------------------------------------------------------- feature calc
def derived_features(ep):
    d = {k: [] for k in DERIVED_KEYS}
    for i in range(1, len(ep)):
        a, b = ep[i - 1], ep[i]
        lat1, lon1, lat2, lon2 = (a.get("lat_deg", NAN), a.get("lon_deg", NAN), b.get("lat_deg", NAN), b.get("lon_deg", NAN))
        t1, t2 = a.get("utc_time", NAN), b.get("utc_time", NAN)
        dt = t2 - t1 if not (math.isnan(t1) or math.isnan(t2)) else NAN
        d["time_delta"].append(dt)
        if not any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
            jump = haversine_m(lat1, lon1, lat2, lon2)
            d["position_jump_m"].append(jump)
            if not math.isnan(dt) and dt > 0:
                sog_c = jump / dt * 1.94384
                d["computed_sog_knots"].append(sog_c)
                sog_r = b.get("sog_knots", NAN)
                d["sog_discrepancy"].append(abs(sog_r - sog_c) if not math.isnan(sog_r) else NAN)
                heading = math.degrees(math.atan2(lon2 - lon1, lat2 - lat1)) % 360
                cog_r = b.get("cog_deg", NAN)
                d["cog_heading_discrepancy"].append(angular_diff(cog_r, heading) if not math.isnan(cog_r) else NAN)
            else:
                for k in ("computed_sog_knots", "sog_discrepancy", "cog_heading_discrepancy"):
                    d[k].append(NAN)
        else:
            for k in ("position_jump_m", "computed_sog_knots", "sog_discrepancy", "cog_heading_discrepancy"):
                d[k].append(NAN)
        c1, c2 = a.get("cog_deg", NAN), b.get("cog_deg", NAN)
        d["cog_change_rate"].append(angular_diff(c2, c1) / dt if not (math.isnan(c1) or math.isnan(c2) or math.isnan(dt) or dt <= 0) else NAN)
    return d


def inter_receiver_features(e1s, e2s):
    out = {k: [] for k in INTER_KEYS}
    if not e1s or not e2s:
        return out
    t2 = np.array([e["utc_time"] for e in e2s])
    for e1 in e1s:
        j = int(np.argmin(np.abs(t2 - e1["utc_time"])))
        if abs(t2[j] - e1["utc_time"]) > EPOCH_MATCH_TOL:
            for k in INTER_KEYS:
                out[k].append(NAN)
            continue
        e2 = e2s[j]
        ll = [e1.get("lat_deg", NAN), e1.get("lon_deg", NAN), e2.get("lat_deg", NAN), e2.get("lon_deg", NAN)]
        out["inter_rx_distance_m"].append(haversine_m(*ll) if not any(math.isnan(v) for v in ll) else NAN)
        s1, s2 = e1.get("sog_knots", NAN), e2.get("sog_knots", NAN)
        out["inter_rx_sog_diff"].append(abs(s1 - s2) if not (math.isnan(s1) or math.isnan(s2)) else NAN)
        out["inter_rx_cog_diff"].append(angular_diff(e1.get("cog_deg", NAN), e2.get("cog_deg", NAN)))
    return out


def agg(values, prefix):
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return {f"{prefix}_{s}": NAN for s in ("mean", "std", "min", "max", "median")}
    return {f"{prefix}_mean": float(np.mean(v)), f"{prefix}_std": float(np.std(v)), f"{prefix}_min": float(np.min(v)),
            f"{prefix}_max": float(np.max(v)), f"{prefix}_median": float(np.median(v))}


def features_for_file(path: Path, window_s: float | None, window_start: float = 0.0) -> dict:
    sent, gsv = read_pcap(path)
    e1 = build_epochs(sent[1], gsv[1], window_s, window_start)
    e2 = build_epochs(sent[2], gsv[2], window_s, window_start)
    primary = e1 or e2
    feat = {}
    if not primary:
        return feat
    for k in RAW_KEYS:
        feat.update(agg([e.get(k, NAN) for e in primary], k))
    d = derived_features(primary)
    for k in DERIVED_KEYS:
        feat.update(agg(d[k], k))
    td = [x for x in d["time_delta"] if not math.isnan(x)]
    feat["clock_regularity"] = float(np.std(td)) if td else NAN
    inter = inter_receiver_features(e1, e2)
    for k in INTER_KEYS:
        feat.update(agg(inter[k], k))
    feat["n_epochs_rx1"], feat["n_epochs_rx2"] = len(e1), len(e2)
    return feat


# ------------------------------------------------------------------ driver
def _meta(entry):
    params = entry.get("parameters", {})
    keys = list(params)
    m = dict(filename=entry["filename"], scenario=entry["scenario"], label=entry["label"], index=int(entry["index"]))
    if keys:
        m["param_1_name"], m["param_1_value"] = keys[0], f(params[keys[0]])
    if len(keys) > 1:
        m["param_2_name"], m["param_2_value"] = keys[1], f(params[keys[1]])
    return m


def _work(args):
    pcap_dir, entry, window_s, window_start = args
    try:
        feat = features_for_file(Path(pcap_dir) / entry["filename"], window_s, window_start)
    except Exception as exc:  # keep the row, record the failure
        feat = {"_error": str(exc)}
    feat.update(_meta(entry))
    return feat


def _run_chunk(chunk, pcap_dir, window_s, n_workers, window_start=0.0):
    """Run one chunk in a fresh pool; on a broken pool, retry with fewer workers, then serially."""
    from concurrent.futures.process import BrokenProcessPool
    workers = n_workers or os.cpu_count() or 1
    while workers >= 1:
        try:
            if workers == 1:
                return [_work((str(pcap_dir), e, window_s, window_start)) for e in chunk]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(_work, ((str(pcap_dir), e, window_s, window_start) for e in chunk), chunksize=16))
        except (BrokenProcessPool, OSError) as exc:
            print(f"  pool broke with {workers} workers ({type(exc).__name__}); retrying with {max(1, workers // 2)}", flush=True)
            workers = max(1, workers // 2) if workers > 1 else 0
    raise RuntimeError("extraction failed even serially")


def extract(window_s: float | None, out_csv: Path, pcap_dir: Path, n_workers: int | None = None,
            chunk_size: int = 2000, window_start: float = 0.0) -> pd.DataFrame:
    """Extract features for every file in dataset.json.

    Files are processed in chunks with a fresh worker pool per chunk (bounds
    memory growth in scapy) and a checkpoint CSV is written after every chunk,
    so an interrupted run resumes where it stopped.
    """
    dataset = json.loads((pcap_dir / "dataset.json").read_text())
    ckpt = out_csv.with_suffix(".partial.csv")
    done_rows = []
    if ckpt.exists():
        done_rows = pd.read_csv(ckpt).to_dict("records")
        done = {r["filename"] for r in done_rows}
        dataset = [e for e in dataset if e["filename"] not in done]
        print(f"[extract] resuming: {len(done)} done, {len(dataset)} remaining", flush=True)
    workers = n_workers or os.cpu_count() or 1
    print(f"[extract] {len(dataset)} files, window={window_s or 'full'} start={window_start:g}s, {workers} workers -> {out_csv}", flush=True)
    t0 = time.time()
    rows = list(done_rows)
    n_new = 0
    for start in range(0, len(dataset), chunk_size):
        chunk = dataset[start:start + chunk_size]
        rows.extend(_run_chunk(chunk, pcap_dir, window_s, n_workers, window_start))
        n_new += len(chunk)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        el = time.time() - t0
        rate = n_new / el
        print(f"  {len(rows):>6}/{len(rows) + len(dataset) - n_new}  {rate:.1f} files/s  "
              f"ETA {(len(dataset) - n_new) / rate / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    errs = df["_error"].notna().sum() if "_error" in df else 0
    if errs:
        print(f"  WARNING: {errs} files failed to parse; rows kept with NaN features")
        df = df.drop(columns=["_error"])
    feat_cols = sorted(c for c in df.columns if c not in META_COLS)
    df = df[[c for c in META_COLS if c in df.columns] + feat_cols]
    if window_s is not None:
        df.insert(len(META_COLS), "window_s", window_s)
        df.insert(len(META_COLS) + 1, "window_start", window_start)
    df.to_csv(out_csv, index=False)
    ckpt.unlink(missing_ok=True)
    print(f"[extract] done in {(time.time()-t0)/60:.1f} min ({n_new} new files), shape {df.shape}")
    return df


def benchmark(pcap_dir: Path, n: int = 200, seed: int = 42) -> None:
    """Serial single-process timing of full-window extraction, for the paper."""
    import random
    dataset = json.loads((pcap_dir / "dataset.json").read_text())
    sample = random.Random(seed).sample(dataset, n)
    times = []
    for e in sample:
        t0 = time.perf_counter()
        features_for_file(pcap_dir / e["filename"], None)
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    print(f"[benchmark] full-window extraction, single process, n={n}: "
          f"median {np.median(a):.0f} ms, mean {a.mean():.0f} ms, IQR {np.percentile(a,25):.0f}-{np.percentile(a,75):.0f} ms")


def window_csv(window_s: int, window_start: float = 0.0) -> Path:
    start = f"s{int(window_start):03d}_" if window_start else ""
    return SETTINGS["data_dir"] / "windows" / f"features_{start}w{int(window_s):03d}.csv"


def onset(pcap_dir: Path, n_per_group: int = 20, seed: int = 42, thresh_m: float = 0.5) -> None:
    """Estimate when the attack starts inside each recording.

    For a sample of spoofed files per scenario, find the first epoch at which
    the inter-receiver distance departs by more than thresh_m from its value
    over the first 5 epochs, and the first epoch at which receiver-1 mean SNR
    departs by more than 3 dB. Prints the distribution per scenario.
    """
    import random
    dataset = json.loads((pcap_dir / "dataset.json").read_text())
    rng = random.Random(seed)
    by = {}
    for e in dataset:
        if e["label"] == "spoofed":
            by.setdefault(e["scenario"], []).append(e)
    for scn, entries in sorted(by.items()):
        onsets_d, onsets_snr = [], []
        for e in rng.sample(entries, min(n_per_group, len(entries))):
            sent, gsv = read_pcap(pcap_dir / e["filename"])
            e1, e2 = build_epochs(sent[1], gsv[1], None), build_epochs(sent[2], gsv[2], None)
            if not e1 or not e2:
                continue
            t_first = e1[0]["utc_time"]
            inter = inter_receiver_features(e1, e2)["inter_rx_distance_m"]
            base = np.nanmedian(inter[:5])
            od = next((e1[i]["utc_time"] - t_first for i, v in enumerate(inter) if not math.isnan(v) and abs(v - base) > thresh_m), None)
            snr = [x.get("mean_snr", NAN) for x in e1]
            base_s = np.nanmedian(snr[:5])
            osn = next((e1[i]["utc_time"] - t_first for i, v in enumerate(snr) if not math.isnan(v) and abs(v - base_s) > 3), None)
            onsets_d.append(od)
            onsets_snr.append(osn)
        def summ(xs):
            v = [x for x in xs if x is not None]
            return f"n={len(v)}/{len(xs)} detected, min {min(v):.0f}s, median {np.median(v):.0f}s, max {max(v):.0f}s" if v else "never"
        print(f"[onset] {scn}: inter-receiver distance: {summ(onsets_d)}")
        print(f"[onset] {scn}: receiver-1 SNR:          {summ(onsets_snr)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=None, help="window length in seconds")
    ap.add_argument("--start", type=float, default=None,
                    help="window start offset in seconds (default: settings window_start, else 0)")
    ap.add_argument("--all-windows", action="store_true")
    ap.add_argument("--onset", type=int, default=None, metavar="N",
                    help="estimate attack onset time from N spoofed files per scenario and exit")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    ap.add_argument("--workers", type=int, default=SETTINGS["n_workers"])
    ap.add_argument("--benchmark", type=int, default=None, metavar="N",
                    help="time full-window extraction serially on N random files and exit")
    a = ap.parse_args(argv)
    if a.benchmark:
        benchmark(SETTINGS["pcap_dir"], a.benchmark)
        return
    if a.onset:
        onset(SETTINGS["pcap_dir"], a.onset)
        return
    start = a.start if a.start is not None else float(SETTINGS.get("window_start") or 0.0)

    jobs = []
    if a.all_windows:
        jobs = [(w, window_csv(w, start)) for w in SETTINGS["window_lengths"] if start + w < 120]
    elif a.window is not None:
        jobs = [(a.window, window_csv(a.window, start))]
    else:
        jobs = [(None, SETTINGS["data_dir"] / "marsim_features.csv")]
        start = 0.0
    for w, out in jobs:
        if out.exists() and not a.force:
            print(f"[extract] {out} exists, skipping (use --force to redo)")
            continue
        extract(w, out, SETTINGS["pcap_dir"], a.workers, window_start=start)


if __name__ == "__main__":
    main()
