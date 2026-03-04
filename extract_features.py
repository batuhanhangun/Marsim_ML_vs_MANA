#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARSIM PCAP Feature Extraction Pipeline
Run with: conda activate marsim_env && python extract_features.py
"""

import os
import sys
import json
import time
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# Suppress scapy's noisy import output
warnings.filterwarnings('ignore')
os.environ['SCAPY_USE_LIBPCAP'] = '0'

from scapy.all import PcapReader, IP, UDP, Raw

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DATASET_DIR = r"D:\IDMAN_Downloads\ZIP\8202936\dataset\dataset"
DATASET_JSON = os.path.join(DATASET_DIR, "dataset.json")
OUTPUT_CSV = os.path.join(DATASET_DIR, "marsim_features.csv")

# Receiver identification
RX1_IP, RX1_PORT = "192.168.0.10", 62996
RX2_IP, RX2_PORT = "192.168.0.11", 62997

# Epoch matching tolerance (seconds)
EPOCH_MATCH_TOL = 0.5

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def angular_diff(a, b):
    """Angular difference handling 360-degree wraparound."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return float('nan')
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def nmea_time_to_seconds(time_str):
    """Convert NMEA UTC time string (HHMMSS.sss) to seconds since midnight."""
    if not time_str or time_str.strip() == '':
        return float('nan')
    try:
        t = float(time_str)
        hours = int(t / 10000)
        minutes = int((t % 10000) / 100)
        seconds = t % 100
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError):
        return float('nan')


def nmea_coord_to_decimal(coord_str, hemisphere):
    """Convert NMEA DDDMM.MMMMM coordinate to decimal degrees."""
    if not coord_str or coord_str.strip() == '':
        return float('nan')
    try:
        val = float(coord_str)
        degrees = int(val / 100)
        minutes = val - degrees * 100
        decimal = degrees + minutes / 60.0
        if hemisphere in ('S', 'W'):
            decimal = -decimal
        return decimal
    except (ValueError, TypeError):
        return float('nan')


def safe_float(val):
    """Safely convert a string to float, returning NaN on failure."""
    if val is None or (isinstance(val, str) and val.strip() == ''):
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


# ===========================================================================
# NMEA SENTENCE PARSERS
# ===========================================================================

def parse_gprmc(fields):
    """
    Parse $GPRMC sentence.
    $GPRMC,UTC,Status,Lat,N/S,Lon,E/W,SOG,COG,Date,MagVar,E/W,Mode*CS
    """
    result = {}
    if len(fields) < 10:
        return result
    result['utc_time'] = nmea_time_to_seconds(fields[1])
    result['lat_deg'] = nmea_coord_to_decimal(fields[3], fields[4]) if len(fields) > 4 else float('nan')
    result['lon_deg'] = nmea_coord_to_decimal(fields[5], fields[6]) if len(fields) > 6 else float('nan')
    result['sog_knots'] = safe_float(fields[7])
    result['cog_deg'] = safe_float(fields[8])
    return result


def parse_gpgga(fields):
    """
    Parse $GPGGA sentence.
    $GPGGA,UTC,Lat,N/S,Lon,E/W,Quality,NumSats,HDOP,Alt,M,Geoid,M,Age,RefID*CS
    """
    result = {}
    if len(fields) < 10:
        return result
    result['utc_time'] = nmea_time_to_seconds(fields[1])
    result['lat_deg'] = nmea_coord_to_decimal(fields[2], fields[3]) if len(fields) > 3 else float('nan')
    result['lon_deg'] = nmea_coord_to_decimal(fields[4], fields[5]) if len(fields) > 5 else float('nan')
    result['fix_quality'] = safe_float(fields[6])
    result['num_satellites'] = safe_float(fields[7])
    result['hdop'] = safe_float(fields[8])
    result['altitude'] = safe_float(fields[9])
    return result


def parse_gpgsa(fields):
    """
    Parse $GPGSA sentence.
    $GPGSA,Mode1,Mode2,SV1,...,SV12,PDOP,HDOP,VDOP*CS
    """
    result = {}
    if len(fields) < 18:
        return result
    result['fix_mode'] = safe_float(fields[2])
    result['pdop'] = safe_float(fields[15])
    result['hdop_gsa'] = safe_float(fields[16])
    # VDOP may have checksum appended
    vdop_str = fields[17].split('*')[0] if len(fields) > 17 else ''
    result['vdop'] = safe_float(vdop_str)
    return result


def parse_gpgsv(fields):
    """
    Parse $GPGSV sentence. Returns list of satellite SNR values.
    $GPGSV,NumMsg,MsgNum,NumSV,[PRN,Elev,Azim,SNR]*4*CS
    """
    snr_values = []
    if len(fields) < 4:
        return snr_values
    # Each satellite block is 4 fields: PRN, Elevation, Azimuth, SNR
    idx = 4  # start after NumMsg, MsgNum, NumSV
    while idx + 3 < len(fields):
        snr_str = fields[idx + 3].split('*')[0]  # handle checksum on last field
        snr = safe_float(snr_str)
        if not math.isnan(snr):
            snr_values.append(snr)
        idx += 4
    return snr_values


def parse_gpvtg(fields):
    """
    Parse $GPVTG sentence.
    $GPVTG,COG_True,T,COG_Mag,M,SOG_knots,N,SOG_kmh,K,Mode*CS
    """
    result = {}
    if len(fields) < 8:
        return result
    result['cog_true'] = safe_float(fields[1])
    result['cog_magnetic'] = safe_float(fields[3])
    result['sog_vtg_knots'] = safe_float(fields[5])
    result['sog_vtg_kmh'] = safe_float(fields[7].split('*')[0])
    return result


# ===========================================================================
# PACKET READING & EPOCH GROUPING
# ===========================================================================

def read_pcap_packets(filepath):
    """
    Read a pcap file using scapy PcapReader (streaming).
    Returns two lists of (utc_seconds, parsed_sentences_dict) per receiver.
    
    Each receiver's data is organized into epochs keyed by UTC time.
    """
    rx1_sentences = defaultdict(dict)  # {utc_time: {sentence_type: parsed_data}}
    rx2_sentences = defaultdict(dict)
    rx1_gsv = defaultdict(list)  # {utc_time: [snr, snr, ...]}
    rx2_gsv = defaultdict(list)

    current_epoch_time = {1: None, 2: None}

    try:
        reader = PcapReader(str(filepath))
    except Exception:
        return {}, {}, {}, {}

    try:
        for pkt in reader:
            if not pkt.haslayer(IP) or not pkt.haslayer(UDP) or not pkt.haslayer(Raw):
                continue

            src_ip = pkt[IP].src
            src_port = pkt[UDP].sport
            try:
                payload = pkt[Raw].load.decode('ascii', errors='ignore').strip()
            except Exception:
                continue

            if not payload or not payload.startswith('$GP'):
                continue

            # Determine receiver
            if src_ip == RX1_IP and src_port == RX1_PORT:
                rx_id = 1
                sentences = rx1_sentences
                gsv_data = rx1_gsv
            elif src_ip == RX2_IP and src_port == RX2_PORT:
                rx_id = 2
                sentences = rx2_sentences
                gsv_data = rx2_gsv
            else:
                continue

            # Split the sentence into fields
            # Remove checksum if present (after *)
            fields = payload.split(',')
            sentence_type = fields[0]

            if sentence_type == '$GPRMC':
                parsed = parse_gprmc(fields)
                if 'utc_time' in parsed and not math.isnan(parsed['utc_time']):
                    epoch_t = parsed['utc_time']
                    current_epoch_time[rx_id] = epoch_t
                    sentences[epoch_t].update(parsed)
                    sentences[epoch_t]['_type_rmc'] = True

            elif sentence_type == '$GPGGA':
                parsed = parse_gpgga(fields)
                if 'utc_time' in parsed and not math.isnan(parsed['utc_time']):
                    epoch_t = parsed['utc_time']
                    current_epoch_time[rx_id] = epoch_t
                    sentences[epoch_t].update(parsed)
                    sentences[epoch_t]['_type_gga'] = True

            elif sentence_type == '$GPGSA':
                parsed = parse_gpgsa(fields)
                epoch_t = current_epoch_time[rx_id]
                if epoch_t is not None:
                    sentences[epoch_t].update(parsed)

            elif sentence_type == '$GPGSV':
                snr_values = parse_gpgsv(fields)
                epoch_t = current_epoch_time[rx_id]
                if epoch_t is not None:
                    gsv_data[epoch_t].extend(snr_values)

            elif sentence_type == '$GPVTG':
                parsed = parse_gpvtg(fields)
                epoch_t = current_epoch_time[rx_id]
                if epoch_t is not None:
                    sentences[epoch_t].update(parsed)

            # Skip $GPGLL (redundant per spec)

    except Exception:
        pass
    finally:
        reader.close()

    return rx1_sentences, rx2_sentences, rx1_gsv, rx2_gsv


# ===========================================================================
# PER-EPOCH FEATURE EXTRACTION
# ===========================================================================

def build_epoch_array(sentences, gsv_data):
    """
    Convert epoch dictionaries into a sorted list of epoch feature dicts.
    Each dict has all raw values for that epoch.
    """
    epochs = []
    for t in sorted(sentences.keys()):
        epoch = dict(sentences[t])
        epoch['utc_time'] = t

        # Add SNR stats
        snrs = gsv_data.get(t, [])
        if snrs:
            epoch['mean_snr'] = np.mean(snrs)
            epoch['std_snr'] = np.std(snrs)
            epoch['min_snr'] = np.min(snrs)
            epoch['max_snr'] = np.max(snrs)
            epoch['num_sats_with_snr'] = len(snrs)
        else:
            epoch['mean_snr'] = float('nan')
            epoch['std_snr'] = float('nan')
            epoch['min_snr'] = float('nan')
            epoch['max_snr'] = float('nan')
            epoch['num_sats_with_snr'] = 0

        epochs.append(epoch)

    return epochs


def compute_derived_features(epochs):
    """
    Compute derived per-epoch features from consecutive epochs.
    Returns lists of derived feature values (one less than number of epochs).
    """
    derived = {
        'position_jump_m': [],
        'computed_sog_knots': [],
        'sog_discrepancy': [],
        'cog_change_rate': [],
        'cog_heading_discrepancy': [],
        'time_delta': [],
    }

    for i in range(1, len(epochs)):
        prev = epochs[i - 1]
        curr = epochs[i]

        lat1 = prev.get('lat_deg', float('nan'))
        lon1 = prev.get('lon_deg', float('nan'))
        lat2 = curr.get('lat_deg', float('nan'))
        lon2 = curr.get('lon_deg', float('nan'))

        t1 = prev.get('utc_time', float('nan'))
        t2 = curr.get('utc_time', float('nan'))

        # Time delta
        dt = t2 - t1 if not (math.isnan(t1) or math.isnan(t2)) else float('nan')
        derived['time_delta'].append(dt)

        # Position jump
        if not any(math.isnan(v) for v in [lat1, lon1, lat2, lon2]):
            jump = haversine_m(lat1, lon1, lat2, lon2)
            derived['position_jump_m'].append(jump)

            # Computed SOG (m/s -> knots: m/s * 1.94384)
            if not math.isnan(dt) and dt > 0:
                computed_sog = (jump / dt) * 1.94384
                derived['computed_sog_knots'].append(computed_sog)

                # SOG discrepancy
                reported_sog = curr.get('sog_knots', float('nan'))
                if not math.isnan(reported_sog):
                    derived['sog_discrepancy'].append(abs(reported_sog - computed_sog))
                else:
                    derived['sog_discrepancy'].append(float('nan'))

                # Heading from positions
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                heading = math.degrees(math.atan2(dlon, dlat)) % 360

                # COG heading discrepancy
                reported_cog = curr.get('cog_deg', float('nan'))
                if not math.isnan(reported_cog):
                    derived['cog_heading_discrepancy'].append(angular_diff(reported_cog, heading))
                else:
                    derived['cog_heading_discrepancy'].append(float('nan'))
            else:
                derived['computed_sog_knots'].append(float('nan'))
                derived['sog_discrepancy'].append(float('nan'))
                derived['cog_heading_discrepancy'].append(float('nan'))
        else:
            derived['position_jump_m'].append(float('nan'))
            derived['computed_sog_knots'].append(float('nan'))
            derived['sog_discrepancy'].append(float('nan'))
            derived['cog_heading_discrepancy'].append(float('nan'))

        # COG change rate
        cog1 = prev.get('cog_deg', float('nan'))
        cog2 = curr.get('cog_deg', float('nan'))
        if not (math.isnan(cog1) or math.isnan(cog2)) and not math.isnan(dt) and dt > 0:
            derived['cog_change_rate'].append(angular_diff(cog2, cog1) / dt)
        else:
            derived['cog_change_rate'].append(float('nan'))

    return derived


# ===========================================================================
# MULTI-RECEIVER FEATURES
# ===========================================================================

def compute_multi_receiver_features(rx1_epochs, rx2_epochs):
    """
    Match epochs from two receivers by UTC timestamp and compute
    inter-receiver comparison features.
    """
    inter_features = {
        'inter_rx_distance_m': [],
        'inter_rx_sog_diff': [],
        'inter_rx_cog_diff': [],
    }

    if not rx1_epochs or not rx2_epochs:
        return inter_features

    # Build sorted time lists for rx2 for matching
    rx2_times = [e['utc_time'] for e in rx2_epochs]
    rx2_by_time = {e['utc_time']: e for e in rx2_epochs}

    for e1 in rx1_epochs:
        t1 = e1['utc_time']

        # Find closest rx2 epoch
        best_t2 = None
        best_diff = float('inf')
        for t2 in rx2_times:
            diff = abs(t1 - t2)
            if diff < best_diff:
                best_diff = diff
                best_t2 = t2

        if best_t2 is None or best_diff > EPOCH_MATCH_TOL:
            inter_features['inter_rx_distance_m'].append(float('nan'))
            inter_features['inter_rx_sog_diff'].append(float('nan'))
            inter_features['inter_rx_cog_diff'].append(float('nan'))
            continue

        e2 = rx2_by_time[best_t2]

        # Inter-receiver distance
        lat1 = e1.get('lat_deg', float('nan'))
        lon1 = e1.get('lon_deg', float('nan'))
        lat2 = e2.get('lat_deg', float('nan'))
        lon2 = e2.get('lon_deg', float('nan'))

        if not any(math.isnan(v) for v in [lat1, lon1, lat2, lon2]):
            inter_features['inter_rx_distance_m'].append(haversine_m(lat1, lon1, lat2, lon2))
        else:
            inter_features['inter_rx_distance_m'].append(float('nan'))

        # SOG difference
        sog1 = e1.get('sog_knots', float('nan'))
        sog2 = e2.get('sog_knots', float('nan'))
        if not (math.isnan(sog1) or math.isnan(sog2)):
            inter_features['inter_rx_sog_diff'].append(abs(sog1 - sog2))
        else:
            inter_features['inter_rx_sog_diff'].append(float('nan'))

        # COG difference
        cog1 = e1.get('cog_deg', float('nan'))
        cog2 = e2.get('cog_deg', float('nan'))
        inter_features['inter_rx_cog_diff'].append(angular_diff(cog1, cog2))

    return inter_features


# ===========================================================================
# AGGREGATION: per-epoch series → per-file summary stats
# ===========================================================================

def safe_agg(values, func):
    """Apply aggregation function ignoring NaNs. Returns NaN if no valid values."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float('nan')
    return func(clean)


def aggregate_epoch_series(epochs, key):
    """Extract a key from epoch dicts and compute summary stats."""
    values = [e.get(key, float('nan')) for e in epochs]
    return aggregate_values(values, key)


def aggregate_values(values, prefix):
    """Compute mean, std, min, max, median for a list of values."""
    return {
        f'{prefix}_mean': safe_agg(values, np.mean),
        f'{prefix}_std': safe_agg(values, np.std),
        f'{prefix}_min': safe_agg(values, np.min),
        f'{prefix}_max': safe_agg(values, np.max),
        f'{prefix}_median': safe_agg(values, np.median),
    }


def aggregate_derived(derived_dict, key):
    """Aggregate a derived feature series."""
    return aggregate_values(derived_dict.get(key, []), key)


# ===========================================================================
# MAIN PER-FILE PROCESSING
# ===========================================================================

def process_one_file(filepath):
    """
    Process a single pcap file and return a dict of aggregated features.
    """
    rx1_sentences, rx2_sentences, rx1_gsv, rx2_gsv = read_pcap_packets(filepath)

    rx1_epochs = build_epoch_array(rx1_sentences, rx1_gsv)
    rx2_epochs = build_epoch_array(rx2_sentences, rx2_gsv)

    features = {}

    # Use receiver 1 as the primary receiver for per-receiver features
    # (both receivers see the same signal in unspoofed case)
    primary = rx1_epochs if rx1_epochs else rx2_epochs

    if not primary:
        return features

    # --- Raw epoch feature aggregation (Receiver 1 / primary) ---
    for key in ['sog_knots', 'cog_deg', 'num_satellites', 'hdop', 'altitude',
                'pdop', 'hdop_gsa', 'vdop', 'fix_mode',
                'mean_snr', 'std_snr', 'min_snr', 'max_snr', 'num_sats_with_snr',
                'sog_vtg_knots', 'sog_vtg_kmh', 'cog_true', 'cog_magnetic']:
        features.update(aggregate_epoch_series(primary, key))

    # --- Derived features (consecutive epochs) ---
    derived = compute_derived_features(primary)
    for key in ['position_jump_m', 'computed_sog_knots', 'sog_discrepancy',
                'cog_change_rate', 'cog_heading_discrepancy', 'time_delta']:
        features.update(aggregate_derived(derived, key))

    # Clock regularity = std of time deltas
    time_deltas = [v for v in derived.get('time_delta', []) if not math.isnan(v)]
    features['clock_regularity'] = np.std(time_deltas) if time_deltas else float('nan')

    # --- Multi-receiver features ---
    inter = compute_multi_receiver_features(rx1_epochs, rx2_epochs)
    for key in ['inter_rx_distance_m', 'inter_rx_sog_diff', 'inter_rx_cog_diff']:
        features.update(aggregate_values(inter.get(key, []), key))

    # Number of epochs (informational)
    features['n_epochs_rx1'] = len(rx1_epochs)
    features['n_epochs_rx2'] = len(rx2_epochs)

    return features


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print("=" * 70)
    print("MARSIM PCAP Feature Extraction Pipeline")
    print("=" * 70)

    # 1. Load dataset.json
    print(f"\nLoading dataset metadata from: {DATASET_JSON}")
    with open(DATASET_JSON, 'r') as f:
        dataset = json.load(f)
    print(f"  Total entries in dataset.json: {len(dataset)}")

    # 2. Process each file
    results = []
    failures = []
    start_time = time.time()

    for i, entry in enumerate(dataset):
        filename = entry['filename']
        filepath = os.path.join(DATASET_DIR, filename)

        if i > 0 and i % 1000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (len(dataset) - i) / rate if rate > 0 else 0
            print(f"  [{i:>6}/{len(dataset)}] "
                  f"Processed {i} files in {elapsed:.0f}s "
                  f"({rate:.1f} files/s, ETA: {eta:.0f}s) | "
                  f"Failures: {len(failures)}")

        try:
            features = process_one_file(filepath)

            # Add metadata
            features['filename'] = filename
            features['scenario'] = entry['scenario']
            features['label'] = entry['label']
            features['index'] = int(entry['index'])

            params = entry.get('parameters', {})
            param_keys = list(params.keys())
            if len(param_keys) >= 1:
                features['param_1_name'] = param_keys[0]
                features['param_1_value'] = safe_float(params[param_keys[0]])
            if len(param_keys) >= 2:
                features['param_2_name'] = param_keys[1]
                features['param_2_value'] = safe_float(params[param_keys[1]])

            results.append(features)

        except Exception as e:
            failures.append((filename, str(e)))
            # Still add a row with metadata and NaN features
            features = {
                'filename': filename,
                'scenario': entry['scenario'],
                'label': entry['label'],
                'index': int(entry['index']),
            }
            params = entry.get('parameters', {})
            param_keys = list(params.keys())
            if len(param_keys) >= 1:
                features['param_1_name'] = param_keys[0]
                features['param_1_value'] = safe_float(params[param_keys[0]])
            if len(param_keys) >= 2:
                features['param_2_name'] = param_keys[1]
                features['param_2_value'] = safe_float(params[param_keys[1]])
            results.append(features)

    total_time = time.time() - start_time

    # 3. Create DataFrame and save
    print(f"\n{'=' * 70}")
    print(f"Processing complete!")
    print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Files processed: {len(results)}")
    print(f"  Failures: {len(failures)}")

    if failures:
        print(f"\n  First 10 failures:")
        for fn, err in failures[:10]:
            print(f"    {fn}: {err}")

    df = pd.DataFrame(results)

    # Reorder columns: metadata first, then features
    meta_cols = ['filename', 'scenario', 'label', 'index',
                 'param_1_name', 'param_1_value', 'param_2_name', 'param_2_value']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    ordered_cols = [c for c in meta_cols if c in df.columns] + sorted(feature_cols)
    df = df[ordered_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Output saved to: {OUTPUT_CSV}")

    # 4. Validation
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  Shape: {df.shape}")
    print(f"\n  Class balance (label):")
    print(df['label'].value_counts().to_string(header=False))

    print(f"\n  Per-scenario count:")
    print(df['scenario'].value_counts().sort_index().to_string(header=False))

    print(f"\n  First 5 rows (metadata columns):")
    print(df[meta_cols].head().to_string())

    # Check for all-NaN columns
    nan_rates = df[feature_cols].isna().mean()
    all_nan = nan_rates[nan_rates == 1.0]
    if len(all_nan) > 0:
        print(f"\n  WARNING: {len(all_nan)} all-NaN feature columns:")
        for col in all_nan.index:
            print(f"    - {col}")
    else:
        print(f"\n  No all-NaN feature columns (good)")

    # NaN rate per column
    print(f"\n  NaN rate per feature column (top 20 highest):")
    top_nan = nan_rates.sort_values(ascending=False).head(20)
    for col, rate in top_nan.items():
        print(f"    {col:45s} {rate:.4f}")

    print(f"\n  Total feature columns: {len(feature_cols)}")
    print(f"  Total columns (incl. metadata): {len(df.columns)}")
    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == '__main__':
    main()
