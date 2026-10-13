"""Loading of the two frozen input files.

marsim_features.csv : 138 features + 8 metadata columns, one row per pcap file.
mana_predictions.csv: MANA binary prediction + per-method trigger flags.

Both files are treated as immutable inputs; their SHA256 hashes are recorded in
data/SHA256SUMS and verified on load.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["filename", "scenario", "label", "index",
             "param_1_name", "param_1_value", "param_2_name", "param_2_value"]
MANA_METHODS = ["pdm", "pcc_sog", "pcc_rot", "pcc_height", "cdm", "cnm"]
SCENARIOS = ["A1", "A2", "A3"]
SCENARIO_PARAMS = {
    "A1": ("distance_to_ship", "time_difference"),
    "A2": ("distance_to_ship", "delay"),
    "A3": ("shift_angle", "shift_speed"),
}


@dataclass
class Dataset:
    meta: pd.DataFrame      # metadata + MANA columns, aligned with X
    X: pd.DataFrame         # raw 138 features
    y: np.ndarray           # 1 = spoofed

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(data_dir: Path, strict: bool = False) -> None:
    """Compare the CSVs against data/SHA256SUMS.

    A regenerated marsim_features.csv will differ byte-for-byte from the frozen
    one (float formatting), so by default a mismatch is a warning, not an error.
    """
    import warnings
    sums = data_dir / "SHA256SUMS"
    if not sums.exists():
        return
    for line in sums.read_text().splitlines():
        digest, name = line.split()
        p = data_dir / Path(name).name
        if not p.exists():
            continue
        actual = _sha256(p)
        if actual != digest:
            msg = f"{p.name} differs from the frozen input recorded in SHA256SUMS"
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg)


def load(data_dir: str | Path, verify: bool = True, features_csv: str | Path | None = None) -> Dataset:
    """Load features + MANA predictions.

    features_csv: alternative feature file (e.g. a windowed extraction). MANA
    predictions are always the full-file ones.
    """
    data_dir = Path(data_dir)
    if verify:
        verify_hashes(data_dir)
    feat = pd.read_csv(features_csv or data_dir / "marsim_features.csv")
    feat = feat.drop(columns=["window_s", "window_start"], errors="ignore")
    mana = pd.read_csv(data_dir / "mana_predictions.csv")
    if "mana_ms" in mana.columns:
        mana = mana.drop(columns=["mana_ms", "error"], errors="ignore")
    if (mana["mana_pred"] == -1).any():
        raise RuntimeError("mana_predictions.csv contains error rows (-1); resolve before evaluating")

    assert feat["filename"].is_unique and mana["filename"].is_unique
    df = feat.merge(mana, on="filename", how="inner", validate="one_to_one")
    assert len(df) == len(feat) == len(mana), "feature/MANA filename mismatch"

    df["index"] = df["index"].astype(int)
    feature_cols = [c for c in feat.columns if c not in META_COLS]
    assert len(feature_cols) == 138, f"expected 138 features, got {len(feature_cols)}"

    meta = df[META_COLS + ["mana_pred", "mana_triggered_methods"] + MANA_METHODS].copy()
    X = df[feature_cols].copy()
    y = (df["label"] == "spoofed").astype(int).to_numpy()
    return Dataset(meta=meta.reset_index(drop=True), X=X.reset_index(drop=True), y=y)
