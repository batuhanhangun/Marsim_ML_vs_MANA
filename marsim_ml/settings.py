"""Project paths and constants.

Override any default by creating config.json in the repository root, e.g.

    {"pcap_dir": "/path/to/MARSIM/dataset", "n_workers": 8}

pcap_dir is the folder that contains dataset.json and the MARSIM pcap files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULTS = {
    # Directory that contains dataset.json and the 43,320 .pcap files
    # (relative paths are resolved against the repository root)
    "pcap_dir": "data/marsim_pcaps",
    # Project root: always the folder containing the marsim_ml package.
    # (Set explicitly in config.json only if you really want somewhere else.)
    "project_dir": None,
    # Number of worker processes for pcap parsing (None = all cores)
    "n_workers": max(1, (os.cpu_count() or 2) - 2),
    # MANA methods.json (relative to project_dir unless absolute)
    "mana_methods_json": "configs/methods.json",
    # Window-sensitivity experiment: windows [window_start, window_start + W].
    # window_start should be the attack onset inside each recording
    # (run `python -m marsim_ml.extract --onset 20` to measure it).
    "window_start": 60,
    "window_lengths": [5, 10, 15, 20, 30, 45, 60],
    # Random search iterations for the 'tuned' configuration
    "tune_n_iter": 30,
    "seed": 42,
}


def _find_config() -> Path | None:
    here = Path(__file__).resolve().parent.parent
    for cand in [here / "config.json", Path.cwd() / "config.json"]:
        if cand.exists():
            return cand
    return None


def load_settings() -> dict:
    s = dict(_DEFAULTS)
    cfg = _find_config()
    if cfg is not None:
        s.update(json.loads(cfg.read_text()))
    pkg_root = Path(__file__).resolve().parent.parent
    s["project_dir"] = Path(s["project_dir"]) if s.get("project_dir") else pkg_root
    pcap = Path(s["pcap_dir"])
    s["pcap_dir"] = pcap if pcap.is_absolute() else s["project_dir"] / pcap
    s["data_dir"] = s["project_dir"] / "data"
    s["results_dir"] = s["project_dir"] / "results"
    s["figures_dir"] = s["project_dir"] / "figures"
    s["tables_dir"] = s["project_dir"] / "tables"
    mj = Path(s["mana_methods_json"])
    s["mana_methods_json"] = mj if mj.is_absolute() else s["project_dir"] / mj
    return s


SETTINGS = load_settings()
