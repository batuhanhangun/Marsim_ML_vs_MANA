"""Train/test split strategies.

All functions return a boolean mask `is_test` aligned with Dataset.meta.

repetition_split
    The primary split: repetition indices 0-15 train, 16-19 test. Every
    parameter combination appears in BOTH partitions (in-distribution).

heldout_param_split
    Generalization split: hold out a set of parameter VALUES for one scenario
    axis so the test set contains attack configurations never seen in training.
    Unspoofed files carry the same parameter labels in MARSIM (they are the
    paired negatives of each attack configuration), so they are held out on the
    same criterion, keeping the test set 50/50 balanced.

heldout_scenario_split
    Leave-one-scenario-out: train on two scenarios, test on the third.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .data import SCENARIO_PARAMS


def repetition_split(meta: pd.DataFrame, test_indices: Iterable[int] = range(16, 20)) -> np.ndarray:
    return meta["index"].isin(list(test_indices)).to_numpy()


def _param_value(meta: pd.DataFrame, param: str) -> pd.Series:
    """Return the value of `param` per row, handling the two metadata slots."""
    v = pd.Series(np.nan, index=meta.index, dtype=float)
    m1 = meta["param_1_name"] == param
    m2 = meta["param_2_name"] == param
    v[m1] = meta.loc[m1, "param_1_value"].astype(float)
    v[m2] = meta.loc[m2, "param_2_value"].astype(float)
    return v


def heldout_param_split(meta: pd.DataFrame, scenario: str, param: str,
                        test_values: Iterable[float],
                        other_scenarios: str = "train") -> np.ndarray:
    """Hold out rows of `scenario` whose `param` is in `test_values`.

    other_scenarios: 'train' keeps A1/A2 (etc.) entirely in training,
                     'drop' excludes them from both partitions.
    Returns is_test mask; rows to exclude entirely are marked with -1 via a
    separate mask returned by heldout_param_masks().
    """
    is_test, _ = heldout_param_masks(meta, scenario, param, test_values, other_scenarios)
    return is_test


def heldout_param_masks(meta, scenario, param, test_values, other_scenarios="train"):
    assert param in SCENARIO_PARAMS[scenario], f"{param} is not a parameter of {scenario}"
    test_values = set(float(v) for v in test_values)
    in_scn = (meta["scenario"] == scenario).to_numpy()
    vals = _param_value(meta, param).to_numpy()
    is_test = in_scn & np.isin(vals, list(test_values))
    if other_scenarios == "drop":
        keep = in_scn
    else:
        keep = np.ones(len(meta), dtype=bool)
    return is_test, keep


def heldout_scenario_split(meta: pd.DataFrame, test_scenario: str) -> np.ndarray:
    return (meta["scenario"] == test_scenario).to_numpy()


def describe_split(meta: pd.DataFrame, y: np.ndarray, is_test: np.ndarray, keep: np.ndarray | None = None) -> dict:
    if keep is None:
        keep = np.ones(len(meta), dtype=bool)
    tr = keep & ~is_test
    te = keep & is_test
    return {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "train_pos_rate": float(y[tr].mean()), "test_pos_rate": float(y[te].mean()),
        "test_scenarios": meta.loc[te, "scenario"].value_counts().to_dict(),
    }
