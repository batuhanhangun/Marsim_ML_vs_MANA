"""Classifier factory.

Each entry: (constructor, default kwargs, needs_scaling). Hyperparameters are
overridable from the run config so a 'baseline' and a 'tuned'
configuration can share code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


@dataclass
class ModelSpec:
    name: str
    kind: str                       # key in _REGISTRY
    params: dict[str, Any] = field(default_factory=dict)
    subsample_train: int | None = None   # e.g. SVM trained on 10k subset
    needs_scaling: bool = False
    has_importance: bool = False


_REGISTRY = {
    "rf": (RandomForestClassifier, False, True),
    "xgb": (XGBClassifier, False, True),
    "lgbm": (LGBMClassifier, False, True),
    "svm": (SVC, True, False),
    "logreg": (LogisticRegression, True, False),
    "knn": (KNeighborsClassifier, True, False),
}

# Default (near-default) hyperparameters used for all experiments
DEFAULT_MODELS = [
    ModelSpec("RF", "rf", dict(n_estimators=500, max_depth=None, min_samples_split=5,
                               min_samples_leaf=2, max_features="sqrt",
                               class_weight="balanced", random_state=42, n_jobs=-1)),
    ModelSpec("XGB", "xgb", dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                 random_state=42, n_jobs=-1, eval_metric="logloss")),
    ModelSpec("LGBM", "lgbm", dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                   random_state=42, n_jobs=-1, verbose=-1)),
    ModelSpec("SVM", "svm", dict(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                                 probability=True, random_state=42),
              subsample_train=10_000),
]

# Search spaces for RandomizedSearchCV
SEARCH_SPACES = {
    "rf": {"n_estimators": [200, 500, 1000], "max_depth": [None, 10, 20, 40],
           "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 0.3]},
    "xgb": {"n_estimators": [200, 500, 1000], "max_depth": [3, 6, 9],
            "learning_rate": [0.03, 0.1, 0.3], "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5]},
    "lgbm": {"n_estimators": [200, 500, 1000], "num_leaves": [15, 31, 63, 127],
             "learning_rate": [0.03, 0.1, 0.3], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "min_child_samples": [10, 20, 50]},
    "svm": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]},
}


def build(spec: ModelSpec):
    cls, needs_scaling, has_imp = _REGISTRY[spec.kind]
    spec.needs_scaling = needs_scaling
    spec.has_importance = has_imp
    return cls(**spec.params)
