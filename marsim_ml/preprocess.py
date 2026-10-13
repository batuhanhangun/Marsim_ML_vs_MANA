"""Preprocessing chain, fitted on the training partition only.

Steps (each toggled by config so the paper can describe exactly what ran):
  1. zero_variance   : drop features with std < eps on train
  1b. impute         : drop features that are NaN in more than max_nan_frac of
                       train rows, then median-impute the rest (train medians).
                       No-op on the full 120 s MARSIM features; needed for
                       short observation windows.
  2. log_transform   : signed log1p on features with |skew| > threshold
  3. winsorize       : clip to [q_low, q_high] train quantiles (optional)
  4. correlation     : drop one of each pair with |r| > threshold.
                       method='greedy'   : scan upper triangle, drop later column (old code)
                       method='iterative': repeatedly drop the feature with the
                                           highest mean |r| among pairs above threshold
  5. scale           : StandardScaler, applied only to models flagged needs_scaling

Returns transformed train/test frames plus a log dict for the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessConfig:
    zero_variance: bool = True
    zero_variance_eps: float = 1e-10
    impute: bool = True
    max_nan_frac: float = 0.5
    log_transform: bool = True
    skew_threshold: float = 5.0
    winsorize: bool = False
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    correlation: bool = True
    corr_threshold: float = 0.95
    corr_method: str = "greedy"  # 'greedy' | 'iterative'


@dataclass
class Preprocessor:
    cfg: PreprocessConfig
    dropped_zero_var: list[str] = field(default_factory=list)
    dropped_nan: list[str] = field(default_factory=list)
    medians: pd.Series | None = None
    _n_imputed: int = 0
    log_cols: list[str] = field(default_factory=list)
    winsor_lower: pd.Series | None = None
    winsor_upper: pd.Series | None = None
    dropped_corr: list[str] = field(default_factory=list)
    final_features: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None

    # ---------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        X = X.copy()
        c = self.cfg
        if c.zero_variance:
            std = X.std()
            self.dropped_zero_var = std[std < c.zero_variance_eps].index.tolist()
            X = X.drop(columns=self.dropped_zero_var)
        if c.impute:
            nan_frac = X.isna().mean()
            self.dropped_nan = nan_frac[nan_frac > c.max_nan_frac].index.tolist()
            X = X.drop(columns=self.dropped_nan)
            self.medians = X.median()
            self._n_imputed = int((X.isna().sum() > 0).sum())
            X = X.fillna(self.medians)
        if c.log_transform:
            sk = X.apply(skew)
            self.log_cols = sk[sk.abs() > c.skew_threshold].index.tolist()
            X = self._apply_log(X)
        if c.winsorize:
            self.winsor_lower = X.quantile(c.winsor_quantiles[0])
            self.winsor_upper = X.quantile(c.winsor_quantiles[1])
            X = X.clip(lower=self.winsor_lower, upper=self.winsor_upper, axis=1)
        if c.correlation:
            self.dropped_corr = self._select_correlated(X)
            X = X.drop(columns=self.dropped_corr)
        self.final_features = list(X.columns)
        self.scaler = StandardScaler().fit(X)
        return self

    def _apply_log(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.log_cols:
            X[col] = np.sign(X[col]) * np.log1p(X[col].abs())
        return X

    def _select_correlated(self, X: pd.DataFrame) -> list[str]:
        corr = X.corr().abs()
        thr = self.cfg.corr_threshold
        if self.cfg.corr_method == "greedy":
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            return [col for col in upper.columns if (upper[col] > thr).any()]
        if self.cfg.corr_method == "iterative":
            dropped = []
            vals = corr.to_numpy(copy=True)
            np.fill_diagonal(vals, 0.0)
            c = pd.DataFrame(vals, index=corr.index, columns=corr.columns)
            while True:
                above = c > thr
                if not above.values.any():
                    break
                # among features participating in a violating pair, drop the one
                # with highest mean |r| to remaining features
                involved = above.any(axis=1)
                cand = c.loc[involved, :].mean(axis=1)
                victim = cand.idxmax()
                dropped.append(victim)
                c = c.drop(index=victim, columns=victim)
            return dropped
        raise ValueError(self.cfg.corr_method)

    # ---------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        X = X.drop(columns=self.dropped_zero_var, errors="ignore")
        if self.cfg.impute and self.medians is not None:
            X = X.drop(columns=self.dropped_nan, errors="ignore").fillna(self.medians)
        X = self._apply_log(X)
        if self.cfg.winsorize and self.winsor_lower is not None:
            X = X.clip(lower=self.winsor_lower, upper=self.winsor_upper, axis=1)
        X = X.drop(columns=self.dropped_corr, errors="ignore")[self.final_features]
        if scale:
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return X

    def summary(self, n_input: int) -> dict:
        return {
            "n_input": n_input,
            "n_zero_variance_dropped": len(self.dropped_zero_var),
            "n_nan_dropped": len(self.dropped_nan),
            "n_features_with_imputed_values": self._n_imputed,
            "n_log_transformed": len(self.log_cols),
            "winsorized": self.cfg.winsorize,
            "n_correlated_dropped": len(self.dropped_corr),
            "corr_method": self.cfg.corr_method,
            "n_final": len(self.final_features),
        }
