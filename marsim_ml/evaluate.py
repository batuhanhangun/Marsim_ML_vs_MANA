"""Evaluation utilities. All functions are pure: predictions in, tables out."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from statsmodels.stats.contingency_tables import mcnemar

from .data import MANA_METHODS, SCENARIO_PARAMS, SCENARIOS


def _metrics(y, pred, prob=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0),
               fpr=fp / (fp + tn) if fp + tn else np.nan,
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y)))
    out["auc"] = roc_auc_score(y, prob) if prob is not None and len(np.unique(y)) > 1 else np.nan
    return out


def overall_table(y, preds: dict, probs: dict, times_ms: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        m = _metrics(y, p, probs.get(name))
        m["ms_per_sample"] = times_ms.get(name, np.nan)
        rows.append({"classifier": name, **m})
    return pd.DataFrame(rows)


def per_scenario_table(meta, y, preds: dict, probs: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        for s in SCENARIOS:
            mask = (meta["scenario"] == s).to_numpy()
            if mask.sum() == 0:
                continue
            pr = probs.get(name)
            m = _metrics(y[mask], p[mask], pr[mask] if pr is not None else None)
            rows.append({"classifier": name, "scenario": s, **m})
    return pd.DataFrame(rows)


def _param_value(meta, param):
    v = pd.Series(np.nan, index=meta.index, dtype=float)
    m1 = meta["param_1_name"] == param
    m2 = meta["param_2_name"] == param
    v[m1] = meta.loc[m1, "param_1_value"].astype(float)
    v[m2] = meta.loc[m2, "param_2_value"].astype(float)
    return v


def param_grid_table(meta, y, preds: dict, scenario: str) -> pd.DataFrame:
    """F1 and recall per (p1, p2) cell for a scenario. Long format."""
    p1n, p2n = SCENARIO_PARAMS[scenario]
    mask = (meta["scenario"] == scenario).to_numpy()
    sub = meta[mask].reset_index(drop=True)
    p1 = _param_value(sub, p1n).to_numpy()
    p2 = _param_value(sub, p2n).to_numpy()
    ys = y[mask]
    rows = []
    for v1 in np.unique(p1):
        for v2 in np.unique(p2):
            cell = (p1 == v1) & (p2 == v2)
            if cell.sum() == 0:
                continue
            row = {p1n: v1, p2n: v2, "n": int(cell.sum())}
            for name, p in preds.items():
                pc = p[mask][cell]
                yc = ys[cell]
                row[f"{name}_f1"] = f1_score(yc, pc, zero_division=0) if len(np.unique(yc)) > 1 else float((yc == pc).mean())
                row[f"{name}_recall"] = recall_score(yc, pc, zero_division=0) if yc.sum() else np.nan
                row[f"{name}_fpr"] = float((pc[yc == 0] == 1).mean()) if (yc == 0).sum() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def param_curve_table(meta, y, preds: dict, scenario: str, along: str) -> pd.DataFrame:
    """Recall / F1 / FPR as a function of one parameter, pooling over the other."""
    mask = (meta["scenario"] == scenario).to_numpy()
    sub = meta[mask].reset_index(drop=True)
    v = _param_value(sub, along).to_numpy()
    ys = y[mask]
    rows = []
    for val in np.unique(v):
        cell = v == val
        row = {along: val, "n": int(cell.sum())}
        for name, p in preds.items():
            pc, yc = p[mask][cell], ys[cell]
            row[f"{name}_f1"] = f1_score(yc, pc, zero_division=0)
            row[f"{name}_recall"] = recall_score(yc, pc, zero_division=0)
            row[f"{name}_fpr"] = float((pc[yc == 0] == 1).mean()) if (yc == 0).sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mana_method_table(meta, y) -> pd.DataFrame:
    rows = []
    fp_total = int(((meta["mana_pred"] == 1) & (y == 0)).sum())
    for s in SCENARIOS + ["all"]:
        mask = np.ones(len(meta), bool) if s == "all" else (meta["scenario"] == s).to_numpy()
        pos, neg = mask & (y == 1), mask & (y == 0)
        for m in MANA_METHODS:
            trig = meta[m].to_numpy() == 1
            rows.append({"scenario": s, "method": m,
                         "tpr": float(trig[pos].mean()) if pos.sum() else np.nan,
                         "fpr": float(trig[neg].mean()) if neg.sum() else np.nan,
                         "share_of_mana_fp": float((trig & neg & (meta["mana_pred"].to_numpy() == 1)).sum() / fp_total) if s == "all" and fp_total else np.nan})
    return pd.DataFrame(rows)


def mcnemar_table(meta, y, preds: dict, baseline: str = "MANA", n_comparisons: int | None = None) -> pd.DataFrame:
    rows = []
    for subset in ["overall"] + SCENARIOS:
        mask = np.ones(len(y), bool) if subset == "overall" else (meta["scenario"] == subset).to_numpy()
        base = preds[baseline][mask] == y[mask]
        for name, p in preds.items():
            if name == baseline:
                continue
            a = p[mask] == y[mask]
            b = base
            n01 = int((a & ~b).sum())   # ML right, baseline wrong
            n10 = int((~a & b).sum())   # ML wrong, baseline right
            table = [[int((a & b).sum()), n01], [n10, int((~a & ~b).sum())]]
            res = mcnemar(table, exact=(n01 + n10) < 25)
            rows.append({"classifier": name, "vs": baseline, "subset": subset,
                         "ml_right_base_wrong": n01, "ml_wrong_base_right": n10,
                         "p_value": float(res.pvalue)})
    df = pd.DataFrame(rows)
    k = n_comparisons or len(df)
    df["p_bonferroni"] = np.minimum(df["p_value"] * k, 1.0)
    return df


def prevalence_table(y, preds: dict, prevalences=(0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5), alpha=0.05) -> pd.DataFrame:
    """Precision (PPV) as a function of true spoofing prevalence.

    For every classifier the FPR is replaced by its one-sided (1 - alpha)
    Clopper-Pearson upper bound, so that models with 0 and 1 false positives
    are compared on the same footing (the bound reduces to the rule of three
    when FP = 0). The point estimate is reported alongside.
    """
    from scipy.stats import beta
    rows = []
    for name, p in preds.items():
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn)
        n_neg = fp + tn
        fpr_point = fp / n_neg
        fpr_upper = beta.ppf(1 - alpha, fp + 1, n_neg - fp)
        for prev in prevalences:
            ppv_bound = (tpr * prev) / (tpr * prev + fpr_upper * (1 - prev))
            ppv_point = (tpr * prev) / (tpr * prev + fpr_point * (1 - prev)) if (tpr * prev + fpr_point * (1 - prev)) > 0 else np.nan
            rows.append({"classifier": name, "prevalence": prev, "tpr": tpr, "fp": int(fp), "n_neg": int(n_neg),
                         "fpr_point": fpr_point, "fpr_upper95": fpr_upper,
                         "ppv": ppv_bound, "ppv_point": ppv_point,
                         "false_alarms_per_1000_alerts": 1000 * (1 - ppv_bound)})
    return pd.DataFrame(rows)


def disagreement(y, a, b) -> dict:
    return {"a_right_b_wrong": int(((a == y) & (b != y)).sum()),
            "b_right_a_wrong": int(((b == y) & (a != y)).sum()),
            "both_wrong": int(((a != y) & (b != y)).sum())}
