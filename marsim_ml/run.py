"""Experiment runner.

    python -m marsim_ml.run --config baseline
    python -m marsim_ml.run --config all              # every config in CONFIGS
    python -m marsim_ml.run --config tuned --n-iter 30

Outputs go to results/<config>/ (see README for the file list).
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from . import evaluate as ev
from .data import SCENARIOS, Dataset, load
from .models import DEFAULT_MODELS, SEARCH_SPACES, ModelSpec, build
from .preprocess import PreprocessConfig, Preprocessor
from .settings import SETTINGS
from .splits import (_param_value, describe_split, heldout_param_masks,
                     heldout_scenario_split, repetition_split)

PP_CODE = dict(winsorize=True, corr_method="greedy")        # default preprocessing
PP_TEXT = dict(winsorize=False, corr_method="iterative")    # sensitivity check

CONFIGS = {
    "baseline": dict(split=dict(kind="repetition"), preprocess=PP_CODE, tune=False),
    "alt_preprocessing": dict(split=dict(kind="repetition"), preprocess=PP_TEXT, tune=False),
    # --- generalization to unseen attack configurations ----------------------
    "heldout_speed_low": dict(split=dict(kind="heldout_param", scenario="A3", param="shift_speed",
                                         test_values=[0, 4, 8, 12]), preprocess=PP_CODE, tune=False),
    "heldout_speed_mid": dict(split=dict(kind="heldout_param", scenario="A3", param="shift_speed",
                                         test_values=[28, 32, 36, 40]), preprocess=PP_CODE, tune=False),
    "heldout_speed_interleaved": dict(split=dict(kind="heldout_param", scenario="A3", param="shift_speed",
                                                 test_values="every_other"), preprocess=PP_CODE, tune=False),
    "heldout_angle_interleaved": dict(split=dict(kind="heldout_param", scenario="A3", param="shift_angle",
                                                 test_values="every_other"), preprocess=PP_CODE, tune=False),
    "heldout_distance_A2": dict(split=dict(kind="heldout_param", scenario="A2", param="distance_to_ship",
                                           test_values="every_other"), preprocess=PP_CODE, tune=False),
    "heldout_scenario_A1": dict(split=dict(kind="heldout_scenario", test_scenario="A1"), preprocess=PP_CODE, tune=False),
    "heldout_scenario_A2": dict(split=dict(kind="heldout_scenario", test_scenario="A2"), preprocess=PP_CODE, tune=False),
    "heldout_scenario_A3": dict(split=dict(kind="heldout_scenario", test_scenario="A3"), preprocess=PP_CODE, tune=False),
    # --- variance across repetition folds (5 folds x 4 repetition indices) ---
    "repetition_cv": dict(split=dict(kind="repetition_cv", n_folds=5), preprocess=PP_CODE, tune=False),
    # --- hyperparameter search ------------------------------------------------
    "tuned": dict(split=dict(kind="repetition"), preprocess=PP_CODE, tune=True),
}


# ------------------------------------------------------------------ splits
def resolve_split(meta: pd.DataFrame, split_cfg: dict):
    kind = split_cfg["kind"]
    keep = np.ones(len(meta), bool)
    if kind == "repetition":
        is_test = repetition_split(meta, split_cfg.get("test_indices", range(16, 20)))
    elif kind == "heldout_param":
        tv = split_cfg["test_values"]
        if tv == "every_other":
            vals = np.unique(_param_value(meta[meta.scenario == split_cfg["scenario"]], split_cfg["param"]).dropna())
            tv = vals[1::2].tolist()
        is_test, keep = heldout_param_masks(meta, split_cfg["scenario"], split_cfg["param"], tv,
                                            split_cfg.get("other_scenarios", "train"))
        split_cfg = {**split_cfg, "test_values": [float(v) for v in tv]}
    elif kind == "heldout_scenario":
        is_test = heldout_scenario_split(meta, split_cfg["test_scenario"])
    else:
        raise ValueError(kind)
    return is_test, keep, split_cfg


# ------------------------------------------------------------------- fit
def fit_predict(spec: ModelSpec, Xtr, ytr, Xte, tune: bool, n_iter: int, seed: int):
    model = build(spec)
    if spec.subsample_train and spec.subsample_train < len(Xtr):
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=spec.subsample_train, random_state=seed, stratify=ytr)
    t0 = time.time()
    best = None
    if tune and spec.kind in SEARCH_SPACES:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(model, SEARCH_SPACES[spec.kind], n_iter=n_iter, cv=cv, scoring="f1",
                                    random_state=seed, n_jobs=1, refit=True)
        search.fit(Xtr, ytr)
        model, best = search.best_estimator_, search.best_params_
    else:
        model.fit(Xtr, ytr)
    train_s = time.time() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    ms = (time.perf_counter() - t0) / len(Xte) * 1000
    prob = model.predict_proba(Xte)[:, 1]
    imp = getattr(model, "feature_importances_", None)
    return pred, prob, ms, train_s, imp, best


# ------------------------------------------------------------ single run
def run_config(ds: Dataset, cfg: dict, out: Path, *, seed: int = 42, n_iter: int = 30,
               skip: list[str] = (), verbose: bool = True) -> dict:
    """Fit all models on one split and write every table. Returns overall metrics."""
    cfg = deepcopy(cfg)
    out.mkdir(parents=True, exist_ok=True)
    is_test, keep, cfg["split"] = resolve_split(ds.meta, cfg["split"])
    tr, te = keep & ~is_test, keep & is_test
    split_info = describe_split(ds.meta, ds.y, is_test, keep)

    pp = Preprocessor(PreprocessConfig(**cfg["preprocess"])).fit(ds.X[tr])
    meta_te, y_te = ds.meta[te].reset_index(drop=True), ds.y[te]
    if verbose:
        print("  split:", split_info)
        print("  preprocess:", pp.summary(ds.X.shape[1]))

    preds, probs, times, train_times, importances, tuned = {}, {}, {}, {}, {}, {}
    for spec in [s for s in deepcopy(DEFAULT_MODELS) if s.name not in skip]:
        build(spec)
        Xtr = pp.transform(ds.X[tr], scale=spec.needs_scaling).to_numpy()
        Xte = pp.transform(ds.X[te], scale=spec.needs_scaling).to_numpy()
        p, pr, ms, ts, imp, bp = fit_predict(spec, Xtr, ds.y[tr], Xte, cfg["tune"], n_iter, seed)
        preds[spec.name], probs[spec.name], times[spec.name], train_times[spec.name] = p, pr, ms, ts
        if imp is not None:
            importances[spec.name] = imp
        if bp is not None:
            tuned[spec.name] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in bp.items()}
        if verbose:
            print(f"  {spec.name:<5} F1={f1_score(y_te, p):.4f}  train {ts:.1f}s")
    preds["MANA"] = meta_te["mana_pred"].to_numpy().astype(int)

    overall = ev.overall_table(y_te, preds, probs, times)
    overall.to_csv(out / "overall.csv", index=False)
    ev.per_scenario_table(meta_te, y_te, preds, probs).to_csv(out / "per_scenario.csv", index=False)
    for s in SCENARIOS:
        if (meta_te.scenario == s).any():
            ev.param_grid_table(meta_te, y_te, preds, s).to_csv(out / f"{s.lower()}_grid.csv", index=False)
    if (meta_te.scenario == "A3").any():
        ev.param_curve_table(meta_te, y_te, preds, "A3", "shift_speed").to_csv(out / "a3_speed_curve.csv", index=False)
        ev.param_curve_table(meta_te, y_te, preds, "A3", "shift_angle").to_csv(out / "a3_angle_curve.csv", index=False)
    ev.mana_method_table(meta_te, y_te).to_csv(out / "mana_methods.csv", index=False)
    ev.mcnemar_table(meta_te, y_te, preds).to_csv(out / "mcnemar.csv", index=False)
    ev.prevalence_table(y_te, preds).to_csv(out / "prevalence.csv", index=False)

    fi = [{"classifier": n, "rank": r, "feature": pp.final_features[i], "importance": float(imp[i])}
          for n, imp in importances.items() for r, i in enumerate(np.argsort(imp)[::-1], 1)]
    pd.DataFrame(fi).to_csv(out / "feature_importance.csv", index=False)

    tp = meta_te[["filename", "scenario", "label", "index", "param_1_name", "param_1_value",
                  "param_2_name", "param_2_value"]].copy()
    tp["y"] = y_te
    for n, p in preds.items():
        tp[f"{n}_pred"] = p
        if n in probs:
            tp[f"{n}_prob"] = probs[n]
    tp.to_csv(out / "test_predictions.csv", index=False)

    (out / "features.txt").write_text("\n".join(pp.final_features))
    json.dump({**cfg, "seed": seed, "skipped": list(skip), "train_time_s": train_times, "tuned_best_params": tuned},
              open(out / "config.json", "w"), indent=2, default=str)
    json.dump(split_info, open(out / "split.json", "w"), indent=2, default=str)
    json.dump({**pp.summary(ds.X.shape[1]), "zero_variance_dropped": pp.dropped_zero_var,
               "log_transformed": pp.log_cols, "correlated_dropped": pp.dropped_corr},
              open(out / "preprocessing.json", "w"), indent=2)
    return overall.set_index("classifier").to_dict("index")


# ------------------------------------------------------- repetition folds
def run_repetition_cv(ds: Dataset, cfg: dict, out: Path, **kw) -> pd.DataFrame:
    """5 folds over repetition indices (0-3, 4-7, ...). Fold 4 == the primary split."""
    n_folds = cfg["split"].get("n_folds", 5)
    per_fold = 20 // n_folds
    rows = []
    for k in range(n_folds):
        idx = list(range(k * per_fold, (k + 1) * per_fold))
        fold_cfg = {**cfg, "split": dict(kind="repetition", test_indices=idx)}
        print(f"  fold {k}: test indices {idx}")
        res = run_config(ds, fold_cfg, out / f"fold{k}", verbose=False, **kw)
        for clf, m in res.items():
            rows.append({"fold": k, "classifier": clf, **m})
    df = pd.DataFrame(rows)
    df.to_csv(out / "folds.csv", index=False)
    summ = df.groupby("classifier")[["f1", "precision", "recall", "fpr"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(out / "summary.csv")
    print(summ.round(4))
    return df


# ------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS) + ["all"])
    ap.add_argument("--data", default=str(SETTINGS["data_dir"]))
    ap.add_argument("--out", default=None, help="defaults to results/<config>")
    ap.add_argument("--n-iter", type=int, default=SETTINGS["tune_n_iter"])
    ap.add_argument("--seed", type=int, default=SETTINGS["seed"])
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args(argv)

    ds = load(a.data)
    names = list(CONFIGS) if a.config == "all" else [a.config]
    for name in names:
        out = Path(a.out) if (a.out and a.config != "all") else SETTINGS["results_dir"] / name
        print(f"=== {name} -> {out}")
        cfg = CONFIGS[name]
        kw = dict(seed=a.seed, n_iter=a.n_iter, skip=a.skip)
        if cfg["split"]["kind"] == "repetition_cv":
            run_repetition_cv(ds, cfg, out, **kw)
        else:
            run_config(ds, cfg, out, **kw)


if __name__ == "__main__":
    main()
