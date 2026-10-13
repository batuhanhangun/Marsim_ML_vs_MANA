"""Emit LaTeX tables from results/ into tables/*.tex.

    python -m marsim_ml.tables

Each file is a bare tabular body (booktabs) meant to be \\input{} inside a
table environment in the manuscript, so captions and labels stay in the .tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .settings import SETTINGS

R, T = SETTINGS["results_dir"], SETTINGS["tables_dir"]
ORDER = ["XGB", "LGBM", "RF", "SVM", "MANA"]
NAME = {"XGB": "XGBoost", "LGBM": "LightGBM", "RF": "Random Forest", "SVM": "SVM", "MANA": "MANA"}


def _w(name: str, body: str):
    T.mkdir(parents=True, exist_ok=True)
    (T / f"{name}.tex").write_text(body)
    print(f"  wrote {name}.tex")


def _f(x, nd=4):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def tab_overall(run="baseline"):
    p = R / run / "overall.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).set_index("classifier")
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Classifier & F1 & Precision & Recall & FPR & AUC & FN / FP \\", r"\midrule"]
    for m in ORDER:
        if m not in d.index:
            continue
        r = d.loc[m]
        lines.append(f"{NAME[m]} & {_f(r.f1)} & {_f(r.precision)} & {_f(r.recall)} & {_f(r.fpr)} & {_f(r.auc)} & {int(r.fn)} / {int(r.fp)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_overall", "\n".join(lines))


def tab_per_scenario(run="baseline"):
    p = R / run / "per_scenario.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).pivot(index="classifier", columns="scenario", values="f1")
    o = pd.read_csv(R / run / "overall.csv").set_index("classifier")["f1"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Classifier & A1 & A2 & A3 & Overall \\", r"\midrule"]
    for m in ORDER:
        if m in d.index:
            lines.append(f"{NAME[m]} & {_f(d.loc[m,'A1'],3)} & {_f(d.loc[m,'A2'],3)} & {_f(d.loc[m,'A3'],3)} & {_f(o[m],3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_per_scenario", "\n".join(lines))


def tab_generalization():
    configs = [("baseline", "Repetition split (in-distribution)"),
               ("heldout_speed_low", "A3 shift speed 0--12 held out"),
               ("heldout_speed_mid", "A3 shift speed 28--40 held out"),
               ("heldout_speed_interleaved", "A3 shift speed, alternate values held out"),
               ("heldout_angle_interleaved", "A3 shift angle, alternate values held out"),
               ("heldout_distance_A2", "A2 distance, alternate values held out"),
               ("heldout_scenario_A1", "Train A2+A3, test A1"),
               ("heldout_scenario_A2", "Train A1+A3, test A2"),
               ("heldout_scenario_A3", "Train A1+A2, test A3")]
    rows = []
    for name, label in configs:
        p = R / name / "overall.csv"
        if p.exists():
            d = pd.read_csv(p).set_index("classifier")["f1"]
            rows.append((label, d))
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Split & XGBoost & LightGBM & RF & SVM & MANA \\", r"\midrule"]
    for label, d in rows:
        lines.append(label + " & " + " & ".join(_f(d.get(m), 3) for m in ORDER) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_generalization", "\n".join(lines))


def tab_repetition_cv():
    p = R / "repetition_cv" / "summary.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).set_index("classifier")
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Classifier & F1 (mean $\pm$ sd) & Recall (mean $\pm$ sd) & FPR (mean $\pm$ sd) \\", r"\midrule"]
    for m in ORDER:
        if m in d.index:
            r = d.loc[m]
            lines.append(f"{NAME[m]} & {r.f1_mean:.4f} $\\pm$ {r.f1_std:.4f} & {r.recall_mean:.4f} $\\pm$ {r.recall_std:.4f} & {r.fpr_mean:.4f} $\\pm$ {r.fpr_std:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_repetition_cv", "\n".join(lines))


def tab_tuned():
    a, b = R / "baseline" / "overall.csv", R / "tuned" / "overall.csv"
    if not (a.exists() and b.exists()):
        return
    da, db = pd.read_csv(a).set_index("classifier"), pd.read_csv(b).set_index("classifier")
    best = json.loads((R / "tuned" / "config.json").read_text()).get("tuned_best_params", {})
    lines = [r"\begin{tabular}{@{}lccp{6cm}@{}}", r"\toprule",
             r"Classifier & F1 (default) & F1 (tuned) & Selected hyperparameters \\", r"\midrule"]
    for m in [m for m in ["XGB", "LGBM", "RF", "SVM"] if m in da.index and m in db.index]:
        hp = ", ".join(f"{k}={v}" for k, v in best.get(m, {}).items()).replace("_", r"\_")
        lines.append(f"{NAME[m]} & {_f(da.loc[m,'f1'])} & {_f(db.loc[m,'f1'])} & {hp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_tuned", "\n".join(lines))


def tab_prevalence(run="baseline"):
    p = R / run / "prevalence.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    prevs = sorted(d.prevalence.unique(), reverse=True)
    lines = [r"\begin{tabular}{@{}l" + "c" * len(prevs) + "@{}}", r"\toprule",
             "Classifier & " + " & ".join(f"{p:g}" for p in prevs) + r" \\", r"\midrule"]
    for m in ORDER:
        s = d[d.classifier == m].set_index("prevalence")["ppv"]
        if len(s):
            lines.append(f"{NAME[m]} & " + " & ".join(_f(s.get(p), 3) for p in prevs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_prevalence", "\n".join(lines))


def tab_window():
    p = R / "window_sensitivity" / "summary.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    ws = sorted(d.window_s.unique())
    lines = [r"\begin{tabular}{@{}l" + "c" * len(ws) + "@{}}", r"\toprule",
             "Classifier & " + " & ".join(f"{w:g}\\,s" for w in ws) + r" \\", r"\midrule"]
    for m in ["XGB", "LGBM", "RF", "SVM"]:
        s = d[d.classifier == m].set_index("window_s")["f1"]
        lines.append(f"{NAME[m]} & " + " & ".join(_f(s.get(w), 3) for w in ws) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_window", "\n".join(lines))


def tab_mcnemar(run="baseline"):
    p = R / run / "mcnemar.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.subset.isin(["overall", "A3"])]
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Classifier & Subset & ML right / MANA wrong & ML wrong / MANA right & $p$ (Bonferroni) \\", r"\midrule"]
    for _, r in d.iterrows():
        lines.append(f"{NAME[r.classifier]} & {r.subset} & {int(r.ml_right_base_wrong)} & {int(r.ml_wrong_base_right)} & ${r.p_bonferroni:.1e}$ \\\\".replace("e-", r"\times 10^{-").replace("$ \\\\", "}$ \\\\") if r.p_bonferroni < 1e-3 else
                     f"{NAME[r.classifier]} & {r.subset} & {int(r.ml_right_base_wrong)} & {int(r.ml_wrong_base_right)} & {r.p_bonferroni:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_mcnemar", "\n".join(lines))


def main():
    print(f"[tables] -> {T}")
    for fn in [tab_overall, tab_per_scenario, tab_generalization, tab_repetition_cv, tab_tuned,
               tab_prevalence, tab_window, tab_mcnemar]:
        fn()


if __name__ == "__main__":
    main()
