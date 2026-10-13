"""Regenerate every figure from results/*.csv.

    python -m marsim_ml.plots            # writes figures/*.pdf and *.png

Figures that depend on a results directory that does not exist yet are skipped
with a message. Fonts are sized for a two-column journal page.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .settings import SETTINGS  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "lines.linewidth": 2.0, "lines.markersize": 6,
})

ML = ["XGB", "LGBM", "RF", "SVM"]
ALL = ML + ["MANA"]
COLOR = {"XGB": "#1b9e77", "LGBM": "#d95f02", "RF": "#7570b3", "SVM": "#e7298a", "MANA": "#222222"}
MARK = {"XGB": "s", "LGBM": "^", "RF": "o", "SVM": "D", "MANA": "X"}
LABEL = {"XGB": "XGBoost", "LGBM": "LightGBM", "RF": "Random Forest", "SVM": "SVM", "MANA": "MANA"}

R = SETTINGS["results_dir"]
F = SETTINGS["figures_dir"]


def _save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def _have(*paths) -> bool:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"  skip: missing {missing[0]}")
        return False
    return True


# ---------------------------------------------------------------- fig 1
def fig_detection_cliff(run="baseline"):
    p = R / run / "a3_speed_curve.csv"
    if not _have(p):
        return
    c = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, ylab in zip(axes, ["recall", "f1"], ["Recall (A3)", "F1 score (A3)"]):
        for m in ALL:
            if f"{m}_{metric}" not in c:
                continue
            ax.plot(c.shift_speed, c[f"{m}_{metric}"], marker=MARK[m], color=COLOR[m], label=LABEL[m],
                    linestyle="--" if m == "MANA" else "-")
        ax.set_xlabel("Shift speed (simulation units)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0.75, 1.02)
        ax.axvspan(-2, 14, color="grey", alpha=0.12, lw=0)
    axes[0].legend(loc="lower right", frameon=True)
    _save(fig, "fig1_detection_cliff")


# ---------------------------------------------------------------- fig 2
def _category(feature: str) -> str:
    from .extract import DERIVED_KEYS, INTER_KEYS, RAW_KEYS
    for k in INTER_KEYS:
        if feature.startswith(k):
            return "Multi-receiver"
    for k in DERIVED_KEYS:
        if feature.startswith(k):
            return "Temporal derived"
    for k in RAW_KEYS:
        if feature.startswith(k):
            return "Per-epoch raw"
    return "Other"


CAT_COLOR = {"Multi-receiver": "#d95f02", "Per-epoch raw": "#1b9e77", "Temporal derived": "#7570b3", "Other": "#999999"}


def fig_feature_importance(run="baseline", clf="RF", top=20):
    p = R / run / "feature_importance.csv"
    if not _have(p):
        return
    fi = pd.read_csv(p)
    fi = fi[fi.classifier == clf].nsmallest(top, "rank").iloc[::-1]
    cats = fi.feature.map(_category)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(fi.feature, fi.importance, color=[CAT_COLOR[c] for c in cats])
    ax.set_xlabel("Gini importance")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in CAT_COLOR.values()]
    ax.legend(handles, CAT_COLOR.keys(), loc="lower right")
    _save(fig, "fig2_feature_importance")


# ---------------------------------------------------------------- fig 3
def fig_heatmap(run="baseline", clf="XGB"):
    p = R / run / "a3_grid.csv"
    if not _have(p):
        return
    g = pd.read_csv(p)
    piv = lambda col: g.pivot(index="shift_angle", columns="shift_speed", values=col)
    a, b = piv(f"{clf}_f1"), piv("MANA_f1")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"wspace": 0.35})
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], a, f"(a) {LABEL[clf]} F1", "viridis", 0.5, 1.0),
        (axes[1], b, "(b) MANA F1", "viridis", 0.5, 1.0),
        (axes[2], a - b, f"(c) {LABEL[clf]} minus MANA", "RdBu", -0.5, 0.5)]:
        im = ax.imshow(mat.values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(0, len(mat.columns), 3))
        ax.set_xticklabels([f"{v:g}" for v in mat.columns[::3]])
        ax.set_yticks(range(0, len(mat.index), 3))
        ax.set_yticklabels([f"{v:g}" for v in mat.index[::3]])
        ax.set_xlabel("Shift speed")
        if ax is axes[0]:
            ax.set_ylabel("Shift angle (deg)")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig3_heatmap_3panel")


# ---------------------------------------------------------------- fig 4
def fig_mana_methods(run="baseline"):
    p = R / run / "mana_methods.csv"
    if not _have(p):
        return
    m = pd.read_csv(p)
    m = m[m.scenario != "all"]
    methods = ["pdm", "pcc_sog", "pcc_rot", "pcc_height", "cdm", "cnm"]
    names = ["PDM", "PCC_sog", "PCC_rot", "PCC_height", "CDM", "CNM"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.26
    for ax, metric, ylab in zip(axes, ["tpr", "fpr"], ["True positive rate", "False positive rate"]):
        for i, s in enumerate(["A1", "A2", "A3"]):
            vals = [m[(m.scenario == s) & (m.method == k)][metric].iloc[0] for k in methods]
            ax.bar(np.arange(len(methods)) + (i - 1) * w, vals, w, label=s)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("(a) Sensitivity of each MANA method")
    axes[1].set_title("(b) False alarms of each MANA method")
    axes[0].legend(title="Scenario")
    _save(fig, "fig4_mana_methods")


# ------------------------------------------------------------ schematics
def _box(ax, xy, w, h, text, fc="#f2f2f2", fs=11):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec="#333", lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    steps = [
        (0.2, "Raw NMEA 0183\n(pcap, 2 receivers,\n120 s)", "#dde8f5"),
        (2.9, "Parse 5 sentence types\nRMC GGA GSA GSV VTG\n1 s epochs", "#dde8f5"),
        (5.6, "138 features\nraw (18x5) + temporal (6x5)\n+ multi-receiver (3x5) + 3", "#e6f2e0"),
        (8.3, "Preprocessing\nzero-var, log1p, winsor,\n|r|>0.95  ->  56 features", "#e6f2e0"),
        (11.0, "Classifiers\nRF, SVM,\nXGBoost, LightGBM", "#fbe6d0"),
    ]
    for x, t, c in steps:
        _box(ax, (x, 0.5), 2.0, 2.2, t, fc=c, fs=10.5)
    for i in range(len(steps) - 1):
        _arrow(ax, (steps[i][0] + 2.0, 1.6), (steps[i + 1][0], 1.6))
    _save(fig, "fig5_pipeline")


def fig_hybrid():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    _box(ax, (0.2, 1.7), 2.2, 1.2, "Preprocessed\nNMEA stream", fc="#dde8f5")
    _box(ax, (3.4, 2.9), 2.6, 1.2, "ML detection engine\n(XGBoost / LightGBM)", fc="#fbe6d0")
    _box(ax, (3.4, 0.5), 2.6, 1.2, "MANA rule engine\n(PDM, PCC, CDM, CNM)", fc="#e6f2e0")
    _box(ax, (7.0, 1.7), 2.4, 1.2, "Hybrid decision layer", fc="#f2f2f2")
    _box(ax, (9.9, 2.9), 1.0, 1.0, "Alert +\nscore", fc="#f9d5d3", fs=10)
    _box(ax, (9.9, 0.7), 1.0, 1.0, "Rule\naudit", fc="#f9d5d3", fs=10)
    _arrow(ax, (2.4, 2.4), (3.4, 3.4))
    _arrow(ax, (2.4, 2.2), (3.4, 1.2))
    _arrow(ax, (6.0, 3.5), (7.0, 2.5))
    _arrow(ax, (6.0, 1.1), (7.0, 2.1))
    _arrow(ax, (9.4, 2.5), (9.9, 3.3))
    _arrow(ax, (9.4, 2.1), (9.9, 1.3))
    ax.text(4.7, 4.35, "spoofing probability", ha="center", fontsize=10, style="italic")
    ax.text(4.7, 0.2, "triggered rules (interpretable)", ha="center", fontsize=10, style="italic")
    _save(fig, "fig6_hybrid_architecture")


# ------------------------------------------------------- new: fig 7, 8, 9
def fig_generalization():
    curve = R / "heldout_speed_low" / "a3_speed_curve.csv"
    if not _have(curve):
        return
    c = pd.read_csv(curve)
    configs = [("heldout_speed_low", "A3 speed\n0-12"), ("heldout_speed_mid", "A3 speed\n28-40"),
               ("heldout_speed_interleaved", "A3 speed\nalternate"), ("heldout_angle_interleaved", "A3 angle\nalternate"),
               ("heldout_distance_A2", "A2 dist.\nalternate"), ("heldout_scenario_A1", "A2+A3\nto A1"),
               ("heldout_scenario_A3", "A1+A2\nto A3")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]
    for m in ALL:
        if f"{m}_recall" not in c:
            continue
        ax.plot(c.shift_speed, c[f"{m}_recall"], marker=MARK[m], color=COLOR[m], label=LABEL[m],
                linestyle="--" if m == "MANA" else "-")
    ax.set_xlabel("Held-out shift speed (unseen in training)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Extrapolation to slower drift")
    ax.legend(loc="lower right")
    ax = axes[1]
    labels, present = [], []
    for name, lab in configs:
        p = R / name / "overall.csv"
        if p.exists():
            present.append(pd.read_csv(p).set_index("classifier")["f1"])
            labels.append(lab)
    w = 0.16
    for i, m in enumerate(ALL):
        ax.bar(np.arange(len(present)) + (i - 2) * w, [d.get(m, np.nan) for d in present], w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("F1 on held-out set")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) F1 under held-out splits (held-out values / scenario)")
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig7_generalization")


def fig_prevalence(run="baseline"):
    p = R / run / "prevalence.csv"
    if not _have(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m in ALL:
        s = d[d.classifier == m].sort_values("prevalence")
        ax.plot(s.prevalence, s.ppv, marker=MARK[m], color=COLOR[m], label=LABEL[m], linestyle="--" if m == "MANA" else "-")
    ax.set_xscale("log")
    ax.set_xlabel("True spoofing prevalence (fraction of 120 s windows)")
    ax.set_ylabel("Precision of alerts (PPV, conservative)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.text(0.02, 0.97, "All curves use the 95% Clopper-Pearson upper bound on FPR",
            transform=ax.transAxes, fontsize=9, va="top", style="italic")
    _save(fig, "fig8_prevalence")


def fig_window():
    p = R / "window_sensitivity" / "summary.csv"
    if not _have(p):
        return
    d = pd.read_csv(p)
    start = float(d["window_start"].max()) if "window_start" in d else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, ylab in zip(axes, ["f1", "a3_recall_speed_le12"], ["Overall F1", "A3 recall, shift speed <= 12"]):
        for m in ML:
            s = d[(d.classifier == m) & (d.window_s < 120)].sort_values("window_s")
            full = d[(d.classifier == m) & (d.window_s >= 120)]
            if len(s):
                ax.plot(s.window_s, s[col], marker=MARK[m], color=COLOR[m], label=LABEL[m])
            if len(full):
                ax.scatter([s.window_s.max() + 10], full[col], marker=MARK[m], facecolors="none",
                           edgecolors=COLOR[m], s=70, linewidths=1.8, zorder=5)
        mana = d[d.classifier == "MANA"]
        if len(mana):
            ax.axhline(mana[col].iloc[-1], color=COLOR["MANA"], ls="--", label="MANA (full recording)")
        ax.set_xlabel(f"Window length after attack onset (s)")
        ax.set_ylabel(ylab)
        ticks = sorted(d[d.window_s < 120].window_s.unique())
        ax.set_xticks(list(ticks) + [max(ticks) + 10])
        ax.set_xticklabels([f"{t:g}" for t in ticks] + ["full"])
    axes[0].legend(loc="center right")
    axes[0].text(0.02, 0.55, "Hollow markers: full 120 s recording\n(60 s clean + 60 s attack)",
                 transform=axes[0].transAxes, fontsize=9, style="italic", va="center")
    _save(fig, "fig9_window_sensitivity")


def fig_repetition_cv():
    p = R / "repetition_cv" / "folds.csv"
    if not _have(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(ALL):
        s = d[d.classifier == m]
        if not len(s):
            continue
        ax.errorbar(i, s.f1.mean(), yerr=s.f1.std(), fmt=MARK[m], color=COLOR[m], capsize=5, ms=9)
        ax.scatter([i] * len(s), s.f1, color=COLOR[m], alpha=0.35, s=25)
    ax.set_xticks(range(len(ALL)))
    ax.set_xticklabels([LABEL[m] for m in ALL])
    ax.set_ylabel("F1 (5 repetition folds)")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig10_repetition_cv")


def fig_tuned():
    a, b = R / "baseline" / "overall.csv", R / "tuned" / "overall.csv"
    if not _have(a, b):
        return
    da, db = pd.read_csv(a).set_index("classifier"), pd.read_csv(b).set_index("classifier")
    ML_ = [m for m in ML if m in da.index and m in db.index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ML_))
    ax.bar(x - 0.2, [da.loc[m, "f1"] for m in ML_], 0.4, label="Default hyperparameters", color="#9ecae1")
    ax.bar(x + 0.2, [db.loc[m, "f1"] for m in ML_], 0.4, label="Randomized search (3-fold CV)", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in ML_])
    ax.set_ylabel("Test F1")
    lo = min(min(da.loc[ML_, "f1"]), min(db.loc[ML_, "f1"]))
    ax.set_ylim(lo - 0.01, 1.001)
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig11_tuned_vs_default")


def main():
    print(f"[plots] -> {F}")
    for fn in [fig_detection_cliff, fig_feature_importance, fig_heatmap, fig_mana_methods, fig_pipeline,
               fig_hybrid, fig_generalization, fig_prevalence, fig_window, fig_repetition_cv, fig_tuned]:
        try:
            fn()
        except Exception as exc:  # one broken figure must not block the rest
            print(f"  ERROR in {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
