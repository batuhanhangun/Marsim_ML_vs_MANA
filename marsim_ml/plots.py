"""Regenerate the paper's data figures from results/*.csv.

    python -m marsim_ml.plots            # writes figures/*.pdf and *.png

Figures are drawn at their printed size (5.4 in for a full-width figure), so
the font sizes below are the sizes that appear in the paper. Figures 1 and 9
(pipeline and hybrid architecture) are drawings and are not generated here.
Figures whose results directory does not exist yet are skipped with a message.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .settings import SETTINGS  # noqa: E402

FULL = 5.4  # full text width of the manuscript, inches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "FreeSans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.title_fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
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


def _line(ax, x, y, m, **kw):
    ax.plot(x, y, marker=MARK[m], color=COLOR[m], label=LABEL[m], linestyle="--" if m == "MANA" else "-", **kw)


# ------------------------------------------------- Figure 2: detection cliff
def fig_detection_cliff(run="baseline"):
    p = R / run / "a3_speed_curve.csv"
    if not _have(p):
        return
    c = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharex=True)
    for ax, metric, ylab in zip(axes, ["recall", "f1"], ["Recall (A3)", "F1 score (A3)"]):
        for m in ALL:
            if f"{m}_{metric}" in c:
                _line(ax, c.shift_speed, c[f"{m}_{metric}"], m)
        ax.set_xlabel("Shift speed (simulation units)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0.75, 1.02)
        ax.axvspan(-2, 14, color="grey", alpha=0.12, lw=0)
    axes[0].legend(loc="lower right", frameon=True)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig2_detection_cliff")


# ------------------------------------------------- Figure 3: MANA methods
def fig_mana_methods(run="baseline"):
    p = R / run / "mana_methods.csv"
    if not _have(p):
        return
    m = pd.read_csv(p)
    m = m[m.scenario != "all"]
    methods = ["pdm", "pcc_sog", "pcc_rot", "pcc_height", "cdm", "cnm"]
    names = ["PDM", "PCC_sog", "PCC_rot", "PCC_height", "CDM", "CNM"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    w = 0.26
    for ax, metric, ylab in zip(axes, ["tpr", "fpr"], ["True positive rate", "False positive rate"]):
        for i, s in enumerate(["A1", "A2", "A3"]):
            vals = [m[(m.scenario == s) & (m.method == k)][metric].iloc[0] for k in methods]
            ax.bar(np.arange(len(methods)) + (i - 1) * w, vals, w, label=s)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(names, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("(a) Spoofed recordings")
    axes[1].set_title("(b) Genuine recordings")
    axes[1].legend(title="Scenario", loc="upper right")  # right half of (b) is empty
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig3_mana_methods")


# ------------------------------------------------- Figure 4: A3 heatmaps
def fig_heatmap(run="baseline", clf="XGB"):
    p = R / run / "a3_grid.csv"
    if not _have(p):
        return
    g = pd.read_csv(p)
    piv = lambda col: g.pivot(index="shift_angle", columns="shift_speed", values=col)
    a, b = piv(f"{clf}_f1"), piv("MANA_f1")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0))
    for ax, mat, title, cmap, vmin, vmax in [
            (axes[0], a, f"(a) {LABEL[clf]} F1", "viridis", 0.5, 1.0),
            (axes[1], b, "(b) MANA F1", "viridis", 0.5, 1.0),
            (axes[2], a - b, f"(c) {LABEL[clf]} minus MANA", "RdBu", -0.5, 0.5)]:
        im = ax.imshow(mat.values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(0, len(mat.columns), 6))
        ax.set_xticklabels([f"{v:g}" for v in mat.columns[::6]])
        ax.set_yticks(range(0, len(mat.index), 6))
        ax.set_yticklabels([f"{v:g}" for v in mat.index[::6]])
        ax.set_xlabel("Shift speed")
        if ax is axes[0]:
            ax.set_ylabel("Shift angle (deg)")
        ax.set_title(title)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.ax.tick_params(labelsize=7, width=0.5)
        cb.outline.set_linewidth(0.5)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig4_heatmap")


# ------------------------------------------------- Figure 5: feature importance
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
    fig, ax = plt.subplots(figsize=(0.8 * FULL, 3.6))
    ax.barh(fi.feature, fi.importance, color=[CAT_COLOR[c] for c in cats], height=0.7)
    ax.set_xlabel("Gini importance")
    ax.set_ylim(-0.6, len(fi) - 0.4)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in CAT_COLOR.values()]
    ax.legend(handles, CAT_COLOR.keys(), loc="lower right")
    _save(fig, "fig5_feature_importance")


# ------------------------------------------------- Figure 6: generalization
def fig_generalization():
    curve = R / "heldout_speed_low" / "a3_speed_curve.csv"
    if not _have(curve):
        return
    c = pd.read_csv(curve)
    configs = [("heldout_speed_low", "A3 speed\n0-12"), ("heldout_speed_mid", "A3 speed\n28-40"),
               ("heldout_speed_interleaved", "A3 speed\nalternate"), ("heldout_angle_interleaved", "A3 angle\nalternate"),
               ("heldout_distance_A2", "A2 dist.\nalternate"), ("heldout_scenario_A1", "A2+A3\nto A1"),
               ("heldout_scenario_A3", "A1+A2\nto A3")]
    fig = plt.figure(figsize=(FULL, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    for m in ALL:
        if f"{m}_recall" in c:
            _line(ax, c.shift_speed, c[f"{m}_recall"], m)
    ax.set_xticks(sorted(c.shift_speed.unique()))
    ax.set_xlabel("Held-out shift speed")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Slower drift than in training")
    handles, names = ax.get_legend_handles_labels()
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")
    lax.legend(handles, names, loc="center", frameon=False)
    ax = fig.add_subplot(gs[1, :])
    labels, present = [], []
    for name, lab in configs:
        p = R / name / "overall.csv"
        if p.exists():
            present.append(pd.read_csv(p).set_index("classifier")["f1"])
            labels.append(lab)
    w = 0.16
    for i, m in enumerate(ALL):
        ax.bar(np.arange(len(present)) + (i - 2) * w, [d.get(m, np.nan) for d in present], w, color=COLOR[m])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 on held-out set")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) F1 on each held-out split")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig6_generalization")


# ------------------------------------------------- Figure 7: observation window
def fig_window():
    p = R / "window_sensitivity" / "summary.csv"
    if not _have(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for ax, col, ylab in zip(axes, ["f1", "a3_recall_speed_le12"], ["Overall F1", "A3 recall, shift speed ≤ 12"]):
        for m in ML:
            s = d[(d.classifier == m) & (d.window_s < 120)].sort_values("window_s")
            full = d[(d.classifier == m) & (d.window_s >= 120)]
            if len(s):
                ax.plot(s.window_s, s[col], marker=MARK[m], color=COLOR[m], label=LABEL[m])
            if len(full):
                ax.scatter([s.window_s.max() + 10], full[col], marker=MARK[m], facecolors="none",
                           edgecolors=COLOR[m], s=22, linewidths=1.0, zorder=5)
        mana = d[d.classifier == "MANA"]
        if len(mana):
            ax.axhline(mana[col].iloc[-1], color=COLOR["MANA"], ls="--", lw=1.0, label="MANA (full recording)")
        ax.set_xlabel("Seconds after attack onset")
        ax.set_ylabel(ylab)
        ticks = sorted(d[d.window_s < 120].window_s.unique())
        ax.set_xticks(list(ticks) + [max(ticks) + 10])
        ax.set_xticklabels([f"{t:g}" for t in ticks] + ["full"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    _save(fig, "fig7_window_sensitivity")


# ------------------------------------------------- Figure 8: prevalence
def fig_prevalence(run="baseline"):
    p = R / run / "prevalence.csv"
    if not _have(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(0.7 * FULL, 2.5))
    # pairs with identical recall and false-positive bound share one curve (as in the table)
    for m, lab in [("XGB", "XGBoost / LightGBM"), ("RF", "Random Forest / SVM"), ("MANA", "MANA")]:
        s = d[d.classifier == m].sort_values("prevalence")
        _line(ax, s.prevalence, s.ppv, m)
        ax.lines[-1].set_label(lab)
    ax.set_xscale("log")
    ax.set_xlabel("Spoofing prevalence (fraction of 120 s windows)")
    ax.set_ylabel("Precision of alerts (PPV)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    _save(fig, "fig8_prevalence")


def main():
    print(f"[plots] -> {F}")
    for fn in [fig_detection_cliff, fig_mana_methods, fig_heatmap, fig_feature_importance,
               fig_generalization, fig_window, fig_prevalence]:
        try:
            fn()
        except Exception as exc:  # one broken figure must not block the rest
            print(f"  ERROR in {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
