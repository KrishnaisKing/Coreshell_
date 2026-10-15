"""
Figure for scripts/validation/symbolic_uncertainty.py: empirical coverage of each
80% interval method per split strategy (top row; the dashed line is the nominal
0.80) and mean interval width (bottom row, target units).
Reads csv/symbolic_uncertainty_summary.csv -> plots/symbolic_uncertainty.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv("csv/symbolic_uncertainty_summary.csv")  # FileNotFoundError if not run yet
targets = list(dict.fromkeys(summary.target))
strategies = list(dict.fromkeys(summary.strategy))
methods = list(dict.fromkeys(summary.method))

# Model colours match every other symbolic-regression figure (blue = SR, orange = XGBoost; aqua
# stays reserved for the NN). The two SR interval methods are variants of one model, so they share
# blue and differ by texture (solid vs 45-degree hatch) -- no third hue validated next to these two.
COLORS = {"SR conformal": "#2a78d6", "SR bootstrap": "#2a78d6", "XGB quantile": "#eb6834"}
HATCH = {"SR conformal": None, "SR bootstrap": "////", "XGB quantile": None}
plt.rcParams["hatch.color"] = "#fcfcfb"
plt.rcParams["hatch.linewidth"] = 1.2
INK, MUTED, GRID = "#21201c", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
LABELS = {"grouped-by-pair": "grouped\nby pair", "leave-materials-out": "leave-\nmaterials-out",
          "Nt range (top 25%)": "Nt\nrange", "barrier range (top 25%)": "barrier\nrange",
          "thickness range (top 25%)": "thickness\nrange"}
n_seeded = int((summary.groupby("strategy", sort=False).n_seeds.max() > 1).sum())

fig, axes = plt.subplots(2, len(targets), figsize=(18, 8.5), dpi=150, sharex=True)
x = np.arange(len(strategies))
w = 0.8 / len(methods)
for col, target in enumerate(targets):
    for row, (metric, label) in enumerate([("coverage", "empirical coverage of 80% interval"),
                                           ("width", "mean interval width (target units)")]):
        ax = axes[row, col]
        for k, m in enumerate(methods):
            s = summary[(summary.target == target) & (summary.method == m)].set_index("strategy").loc[strategies]
            vals = s[f"{metric}_mean"].to_numpy()
            err = s["coverage_std"].fillna(0).to_numpy() if metric == "coverage" else None
            ax.bar(x + (k - (len(methods) - 1) / 2) * w, vals, w - 0.03, color=COLORS[m], hatch=HATCH[m],
                   label=m, yerr=err, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        if metric == "coverage":
            ax.axhline(0.8, color=INK, linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_title(target, fontsize=12, color=INK)
        ax.set_ylabel(label)
        ax.axvline(n_seeded - 0.5, color=MUTED, linewidth=0.8, linestyle=":")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == 1:
            ax.set_xticks(x, [LABELS.get(s, s) for s in strategies], fontsize=8)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle("80% prediction intervals: symbolic regression (conformal, bootstrap) vs XGBoost quantile regression\n"
             "dashed = nominal 0.80 · left of dotted line: mean ± std over seeds 0–9 · right: single range splits",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig("plots/symbolic_uncertainty.png", dpi=150, bbox_inches="tight")
print("Wrote plots/symbolic_uncertainty.png")
