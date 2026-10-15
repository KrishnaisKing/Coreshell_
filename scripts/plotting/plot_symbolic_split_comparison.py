"""
Figure for scripts/validation/symbolic_split_comparison.py, kept separate because that
script takes ~40 min (NN-dominated) and the figure shouldn't need a rerun to restyle.
Reads csv/symbolic_split_comparison_summary.csv -> plots/symbolic_split_comparison.png.

R^2 axes are clipped at R2_FLOOR: a few range-extrapolation fits reach R^2 ~ -1e6 and
would flatten every other bar. Clipped bars carry their true value as a label.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY = "csv/symbolic_split_comparison_summary.csv"
R2_FLOOR = -1.0

summary = pd.read_csv(SUMMARY)  # raises FileNotFoundError if the comparison hasn't run
targets = list(dict.fromkeys(summary.target))
strategies = list(dict.fromkeys(summary.strategy))
models = list(dict.fromkeys(summary.model))

# Validated categorical palette (dataviz skill, light mode): slot 1 blue, 2 orange, 3 aqua.
COLORS = {"symbolic (ours)": "#2a78d6", "XGBoost": "#eb6834", "NN": "#1baf7a"}
INK, MUTED, GRID = "#21201c", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
LABELS = {
    "random (leaky)": "random\n(leaky)",
    "grouped-by-pair": "grouped\nby pair",
    "leave-materials-out": "leave-\nmaterials-out",
    "Nt range (top 25%)": "Nt\nrange\n(top 25%)",
    "barrier range (top 25%)": "barrier\nrange\n(top 25%)",
    "thickness range (top 25%)": "thickness\nrange\n(top 25%)",
}
N_SEEDED = int((summary.groupby("strategy", sort=False).n_seeds.max() > 1).sum())


def fmt(v):
    return f"{v:.1e}" if abs(v) >= 1000 else f"{v:.1f}"


fig, axes = plt.subplots(2, len(targets), figsize=(18, 9), dpi=150, sharex=True)
width = 0.26
x = np.arange(len(strategies))
for col, target in enumerate(targets):
    for row, metric in enumerate(["r2", "mae"]):
        ax = axes[row, col]
        for k, model_name in enumerate(models):
            s = summary[(summary.target == target) & (summary.model == model_name)].set_index("strategy")
            means = s.loc[strategies, f"{metric}_mean"].to_numpy()
            stds = s.loc[strategies, f"{metric}_std"].fillna(0).to_numpy()
            xs = x + (k - 1) * width
            if metric == "r2":
                shown = np.maximum(means, R2_FLOOR)
                err_lo = np.minimum(stds, shown - R2_FLOOR)
                ax.bar(xs, shown, width - 0.03, color=COLORS[model_name], label=model_name,
                       yerr=[err_lo, np.minimum(stds, 1 - shown)],
                       error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
                for xi, m in zip(xs, means):
                    if m < R2_FLOOR:
                        ax.text(xi, R2_FLOOR + 0.03, fmt(m), rotation=90, ha="center", va="bottom",
                                fontsize=7, color=INK)
            else:
                ax.bar(xs, means, width - 0.03, color=COLORS[model_name], label=model_name, yerr=stds,
                       error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        if metric == "r2":
            ax.set_ylim(R2_FLOOR, 1.0)
            ax.set_title(target, fontsize=12, color=INK)
            ax.set_ylabel(f"test R² (clipped at {R2_FLOOR:g})")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("test MAE (log scale)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(N_SEEDED - 0.5, color=MUTED, linewidth=0.8, linestyle=":")
        if row == 1:
            ax.set_xticks(x, [LABELS.get(s, s) for s in strategies], fontsize=8)
axes[0, 0].legend(frameon=False, loc="lower left")
fig.suptitle("Symbolic regression vs XGBoost vs NN across split strategies\n"
             "left of dotted line: mean ± std over seeds 0–9 · right: deterministic range-extrapolation splits",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig("plots/symbolic_split_comparison.png", dpi=150, bbox_inches="tight")
print("Wrote plots/symbolic_split_comparison.png")
