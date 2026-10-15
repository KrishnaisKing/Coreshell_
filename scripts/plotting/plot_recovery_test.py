"""
Figure for scripts/symbolic/recovery_test.py.
Reads csv/recovery_test.csv and csv/recovery_extrapolation.csv -> plots/recovery_test.png.

Panels A-B colour the three TARGETS (palette slots 4-6, plus distinct markers since
two of them sit below 3:1 contrast); panel C colours the three MODELS with the same
slots 1-3 as every other symbolic-regression figure, so no colour means two things.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

recov = pd.read_csv("csv/recovery_test.csv")  # FileNotFoundError if recovery_test.py hasn't run
ext = pd.read_csv("csv/recovery_extrapolation.csv")
targets = list(dict.fromkeys(recov.target))
in_library = recov.groupby("target").truth_in_library.first()
strategies = list(dict.fromkeys(ext.strategy))
models = list(dict.fromkeys(ext.model))
n_seeds = recov.seed.nunique()

# Validated categorical palette (dataviz skill, light mode).
MODEL_COLORS = {"symbolic (ours)": "#2a78d6", "XGBoost": "#eb6834", "NN": "#1baf7a"}
TARGET_STYLE = {"log10_retention_tau_s": ("#eda100", "o"), "log10_on_off_ratio": ("#e87ba4", "s"),
                "hysteresis_window_V": ("#008300", "^")}
INK, MUTED, GRID = "#21201c", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
SHORT = {"log10_retention_tau_s": "retention", "log10_on_off_ratio": "on/off", "hysteresis_window_V": "hysteresis"}


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), dpi=150, gridspec_kw={"width_ratios": [1, 1, 1.7]})

ax = axes[0]
for target in [t for t in targets if in_library[t]]:
    c, m = TARGET_STYLE[target]
    g = recov[recov.target == target].groupby("noise_eta").recovered.mean()
    ax.plot(g.index, g.values, marker=m, ms=8, lw=2, color=c, label=SHORT[target])
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("noise σ / std(clean label)")
ax.set_ylabel(f"true-law recovery rate ({n_seeds} seeds)")
ax.set_title("A. Recovering a known in-library law", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)
style(ax)

ax = axes[1]
for target in targets:
    c, m = TARGET_STYLE[target]
    g = recov[recov.target == target].groupby("noise_eta").r2_vs_clean.agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker=m, ms=8, lw=2, capsize=3, color=c,
                label=SHORT[target] + ("" if in_library[target] else " (truth NOT in library)"))
ax.set_ylim(min(0.4, recov.r2_vs_clean.min() - 0.05), 1.02)
ax.set_xlabel("noise σ / std(clean label)")
ax.set_ylabel("test R² vs the noise-free truth")
ax.set_title("B. Accuracy against the clean law", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="center right")
style(ax)

ax = axes[2]
pairs = [(t, s) for t in targets for s in strategies]
xs = np.arange(len(pairs))
w = 0.8 / len(models)
for k, model_name in enumerate(models):
    vals = np.array([ext[(ext.target == t) & (ext.strategy == s) & (ext.model == model_name)].r2_vs_clean.iloc[0]
                     for t, s in pairs])
    pos = xs + (k - (len(models) - 1) / 2) * w
    ax.bar(pos, np.maximum(vals, -1), w - 0.03, color=MODEL_COLORS[model_name], label=model_name)
    for xi, v in zip(pos, vals):
        if v < -1:
            ax.text(xi, -0.97, f"{v:.1f}", rotation=90, fontsize=7, ha="center", va="bottom", color=INK)
for i in range(1, len(targets)):
    ax.axvline(i * len(strategies) - 0.5, color=MUTED, linewidth=0.8, linestyle=":")
ax.set_xticks(xs, [f"{SHORT[t]}\n{s.split(' range')[0]}\nrange" for t, s in pairs], fontsize=8)
ax.set_ylim(-1, 1.02)
ax.set_ylabel("test R² vs clean truth (clipped at -1)")
ax.set_title("C. Extrapolation when the answer is known (noise σ = 0.1·std)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="lower left")
style(ax)

fig.suptitle("Known-equation recovery test on the curated (cited-property) candidate set", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig("plots/recovery_test.png", dpi=150, bbox_inches="tight")
print("Wrote plots/recovery_test.png")
