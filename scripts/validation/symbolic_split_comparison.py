"""
Methodology-paper step 2: does the physics-constrained symbolic regressor
(scripts/symbolic_model.py) generalize better than XGBoost / the NN where it
matters -- out of distribution -- and what does it cost in-distribution?

Same three models, same three targets, six split strategies:

  random            random_row_split()          deliberately leaky baseline
  grouped-by-pair   candidate_split()           pipeline default (offset-space cluster holdout)
  leave-materials   leave_materials_out_split() unseen materials in either role
  Nt range          feature_range_split(Nt_cm3)            top 25% trap density held out
  barrier range     feature_range_split(max |offset|)      top 25% barrier height held out
  thickness range   feature_range_split(shell_thick_nm)    top 25% shell thickness held out

The first three are re-drawn under seeds 0..9 -- never seed 42, whose test set
shaped the XGBoost hyperparameters (see repeated_split_evaluation.py) -- and
reported as mean +/- std. The range splits are deterministic (one run each).
Every model sees the same split; the NN uses nn.py's exact training loop and
preprocessing, with its inner validation carve-out and init seeded per split.

The range splits are new here: trees cannot predict outside the target range
they were trained on, so a held-out high-Nt or high-barrier slab is the direct
test of the extrapolation claim. Their R^2 is computed on a range-restricted test
set (lower variance), so read MAE alongside it.

Writes csv/symbolic_split_comparison.csv (one row per fit),
csv/symbolic_split_comparison_summary.csv and plots/symbolic_split_comparison.png.
"""

import time
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import (load_dataset, candidate_split, random_row_split, leave_materials_out_split,
                         feature_range_split)
from nn_model import (TORCH_THREADS, add_engineered_features, feature_columns, inner_split_and_scaler,
                      train_tabular_resnet, predict)
from symbolic_model import SparseSymbolicRegressor, CONSTRAINTS

torch.set_num_threads(TORCH_THREADS)

SEEDS = list(range(10))
TARGETS = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]
OUT_CSV = "csv/symbolic_split_comparison.csv"

df = add_engineered_features(load_dataset())
df["max_abs_offset_eV"] = np.maximum(df["dE_LUMO_eV"].abs(), df["dE_HOMO_eV"].abs())

# XGBoost: identical features/hyperparameters to split_strategy_comparison.py
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
xgb_features = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols
nn_features = feature_columns(df)

STRATEGIES = {
    "random (leaky)": (random_row_split, SEEDS),
    "grouped-by-pair": (candidate_split, SEEDS),
    "leave-materials-out": (leave_materials_out_split, SEEDS),
    "Nt range (top 25%)": (partial(feature_range_split, column="Nt_cm3"), [0]),
    "barrier range (top 25%)": (partial(feature_range_split, column="max_abs_offset_eV"), [0]),
    "thickness range (top 25%)": (partial(feature_range_split, column="shell_thick_nm"), [0]),
}


def fit_xgb(train_df, test_df, target, seed):
    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.7, random_state=seed)
    model.fit(train_df[xgb_features], train_df[target])
    return model.predict(test_df[xgb_features]), {}


def fit_nn(train_df, test_df, target, seed):
    inner_train_df, val_df, scaler = inner_split_and_scaler(train_df, nn_features, seed)
    torch.manual_seed(seed)
    model, best_epoch, _, _ = train_tabular_resnet(
        scaler.transform(inner_train_df[nn_features].values), inner_train_df[target].values.astype(np.float32),
        scaler.transform(val_df[nn_features].values), val_df[target].values.astype(np.float32),
        shuffle_seed=seed)
    return predict(model, scaler.transform(test_df[nn_features].values)), {"nn_best_epoch": best_epoch}


def fit_sr(train_df, test_df, target, seed):
    model = SparseSymbolicRegressor(constraints=CONSTRAINTS[target])
    model.fit(train_df, train_df[target], groups=train_df["candidate_id"])
    return model.predict(test_df), {"sr_n_terms": model.n_terms_, "sr_terms": " | ".join(model.terms()),
                                    "sr_constraints_satisfied": model.constraints_satisfied()}


MODELS = {"symbolic (ours)": fit_sr, "XGBoost": fit_xgb, "NN": fit_nn}

rows = []
t_start = time.perf_counter()
for strat_name, (split_fn, seeds) in STRATEGIES.items():
    for seed in seeds:
        train_df, test_df = split_fn(df, random_state=seed)
        for target in TARGETS:
            y = test_df[target].to_numpy()
            for model_name, fit in MODELS.items():
                pred, extra = fit(train_df, test_df, target, seed)
                rows.append({"strategy": strat_name, "seed": seed, "target": target, "model": model_name,
                             "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred),
                             # share of test predictions above the highest training label. Not zero
                             # for XGBoost: summed boosted trees can overshoot the label maximum
                             # slightly, which counts when test labels sit at the target ceiling
                             "frac_pred_above_train_max": float(np.mean(pred > train_df[target].max())),
                             "n_train_rows": len(train_df), "n_test_rows": len(test_df), **extra})
        done = pd.DataFrame(rows)
        done.to_csv(OUT_CSV, index=False)
        last = done[(done.strategy == strat_name) & (done.seed == seed)]
        print(f"[{time.perf_counter() - t_start:5.0f}s] {strat_name} seed {seed}: " + "  ".join(
            f"{t.split('_')[0] if t != 'log10_on_off_ratio' else 'onoff'}/"
            + "/".join(f"{r2:.3f}" for r2 in last[last.target == t].r2) for t in TARGETS)
            + "   (R^2 sr/xgb/nn)")

results = pd.DataFrame(rows)
summary = (results.groupby(["strategy", "target", "model"], sort=False)
           .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                n_seeds=("seed", "nunique"))
           .reset_index())
summary.to_csv("csv/symbolic_split_comparison_summary.csv", index=False)
print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------------------- figure
# Validated categorical palette (dataviz skill, light mode): slot 1 blue, 2 orange, 3 aqua.
COLORS = {"symbolic (ours)": "#2a78d6", "XGBoost": "#eb6834", "NN": "#1baf7a"}
INK, MUTED, GRID = "#21201c", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

strategies = list(STRATEGIES)
fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=150, sharex=True)
width = 0.26
x = np.arange(len(strategies))
for col, target in enumerate(TARGETS):
    for row, (metric, label) in enumerate([("r2", "test R²"), ("mae", "test MAE")]):
        ax = axes[row, col]
        for k, model_name in enumerate(MODELS):
            s = summary[(summary.target == target) & (summary.model == model_name)].set_index("strategy")
            means = s.loc[strategies, f"{metric}_mean"].to_numpy()
            stds = s.loc[strategies, f"{metric}_std"].fillna(0).to_numpy()
            ax.bar(x + (k - 1) * width, means, width - 0.03, color=COLORS[model_name], label=model_name,
                   yerr=stds, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        if metric == "r2":
            ax.set_ylim(min(0, summary[summary.target == target].r2_mean.min() - 0.05), 1.0)
            ax.set_title(target, fontsize=12, color=INK)
        ax.set_ylabel(label)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(2.5, color=MUTED, linewidth=0.8, linestyle=":")
        if row == 1:
            ax.set_xticks(x, [s.replace(" (", "\n(") for s in strategies], rotation=0, fontsize=8)
axes[0, 0].legend(frameon=False, loc="lower left")
fig.suptitle("Symbolic regression vs XGBoost vs NN across split strategies\n"
             "left of dotted line: mean ± std over seeds 0–9 · right: deterministic range-extrapolation splits",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig("plots/symbolic_split_comparison.png", dpi=150, bbox_inches="tight")
print("\nWrote csv/symbolic_split_comparison.csv, csv/symbolic_split_comparison_summary.csv, "
      "plots/symbolic_split_comparison.png")
