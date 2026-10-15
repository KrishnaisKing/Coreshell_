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
csv/symbolic_split_comparison_summary.csv; the figure is drawn separately by
scripts/plotting/plot_symbolic_split_comparison.py so restyling it doesn't need this ~40-min rerun.
"""

import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import (load_dataset, candidate_split, random_row_split, leave_materials_out_split,
                         feature_range_split)
from model_fits import MODELS

SEEDS = list(range(10))
TARGETS = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]
OUT_CSV = "csv/symbolic_split_comparison.csv"

df = load_dataset()
df["max_abs_offset_eV"] = np.maximum(df["dE_LUMO_eV"].abs(), df["dE_HOMO_eV"].abs())

STRATEGIES = {
    "random (leaky)": (random_row_split, SEEDS),
    "grouped-by-pair": (candidate_split, SEEDS),
    "leave-materials-out": (leave_materials_out_split, SEEDS),
    "Nt range (top 25%)": (partial(feature_range_split, column="Nt_cm3"), [0]),
    "barrier range (top 25%)": (partial(feature_range_split, column="max_abs_offset_eV"), [0]),
    "thickness range (top 25%)": (partial(feature_range_split, column="shell_thick_nm"), [0]),
}

rows = []
t_start = time.perf_counter()
for strat_name, (split_fn, seeds) in STRATEGIES.items():
    for seed in seeds:
        train_df, test_df = split_fn(df, random_state=seed)
        for target in TARGETS:
            y = test_df[target].to_numpy()
            for model_name, fit in MODELS.items():
                pred, extra = fit(train_df, test_df, target, seed)
                extra = {k: v for k, v in extra.items() if k in ("nn_best_epoch", "sr_n_terms", "sr_terms",
                                                                 "sr_constraints_satisfied")}
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

print("\nWrote csv/symbolic_split_comparison.csv and csv/symbolic_split_comparison_summary.csv "
      "(figure: python scripts/plotting/plot_symbolic_split_comparison.py)")
