"""
Two problems this addresses together:

1. No confidence interval: kmeans_xgboost_train.py reports one test R^2 per
   target from one candidate_split() at RANDOM_STATE=42 -- a single draw, not a
   distribution.

2. Test-set contamination: this file's own docstring says max_depth, colsample_bytree
   and the clustering-based split design were chosen "to fix horizontal prediction
   plateaus" and "eliminate extreme bimodal gap artifacts" -- those are behaviors
   that were only visible by looking at predictions on candidate_split()'s test set
   at seed 42. That means the reported 0.925 / 0.976 / 0.815 test scores are
   development numbers: the hyperparameters were shaped by seeing them.

Re-evaluating the SAME fixed hyperparameters under many other seeds' splits is a
legitimate (if partial) fix for #2: those other seeds' test candidates were never
looked at while max_depth/colsample_bytree were chosen, so their scores are not
contaminated the way seed 42's are. If seed 42 turns out to be an optimistic
outlier relative to the other seeds, that tells you the reported numbers overstate
real performance; if it sits inside the spread, the contamination concern is
smaller in practice than the methodology gap alone would suggest.

This does NOT fix contamination for retraining decisions made after this point --
any future hyperparameter change should be validated against seeds not used to
choose it, ideally via a true nested CV.

Writes csv/repeated_split_evaluation.csv and plots/repeated_split_evaluation.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

from split_utils import load_dataset, candidate_split

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
DEVELOPMENT_SEED = 42  # the seed used everywhere else in this repo; reported separately, not averaged in

plt.style.use('seaborn-v0_8-whitegrid')

df = load_dataset()
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

targets = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]

rows = []
for seed in SEEDS + [DEVELOPMENT_SEED]:
    train_df, test_df = candidate_split(df, random_state=seed)
    for target in targets:
        if target not in df.columns:
            continue
        X_train, y_train = train_df[feature_cols], train_df[target]
        X_test, y_test = test_df[feature_cols], test_df[target]

        # Same fixed hyperparameters as kmeans_xgboost_train.py -- unchanged here on purpose.
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, random_state=42,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "seed": seed,
            "is_development_seed": seed == DEVELOPMENT_SEED,
            "target": target,
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
        })
    print(f"seed {seed} done" + (" (development seed, used to choose hyperparameters)" if seed == DEVELOPMENT_SEED else ""))

results = pd.DataFrame(rows)
results.to_csv("csv/repeated_split_evaluation.csv", index=False)

print("\n=== R^2 across 10 unseen seeds (development seed 42 excluded from stats) ===")
unseen = results[~results["is_development_seed"]]
for target in targets:
    sub = unseen[unseen["target"] == target]["r2"]
    dev_r2 = results[(results["target"] == target) & (results["is_development_seed"])]["r2"].iloc[0]
    print(f"[{target:22s}] unseen-seed mean={sub.mean():.4f} std={sub.std():.4f} "
          f"min={sub.min():.4f} max={sub.max():.4f}  |  seed-42 (development) = {dev_r2:.4f}")

# ==========================================
# Figure: R^2 spread per target, seed 42 marked separately
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
sns.boxplot(data=unseen, x="target", y="r2", ax=ax, color="#93C5FD", showfliers=False)
sns.stripplot(data=unseen, x="target", y="r2", ax=ax, color="#1E3A8A", alpha=0.6, size=5, jitter=0.15)
dev_points = results[results["is_development_seed"]]
for i, target in enumerate(targets):
    r2 = dev_points[dev_points["target"] == target]["r2"].iloc[0]
    ax.scatter([i], [r2], color="#EF4444", marker="D", s=90, zorder=5,
               label="seed 42 (used to choose hyperparameters)" if i == 0 else None)
ax.set_title("Test $R^2$ across 10 unseen seeds vs. the development seed", fontweight="bold", fontsize=12)
ax.set_xlabel("Target", fontweight="bold", fontsize=10)
ax.set_ylabel("Test $R^2$", fontweight="bold", fontsize=10)
ax.legend(loc="lower left", frameon=True, facecolor="white")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig("plots/repeated_split_evaluation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nDone. Wrote csv/repeated_split_evaluation.csv and plots/repeated_split_evaluation.png")
