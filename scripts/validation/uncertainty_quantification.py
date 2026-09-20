"""
Tier 2 fix: adds prediction intervals, not just point estimates. Nothing else in
this pipeline produces a confidence interval, which matters most for the stated
end use (ranking candidates for synthesis) -- a ranked list from point estimates
alone can't distinguish "confidently good" from "who knows, could be anything."

Approach: XGBoost's native multi-quantile objective (reg:quantileerror,
quantile_alpha=[0.1, 0.5, 0.9]), trained with the same hyperparameters, feature
set, and candidate_split() as kmeans_xgboost_train.py, so this is genuinely
"add uncertainty to the existing model" rather than a separate exercise. No new
dependency: this has been in xgboost's sklearn API since 2.0 (installed here:
3.4.1), so no ensembles/GPs needed for a first pass.

For each target, reports:
  - empirical coverage: fraction of test y_true inside [q10, q90] (well
    calibrated should be close to 80%)
  - mean interval width (in the target's own units -- log10 for on/off ratio and
    retention, volts for hysteresis)
  - q50 MAE, to sanity-check the median prediction is in the same ballpark as
    kmeans_xgboost_train.py's point-estimate MAE

Writes csv/uncertainty_quantification.csv (per-candidate) and
plots/uncertainty_quantification.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import RANDOM_STATE, load_dataset, candidate_split

plt.style.use('seaborn-v0_8-whitegrid')

QUANTILES = [0.1, 0.5, 0.9]
NOMINAL_COVERAGE = QUANTILES[-1] - QUANTILES[0]  # 0.8

df = load_dataset()
train_df, test_df = candidate_split(df)

band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

targets = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]

all_rows = []
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

for ax, target in zip(axes, targets):
    if target not in df.columns:
        continue
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    q10, q50, q90 = model.predict(X_test).T

    within = (y_test.values >= q10) & (y_test.values <= q90)
    coverage = within.mean()
    mean_width = (q90 - q10).mean()
    q50_mae = mean_absolute_error(y_test, q50)

    print(f"[{target:22s}] empirical coverage = {coverage:.3f} (nominal {NOMINAL_COVERAGE:.2f})  "
          f"mean interval width = {mean_width:.3f}  q50 MAE = {q50_mae:.4f}")

    out = pd.DataFrame({
        "candidate_id": test_df["candidate_id"].values, "target": target,
        "y_true": y_test.values, "q10": q10, "q50": q50, "q90": q90, "within_interval": within,
    })
    all_rows.append(out)

    order = np.argsort(q50)
    x = np.arange(len(order))
    ax.fill_between(x, q10[order], q90[order], color="#93C5FD", alpha=0.6, label="80% interval [q10, q90]")
    ax.plot(x, q50[order], color="#1E3A8A", linewidth=1, label="q50 (median prediction)")
    ax.scatter(x, y_test.values[order], color="#EF4444", s=4, alpha=0.5, label="actual", zorder=5)
    ax.set_title(f"{target}\ncoverage={coverage:.2f} (nominal {NOMINAL_COVERAGE:.2f}), "
                 f"mean width={mean_width:.2f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("test candidates, sorted by predicted median", fontsize=8)
    ax.set_ylabel(target, fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

pd.concat(all_rows, ignore_index=True).to_csv("csv/uncertainty_quantification.csv", index=False)
plt.tight_layout()
fig.savefig("plots/uncertainty_quantification.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nDone. Wrote csv/uncertainty_quantification.csv and plots/uncertainty_quantification.png")
