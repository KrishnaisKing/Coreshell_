"""
Tier 2 fix: VIF (variance inflation factor) check across the full feature set,
formalizing what was previously only checked via pairwise Pearson correlations.

VIF_i = 1 / (1 - R^2_i), where R^2_i comes from regressing feature i on every
other feature. VIF > ~5-10 is the conventional rule-of-thumb threshold for
"this feature's coefficient would be unstable in a linear/interpretable model."
No statsmodels dependency -- this project has no package manifest and adding one
just for variance_inflation_factor isn't worth it when it's a few lines of
sklearn.LinearRegression.

Computed two ways, because the band_align_* one-hots have TWO independent
sources of inflated VIF that would otherwise be conflated:
  1. The already-documented physical redundancy: band_alignment is a
     deterministic function of dE_LUMO_eV/dE_HOMO_eV (see CLAUDE.md).
  2. The "dummy variable trap": with all 4 band_alignment categories one-hot
     encoded and an intercept in the regression, the 4 columns sum to exactly 1
     for every row -- a perfect linear dependency that exists for ANY complete
     one-hot encoding, independent of what the categories mean physically.

  Pass 1 ("as used in the pipeline"): exact feature_cols from
    kmeans_xgboost_train.py, all 4 one-hots included -- shows the compounded
    picture that's actually fed to the models today.
  Pass 2 ("one-hot reference dropped"): drops one band_align_* column (the
    standard k-1 dummy convention), removing the dummy-trap artifact so the
    VIF that's left is attributable to the physical redundancy alone, and so
    the *other* continuous features' mutual multicollinearity (independent of
    band_alignment entirely) is actually visible instead of being swamped.

Writes csv/multicollinearity_check.csv and plots/multicollinearity_check.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from split_utils import load_dataset

plt.style.use('seaborn-v0_8-whitegrid')

VIF_WARN = 5.0
VIF_SEVERE = 10.0

df = load_dataset()
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols


def compute_vif(X: pd.DataFrame) -> pd.Series:
    # VIF's true value is scale-invariant -- it only depends on the correlation
    # structure between predictors, not their units. But the LEAST-SQUARES FIT
    # used to compute it is not numerically scale-invariant: Nt_cm3 spans
    # 1e15-1e19 while every other feature here is O(1)-O(300), a condition
    # number far beyond float64's ~15-16 digits of precision. Fitting
    # unstandardized gave R^2 ~ 0.00002 for every single feature (garbage --
    # the solver lost the small-magnitude coefficients in the noise floor),
    # which is what an all-VIF-exactly-1.0 plot actually means: numerical
    # failure, not "no collinearity." Standardizing first fixes this without
    # changing the true VIF values at all.
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    vifs = {}
    for col in Xs.columns:
        y = Xs[col].values
        others = Xs.drop(columns=[col]).values
        r2 = LinearRegression().fit(others, y).score(others, y)
        # r2 can come back fractionally above 1 or exactly 1 from float error on
        # an exact linear dependency -- clip so VIF reports as a large finite
        # number (informative) rather than a divide-by-zero inf (not).
        r2 = min(r2, 1 - 1e-10)
        vifs[col] = 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


X_full = df[feature_cols].dropna()
vif_as_used = compute_vif(X_full)

reference_col = "band_align_Type_II_staggered"  # most common class -> natural reference
cols_dropped = [c for c in feature_cols if c != reference_col]
X_dropped = df[cols_dropped].dropna()
vif_dropped_ref = compute_vif(X_dropped)

print("=== VIF as used in the pipeline (all 4 band_align_* one-hots) ===")
print(vif_as_used.round(2).to_string())
print(f"\n=== VIF with one-hot reference category dropped ({reference_col}) ===")
print(vif_dropped_ref.round(2).to_string())

results = pd.concat([
    vif_as_used.rename("vif").reset_index().rename(columns={"index": "feature"}).assign(pass_name="as_used"),
    vif_dropped_ref.rename("vif").reset_index().rename(columns={"index": "feature"}).assign(pass_name="reference_dropped"),
])
results.to_csv("csv/multicollinearity_check.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
for ax, series, title in [
    (axes[0], vif_as_used, "VIF as used in the pipeline\n(all 4 band_align_* one-hots)"),
    (axes[1], vif_dropped_ref, f"VIF with one-hot reference dropped\n({reference_col} excluded)"),
]:
    capped = series.clip(upper=50)  # cap display only, raw values are in the CSV
    colors = ["#EF4444" if v >= VIF_SEVERE else ("#F59E0B" if v >= VIF_WARN else "#10B981") for v in series]
    ax.barh(capped.index[::-1], capped.values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
    ax.axvline(VIF_WARN, color="gray", linestyle="--", linewidth=1, label=f"VIF={VIF_WARN:.0f} (watch)")
    ax.axvline(VIF_SEVERE, color="black", linestyle="--", linewidth=1, label=f"VIF={VIF_SEVERE:.0f} (severe)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("VIF (capped at 50 for display; see CSV for exact values)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig("plots/multicollinearity_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nDone. Wrote csv/multicollinearity_check.csv and plots/multicollinearity_check.png")
