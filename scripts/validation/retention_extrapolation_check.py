"""
Diagnoses why log10_retention_tau_s's held-out R^2 collapses under the pipeline's
default candidate_split() (0.815) but not under leave_materials_out_split() (0.982),
while hysteresis_window_V and log10_on_off_ratio are stable across every split
strategy (see split_strategy_comparison.py).

Hypothesis: candidate_split() clusters candidates by material features
(dE_LUMO_eV, dE_HOMO_eV -- the only 2 of the 4 intended columns present in this
dataset) and holds out whole clusters. Permutation importance
(csv/permutation_importance_nn_log10_retention_tau_s.csv) shows retention is
driven by exactly those two features, while hysteresis/on-off ratio are driven by
trap density, which is sampled independently of cluster membership. So holding
out whole offset-space clusters is a genuine extrapolation test for retention and
a near-irrelevant one for the other two targets.

Test: for each grouped-by-pair test candidate, find its nearest training
candidate in scaled (dE_LUMO_eV, dE_HOMO_eV) space, and correlate that distance
with the XGBoost model's actual retention prediction error.

Writes csv/retention_extrapolation_check.csv and
plots/retention_extrapolation_check.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import load_dataset, candidate_split, leave_materials_out_split

plt.style.use('seaborn-v0_8-whitegrid')

MATERIAL_COLS = ['dE_LUMO_eV', 'dE_HOMO_eV']

df = load_dataset()


def nearest_train_distance(split_fn):
    train_df, test_df = split_fn(df)
    cand_train = train_df.groupby('candidate_id')[MATERIAL_COLS].first()
    cand_test = test_df.groupby('candidate_id')[MATERIAL_COLS].first()
    scaler = StandardScaler().fit(cand_train[MATERIAL_COLS])
    tree = cKDTree(scaler.transform(cand_train[MATERIAL_COLS]))
    dist, _ = tree.query(scaler.transform(cand_test[MATERIAL_COLS]), k=1)
    return pd.DataFrame({'candidate_id': cand_test.index, 'nn_dist_to_train': dist})


grouped_dist = nearest_train_distance(candidate_split)
lomo_dist = nearest_train_distance(leave_materials_out_split)

print("Nearest-train-neighbor distance in (dE_LUMO_eV, dE_HOMO_eV) space:")
for name, d in [("grouped-by-pair", grouped_dist), ("leave-materials-out", lomo_dist)]:
    print(f"  {name}: mean={d.nn_dist_to_train.mean():.3f} median={d.nn_dist_to_train.median():.3f} "
          f"p90={d.nn_dist_to_train.quantile(0.9):.3f}")

pred = pd.read_csv("csv/predictions_xgb_log10_retention_tau_s.csv")
pred["abs_err"] = (pred["y_true"] - pred["y_pred"]).abs()
cand_err = pred.groupby("candidate_id")["abs_err"].mean().reset_index()

merged = grouped_dist.merge(cand_err, on="candidate_id", how="inner")
merged.to_csv("csv/retention_extrapolation_check.csv", index=False)

pearson = merged["nn_dist_to_train"].corr(merged["abs_err"])
spearman = merged["nn_dist_to_train"].corr(merged["abs_err"], method="spearman")
median_dist = merged["nn_dist_to_train"].median()
near = merged[merged["nn_dist_to_train"] <= median_dist]["abs_err"]
far = merged[merged["nn_dist_to_train"] > median_dist]["abs_err"]

print(f"\ncorrelation(distance to nearest train candidate, |retention error|):")
print(f"  Pearson = {pearson:.3f}, Spearman = {spearman:.3f}")
print(f"  mean |error|, near half (dist<={median_dist:.3f}): {near.mean():.3f} (n={len(near)})")
print(f"  mean |error|, far half  (dist> {median_dist:.3f}): {far.mean():.3f} (n={len(far)})")

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
ax.scatter(merged["nn_dist_to_train"], merged["abs_err"], color='#3B82F6', edgecolors='#1E3A8A',
           alpha=0.6, s=30)
ax.set_xlabel("Distance to nearest training candidate\n(scaled dE_LUMO_eV, dE_HOMO_eV space)",
              fontweight='bold', fontsize=10)
ax.set_ylabel(r"$|y_{\mathrm{true}} - y_{\mathrm{pred}}|$ (log10 retention time)",
              fontweight='bold', fontsize=10)
ax.set_title(f"Retention error vs. feature-space extrapolation\n(Pearson r = {pearson:.2f})",
             fontweight='bold', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig("plots/retention_extrapolation_check.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("\nDone. Wrote csv/retention_extrapolation_check.csv and plots/retention_extrapolation_check.png")
