"""
Tier 1 fix (core-validation-gaps): quantifies the leakage story instead of just
asserting it. Trains the same XGBoost model, same hyperparameters, same three
targets, under three split strategies:

  random    - plain row-level random split (deliberately leaky baseline)
  grouped   - the pipeline's current candidate-pair-grouped, retention-stratified
              split (kmeans_xgboost_train.py / nn.py's candidate_split())
  lomo      - leave-materials-out: held-out materials never appear in training at
              all, in either role -- the strictest generalization test

Doc reference: Publication possibilities_figures.docx essential figure #5
("split-strategy comparison ... makes the leakage story explicit and quantified
rather than asserted in text").

Writes csv/split_strategy_comparison.csv and plots/split_strategy_comparison.png.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

from split_utils import RANDOM_STATE, load_dataset, candidate_split, random_row_split, leave_materials_out_split

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

df = load_dataset()

band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

targets = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]

strategies = {
    "random (leaky baseline)": random_row_split,
    "grouped-by-pair (pipeline default)": candidate_split,
    "leave-materials-out": leave_materials_out_split,
}

rows = []
for strat_name, split_fn in strategies.items():
    train_df, test_df = split_fn(df)
    print(f"\n=== {strat_name} === train rows={len(train_df)} test rows={len(test_df)} "
          f"(train candidates={train_df['candidate_id'].nunique()}, "
          f"test candidates={test_df['candidate_id'].nunique()})")

    for target in targets:
        if target not in df.columns:
            continue
        X_train, y_train = train_df[feature_cols], train_df[target]
        X_test, y_test = test_df[feature_cols], test_df[target]

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rows.append({
            "strategy": strat_name,
            "target": target,
            "r2": r2,
            "mae": mae,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
        })
        print(f"  [{target:22s}] R^2 = {r2:.4f}  MAE = {mae:.4f}")

results = pd.DataFrame(rows)
results.to_csv("csv/split_strategy_comparison.csv", index=False)

# ==========================================
# Figure: grouped bars, target on x, split strategy as hue
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
sns.barplot(data=results, x="target", y="r2", hue="strategy", ax=ax,
            palette=["#EF4444", "#3B82F6", "#10B981"], edgecolor="black", linewidth=0.5)
ax.set_title("Test $R^2$ by split strategy", fontweight="bold", fontsize=12)
ax.set_xlabel("Target", fontweight="bold", fontsize=10)
ax.set_ylabel("Test $R^2$", fontweight="bold", fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(title=None, loc="lower left", frameon=True, facecolor="white", edgecolor="lightgray")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig("plots/split_strategy_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nDone. Wrote csv/split_strategy_comparison.csv and plots/split_strategy_comparison.png")
