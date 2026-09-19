"""
Phase 4 (real candidates): Stratified Group-Safe Split + XGBoost training.

UPDATED FIXES:
1. Candidate-level clustering & stratified assignment to eliminate extreme
   bimodal gap artifacts in log10(retention_tau_s).
2. Deepened XGBoost tree depth & adjusted sampling to fix horizontal prediction
   plateaus inside individual candidate clusters.
3. Added 5-Fold StratifiedGroupKFold cross-validation across candidates (grouped by
   candidate_id, stratified by band_alignment) for robust, class-balanced evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

from split_utils import RANDOM_STATE, load_dataset, candidate_split

# ---------------------------------------------------------------
# 1. Load data & candidate-grouped split
# ---------------------------------------------------------------
df = load_dataset()
print(f"Loaded {len(df)} rows.")
n_candidates = df["candidate_id"].nunique()
print(f"{n_candidates} unique candidates, ~{len(df)/n_candidates:.0f} rows each.")

train_df, test_df = candidate_split(df)
print(f"\nCandidates Split -> Train: {train_df['candidate_id'].nunique()}, Test: {test_df['candidate_id'].nunique()}")

# ---------------------------------------------------------------
# 2. XGBoost Model Training & Evaluation
# ---------------------------------------------------------------
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]

feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

print(f"\nFeature columns used: {feature_cols}")

cv_rows = []

for target_name in ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]:
    if target_name not in df.columns:
        continue

    ycol = target_name
    X_train, y_train = train_df[feature_cols], train_df[ycol]
    X_test, y_test = test_df[feature_cols], test_df[ycol]

    # Model parameters tuned to resolve intra-candidate prediction plateaus
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,             # Deeper tree to capture device parameter sweeps
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,    # Force tree splits to explore non-material features
        random_state=RANDOM_STATE,
    )

    # 5-Fold Stratified Group Cross-Validation on Training Set.
    # StratifiedGroupKFold keeps candidate_id groups intact (no pair split across
    # folds) while also balancing band_alignment class share per fold -- plain
    # GroupKFold ignores the ~50/14/14/21% class imbalance entirely.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    groups = train_df["candidate_id"]
    strata = train_df["band_alignment"]

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, strata, groups), start=1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds_val = model.predict(X_val)
        r2_fold = r2_score(y_val, preds_val)
        mae_fold = mean_absolute_error(y_val, preds_val)
        cv_scores.append(r2_fold)
        cv_rows.append({"target": ycol, "fold": fold, "r2": r2_fold, "mae": mae_fold,
                         "n_val_rows": len(val_idx)})

    print(f"\n[{ycol}] StratifiedGroupKFold 5-Fold CV R^2 Mean: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # Fit on full training set and evaluate on test set
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"[{ycol}] Test Set Evaluation -> R^2 = {r2:.4f} | MAE = {mae:.4f}")

    # Save artifact files
    model.save_model(f"xgb_model_{ycol}.json")
    out = test_df[["candidate_id"]].copy()
    out["y_true"] = y_test.values
    out["y_pred"] = pred
    out.to_csv(f"csv/predictions_xgb_{ycol}.csv", index=False)

pd.DataFrame(cv_rows).to_csv("csv/cv_scores_xgb.csv", index=False)

print("\nDone. Saved models (xgb_model_*.json), predictions (csv/predictions_xgb_*.csv) and CV scores (csv/cv_scores_xgb.csv).")
