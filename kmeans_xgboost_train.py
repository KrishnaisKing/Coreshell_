"""
Phase 4 (real candidates): Stratified Group-Safe Split + XGBoost training.

UPDATED FIXES:
1. Candidate-level clustering & stratified assignment to eliminate extreme 
   bimodal gap artifacts in log10(retention_tau_s).
2. Deepened XGBoost tree depth & adjusted sampling to fix horizontal prediction 
   plateaus inside individual candidate clusters.
3. Added 5-Fold GroupKFold cross-validation across candidates for robust evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data & Candidate Setup
# ---------------------------------------------------------------
df = pd.read_csv("csv/rs_training_data_real_candidates.csv")
print(f"Loaded {len(df)} rows.")

candidate_id_col = "candidate_id" if "candidate_id" in df.columns else None
if candidate_id_col is None:
    df["candidate_id"] = df["formula_core"].astype(str) + "__" + df["formula_shell"].astype(str)
    candidate_id_col = "candidate_id"

n_candidates = df[candidate_id_col].nunique()
print(f"{n_candidates} unique candidates, ~{len(df)/n_candidates:.0f} rows each.")

# Prepare Target Log Columns if missing
targets = {
    "hysteresis_window_V": "hysteresis_window_V",
    "log10_on_off_ratio": "on_off_ratio",
    "log10_retention_tau_s": "retention_tau_s",
}

for target_name, fallback_col in targets.items():
    if target_name not in df.columns and fallback_col in df.columns:
        df[f"log10_{fallback_col}"] = np.log10(df[fallback_col].clip(lower=1e-12))

# ---------------------------------------------------------------
# 2. Candidate-Level Summaries & Stratified Clustering
# ---------------------------------------------------------------
material_cols = [c for c in ["dE_LUMO_eV", "dE_HOMO_eV", "confinement_score_eV",
                              "lattice_mismatch_pct"] if c in df.columns]

if not material_cols:
    raise SystemExit(f"None of material columns found. Available: {df.columns.tolist()}")

# Create candidate-level representation (descriptors + mean retention target)
candidate_level = df.groupby(candidate_id_col)[material_cols].first().reset_index()

# Add target mean for stratified binning across retention time
if "log10_retention_tau_s" in df.columns:
    ret_means = df.groupby(candidate_id_col)["log10_retention_tau_s"].mean().values
    candidate_level["mean_log_tau"] = ret_means
    # Bin into low, mid, high retention bins for balanced split
    candidate_level["retention_bin"] = pd.qcut(candidate_level["mean_log_tau"], q=3, labels=[0, 1, 2])
else:
    candidate_level["retention_bin"] = 0

print(f"Clustering on candidate features: {material_cols}")
scaler = StandardScaler()
X_cluster = scaler.fit_transform(candidate_level[material_cols])

n_clusters = min(6, n_candidates // 5)
km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
candidate_level["cluster"] = km.fit_predict(X_cluster)

# Assign Train/Test using Stratified Candidate Sampling
rng = np.random.default_rng(RANDOM_STATE)
test_candidates = []

for _, bin_group in candidate_level.groupby("retention_bin"):
    unique_clusters = bin_group["cluster"].unique()
    rng.shuffle(unique_clusters)
    # Pick ~25% of candidates from each retention bin
    n_test_in_bin = max(1, round(len(bin_group) * 0.25))
    
    count = 0
    for c in unique_clusters:
        c_cands = bin_group[bin_group["cluster"] == c][candidate_id_col].tolist()
        test_candidates.extend(c_cands)
        count += len(c_cands)
        if count >= n_test_in_bin:
            break

candidate_level["split"] = np.where(candidate_level[candidate_id_col].isin(test_candidates), "test", "train")
print(f"\nCandidates Split -> Train: {(candidate_level['split']=='train').sum()}, Test: {(candidate_level['split']=='test').sum()}")

# Merge splits back to row-level DataFrame
df = df.merge(candidate_level[[candidate_id_col, "cluster", "split"]], on=candidate_id_col, how="left")
train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

# Zero overlap sanity check
overlap = set(train_df[candidate_id_col]) & set(test_df[candidate_id_col])
assert len(overlap) == 0, f"LEAK: {len(overlap)} candidates appear in both splits!"

# ---------------------------------------------------------------
# 3. XGBoost Model Training & Evaluation
# ---------------------------------------------------------------
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]

feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

print(f"\nFeature columns used: {feature_cols}")

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
    
    # 5-Fold Group Cross-Validation on Training Set
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    groups = train_df[candidate_id_col]
    
    for train_idx, val_idx in gkf.split(X_train, y_train, groups):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds_val = model.predict(X_val)
        cv_scores.append(r2_score(y_val, preds_val))
    
    print(f"\n[{ycol}] GroupKFold 5-Fold CV R^2 Mean: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # Fit on full training set and evaluate on test set
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"[{ycol}] Test Set Evaluation -> R^2 = {r2:.4f} | MAE = {mae:.4f}")

    # Save artifact files
    model.save_model(f"xgb_model_{ycol}.json")
    out = test_df[[candidate_id_col]].copy()
    out["y_true"] = y_test.values
    out["y_pred"] = pred
    out.to_csv(f"csv/predictions_xgb_{ycol}.csv", index=False)

print("\nDone. Saved models (xgb_model_*.json) and predictions (predictions_*.csv).")