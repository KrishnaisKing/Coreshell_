"""
Compute permutation feature importance for already-trained nn_model_*.pt files
without retraining. Reproduces the deterministic candidate split and feature
scaler from nn.py, loads each saved model, and writes
csv/permutation_importance_nn_{target}.csv.

Run this whenever nn_model_*.pt exists but the matching importance CSV doesn't
(e.g. models trained before nn.py started writing importances itself).
"""

import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import RANDOM_STATE, load_dataset, candidate_split
from nn_model import add_engineered_features, feature_columns, TabularResNet, permutation_importance

df = add_engineered_features(load_dataset())
train_df, test_df = candidate_split(df)
feature_cols = feature_columns(df)

scaler = StandardScaler().fit(train_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)

for target_col in ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]:
    model_path = f"nn_model_{target_col}.pt"
    if not os.path.exists(model_path):
        print(f"[{target_col}] {model_path} not found, skipping.")
        continue

    # Guard: the saved predictions must come from the same test candidates we reproduced,
    # otherwise the importance numbers would be computed against the wrong rows.
    pred_path = f"csv/predictions_nn_{target_col}.csv"
    if os.path.exists(pred_path):
        saved_ids = set(pd.read_csv(pred_path)["candidate_id"])
        if saved_ids != set(test_df["candidate_id"]):
            raise SystemExit(f"[{target_col}] reproduced test split does not match {pred_path}; "
                             "re-run nn.py instead of using this script.")

    model = TabularResNet(input_dim=len(feature_cols), hidden_dim=128, num_blocks=2, dropout=0.1)
    model.load_state_dict(torch.load(model_path))

    imp = permutation_importance(model, X_test, test_df[target_col].values, feature_cols,
                                 random_state=RANDOM_STATE)
    out_path = f"csv/permutation_importance_nn_{target_col}.csv"
    imp.to_csv(out_path, index=False)
    print(f"[{target_col}] wrote {out_path}")
    print(imp.head(6).to_string(index=False))
