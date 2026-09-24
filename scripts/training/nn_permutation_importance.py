"""
Compute permutation feature importance for already-trained nn_model_*.pt files
without retraining. Reproduces the deterministic candidate split and the exact
feature preprocessing nn.py trained with (via nn_model.inner_split_and_scaler),
loads each saved model, and writes csv/permutation_importance_nn_{target}.csv.

Run this whenever nn_model_*.pt exists but the matching importance CSV doesn't
(e.g. models trained before nn.py started writing importances itself).
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import RANDOM_STATE, load_dataset, candidate_split
from nn_model import (TORCH_THREADS, add_engineered_features, feature_columns, inner_split_and_scaler,
                      TabularResNet, predict, permutation_importance)

torch.set_num_threads(TORCH_THREADS)

# float32 inference noise sits around 1e-7; anything near 1e-3 means the inputs differ.
REPRODUCTION_TOL = 1e-4

df = add_engineered_features(load_dataset())
train_df, test_df = candidate_split(df)
feature_cols = feature_columns(df)
_, _, scaler = inner_split_and_scaler(train_df, feature_cols, RANDOM_STATE)
X_test = scaler.transform(test_df[feature_cols].values)

for target_col in ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]:
    model_path = f"nn_model_{target_col}.pt"
    if not os.path.exists(model_path):
        print(f"[{target_col}] {model_path} not found, skipping.")
        continue

    model = TabularResNet(input_dim=len(feature_cols), hidden_dim=128, num_blocks=2, dropout=0.1)
    model.load_state_dict(torch.load(model_path))

    # Guard: re-predict the test set and require it to match the predictions nn.py saved.
    # Checking candidate IDs alone is not enough -- it passed while this script was
    # feeding the models inputs scaled differently from training, which silently
    # lowered every model's test R^2. Matching predictions verifies the whole chain
    # (split, feature engineering, scaler, weights) at once.
    pred_path = f"csv/predictions_nn_{target_col}.csv"
    if not os.path.exists(pred_path):
        raise SystemExit(f"[{target_col}] {pred_path} not found; can't verify preprocessing matches "
                         "training. Re-run nn.py instead.")
    saved = pd.read_csv(pred_path)
    if list(saved["candidate_id"]) != list(test_df["candidate_id"]):
        raise SystemExit(f"[{target_col}] reproduced test rows don't match {pred_path}; re-run nn.py instead.")
    max_diff = np.abs(saved["y_pred"].values - predict(model, X_test)).max()
    if max_diff > REPRODUCTION_TOL:
        raise SystemExit(f"[{target_col}] saved model does not reproduce {pred_path} (max |diff| = "
                         f"{max_diff:.2e}); preprocessing has drifted from nn.py. Re-run nn.py instead.")

    # float32, matching nn.py's y_test exactly, so this reproduces nn.py's importance CSV bit-for-bit.
    y_test = test_df[target_col].values.astype(np.float32)
    imp = permutation_importance(model, X_test, y_test, feature_cols, random_state=RANDOM_STATE)
    out_path = f"csv/permutation_importance_nn_{target_col}.csv"
    imp.to_csv(out_path, index=False)
    print(f"[{target_col}] reproduced saved predictions (max |diff| = {max_diff:.1e}); wrote {out_path}")
    print(imp.head(6).to_string(index=False))
