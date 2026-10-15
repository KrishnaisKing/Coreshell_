"""
Phase 4 Neural Network Counterpart (PyTorch MLP)
Stratified Group-Safe Split + Hyperparameter Tuned for High R^2
"""

import torch

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import RANDOM_STATE, load_dataset, candidate_split
from nn_model import (TORCH_THREADS, add_engineered_features, feature_columns, inner_split_and_scaler,
                      train_tabular_resnet, predict, permutation_importance)

torch.set_num_threads(TORCH_THREADS)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------
# 1. Load Data, Feature Engineering & candidate-grouped split
# ---------------------------------------------------------------
df = add_engineered_features(load_dataset())
train_df, test_df = candidate_split(df)

# Inner validation split, carved out of train_df only -- test_df is never touched
# until final evaluation. Plain GroupShuffleSplit, not candidate_split()'s
# clustering/stratification: that machinery's greedy overshoot is calibrated for
# ~2000 candidates and is far worse at this scale (it gave 51.6% to validation
# instead of the requested 15% when tried here). Validation just needs a clean,
# correctly-sized, same-distribution held-out slice for early stopping -- it isn't
# meant to be another extrapolation test.
feature_cols = feature_columns(df)
# Scaler is fit on inner_train only, so validation and test are both genuinely held
# out from anything the model or scaler has seen. Built in nn_model.py so
# nn_permutation_importance.py reproduces exactly the same preprocessing.
inner_train_df, val_df, scaler = inner_split_and_scaler(train_df, feature_cols, RANDOM_STATE)

# ---------------------------------------------------------------
# 2. Training Loop (early-stopped on a held-out validation fold)
# ---------------------------------------------------------------
target_cols = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]
early_stop_log = []

for target_col in target_cols:
    if target_col not in df.columns:
        continue

    X_train = scaler.transform(inner_train_df[feature_cols].values)
    y_train = inner_train_df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df[target_col].values.astype(np.float32).reshape(-1, 1)

    model, best_epoch, stopped_epoch, best_val_r2 = train_tabular_resnet(
        X_train, y_train, X_val, y_val, shuffle_seed=RANDOM_STATE, log_name=target_col)
    early_stop_log.append({"target": target_col, "best_epoch": best_epoch,
                            "stopped_epoch": stopped_epoch, "best_val_r2": best_val_r2})

    # Predict & Evaluate on the untouched test set, using the best-validation-epoch weights
    preds = predict(model, X_test)
    y_true_flat = y_test.ravel()
    r2 = r2_score(y_true_flat, preds)
    mae = mean_absolute_error(y_true_flat, preds)
    print(f"[{target_col}] NN Evaluation (best epoch {best_epoch}, val R^2 = {best_val_r2:.4f}) "
          f"-> test R^2 = {r2:.4f} | MAE = {mae:.4f}")

    # Save outputs for parity plotting
    out = test_df[["candidate_id"]].copy()
    out["y_true"] = y_true_flat
    out["y_pred"] = preds
    out.to_csv(f"csv/predictions_nn_{target_col}.csv", index=False)
    torch.save(model.state_dict(), f"nn_model_{target_col}.pt")

    imp = permutation_importance(model, X_test, y_true_flat, feature_cols, random_state=RANDOM_STATE)
    imp.to_csv(f"csv/permutation_importance_nn_{target_col}.csv", index=False)

pd.DataFrame(early_stop_log).to_csv("csv/nn_early_stopping_log.csv", index=False)
print("\nDone. Neural network training complete.")
