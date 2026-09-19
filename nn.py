"""
Phase 4 Neural Network Counterpart (PyTorch MLP)
Stratified Group-Safe Split + Hyperparameter Tuned for High R^2
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from split_utils import RANDOM_STATE, load_dataset, candidate_split
from nn_model import add_engineered_features, feature_columns, TabularResNet, predict, permutation_importance

# The model is tiny (14 inputs, 128 hidden); letting torch spread each op across all
# cores costs more in thread contention than it gains. 4 threads beats 16 by ~2x here.
torch.set_num_threads(4)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

EPOCHS = 200
BATCH_SIZE = 128
LR = 2e-3
VAL_FRAC = 0.15          # carved out of train_df, grouped by candidate, for early stopping only
WARMUP_EPOCHS = 40       # don't let patience start counting until training has had a chance to anneal
PATIENCE = 30            # post-warmup epochs without validation R^2 improvement before stopping
MIN_DELTA = 1e-4

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
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=RANDOM_STATE + 1)
inner_idx, val_idx = next(gss.split(train_df, groups=train_df["candidate_id"]))
inner_train_df = train_df.iloc[inner_idx].copy()
val_df = train_df.iloc[val_idx].copy()
assert set(inner_train_df["candidate_id"]) & set(val_df["candidate_id"]) == set()
feature_cols = feature_columns(df)

# ---------------------------------------------------------------
# 2. Training Loop (early-stopped on a held-out validation fold)
# ---------------------------------------------------------------
target_cols = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]
early_stop_log = []

for target_col in target_cols:
    if target_col not in df.columns:
        continue

    # Scale Features -- scaler fit on inner_train only, so validation and test
    # are both genuinely held out from anything the model or scaler has seen.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(inner_train_df[feature_cols].values)
    y_train = inner_train_df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train)
    n_train = len(X_train_t)

    # Initialize Model
    model = TabularResNet(input_dim=len(feature_cols), hidden_dim=128, num_blocks=2, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train. Manual index batching instead of DataLoader: the per-batch DataLoader
    # overhead was a large fraction of step time for a model this small.
    shuffle_gen = torch.Generator().manual_seed(RANDOM_STATE)
    t0 = time.perf_counter()
    best_val_r2 = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improve = 0
    stopped_epoch = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, generator=shuffle_gen)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()

        val_r2 = r2_score(y_val.ravel(), predict(model, X_val))
        improved = val_r2 > best_val_r2 + MIN_DELTA
        if improved:
            best_val_r2 = val_r2
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Patience only starts counting after warmup: with CosineAnnealingLR(T_max=EPOCHS),
        # the LR hasn't annealed yet in the first ~40 epochs and val R^2 is dominated by
        # noise from the still-high LR, not genuine convergence -- letting patience run from
        # epoch 1 stopped training at epoch ~20 in practice, before it had gotten anywhere.
        if epoch > WARMUP_EPOCHS:
            epochs_without_improve = 0 if improved else epochs_without_improve + 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"[{target_col}] epoch {epoch:3d}/{EPOCHS}  train MSE = {epoch_loss / n_train:.4f}  "
                  f"val R^2 = {val_r2:.4f}  best = {best_val_r2:.4f}@{best_epoch}  "
                  f"({time.perf_counter() - t0:.0f}s elapsed)")

        if epoch > WARMUP_EPOCHS and epochs_without_improve >= PATIENCE:
            stopped_epoch = epoch
            print(f"[{target_col}] early stopping at epoch {epoch} "
                  f"(no val R^2 improvement for {PATIENCE} epochs after warmup)")
            break

    model.load_state_dict(best_state)
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
