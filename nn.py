"""
Phase 4 Neural Network Counterpart (PyTorch MLP)
Stratified Group-Safe Split + Hyperparameter Tuned for High R^2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from split_utils import RANDOM_STATE, load_dataset, candidate_split
from nn_model import add_engineered_features, feature_columns, TabularResNet, predict, permutation_importance

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------
# 1. Load Data, Feature Engineering & candidate-grouped split
# ---------------------------------------------------------------
df = add_engineered_features(load_dataset())
train_df, test_df = candidate_split(df)
feature_cols = feature_columns(df)

# ---------------------------------------------------------------
# 2. Training Loop
# ---------------------------------------------------------------
target_cols = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]

for target_col in target_cols:
    if target_col not in df.columns:
        continue

    # Scale Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df[target_col].values.astype(np.float32).reshape(-1, 1)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Initialize Model
    model = TabularResNet(input_dim=len(feature_cols), hidden_dim=128, num_blocks=2, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Train
    model.train()
    for epoch in range(200):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Predict & Evaluate
    preds = predict(model, X_test)
    y_true_flat = y_test.ravel()
    r2 = r2_score(y_true_flat, preds)
    mae = mean_absolute_error(y_true_flat, preds)
    print(f"[{target_col}] NN Evaluation -> R^2 = {r2:.4f} | MAE = {mae:.4f}")

    # Save outputs for parity plotting
    out = test_df[["candidate_id"]].copy()
    out["y_true"] = y_true_flat
    out["y_pred"] = preds
    out.to_csv(f"csv/predictions_nn_{target_col}.csv", index=False)
    torch.save(model.state_dict(), f"nn_model_{target_col}.pt")

    imp = permutation_importance(model, X_test, y_true_flat, feature_cols, random_state=RANDOM_STATE)
    imp.to_csv(f"csv/permutation_importance_nn_{target_col}.csv", index=False)

print("\nDone. Neural network training complete.")
