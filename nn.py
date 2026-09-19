"""
Phase 4 Neural Network Counterpart (PyTorch MLP)
Stratified Group-Safe Split + Hyperparameter Tuned for High R^2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------
# 1. Load Data & Feature Engineering
# ---------------------------------------------------------------
df = pd.read_csv("csv/rs_training_data_real_candidates.csv")

candidate_id_col = "candidate_id" if "candidate_id" in df.columns else None
if candidate_id_col is None:
    df["candidate_id"] = df["formula_core"].astype(str) + "__" + df["formula_shell"].astype(str)
    candidate_id_col = "candidate_id"

# Log transforms for target variables
targets = {
    "hysteresis_window_V": "hysteresis_window_V",
    "log10_on_off_ratio": "on_off_ratio",
    "log10_retention_tau_s": "retention_tau_s",
}
for target_name, fallback_col in targets.items():
    if target_name not in df.columns and fallback_col in df.columns:
        df[f"log10_{fallback_col}"] = np.log10(df[fallback_col].clip(lower=1e-12))

# Physics-informed non-linear features to boost NN performance
if "dE_LUMO_eV" in df.columns:
    df["exp_dE_LUMO"] = np.exp(df["dE_LUMO_eV"].clip(upper=5.0))
if "dE_HOMO_eV" in df.columns:
    df["exp_dE_HOMO"] = np.exp(df["dE_HOMO_eV"].clip(upper=5.0))
if "Nt_cm3" in df.columns:
    df["log10_Nt"] = np.log10(df["Nt_cm3"].clip(lower=1e10))

# ---------------------------------------------------------------
# 2. Candidate-Level Stratified Clustering Split
# ---------------------------------------------------------------
material_cols = [c for c in ["dE_LUMO_eV", "dE_HOMO_eV", "confinement_score_eV", "lattice_mismatch_pct"] if c in df.columns]
candidate_level = df.groupby(candidate_id_col)[material_cols].first().reset_index()

if "log10_retention_tau_s" in df.columns:
    ret_means = df.groupby(candidate_id_col)["log10_retention_tau_s"].mean().values
    candidate_level["mean_log_tau"] = ret_means
    candidate_level["retention_bin"] = pd.qcut(candidate_level["mean_log_tau"], q=3, labels=[0, 1, 2])
else:
    candidate_level["retention_bin"] = 0

km_scaler = StandardScaler()
X_cluster = km_scaler.fit_transform(candidate_level[material_cols])
n_clusters = min(6, len(candidate_level) // 5)
km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
candidate_level["cluster"] = km.fit_predict(X_cluster)

rng = np.random.default_rng(RANDOM_STATE)
test_candidates = []
for _, bin_group in candidate_level.groupby("retention_bin"):
    unique_clusters = bin_group["cluster"].unique()
    rng.shuffle(unique_clusters)
    n_test_in_bin = max(1, round(len(bin_group) * 0.25))
    count = 0
    for c in unique_clusters:
        c_cands = bin_group[bin_group["cluster"] == c][candidate_id_col].tolist()
        test_candidates.extend(c_cands)
        count += len(c_cands)
        if count >= n_test_in_bin:
            break

candidate_level["split"] = np.where(candidate_level[candidate_id_col].isin(test_candidates), "test", "train")
df = df.merge(candidate_level[[candidate_id_col, "split"]], on=candidate_id_col, how="left")

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

# ---------------------------------------------------------------
# 3. PyTorch Residual MLP Architecture
# ---------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ---------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]

feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "exp_dE_LUMO", "exp_dE_HOMO",
    "shell_thick_nm", "core_radius_nm", "Nt_cm3", "log10_Nt",
    "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

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
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()

    y_true_flat = y_test.ravel()
    r2 = r2_score(y_true_flat, preds)
    mae = mean_absolute_error(y_true_flat, preds)
    print(f"[{target_col}] NN Evaluation -> R^2 = {r2:.4f} | MAE = {mae:.4f}")

    # Save outputs for parity plotting
    out = test_df[[candidate_id_col]].copy()
    out["y_true"] = y_true_flat
    out["y_pred"] = preds
    out.to_csv(f"csv/predictions_nn_{target_col}.csv", index=False)
    torch.save(model.state_dict(), f"nn_model_{target_col}.pt")

print("\nDone. Neural network training complete.")