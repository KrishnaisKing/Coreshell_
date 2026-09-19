"""
TabularResNet architecture, NN feature engineering and permutation importance,
shared by nn.py (training) and nn_permutation_importance.py (post-hoc analysis
of saved nn_model_*.pt files).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

BASE_FEATURE_COLS = [
    "dE_LUMO_eV", "dE_HOMO_eV", "exp_dE_LUMO", "exp_dE_HOMO",
    "shell_thick_nm", "core_radius_nm", "Nt_cm3", "log10_Nt",
    "eps_shell", "Vmax_V", "lattice_mismatch_pct",
]


def add_engineered_features(df):
    if "dE_LUMO_eV" in df.columns:
        df["exp_dE_LUMO"] = np.exp(df["dE_LUMO_eV"].clip(upper=5.0))
    if "dE_HOMO_eV" in df.columns:
        df["exp_dE_HOMO"] = np.exp(df["dE_HOMO_eV"].clip(upper=5.0))
    if "Nt_cm3" in df.columns:
        df["log10_Nt"] = np.log10(df["Nt_cm3"].clip(lower=1e10))
    return df


def feature_columns(df):
    band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
    return [c for c in BASE_FEATURE_COLS if c in df.columns] + band_align_cols


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


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()


def permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=0):
    """
    Mean drop in test R^2 when each feature column is shuffled. Returns a DataFrame
    with columns feature, importance_mean, importance_std sorted descending.
    """
    import pandas as pd
    rng = np.random.default_rng(random_state)
    base = r2_score(y, predict(model, X))
    rows = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - r2_score(y, predict(model, Xp)))
        rows.append({"feature": name, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
