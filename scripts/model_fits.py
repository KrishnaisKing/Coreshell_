"""
The three model families compared in the methodology paper, each as
fit(train_df, test_df, target, seed) -> (test predictions, extra info dict),
shared by symbolic_split_comparison.py and recovery_test.py so every comparison
fits the models identically:

  fit_sr   physics-constrained symbolic regression (symbolic_model.py)
  fit_xgb  XGBoost with split_strategy_comparison.py's fixed features/hyperparameters
  fit_nn   TabularResNet with nn.py's exact preprocessing and training loop
"""

import numpy as np
import torch
import xgboost as xgb

from nn_model import TORCH_THREADS, add_engineered_features, feature_columns, inner_split_and_scaler, \
    train_tabular_resnet, predict
from symbolic_model import SparseSymbolicRegressor, CONSTRAINTS

torch.set_num_threads(TORCH_THREADS)

XGB_BASE_FEATURES = ["dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
                     "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct"]


def xgb_features(df):
    return [c for c in XGB_BASE_FEATURES if c in df.columns] + \
        [c for c in df.columns if c.startswith("band_align_")]


def fit_xgb(train_df, test_df, target, seed):
    cols = xgb_features(train_df)
    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.7, random_state=seed)
    model.fit(train_df[cols], train_df[target])
    return model.predict(test_df[cols]), {}


def fit_nn(train_df, test_df, target, seed):
    train_df, test_df = add_engineered_features(train_df.copy()), add_engineered_features(test_df.copy())
    cols = feature_columns(train_df)
    inner_train_df, val_df, scaler = inner_split_and_scaler(train_df, cols, seed)
    torch.manual_seed(seed)
    model, best_epoch, _, _ = train_tabular_resnet(
        scaler.transform(inner_train_df[cols].values), inner_train_df[target].values.astype(np.float32),
        scaler.transform(val_df[cols].values), val_df[target].values.astype(np.float32),
        shuffle_seed=seed)
    return predict(model, scaler.transform(test_df[cols].values)), {"nn_best_epoch": best_epoch}


def fit_sr(train_df, test_df, target, seed, constraints=None):
    """constraints=None uses CONSTRAINTS[target] (the project's expected physics signs)."""
    model = SparseSymbolicRegressor(constraints=CONSTRAINTS.get(target, []) if constraints is None else constraints)
    model.fit(train_df, train_df[target], groups=train_df["candidate_id"])
    return model.predict(test_df), {"sr_n_terms": model.n_terms_, "sr_terms": " | ".join(model.terms()),
                                    "sr_constraints_satisfied": model.constraints_satisfied(),
                                    "sr_equation": model.equation(target=target), "sr_model": model}


MODELS = {"symbolic (ours)": fit_sr, "XGBoost": fit_xgb, "NN": fit_nn}
