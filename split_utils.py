"""
Shared data loading and candidate-grouped train/test split used by
kmeans_xgboost_train.py, nn.py and nn_permutation_importance.py.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = "csv/rs_training_data_real_candidates.csv"

MATERIAL_COLS = ["dE_LUMO_eV", "dE_HOMO_eV", "confinement_score_eV", "lattice_mismatch_pct"]

TARGETS = {
    "hysteresis_window_V": "hysteresis_window_V",
    "log10_on_off_ratio": "on_off_ratio",
    "log10_retention_tau_s": "retention_tau_s",
}


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = df["formula_core"].astype(str) + "__" + df["formula_shell"].astype(str)
    for target_name, fallback_col in TARGETS.items():
        if target_name not in df.columns and fallback_col in df.columns:
            df[f"log10_{fallback_col}"] = np.log10(df[fallback_col].clip(lower=1e-12))
    return df


def candidate_split(df, test_frac=0.25, random_state=RANDOM_STATE):
    """
    Candidate-grouped, retention-stratified split. Whole (core, shell) pairs are
    held out together so no candidate's device-parameter rows leak across the split.
    Returns (train_df, test_df).
    """
    material_cols = [c for c in MATERIAL_COLS if c in df.columns]
    if not material_cols:
        raise SystemExit(f"None of material columns found. Available: {df.columns.tolist()}")

    candidate_level = df.groupby("candidate_id")[material_cols].first().reset_index()

    if "log10_retention_tau_s" in df.columns:
        candidate_level["mean_log_tau"] = df.groupby("candidate_id")["log10_retention_tau_s"].mean().values
        candidate_level["retention_bin"] = pd.qcut(candidate_level["mean_log_tau"], q=3, labels=[0, 1, 2])
    else:
        candidate_level["retention_bin"] = 0

    X_cluster = StandardScaler().fit_transform(candidate_level[material_cols])
    n_clusters = min(6, len(candidate_level) // 5)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    candidate_level["cluster"] = km.fit_predict(X_cluster)

    rng = np.random.default_rng(random_state)
    test_candidates = []
    for _, bin_group in candidate_level.groupby("retention_bin"):
        unique_clusters = bin_group["cluster"].unique()
        rng.shuffle(unique_clusters)
        n_test_in_bin = max(1, round(len(bin_group) * test_frac))
        count = 0
        for c in unique_clusters:
            c_cands = bin_group[bin_group["cluster"] == c]["candidate_id"].tolist()
            test_candidates.extend(c_cands)
            count += len(c_cands)
            if count >= n_test_in_bin:
                break

    test_set = set(test_candidates)
    train_df = df[~df["candidate_id"].isin(test_set)].copy()
    test_df = df[df["candidate_id"].isin(test_set)].copy()

    overlap = set(train_df["candidate_id"]) & set(test_df["candidate_id"])
    assert len(overlap) == 0, f"LEAK: {len(overlap)} candidates appear in both splits!"

    return train_df, test_df


def random_row_split(df, test_frac=0.25, random_state=RANDOM_STATE):
    """
    Deliberately leaky baseline: a plain row-level random split with no grouping
    by candidate or material at all. Rows from the same candidate_id (near-duplicate
    device-parameter sweeps of the same material pair) can land on both sides,
    inflating apparent test performance. This exists only to quantify, in the
    split-strategy comparison, how optimistic that inflation is relative to
    candidate_split() and leave_materials_out_split() -- never use it for a result
    meant to represent real generalization.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(df))
    n_test = round(len(df) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def leave_materials_out_split(df, test_frac=0.25, random_state=RANDOM_STATE):
    """
    Holds out entire MATERIALS, not just pairs, so a test-set material never
    appears in training at all, in either the core or shell role. This is
    strictly harder than candidate_split(): that function only guarantees an
    unseen (core, shell) COMBINATION -- a material can still appear in training
    paired with something else. 840 of this dataset's 1444 distinct materials
    appear in both the core and shell role across different rows, so the held-out
    pool is drawn from the union of formula_core/formula_shell, not either alone.

    Because held-out materials pull in every row that references them (in
    whichever role), the resulting test fraction is not exactly test_frac -- it's
    a side effect of how densely the held-out materials happen to be reused
    across candidate pairs, not a tunable target.
    """
    all_materials = pd.unique(pd.concat([df["formula_core"], df["formula_shell"]]))
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(all_materials)
    n_test_materials = round(len(shuffled) * test_frac)
    test_materials = set(shuffled[:n_test_materials])

    is_test_row = df["formula_core"].isin(test_materials) | df["formula_shell"].isin(test_materials)
    test_df = df[is_test_row].copy()
    train_df = df[~is_test_row].copy()

    train_materials = set(train_df["formula_core"]) | set(train_df["formula_shell"])
    overlap = train_materials & test_materials
    assert len(overlap) == 0, f"LEAK: {len(overlap)} materials appear in both splits!"

    return train_df, test_df
