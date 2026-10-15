"""
Methodology-paper step 4: 80% prediction intervals for the symbolic-regression
equations, compared with XGBoost quantile regression (uncertainty_quantification.py's
method), on the splits where calibration actually gets tested.

Three interval methods, all at nominal 80% coverage:

  SR conformal   Split conformal: 20% of the TRAINING candidates (grouped by
                 candidate_id) are held out as a calibration set; the equation is fit
                 on the rest and the interval is prediction +/- the finite-sample
                 corrected 80% quantile of |calibration residual|. Distribution-free
                 coverage, but only under exchangeability -- it is NOT promised under
                 extrapolation, which is exactly what the range splits test.
  SR bootstrap   Candidate bootstrap: B refits on candidate-resampled training sets,
                 each re-running SIS + L0 term selection (number of terms fixed at the
                 full fit's choice), so model-selection uncertainty is included. Each
                 refit's prediction is paired with a residual drawn from that refit's
                 out-of-bag rows; the interval is the 10th-90th percentile of those draws.
  XGB quantile   reg:quantileerror with alpha = 0.1/0.5/0.9, same hyperparameters as
                 kmeans_xgboost_train.py (the existing Tier 2 approach).

Splits: grouped-by-pair and leave-materials-out (seeds 0..9) and the three range
splits. CLAUDE.md flags that XGB's retention coverage (0.632 at seed 42, grouped)
had never been checked under leave-materials-out; this does that too.

Writes csv/symbolic_uncertainty.csv (one row per split x target x method) and
csv/symbolic_uncertainty_summary.csv; figure by
scripts/plotting/plot_symbolic_uncertainty.py.
"""

import time
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import load_dataset, candidate_split, leave_materials_out_split, feature_range_split
from symbolic_model import SparseSymbolicRegressor, CONSTRAINTS
from model_fits import xgb_features

SEEDS = list(range(10))
TARGETS = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]
ALPHA = 0.2  # 1 - nominal coverage
N_BOOT = 30
CAL_FRAC = 0.2
OUT_CSV = "csv/symbolic_uncertainty.csv"

df = load_dataset()
df["max_abs_offset_eV"] = np.maximum(df["dE_LUMO_eV"].abs(), df["dE_HOMO_eV"].abs())

STRATEGIES = {
    "grouped-by-pair": (candidate_split, SEEDS),
    "leave-materials-out": (leave_materials_out_split, SEEDS),
    "Nt range (top 25%)": (partial(feature_range_split, column="Nt_cm3"), [0]),
    "barrier range (top 25%)": (partial(feature_range_split, column="max_abs_offset_eV"), [0]),
    "thickness range (top 25%)": (partial(feature_range_split, column="shell_thick_nm"), [0]),
}


def sr(target):
    return SparseSymbolicRegressor(constraints=CONSTRAINTS[target])


def conformal_interval(train_df, test_df, target, seed):
    fit_idx, cal_idx = next(GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=seed)
                            .split(train_df, groups=train_df["candidate_id"]))
    fit_df, cal_df = train_df.iloc[fit_idx], train_df.iloc[cal_idx]
    model = sr(target).fit(fit_df, fit_df[target], groups=fit_df["candidate_id"])
    scores = np.abs(cal_df[target].to_numpy() - model.predict(cal_df))
    n = len(scores)
    q = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
    pred = model.predict(test_df)
    return pred - q, pred, pred + q


def bootstrap_interval(train_df, test_df, target, seed):
    full = sr(target).fit(train_df, train_df[target], groups=train_df["candidate_id"])
    rng = np.random.default_rng(seed)
    cands = train_df["candidate_id"].unique()
    rows_of = train_df.groupby("candidate_id").indices
    draws = []
    for _ in range(N_BOOT):
        picked = rng.choice(cands, size=len(cands), replace=True)
        idx = np.concatenate([rows_of[c] for c in picked])
        oob = np.setdiff1d(np.arange(len(train_df)), idx)
        boot_df = train_df.iloc[idx]
        m = sr(target).fit(boot_df, boot_df[target], n_terms=full.n_terms_)
        oob_df = train_df.iloc[oob]
        resid = oob_df[target].to_numpy() - m.predict(oob_df)
        draws.append(m.predict(test_df)[:, None] + rng.choice(resid, size=(len(test_df), 20)))
    draws = np.concatenate(draws, axis=1)
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=1)
    return lo, full.predict(test_df), hi


def xgb_quantile_interval(train_df, test_df, target, seed):
    cols = xgb_features(train_df)
    model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=[ALPHA / 2, 0.5, 1 - ALPHA / 2],
                             n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.7, random_state=seed)
    model.fit(train_df[cols], train_df[target])
    lo, mid, hi = model.predict(test_df[cols]).T
    return lo, mid, hi


METHODS = {"SR conformal": conformal_interval, "SR bootstrap": bootstrap_interval,
           "XGB quantile": xgb_quantile_interval}

rows = []
t0 = time.perf_counter()
for strat_name, (split_fn, seeds) in STRATEGIES.items():
    for seed in seeds:
        train_df, test_df = split_fn(df, random_state=seed)
        for target in TARGETS:
            y = test_df[target].to_numpy()
            for method_name, method in METHODS.items():
                lo, mid, hi = method(train_df, test_df, target, seed)
                inside = (y >= lo) & (y <= hi)
                rows.append({"strategy": strat_name, "seed": seed, "target": target, "method": method_name,
                             "coverage": inside.mean(), "mean_width": float(np.mean(hi - lo)),
                             "median_mae": float(np.mean(np.abs(y - mid))),
                             # where misses fall: above the interval vs below it
                             "frac_above": float(np.mean(y > hi)), "frac_below": float(np.mean(y < lo)),
                             "n_test_rows": len(test_df)})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        last = pd.DataFrame(rows)
        last = last[(last.strategy == strat_name) & (last.seed == seed)]
        print(f"[{time.perf_counter() - t0:5.0f}s] {strat_name} seed {seed}: " + "  ".join(
            f"{t.split('_')[0] if t != 'log10_on_off_ratio' else 'onoff'}/"
            + "/".join(f"{c:.2f}" for c in last[last.target == t].coverage) for t in TARGETS)
            + "   (coverage conformal/bootstrap/xgbq)")

results = pd.DataFrame(rows)
summary = (results.groupby(["strategy", "target", "method"], sort=False)
           .agg(coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
                width_mean=("mean_width", "mean"), median_mae=("median_mae", "mean"),
                frac_above=("frac_above", "mean"), frac_below=("frac_below", "mean"),
                n_seeds=("seed", "nunique"))
           .reset_index())
summary.to_csv("csv/symbolic_uncertainty_summary.csv", index=False)
print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nWrote csv/symbolic_uncertainty.csv and csv/symbolic_uncertainty_summary.csv "
      "(figure: python scripts/plotting/plot_symbolic_uncertainty.py)")
