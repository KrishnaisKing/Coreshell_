"""
Methodology-paper step 1: fit the physics-constrained sparse symbolic regressor
(scripts/symbolic_model.py) on the pipeline's default candidate_split() at
RANDOM_STATE, one equation per target, and report the equation next to the best
equation found *without* the physics constraints (so it's visible whether the
constraints changed anything).

Writes csv/symbolic_equations.csv and csv/predictions_sr_{target}.csv (same
candidate_id/y_true/y_pred layout as predictions_{xgb,nn}_*.csv).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import load_dataset, candidate_split
from symbolic_model import SparseSymbolicRegressor, CONSTRAINTS

TARGETS = ["hysteresis_window_V", "log10_on_off_ratio", "log10_retention_tau_s"]

df = load_dataset()
train_df, test_df = candidate_split(df)

rows = []
for target in TARGETS:
    model = SparseSymbolicRegressor(constraints=CONSTRAINTS[target])
    model.fit(train_df, train_df[target], groups=train_df["candidate_id"])
    pred = model.predict(test_df)
    y = test_df[target].to_numpy()
    m = model.model_
    unconstrained = {"idx": m["unconstrained_idx"]}
    r2, mae = r2_score(y, pred), mean_absolute_error(y, pred)

    print(f"\n[{target}]  {model.n_terms_} terms (inner-CV RMSE by d: "
          + ", ".join(f"{d}:{v[0]:.3f}" for d, v in model.cv_rmse_.items()) + ")")
    print(f"  {model.equation(target=target)}")
    print(f"  test R^2 = {r2:.4f}  MAE = {mae:.4f}")
    print(f"  constraints: " + ", ".join(f"{k} on {frac:.1%} of rows" + ("" if active else " (driver absent from equation)")
                                      for k, (frac, active) in m["constraint_report"].items())
          + (f"  (best unconstrained subset {'was' if m['constraint_rank'] == 0 else 'was NOT'} admissible)"
             if model.constraints_satisfied() else "  ** NO admissible subset in top candidates **"))
    if m["constraint_rank"] != 0:
        print(f"  unconstrained best terms: {[model.lib_[i].name for i in unconstrained['idx']]}")

    rows.append({
        "target": target,
        "n_terms": model.n_terms_,
        "equation": model.equation(target=target),
        "terms": " | ".join(model.terms()),
        "test_r2": r2,
        "test_mae": mae,
        "constraints_satisfied": model.constraints_satisfied(),
        "constraint_rank": m["constraint_rank"],
        "constraint_report": "; ".join(f"{k}: {frac:.3f}" + ("" if active else " (driver absent)")
                                      for k, (frac, active) in m["constraint_report"].items()),
        "unconstrained_terms": " | ".join(model.lib_[i].name for i in unconstrained["idx"]),
        **{f"inner_cv_rmse_d{d}": v[0] for d, v in model.cv_rmse_.items()},
        "library_size": len(model.lib_),
    })

    out = test_df[["candidate_id"]].copy()
    out["y_true"] = y
    out["y_pred"] = pred
    out.to_csv(f"csv/predictions_sr_{target}.csv", index=False)

pd.DataFrame(rows).to_csv("csv/symbolic_equations.csv", index=False)
print("\nWrote csv/symbolic_equations.csv and csv/predictions_sr_*.csv")
