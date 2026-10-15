"""
Methodology-paper step 3: known-equation recovery test.

The faculty dataset's labels come from an unknown generator, so recovering
"an equation" from them can't show the method finds the RIGHT equation. Here the
truth is known. Labels for curated_dataset/candidates.csv (cited, physical material
values; no labels of its own) are generated from three equations stated below,
before any fitting, plus Gaussian noise.

Ground truths (kappa*t*sqrt(B) is the WKB exponent defined in symbolic_model.py;
typeI = 1 for Type I pairs, else 0; Bmin/Bmax = min/max(|dEc|, |dEv|)):

  retention, IN library:
      log10 tau = -2 + 1.0*typeI + 0.137*typeI*wkb_min
      Escape by tunneling through the lower confining barrier: log10 of the WKB
      transmission is wkb/ln10, scaled by sqrt(m*/m_e) = sqrt(0.1) for an
      effective mass of 0.1 m_e (0.4343 * 0.316 = 0.137). Without a confining
      barrier (Type II/III) charge leaks out fast (tau = 10 ms).
  on/off ratio, IN library:
      log10 on/off = -2.5 + 0.30*lgNt - 0.03*wkb_max
      ON current grows with trap density; the tunneling penalty through the
      larger barrier lowers it (so on/off falls with shell thickness).
  hysteresis, NOT in library:
      dV = Vmax * x / (1 + x),   x = q*Nt*t^2 / (2*eps0*eps*Vmax)
      Trapped space charge q*Nt in a shell of thickness t shifts the voltage by
      q*Nt*t^2/(2*eps0*eps), saturating at the sweep amplitude. It is linear in
      Nt, but the library only contains log10(Nt) -- this tests what the method
      does when the true law is outside its search space.

All three truths satisfy the project's physics sign constraints.

Part A (recovery): for each noise level (sigma = eta * std of the clean label) and
10 seeds, fit on a 75% candidate-grouped train split and check whether the
equation recovers the true law: same number of terms, and every true term in the
linear span of the selected descriptors (see recovered()). Also reports test R^2
against the CLEAN labels.

Note on the hysteresis truth's magnitude: with physical constants and this data's
ranges, x << 1 almost everywhere, so dV spans only 0-0.08 V and is effectively
linear in Nt (the saturation rarely engages). It still serves its purpose -- a law
outside the library -- but it is not on the faculty data's 0-3.2 V scale.

Part B (extrapolation where the answer is known): eta = 0.1, the Nt / barrier /
thickness range splits, symbolic vs XGBoost vs NN, scored against the clean labels.

Writes csv/recovery_test.csv and csv/recovery_extrapolation.csv; the figure is
scripts/plotting/plot_recovery_test.py.
"""

import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from Prepare_real_candidates import RENAME_MAP
from split_utils import feature_range_split
from symbolic_model import primaries, build_library, WKB_KAPPA, TYPE_I_COLS
from model_fits import MODELS, fit_sr

CURATED = "curated_dataset/candidates.csv"
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4]
SEEDS = list(range(10))
EXTRAP_NOISE = 0.1
Q, EPS0 = 1.602176634e-19, 8.8541878128e-12

BAND_ALIGN_COLS = {
    "Type I (straddling, core confines shell)": "band_align_Type_I_straddling_core_confines_shell",
    "Type I (straddling, shell confines core)": "band_align_Type_I_straddling_shell_confines_core",
    "Type II (staggered)": "band_align_Type_II_staggered",
    "Type III (broken gap)": "band_align_Type_III_broken_gap",
}

# ---------------------------------------------------------------- data + ground truth
df = pd.read_csv(CURATED).rename(columns=RENAME_MAP)
df["candidate_id"] = df["core_material"].astype(str) + "__" + df["shell_material"].astype(str)
df["formula_core"], df["formula_shell"] = df["core_material"], df["shell_material"]
for label, col in BAND_ALIGN_COLS.items():
    df[col] = (df["band_alignment"] == label).astype(int)
df["max_abs_offset_eV"] = np.maximum(df["dE_LUMO_eV"].abs(), df["dE_HOMO_eV"].abs())

prims = primaries(df)
P = {k: v[0] for k, v in prims.items()}
x = Q * (df["Nt_cm3"] * 1e6) * (df["shell_thick_nm"] * 1e-9) ** 2 / (2 * EPS0 * df["eps_shell"] * df["Vmax_V"])
TRUTH = {
    "log10_retention_tau_s": {"clean": -2 + 1.0 * P["typeI"] + 0.137 * P["typeI"] * P["wkb_min"],
                              "terms": ["typeI", "typeI*wkb_min"], "coefs": [1.0, 0.137],
                              "in_library": True},
    "log10_on_off_ratio": {"clean": -2.5 + 0.30 * P["lgNt"] - 0.03 * P["wkb_max"],
                           "terms": ["lgNt", "wkb_max"], "coefs": [0.30, -0.03], "in_library": True},
    "hysteresis_window_V": {"clean": (df["Vmax_V"] * x / (1 + x)).to_numpy(),
                            "terms": None, "coefs": None, "in_library": False},
}
library = {f.name: f for f in build_library(prims)}
for t in TRUTH.values():
    assert t["terms"] is None or all(n in library for n in t["terms"]), t["terms"]
for target, t in TRUTH.items():
    c = t["clean"]
    print(f"{target:22s} clean label range {c.min():.3f}..{c.max():.3f}  (std {c.std():.3f})")


def with_labels(frame, eta, seed):
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for target, t in TRUTH.items():
        clean = np.asarray(t["clean"])[frame.index]
        out[target + "__clean"] = clean
        out[target] = clean + rng.normal(0, eta * np.std(t["clean"]), len(frame))
    return out


def recovered(model, train_df, true_terms):
    """True iff the equation has as many terms as the truth AND every true term lies in the
    span of the selected descriptors (R^2 > 0.999 regressing it on them plus an intercept).
    Span, not one-to-one name matching: a linear model in {lgNt - wkb_max, lgNt} is the same
    family of functions as one in {lgNt, wkb_max}, and the library contains both forms."""
    if true_terms is None or model.n_terms_ != len(true_terms):
        return False
    p = primaries(train_df)
    S = np.column_stack([np.ones(len(train_df))] + [model.lib_[i].evaluate(p) for i in model.model_["idx"]])
    for name in true_terms:
        tv = library[name].evaluate(p)
        resid = tv - S @ np.linalg.lstsq(S, tv, rcond=None)[0]
        if 1 - resid.var() / tv.var() < 0.999:
            return False
    return True


# ---------------------------------------------------------------- Part A: recovery vs noise
rows = []
t0 = time.perf_counter()
for eta in NOISE_LEVELS:
    for seed in SEEDS:
        data = with_labels(df, eta, seed)
        tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                              .split(data, groups=data["candidate_id"]))
        train_df, test_df = data.iloc[tr_idx], data.iloc[te_idx]
        for target, t in TRUTH.items():
            pred, extra = fit_sr(train_df, test_df, target, seed)
            model = extra["sr_model"]
            row = {"noise_eta": eta, "seed": seed, "target": target, "truth_in_library": t["in_library"],
                   "recovered": recovered(model, train_df, t["terms"]),
                   "n_terms": model.n_terms_, "terms": extra["sr_terms"], "equation": extra["sr_equation"],
                   "r2_vs_clean": r2_score(test_df[target + "__clean"], pred),
                   "mae_vs_clean": mean_absolute_error(test_df[target + "__clean"], pred)}
            if t["terms"] is not None and set(model.terms()) == set(t["terms"]):
                _, coef = model.raw_coefficients()
                fitted = dict(zip(model.terms(), coef))
                row["max_rel_coef_error"] = float(max(abs(fitted[n] - c) / abs(c)
                                                      for n, c in zip(t["terms"], t["coefs"])))
            rows.append(row)
    print(f"[{time.perf_counter() - t0:4.0f}s] eta={eta}: " + "  ".join(
        f"{tg.split('_')[0] if tg != 'log10_on_off_ratio' else 'onoff'} recovered "
        f"{sum(r['recovered'] for r in rows if r['noise_eta'] == eta and r['target'] == tg)}/{len(SEEDS)}"
        for tg in TRUTH))
recov = pd.DataFrame(rows)
recov.to_csv("csv/recovery_test.csv", index=False)

print("\nMost common term set per (target, noise):")
for (target, eta), g in recov.groupby(["target", "noise_eta"]):
    top = g.terms.value_counts()
    print(f"  {target:22s} eta={eta:<4}  {top.index[0]}  [{top.iloc[0]}/{len(g)}]   "
          f"recovered {g.recovered.mean():.0%}  R2 vs clean {g.r2_vs_clean.mean():.3f}")
first = recov[(recov.noise_eta == 0.1) & (recov.seed == 0)]
print("\nExample equations at eta=0.1, seed 0:")
for _, r in first.iterrows():
    print(f"  {r.equation}")

# ---------------------------------------------------------------- Part B: extrapolation with a known answer
data = with_labels(df, EXTRAP_NOISE, 0)
splits = {
    "Nt range (top 25%)": partial(feature_range_split, column="Nt_cm3"),
    "barrier range (top 25%)": partial(feature_range_split, column="max_abs_offset_eV"),
    "thickness range (top 25%)": partial(feature_range_split, column="shell_thick_nm"),
}
ext_rows = []
for split_name, split_fn in splits.items():
    train_df, test_df = split_fn(data)
    for target in TRUTH:
        y_clean = test_df[target + "__clean"].to_numpy()
        for model_name, fit in MODELS.items():
            pred, extra = fit(train_df, test_df, target, 0)
            ext_rows.append({"strategy": split_name, "target": target, "model": model_name,
                             "r2_vs_clean": r2_score(y_clean, pred), "mae_vs_clean": mean_absolute_error(y_clean, pred),
                             "sr_terms": extra.get("sr_terms", ""), "n_train_rows": len(train_df),
                             "n_test_rows": len(test_df)})
    print(f"[{time.perf_counter() - t0:4.0f}s] {split_name} done")
ext = pd.DataFrame(ext_rows)
ext.to_csv("csv/recovery_extrapolation.csv", index=False)
print("\n" + ext.pivot_table(index=["target", "strategy"], columns="model", values="r2_vs_clean", sort=False)
      .round(3).to_string())

print("\nWrote csv/recovery_test.csv and csv/recovery_extrapolation.csv "
      "(figure: python scripts/plotting/plot_recovery_test.py)")
