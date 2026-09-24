"""
Tier 2 fix: formalizes the physics-side monotonicity checks from
Publication possibilities_figures.docx as a saved, reproducible script instead
of one-off analysis in conversation. These are the "does the data behave the way
band-engineering/tunneling theory says it should" sanity checks -- necessary
before any physics-trend claim goes in a paper.

Each check: bin the driver variable into 10 quantile bins, compute the target's
mean per bin, and report Spearman rank correlation (a direct monotonicity
measure, unlike Pearson which tests linearity) between the raw driver and target.
A check "passes" if the sign of the Spearman correlation matches the
theoretically-expected direction and |rho| clears a small threshold (0.05) --
this is deliberately loose since the point is catching a *reversed* relationship,
not requiring a strong one.

Writes csv/physics_sanity_checks.csv and plots/physics_sanity_checks.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import load_dataset

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

RHO_THRESHOLD = 0.05
N_BINS = 10

df = load_dataset()
df["log10_Nt"] = np.log10(df["Nt_cm3"].clip(lower=1e10))
df["abs_dE_LUMO"] = df["dE_LUMO_eV"].abs()
df["abs_dE_HOMO"] = df["dE_HOMO_eV"].abs()

# |dE_LUMO_eV| / |dE_HOMO_eV| is only a confining barrier height for Type I pairs, where
# both offsets share a sign and the shell (or core) straddles the other's gap. For Type
# II/III the offsets have mixed sign or near-zero means, so |offset| is not a barrier in
# the confinement sense -- a barrier-height check run on all rows mixes physically
# different quantities. An earlier version of this script did exactly that; its
# conclusion survives restriction to Type I, but the check is only meaningful there.
IS_TYPE_I = df["band_alignment"].str.startswith("Type I ")
SUBSETS = {"all": df, "type_I": df[IS_TYPE_I], "type_II_III": df[~IS_TYPE_I]}

# (driver column, target column, expected sign, row subset, human-readable description)
CHECKS = [
    ("log10_Nt", "hysteresis_window_V", +1, "all",
     "hysteresis_window_V should increase with trap density (more traps -> more charge stored per sweep)"),
    ("abs_dE_LUMO", "hysteresis_window_V", +1, "type_I",
     "Type I only: hysteresis_window_V should increase with electron barrier height |dE_LUMO_eV| "
     "(deeper trap -> slower detrapping -> wider hysteresis)"),
    ("abs_dE_HOMO", "hysteresis_window_V", +1, "type_I",
     "Type I only: hysteresis_window_V should increase with hole barrier height |dE_HOMO_eV|, same mechanism"),
    ("shell_thick_nm", "log10_on_off_ratio", -1, "all",
     "on_off_ratio should fall as shell_thick_nm grows (thicker tunneling barrier -> less charge transfer)"),
    ("shell_thick_nm", "log10_retention_tau_s", +1, "all",
     "retention_tau_s should rise as shell_thick_nm grows (thicker barrier -> harder for charge to escape)"),
]

# Not pass/fail: physics expects |offset| to matter much LESS for Type II/III (no confining
# barrier), so a correlation as strong as Type I's is itself evidence that whatever generated
# this data treats |offset| as a barrier regardless of alignment type -- which bears on the
# inverted band_alignment ordering in the last panel.
DIAGNOSTICS = [
    ("abs_dE_LUMO", "hysteresis_window_V", "type_II_III",
     "Type II/III only: |dE_LUMO_eV| is not a confining barrier here; physics expects a weaker dependence than Type I"),
    ("abs_dE_HOMO", "hysteresis_window_V", "type_II_III",
     "Type II/III only: |dE_HOMO_eV| is not a confining barrier here; physics expects a weaker dependence than Type I"),
]


def spearman_rank_corr(x, y):
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    return rx.corr(ry)


def binned_trend(x, y, n_bins=N_BINS):
    bins = pd.qcut(x, q=n_bins, duplicates="drop")
    return pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin", observed=True)["y"].agg(["mean", "count"])


results = []
fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
axes_flat = axes.flatten()

for i, (driver, target, expected_sign, subset, description) in enumerate(CHECKS):
    sub = SUBSETS[subset][[driver, target]].dropna()
    rho = spearman_rank_corr(sub[driver], sub[target])
    passed = (np.sign(rho) == expected_sign) and (abs(rho) >= RHO_THRESHOLD)
    results.append({
        "kind": "check", "driver": driver, "target": target, "subset": subset, "n_rows": len(sub),
        "expected_sign": expected_sign, "spearman_rho": rho, "passed": passed, "description": description,
    })
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {target} vs {driver} [{subset}, n={len(sub)}]: Spearman rho = {rho:+.3f} "
          f"(expected sign {expected_sign:+d})")
    print(f"       {description}")

    trend = binned_trend(sub[driver], sub[target])
    ax = axes_flat[i]
    ax.plot(range(len(trend)), trend["mean"], marker="o", color="#3B82F6" if passed else "#EF4444")
    ax.set_title(f"{target}\nvs {driver} [{subset}] (rho={rho:+.2f}, {status})", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{driver} decile (low -> high)", fontsize=8)
    ax.set_ylabel(f"mean {target}", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

# Sixth panel: the band_alignment ordering check carried over from the earlier
# investigation (kept here so all physics-sanity checks live in one artifact).
ax = axes_flat[5]
order = ["Type I (straddling, core confines shell)", "Type I (straddling, shell confines core)",
         "Type II (staggered)", "Type III (broken gap)"]
means = df.groupby("band_alignment")["hysteresis_window_V"].mean().reindex(order)
colors = ["#EF4444"] * len(means)  # flagged red: this ordering is known-inverted, not a pass/fail bar chart
ax.bar(range(len(means)), means.values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(means)))
ax.set_xticklabels(["Type I\n(core confines)", "Type I\n(shell confines)", "Type II", "Type III"], fontsize=7)
ax.set_title("hysteresis_window_V by band_alignment\n(increasing toward Type III -- INVERTED vs theory)",
             fontsize=9, fontweight="bold", color="#B91C1C")
ax.set_ylabel("mean hysteresis_window_V", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)
results.append({
    "kind": "check", "driver": "band_alignment", "target": "hysteresis_window_V", "subset": "all",
    "n_rows": len(df), "expected_sign": -1, "spearman_rho": np.nan, "passed": False,
    "description": "Type III (least confinement) should show the LEAST hysteresis, not the most -- "
                   "confirmed inverted (see CLAUDE.md 'Verified data/pipeline caveats'); data provenance is "
                   "AI-assisted Materials Project extraction, mechanism still unconfirmed.",
})

plt.tight_layout()
fig.savefig("plots/physics_sanity_checks.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nDiagnostics (not pass/fail):")
for driver, target, subset, description in DIAGNOSTICS:
    sub = SUBSETS[subset][[driver, target]].dropna()
    rho = spearman_rank_corr(sub[driver], sub[target])
    type_i_rho = next(r["spearman_rho"] for r in results
                      if r["driver"] == driver and r["target"] == target and r["subset"] == "type_I")
    results.append({
        "kind": "diagnostic", "driver": driver, "target": target, "subset": subset, "n_rows": len(sub),
        "expected_sign": np.nan, "spearman_rho": rho, "passed": np.nan, "description": description,
    })
    print(f"  {target} vs {driver} [{subset}, n={len(sub)}]: rho = {rho:+.3f}  (Type I: {type_i_rho:+.3f})")
    print(f"       {description}")

results_df = pd.DataFrame(results)
results_df.to_csv("csv/physics_sanity_checks.csv", index=False)

checks = results_df[results_df["kind"] == "check"]
# astype(bool) is required: diagnostic rows put NaN in "passed", making it an object column, and
# summing numpy bools inside an object column ORs them (True + True == True) instead of counting.
n_pass = int(checks["passed"].astype(bool).sum())
print(f"\n{n_pass}/{len(checks)} checks passed. Wrote csv/physics_sanity_checks.csv and "
      f"plots/physics_sanity_checks.png")
