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

from split_utils import load_dataset

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

RHO_THRESHOLD = 0.05
N_BINS = 10

df = load_dataset()
df["log10_Nt"] = np.log10(df["Nt_cm3"].clip(lower=1e10))
df["abs_dE_LUMO"] = df["dE_LUMO_eV"].abs()
df["abs_dE_HOMO"] = df["dE_HOMO_eV"].abs()

# (driver column, target column, expected sign, human-readable description)
CHECKS = [
    ("log10_Nt", "hysteresis_window_V", +1,
     "hysteresis_window_V should increase with trap density (more traps -> more charge stored per sweep)"),
    ("abs_dE_LUMO", "hysteresis_window_V", +1,
     "hysteresis_window_V should increase with electron barrier height |dE_LUMO_eV| (deeper trap -> slower "
     "detrapping -> wider hysteresis)"),
    ("abs_dE_HOMO", "hysteresis_window_V", +1,
     "hysteresis_window_V should increase with hole barrier height |dE_HOMO_eV|, same mechanism"),
    ("shell_thick_nm", "log10_on_off_ratio", -1,
     "on_off_ratio should fall as shell_thick_nm grows (thicker tunneling barrier -> less charge transfer)"),
    ("shell_thick_nm", "log10_retention_tau_s", +1,
     "retention_tau_s should rise as shell_thick_nm grows (thicker barrier -> harder for charge to escape)"),
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

for i, (driver, target, expected_sign, description) in enumerate(CHECKS):
    sub = df[[driver, target]].dropna()
    rho = spearman_rank_corr(sub[driver], sub[target])
    passed = (np.sign(rho) == expected_sign) and (abs(rho) >= RHO_THRESHOLD)
    results.append({
        "driver": driver, "target": target, "expected_sign": expected_sign,
        "spearman_rho": rho, "passed": passed, "description": description,
    })
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {target} vs {driver}: Spearman rho = {rho:+.3f} (expected sign {expected_sign:+d})")
    print(f"       {description}")

    trend = binned_trend(sub[driver], sub[target])
    ax = axes_flat[i]
    ax.plot(range(len(trend)), trend["mean"], marker="o", color="#3B82F6" if passed else "#EF4444")
    ax.set_title(f"{target}\nvs {driver} (rho={rho:+.2f}, {status})", fontsize=9, fontweight="bold")
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
    "driver": "band_alignment", "target": "hysteresis_window_V", "expected_sign": -1,
    "spearman_rho": np.nan, "passed": False,
    "description": "Type III (least confinement) should show the LEAST hysteresis, not the most -- "
                   "confirmed inverted (see CLAUDE.md 'Verified data/pipeline caveats'); blocked on "
                   "asking faculty how the data was generated, not fixable from code/data alone.",
})

plt.tight_layout()
fig.savefig("plots/physics_sanity_checks.png", dpi=150, bbox_inches="tight")
plt.close(fig)

results_df = pd.DataFrame(results)
results_df.to_csv("csv/physics_sanity_checks.csv", index=False)

n_pass = results_df["passed"].sum()
print(f"\n{n_pass}/{len(results_df)} checks passed. Wrote csv/physics_sanity_checks.csv and "
      f"plots/physics_sanity_checks.png")
