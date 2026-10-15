"""
Tier 3: model vs. real-experiment benchmark.

Compares the trained models' predictions against measured on/off ratios and
retention times of real 2-terminal core/shell quantum-dot memory devices from
the literature (literature_analysis/measured_devices.csv).

Real devices can't be scored with a single point prediction, because papers never
report several of the model's inputs -- above all trap density (Nt_cm3), the
dominant driver of on/off ratio. So every unreported input is swept over a range
and the model's predictions are summarised as a band. A measurement is
"reproducible" by the model if it falls inside that band; the Nt values that
reproduce it are reported too.

Each device is run twice, with material inputs from two sources:
  dataset    -- band offsets / dielectric constant computed from the training
                dataset's own (AI-extracted) values for these materials, exactly as
                a generated candidate for this pair would have looked.
  literature -- band offsets / dielectric constant from cited literature.
The provenance check compares the two directly.

Inputs (hand-curated, sources in literature_analysis/SOURCES.md):
  literature_analysis/measured_devices.csv
  literature_analysis/literature_material_constants.csv
  literature_analysis/literature_pair_offsets.csv
Outputs:
  literature_analysis/provenance_check.csv
  literature_analysis/benchmark_inputs.csv
  literature_analysis/benchmark_results.csv
  literature_analysis/model_vs_experiment.png

Three models: the saved XGBoost and NN, and the physics-constrained symbolic
regression equations (scripts/symbolic_model.py), fit here on the same seed-42
candidate_split() training set (deterministic, ~5 s). The equations can predict
outside the training label range, so for them only, a measurement the in-range Nt
sweep can't reach is also checked against an extended sweep up to 1e21 cm^-3 --
reported separately (Nt_extended_reproducing), since that is extrapolation beyond
every trap density in the training data.

hysteresis_window_V is deliberately not benchmarked: papers report set/reset
voltages of bipolar sweeps (a 3.6-7 V span here), which is a different quantity
from this project's same-current branch separation, and also exceeds the label
range the model was trained on.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import RANDOM_STATE, load_dataset, candidate_split
from nn_model import (TORCH_THREADS, add_engineered_features, feature_columns, inner_split_and_scaler,
                      TabularResNet, predict)
from symbolic_model import SparseSymbolicRegressor, CONSTRAINTS

torch.set_num_threads(TORCH_THREADS)

LIT_DIR = "literature_analysis"
TARGETS = {"log10_on_off_ratio": "on_off_ratio", "log10_retention_tau_s": "retention_s"}
BAND_ALIGN_COLS = {
    "Type I (straddling, core confines shell)": "band_align_Type_I_straddling_core_confines_shell",
    "Type I (straddling, shell confines core)": "band_align_Type_I_straddling_shell_confines_core",
    "Type II (staggered)": "band_align_Type_II_staggered",
    "Type III (broken gap)": "band_align_Type_III_broken_gap",
}

# Assumptions for inputs the papers don't report. Commercial CdSe/ZnS dots carry
# roughly 1-3 ZnS monolayers (~0.31 nm each), hence a 0.3-1.0 nm shell; when even the
# total diameter is missing, core radius spans the common 1.5-3 nm range. Sweep
# voltage, when unreported, spans the full training range.
SHELL_GRID_NM = [0.3, 0.5, 0.7, 1.0]
CORE_GRID_NM = [1.5, 2.0, 2.5, 3.0]
VMAX_GRID_V = [1.0, 2.0, 3.0, 4.0, 5.0]
N_NT = 25
NT_EXTENDED_MAX = 1e21  # symbolic-regression-only extrapolation check, see docstring
MODEL_NAMES = ("XGBoost", "NN", "Symbolic")
MATCH_TOL_LOG10 = np.log10(2)  # "reproduces the measurement" = within a factor of 2


def classify(core_chi, core_eg, shell_chi, shell_eg):
    """The dataset's own band-alignment rule, reverse-engineered and verified to
    reproduce all 8000 rows: same-sign offsets are Type I, mixed signs are Type II,
    and Type III when one conduction band lies below the other's valence band."""
    dEc = shell_chi - core_chi
    dEv = (core_chi + core_eg) - (shell_chi + shell_eg)
    if dEc < 0 and dEv < 0:
        return "Type I (straddling, shell confines core)"
    if dEc > 0 and dEv > 0:
        return "Type I (straddling, core confines shell)"
    core_cb, core_vb = -core_chi, -(core_chi + core_eg)
    shell_cb, shell_vb = -shell_chi, -(shell_chi + shell_eg)
    if core_cb < shell_vb or shell_cb < core_vb:
        return "Type III (broken gap)"
    return "Type II (staggered)"


def material_from_dataset(raw, name, prefer="core"):
    """A material's Eg/chi/eps as the dataset records it. Looks in the preferred role first:
    the dielectric constant only exists on shell rows, so a shell must be looked up as a
    shell -- ZnSe is in the dataset as both, and taking its core row gave eps = NaN, which
    XGBoost silently accepted as 'missing' instead of failing."""
    for role in ((prefer, "shell" if prefer == "core" else "core")):
        rows = raw[raw[f"{role}_material"] == name]
        if len(rows):
            r = rows.iloc[0]
            return {"Eg": r[f"{role}_Eg_eV"], "chi": r[f"{role}_chi_eV"],
                    "chi_estimated": bool(r[f"{role}_chi_is_estimated"]),
                    "eps": r["shell_dielectric_constant_est"] if role == "shell" else np.nan,
                    "eps_source": r["shell_dielectric_constant_source"] if role == "shell" else "",
                    "mp_id": r[f"{role}_material_id"]}
    raise SystemExit(f"{name} does not appear in the training dataset in any role.")


# ---------------------------------------------------------------
# 1. Data, models, and the exact NN preprocessing used in training
# ---------------------------------------------------------------
raw = pd.read_csv("csv/synthetic_rs_dataset_fixed__1_.csv")
df = load_dataset()
train_df, _ = candidate_split(df)
devices = pd.read_csv(f"{LIT_DIR}/measured_devices.csv")
lit_consts = pd.read_csv(f"{LIT_DIR}/literature_material_constants.csv")
lit_pairs = pd.read_csv(f"{LIT_DIR}/literature_pair_offsets.csv").set_index("pair")

nn_df = add_engineered_features(df.copy())
nn_feature_cols = feature_columns(nn_df)
_, _, nn_scaler = inner_split_and_scaler(add_engineered_features(train_df.copy()), nn_feature_cols, RANDOM_STATE)

models = {}
for target in TARGETS:
    xm = xgb.XGBRegressor()
    xm.load_model(f"xgb_model_{target}.json")
    nm = TabularResNet(input_dim=len(nn_feature_cols), hidden_dim=128, num_blocks=2, dropout=0.1)
    nm.load_state_dict(torch.load(f"nn_model_{target}.pt"))
    models[(target, "XGBoost")] = ("xgb", xm)
    models[(target, "NN")] = ("nn", nm)
    sm = SparseSymbolicRegressor(constraints=CONSTRAINTS[target])
    sm.fit(train_df, train_df[target], groups=train_df["candidate_id"])
    print(f"[symbolic] {sm.equation(target=target)}")
    models[(target, "Symbolic")] = ("sr", sm)

train_ranges = {c: (df[c].min(), df[c].max()) for c in
                ["dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm", "Nt_cm3", "eps_shell", "Vmax_V"]}
target_ranges = {t: (df[t].min(), df[t].max()) for t in TARGETS}
nt_grid = np.logspace(np.log10(train_ranges["Nt_cm3"][0]), np.log10(train_ranges["Nt_cm3"][1]), N_NT)
nt_grid_extended = np.logspace(np.log10(train_ranges["Nt_cm3"][1]), np.log10(NT_EXTENDED_MAX), 13)[1:]

# ---------------------------------------------------------------
# 2. Provenance check: dataset-extracted values vs literature
# ---------------------------------------------------------------
prov_rows = []
prop_key = {"band_gap": "Eg", "electron_affinity": "chi", "static_dielectric_constant": "eps"}
for _, c in lit_consts.iterrows():
    m = material_from_dataset(raw, c["material"],
                              prefer="shell" if c["property"] == "static_dielectric_constant" else "core")
    val = m[prop_key[c["property"]]]
    lo, hi = c["literature_low"], c["literature_high"]
    prov_rows.append({
        "kind": "material", "item": c["material"], "property": c["property"],
        "dataset_value": round(float(val), 4), "literature_low": lo, "literature_high": hi,
        "within_literature_range": bool(lo <= val <= hi),
        "dataset_note": f"{m['mp_id']}" + (f"; chi_is_estimated={m['chi_estimated']}" if c["property"] == "electron_affinity"
                                           else (f"; source={m['eps_source']}" if c["property"] == "static_dielectric_constant" else "")),
        "literature_sources": c["sources"],
    })

pair_inputs = {}
for pair, p in lit_pairs.iterrows():
    core = material_from_dataset(raw, p["core"], prefer="core")
    shell = material_from_dataset(raw, p["shell"], prefer="shell")
    dEc = shell["chi"] - core["chi"]
    dEv = (core["chi"] + core["Eg"]) - (shell["chi"] + shell["Eg"])
    ds_class = classify(core["chi"], core["Eg"], shell["chi"], shell["Eg"])
    for prop, val, lo, hi in [("dE_LUMO_eV", dEc, p["dE_LUMO_low"], p["dE_LUMO_high"]),
                              ("dE_HOMO_eV", dEv, p["dE_HOMO_low"], p["dE_HOMO_high"])]:
        prov_rows.append({"kind": "pair", "item": pair, "property": prop, "dataset_value": round(float(val), 4),
                          "literature_low": lo, "literature_high": hi, "within_literature_range": bool(lo <= val <= hi),
                          "dataset_note": "computed from the dataset's chi/Eg with its own convention",
                          "literature_sources": p["sources"]})
    prov_rows.append({"kind": "pair", "item": pair, "property": "band_alignment", "dataset_value": ds_class,
                      "literature_low": p["band_alignment_literature"], "literature_high": p["band_alignment_literature"],
                      "within_literature_range": ds_class == p["band_alignment_literature"],
                      "dataset_note": "the dataset's own classification rule applied to its own values",
                      "literature_sources": p["sources"]})

    lit_eps_row = lit_consts[(lit_consts.material == p["shell"]) & (lit_consts.property == "static_dielectric_constant")]
    lit_eps = ([lit_eps_row.iloc[0]["literature_low"], lit_eps_row.iloc[0]["literature_high"]]
               if len(lit_eps_row) else [shell["eps"]])
    pair_inputs[pair] = {
        "dataset": {"dE_LUMO": [dEc], "dE_HOMO": [dEv], "eps": [shell["eps"]], "band_alignment": ds_class},
        "literature": {"dE_LUMO": [p["dE_LUMO_low"], (p["dE_LUMO_low"] + p["dE_LUMO_high"]) / 2, p["dE_LUMO_high"]],
                       "dE_HOMO": [p["dE_HOMO_low"], (p["dE_HOMO_low"] + p["dE_HOMO_high"]) / 2, p["dE_HOMO_high"]],
                       "eps": lit_eps, "band_alignment": p["band_alignment_literature"],
                       "eps_note": "" if len(lit_eps_row) else "no cited literature value; dataset value used"},
    }

prov = pd.DataFrame(prov_rows)
prov.to_csv(f"{LIT_DIR}/provenance_check.csv", index=False)
print("=== Provenance: dataset-extracted values vs literature ===")
print(prov[["item", "property", "dataset_value", "literature_low", "literature_high", "within_literature_range"]]
      .to_string(index=False))

# ---------------------------------------------------------------
# 3. Build each device's input grid and predict
# ---------------------------------------------------------------
def device_grid(dev, source, nt_values=None):
    pin = pair_inputs[f"{dev['core']}/{dev['shell']}"][source]
    if pd.notna(dev["qd_diameter_nm"]):
        geoms = [(dev["qd_diameter_nm"] / 2 - t, t) for t in SHELL_GRID_NM]
    else:
        geoms = list(itertools.product(CORE_GRID_NM, SHELL_GRID_NM))
    vmaxes = [dev["sweep_vmax_V"]] if pd.notna(dev["sweep_vmax_V"]) else VMAX_GRID_V
    rows = []
    for dEc, dEv, eps, (r, t), vmax, nt in itertools.product(pin["dE_LUMO"], pin["dE_HOMO"], pin["eps"],
                                                             geoms, vmaxes,
                                                             nt_grid if nt_values is None else nt_values):
        row = {"dE_LUMO_eV": dEc, "dE_HOMO_eV": dEv, "shell_thick_nm": t, "core_radius_nm": r,
               "Nt_cm3": nt, "eps_shell": eps, "Vmax_V": vmax}
        row.update({col: int(label == pin["band_alignment"]) for label, col in BAND_ALIGN_COLS.items()})
        rows.append(row)
    grid = pd.DataFrame(rows)
    if grid.isna().any().any():  # XGBoost would treat a NaN as "missing" and predict anyway
        raise SystemExit(f"{dev['device_id']} ({source}): missing input values in "
                         f"{grid.columns[grid.isna().any()].tolist()}")
    return grid, pin


def run_model(kind, model, grid):
    if kind == "xgb":
        return model.predict(grid[model.get_booster().feature_names])
    if kind == "sr":
        return model.predict(grid)
    g = add_engineered_features(grid.copy())
    return predict(model, nn_scaler.transform(g[nn_feature_cols].values))


result_rows, input_rows, plot_data = [], [], []
for _, dev in devices[devices["modelled"].astype(str) == "True"].iterrows():
    for source in ("dataset", "literature"):
        grid, pin = device_grid(dev, source)
        out_of_range = [f"{c}={grid[c].min():.3g}..{grid[c].max():.3g}" for c, (lo, hi) in train_ranges.items()
                        if grid[c].min() < lo - 1e-9 or grid[c].max() > hi + 1e-9]
        input_rows.append({
            "device_id": dev["device_id"], "inputs": source, "grid_points": len(grid),
            "band_alignment": pin["band_alignment"],
            "dE_LUMO_eV": "/".join(f"{v:.3g}" for v in sorted(set(grid.dE_LUMO_eV))),
            "dE_HOMO_eV": "/".join(f"{v:.3g}" for v in sorted(set(grid.dE_HOMO_eV))),
            "eps_shell": "/".join(f"{v:.3g}" for v in sorted(set(grid.eps_shell))),
            "core_radius_nm": f"{grid.core_radius_nm.min():.3g}-{grid.core_radius_nm.max():.3g}",
            "shell_thick_nm": f"{grid.shell_thick_nm.min():.3g}-{grid.shell_thick_nm.max():.3g}",
            "Vmax_V": "/".join(f"{v:g}" for v in sorted(set(grid.Vmax_V))),
            "Nt_cm3": f"{grid.Nt_cm3.min():.2e}-{grid.Nt_cm3.max():.2e} ({N_NT} log steps)",
            "outside_training_range": "; ".join(out_of_range) or "none",
            "note": pin.get("eps_note", ""),
        })
        for target, meas_col in TARGETS.items():
            if pd.isna(dev[meas_col]):
                continue
            meas = np.log10(dev[meas_col])
            # String compare: the column has blanks, so pandas may read it as text, and bool("False") is True.
            lower_bound = target == "log10_retention_tau_s" and str(dev["retention_is_lower_bound"]) == "True"
            for model_name in MODEL_NAMES:
                kind, model = models[(target, model_name)]
                pred = run_model(kind, model, grid)
                q = np.percentile(pred, [0, 10, 50, 90, 100])
                row = {"device_id": dev["device_id"], "target": target, "model": model_name, "inputs": source,
                       "measured_log10": round(meas, 3), "measured_is_lower_bound": lower_bound,
                       "pred_min": q[0], "pred_p10": q[1], "pred_median": q[2], "pred_p90": q[3], "pred_max": q[4],
                       "training_label_min": target_ranges[target][0], "training_label_max": target_ranges[target][1]}
                if lower_bound:
                    consistent = pred >= meas
                    row.update({"verdict": "consistent" if q[2] >= meas else "inconsistent",
                                "frac_grid_consistent": consistent.mean(),
                                "median_minus_measured": q[2] - meas})
                else:
                    match = np.abs(pred - meas) <= MATCH_TOL_LOG10
                    row.update({"verdict": ("within p10-p90" if q[1] <= meas <= q[3] else
                                            "within full range" if q[0] <= meas <= q[4] else
                                            "outside predicted range"),
                                "frac_grid_matching_x2": match.mean(),
                                "median_minus_measured": q[2] - meas,
                                "Nt_reproducing_measurement": (f"{grid.Nt_cm3[match].min():.1e}-{grid.Nt_cm3[match].max():.1e}"
                                                               if match.any() else "none in training range")})
                    if kind == "sr" and not match.any():
                        ext_grid, _ = device_grid(dev, source, nt_grid_extended)
                        ext_match = np.abs(run_model(kind, model, ext_grid) - meas) <= MATCH_TOL_LOG10
                        row["Nt_extended_reproducing"] = (
                            f"{ext_grid.Nt_cm3[ext_match].min():.1e}-{ext_grid.Nt_cm3[ext_match].max():.1e}"
                            if ext_match.any() else f"none up to {NT_EXTENDED_MAX:.0e}")
                result_rows.append(row)
                plot_data.append(row)

pd.DataFrame(input_rows).to_csv(f"{LIT_DIR}/benchmark_inputs.csv", index=False)
results = pd.DataFrame(result_rows)
results.to_csv(f"{LIT_DIR}/benchmark_results.csv", index=False, float_format="%.4g")

print("\n=== Benchmark: measured vs predicted (log10 units) ===")
cols = ["device_id", "target", "model", "inputs", "measured_log10", "pred_p10", "pred_median", "pred_p90", "verdict"]
print(results[cols].round(2).to_string(index=False))

# ---------------------------------------------------------------
# 4. Figure: predicted bands vs measured values
# ---------------------------------------------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SOURCE_COLOR = {"dataset": "#2a78d6", "literature": "#eb6834"}  # validated palette slots 1-2
MODEL_MARKER = {"XGBoost": "o", "NN": "s", "Symbolic": "D"}
OFFSETS = {("XGBoost", "dataset"): -0.35, ("XGBoost", "literature"): -0.21,
           ("NN", "dataset"): -0.07, ("NN", "literature"): 0.07,
           ("Symbolic", "dataset"): 0.21, ("Symbolic", "literature"): 0.35}
TITLES = {"log10_on_off_ratio": "On/off ratio", "log10_retention_tau_s": "Retention time (s)"}

pdata = pd.DataFrame(plot_data)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), dpi=200, gridspec_kw={"width_ratios": [5, 2.4]})
fig.patch.set_facecolor(SURFACE)
for ax, target in zip(axes, TARGETS):
    sub = pdata[pdata.target == target]
    dev_ids = list(dict.fromkeys(sub.device_id))
    ax.set_facecolor(SURFACE)
    lo, hi = target_ranges[target]
    ax.axhspan(lo, hi, color="#efeeea", zorder=0)
    if target == "log10_on_off_ratio":
        # Tree models can't predict above the highest label they were trained on.
        ax.annotate(f"highest on/off in training data (~{10 ** hi:,.0f})", (len(dict.fromkeys(sub.device_id)) - 0.42, hi),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)
    for i, dev_id in enumerate(dev_ids):
        for (model_name, source), dx in OFFSETS.items():
            r = sub[(sub.device_id == dev_id) & (sub.model == model_name) & (sub.inputs == source)].iloc[0]
            x, c = i + dx, SOURCE_COLOR[source]
            ax.plot([x, x], [r.pred_min, r.pred_max], color=c, lw=1, solid_capstyle="round", zorder=2)
            ax.plot([x, x], [r.pred_p10, r.pred_p90], color=c, lw=4, solid_capstyle="round", zorder=3)
            ax.plot(x, r.pred_median, marker=MODEL_MARKER[model_name], ms=8, color=c,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        m = sub[sub.device_id == dev_id].iloc[0]
        if m.measured_is_lower_bound:
            ax.plot(i, m.measured_log10, marker="^", ms=13, color=INK, markeredgecolor=SURFACE, zorder=5)
            ax.annotate("≥", (i, m.measured_log10), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=11, color=INK, fontweight="bold")
        else:
            ax.plot(i, m.measured_log10, marker="*", ms=15, color=INK, markeredgecolor=SURFACE, zorder=5)
    ax.set_xticks(range(len(dev_ids)))
    ax.set_xticklabels([d.replace("_", "\n", 1) for d in dev_ids], fontsize=8, color=INK_2)
    ax.set_xlim(-0.6, len(dev_ids) - 0.4)
    ax.set_ylabel(f"log10 {TITLES[target].lower()}", fontsize=10, color=INK_2)
    ax.set_title(TITLES[target], fontsize=12, fontweight="bold", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)

legend_items = [
    Line2D([], [], color=SOURCE_COLOR["dataset"], lw=4, label="Inputs from training dataset's own values"),
    Line2D([], [], color=SOURCE_COLOR["literature"], lw=4, label="Inputs from literature values"),
    Line2D([], [], color=INK_2, marker="o", lw=0, ms=8, label="XGBoost median"),
    Line2D([], [], color=INK_2, marker="s", lw=0, ms=8, label="NN median"),
    Line2D([], [], color=INK_2, marker="D", lw=0, ms=7, label="Symbolic-regression median"),
    Line2D([], [], color=INK, marker="*", lw=0, ms=14, label="Measured"),
    Line2D([], [], color=INK, marker="^", lw=0, ms=11, label="Measured lower bound (stable for at least)"),
    plt.Rectangle((0, 0), 1, 1, color="#efeeea", label="Range of training labels"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, fontsize=9,
           labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Model predictions vs. measured core/shell QD memory devices", x=0.01, ha="left",
             fontsize=14, fontweight="bold", color=INK)
fig.text(0.01, 0.905, "Bars span predictions over every unreported input (trap density, shell thickness, "
         "sweep voltage): thin = full range, thick = 10th-90th percentile.", fontsize=9, color=INK_2)
plt.tight_layout(rect=(0, 0.07, 1, 0.9))
fig.savefig(f"{LIT_DIR}/model_vs_experiment.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print(f"\nWrote {LIT_DIR}/provenance_check.csv, benchmark_inputs.csv, benchmark_results.csv, model_vs_experiment.png")
