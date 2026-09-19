"""
Phase 3.5 -- Harmonize the real MP-screened candidate dataset
(synthetic_rs_dataset_fixed.csv) into the schema kmeans_xgboost_train.py
and nn.py were written against.

Why this exists: rs_simulator.py's LHS-sampled dataset used generic
device-parameter names (dE_LUMO_eV, dE_HOMO_eV, ...) because it wasn't
tied to any particular material pair. Once real MP candidates entered
the pipeline, the dataset picked up material-screening column names
instead (dEc_eV, dEv_eV, core_material, shell_material, ...) plus new
columns the simulator-only version never had (band_alignment,
core_class/shell_class, endurance_cycles). This script is the missing
adapter between the two -- it does NOT change any physics or labels,
it only renames/derives so the existing training code runs unmodified
against real candidates.

Column mapping:
    core_material, shell_material  -> candidate_id (formula_core__formula_shell)
    dEc_eV                         -> dE_LUMO_eV   (electron/conduction offset)
    dEv_eV                         -> dE_HOMO_eV   (hole/valence offset)
    shell_thickness_nm             -> shell_thick_nm
    trap_density_cm3               -> Nt_cm3
    shell_dielectric_constant_est  -> eps_shell
    voltage_max_V                  -> Vmax_V
    retention_time_s               -> retention_tau_s

Not available in this dataset (left absent, downstream scripts already
guard with "if c in df.columns"):
    confinement_score_eV   -- no single scalar confinement score was
                              computed for real candidates; dE_LUMO_eV/
                              dE_HOMO_eV carry this information instead.
    lattice_mismatch_pct   -- produced by latticmismatch.py, which needs
                              a live Materials Project API call
                              (MP_API_KEY + network to materialsproject.org).
                              This sandbox has no route to that host, so
                              lattice mismatch is left out here. Run
                              latticmismatch.py separately wherever that
                              network access exists, then re-merge on
                              (core_material_id, shell_material_id)
                              before re-running training if you want it
                              as a feature.

Added (not in the old LHS-only pipeline, but present here and useful):
    band_alignment  -- Type I/II/III classification from the MP screen.
                        One-hot encoded into band_align_* columns since
                        it's a first-order driver of device physics
                        (Type I traps confine cleanly; Type II/III change
                        the tunneling picture rs_simulator.py assumes).
                        Included as engineered features below.
"""

import numpy as np
import pandas as pd

IN_PATH = "csv/synthetic_rs_dataset_fixed__1_.csv"
OUT_PATH = "csv/rs_training_data_real_candidates.csv"

RENAME_MAP = {
    "dEc_eV": "dE_LUMO_eV",
    "dEv_eV": "dE_HOMO_eV",
    "shell_thickness_nm": "shell_thick_nm",
    "trap_density_cm3": "Nt_cm3",
    "shell_dielectric_constant_est": "eps_shell",
    "voltage_max_V": "Vmax_V",
    "retention_time_s": "retention_tau_s",
}


def harmonize(in_path=IN_PATH, out_path=OUT_PATH):
    df = pd.read_csv(in_path)
    n_raw = len(df)

    df = df.rename(columns=RENAME_MAP)

    # candidate_id: one real material pair, ~4 device-parameter rows each
    df["formula_core"] = df["core_material"].astype(str)
    df["formula_shell"] = df["shell_material"].astype(str)
    df["candidate_id"] = df["formula_core"] + "__" + df["formula_shell"]

    # log10 target the training scripts expect to find or create
    if "log10_on_off_ratio" not in df.columns and "on_off_ratio" in df.columns:
        df["log10_on_off_ratio"] = np.log10(df["on_off_ratio"].clip(lower=1e-12))

    # sanity: required numeric columns must be finite
    required = ["dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
                "Nt_cm3", "eps_shell", "Vmax_V", "hysteresis_window_V",
                "on_off_ratio", "retention_tau_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Harmonization failed, missing expected columns: {missing}")

    n_bad = df[required].isna().any(axis=1).sum()
    if n_bad:
        print(f"Dropping {n_bad} rows with NaNs in required numeric columns.")
        df = df.dropna(subset=required)

    # band_alignment: one-hot encode as engineered features (see docstring)
    if "band_alignment" in df.columns:
        dummies = pd.get_dummies(df["band_alignment"], prefix="band_align")
        dummies.columns = [
            c.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            for c in dummies.columns
        ]
        df = pd.concat([df, dummies.astype(int)], axis=1)

    df.to_csv(out_path, index=False)
    n_cand = df["candidate_id"].nunique()
    print(f"Harmonized {n_raw} raw rows -> {len(df)} usable rows, {n_cand} unique candidates.")
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    harmonize()