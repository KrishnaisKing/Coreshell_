"""
Option B -- build a core/shell candidate set from measured, cited material properties.

Reads  curated_dataset/material_properties.csv  (one row per material x property, every value cited;
see curated_dataset/SOURCES.md) and writes  curated_dataset/candidates.csv  in the same column layout
as the faculty-provided csv/synthetic_rs_dataset_fixed__1_.csv, so the existing pipeline can consume it
once labels exist.

What it does:
  - Uses only "handbook" (Ioffe NSM) and "literature_measured" (measured values read from a full-text,
    peer-reviewed source) rows; handbook wins where both exist. "secondary" and "conflicting" rows are
    recorded in the CSV for reference but never enter a candidate -- see SOURCES.md.
  - A material needs band gap, electron affinity and lattice constant to be used at all, and also a
    static dielectric constant to be used as a shell (the only dielectric constant the models see is
    eps_shell). Materials without one appear only as cores.
  - Builds every ordered (core, shell) pair of distinct eligible materials.
  - Computes dEc_eV / dEv_eV and band_alignment with exactly the raw dataset's convention and rule
    (the rule is first re-checked against all 8000 raw rows, and the script stops if it disagrees).
  - Computes lattice_mismatch_pct = (a_shell - a_core) / a_core * 100, using the cubic-equivalent
    a_c = sqrt(2) * a_hex for wurtzite (nearest-neighbour spacing match between the (111) cubic
    and (0001) hexagonal planes).
  - Samples 4 device rows per pair (core radius, shell thickness, trap density, sweep voltage) with
    a seeded Latin hypercube over the raw dataset's own ranges.

What it does NOT do: compute device labels. hysteresis_window_V, on_off_ratio, retention_time_s and
endurance_cycles are left empty (labels_status = "pending") -- no generator for them exists in this
repo, and inventing one is a separate decision.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

PROPS_PATH = "curated_dataset/material_properties.csv"
RAW_PATH = "csv/synthetic_rs_dataset_fixed__1_.csv"
OUT_PATH = "curated_dataset/candidates.csv"

REQUIRED = ["band_gap", "electron_affinity", "lattice_constant_a"]
SHELL_REQUIRED = REQUIRED + ["static_dielectric_constant"]
ALL_PROPS = SHELL_REQUIRED
USED_QUALITIES = ["handbook", "literature_measured"]  # in priority order
ROWS_PER_PAIR = 4
LHS_SEED = 42

# Same ranges as the raw dataset (checked: its min/max sit at these bounds).
CORE_RADIUS_NM = (1.5, 6.0)
SHELL_THICKNESS_NM = (0.3, 3.0)
LOG10_TRAP_DENSITY = (15.0, 19.0)
VOLTAGE_MAX_V = (1.0, 5.0)

LABEL_COLS = ["hysteresis_window_V", "on_off_ratio", "retention_time_s", "endurance_cycles"]


def classify(core_chi, core_eg, shell_chi, shell_eg):
    """The raw dataset's band-alignment rule (same as scripts/literature/model_vs_experiment.py):
    same-sign offsets are Type I, mixed signs are Type II, and Type III when one conduction band
    lies below the other's valence band."""
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


def check_rule_against_raw():
    raw = pd.read_csv(RAW_PATH)
    got = [classify(*r) for r in raw[["core_chi_eV", "core_Eg_eV", "shell_chi_eV", "shell_Eg_eV"]].to_numpy()]
    n_bad = int((pd.Series(got, index=raw.index) != raw["band_alignment"]).sum())
    if n_bad:
        raise SystemExit(f"classify() disagrees with {n_bad} of {len(raw)} raw labels -- rule has drifted.")
    print(f"Band-alignment rule reproduces all {len(raw)} raw dataset labels.")


def eligible_materials(props):
    used = props[props["quality"].isin(USED_QUALITIES)].copy()
    used["rank"] = used["quality"].map(USED_QUALITIES.index)
    used = used.sort_values("rank").drop_duplicates(["material", "property"])  # handbook wins
    table = used.pivot(index="material", columns="property", values="value").reindex(columns=ALL_PROPS)
    structure = used.groupby("material")["structure"].first()
    quality = used.pivot(index="material", columns="property", values="quality").reindex(columns=ALL_PROPS)
    citations = (used.assign(cite=used["source_key"] + used["cited_reference"].fillna("").map(
        lambda c: f" ({c})" if c else ""))
        .pivot(index="material", columns="property", values="cite").reindex(columns=ALL_PROPS))

    complete = table[REQUIRED].notna().all(axis=1)
    for m in table.index[~complete]:
        missing = [p for p in REQUIRED if pd.isna(table.loc[m, p])]
        print(f"  skipped {m}: no usable value for {', '.join(missing)}")
    unused_only = sorted(set(props["material"]) - set(table.index))
    if unused_only:
        print(f"  skipped (secondary/conflicting values only): {', '.join(unused_only)}")

    table = table.loc[complete].copy()
    for m in table.index[table["static_dielectric_constant"].isna()]:
        print(f"  {m}: no usable static dielectric constant -> core only")
    table["can_be_shell"] = table["static_dielectric_constant"].notna()
    table["all_handbook"] = quality.loc[table.index, REQUIRED].eq("handbook").all(axis=1) & (
        quality.loc[table.index, "static_dielectric_constant"].fillna("handbook") == "handbook")
    table["structure"] = structure
    table["a_cubic_equiv_A"] = np.where(table["structure"] == "wurtzite",
                                        np.sqrt(2) * table["lattice_constant_a"], table["lattice_constant_a"])
    return table, citations


def build():
    check_rule_against_raw()
    props = pd.read_csv(PROPS_PATH)
    mats, cites = eligible_materials(props)
    print(f"{len(mats)} eligible materials ({mats['can_be_shell'].sum()} usable as shells): "
          f"{', '.join(mats.index)}")

    pairs = []
    for core in mats.index:
        for shell in mats.index:
            if core == shell or not mats.loc[shell, "can_be_shell"]:
                continue
            c, s = mats.loc[core], mats.loc[shell]
            pairs.append({
                "core_material": core, "core_class": "inorganic", "core_material_id": "",
                "shell_material": shell, "shell_class": "inorganic", "shell_material_id": "",
                "core_Eg_eV": c["band_gap"], "core_chi_eV": c["electron_affinity"], "core_chi_is_estimated": False,
                "shell_Eg_eV": s["band_gap"], "shell_chi_eV": s["electron_affinity"], "shell_chi_is_estimated": False,
                "shell_dielectric_constant_est": s["static_dielectric_constant"],
                "shell_dielectric_constant_source": "measured_static (Ioffe NSM)",
                "band_alignment": classify(c["electron_affinity"], c["band_gap"],
                                           s["electron_affinity"], s["band_gap"]),
                "dEc_eV": s["electron_affinity"] - c["electron_affinity"],
                "dEv_eV": (c["electron_affinity"] + c["band_gap"]) - (s["electron_affinity"] + s["band_gap"]),
                "core_structure": c["structure"], "shell_structure": s["structure"],
                "core_dielectric_constant": c["static_dielectric_constant"],
                "core_lattice_a_A": c["lattice_constant_a"], "shell_lattice_a_A": s["lattice_constant_a"],
                "lattice_mismatch_pct": (s["a_cubic_equiv_A"] - c["a_cubic_equiv_A"]) / c["a_cubic_equiv_A"] * 100,
                "core_sources": "; ".join(f"{p}: {cites.loc[core, p]}" for p in ALL_PROPS
                                          if pd.notna(cites.loc[core, p])),
                "shell_sources": "; ".join(f"{p}: {cites.loc[shell, p]}" for p in SHELL_REQUIRED),
                "pair_quality": "handbook" if (c["all_handbook"] and s["all_handbook"]) else "literature_measured",
            })
    pairs = pd.DataFrame(pairs)

    sample = qmc.LatinHypercube(d=4, seed=LHS_SEED).random(len(pairs) * ROWS_PER_PAIR)
    lo = np.array([CORE_RADIUS_NM[0], SHELL_THICKNESS_NM[0], LOG10_TRAP_DENSITY[0], VOLTAGE_MAX_V[0]])
    hi = np.array([CORE_RADIUS_NM[1], SHELL_THICKNESS_NM[1], LOG10_TRAP_DENSITY[1], VOLTAGE_MAX_V[1]])
    device = qmc.scale(sample, lo, hi)

    df = pairs.loc[pairs.index.repeat(ROWS_PER_PAIR)].reset_index(drop=True)
    df["core_radius_nm"] = device[:, 0]
    df["shell_thickness_nm"] = device[:, 1]
    df["trap_density_cm3"] = 10 ** device[:, 2]
    df["voltage_max_V"] = device[:, 3]
    for col in LABEL_COLS:
        df[col] = np.nan
    df["Confinement_Asymmetry"] = np.nan
    df["labels_status"] = "pending"

    raw_cols = list(pd.read_csv(RAW_PATH, nrows=0).columns)
    extra = [c for c in df.columns if c not in raw_cols]
    df = df[raw_cols + extra]

    inputs = [c for c in df.columns if c not in LABEL_COLS + ["Confinement_Asymmetry", "core_material_id",
                                                              "shell_material_id", "core_dielectric_constant"]]
    if df[inputs].isna().any().any():
        raise SystemExit(f"NaN in input columns: {df[inputs].columns[df[inputs].isna().any()].tolist()}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(pairs)} pairs x {ROWS_PER_PAIR} device rows = {len(df)} rows")
    print(f"Pairs built only from handbook values: {(pairs['pair_quality'] == 'handbook').sum()}")
    print("\nBand alignment (pairs):")
    print(pairs["band_alignment"].value_counts().to_string())
    print(f"\n|lattice mismatch| range: {pairs['lattice_mismatch_pct'].abs().min():.2f}"
          f"-{pairs['lattice_mismatch_pct'].abs().max():.1f}%; "
          f"pairs within 5%: {(pairs['lattice_mismatch_pct'].abs() <= 5).sum()}")
    print(f"Labels: all {len(LABEL_COLS)} label columns empty (labels_status = 'pending').")
    return df


if __name__ == "__main__":
    build()
