# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An ML pipeline that predicts resistive-switching (memristor-like) device metrics — `hysteresis_window_V`,
`log10_on_off_ratio`, `log10_retention_tau_s` — for Type I core/shell nanocrystal candidates, from
material/device descriptors (band offsets, shell thickness, trap density, dielectric constant, lattice
mismatch, etc.). There is no package manifest (no `requirements.txt`/`pyproject.toml`) and no test suite;
it's a flat set of standalone scripts run in sequence, each reading/writing by hardcoded relative path (no
CLI args). Layout: scripts and model artifacts (`xgb_model_*.json`, `nn_model_*.pt`) live in the root,
all CSVs live in `csv/`, all figures in `plots/`. Scripts must be run from the repo root.

## Environment

Dependencies are already installed in `.venv` (Python 3.14): pandas, numpy, scikit-learn, xgboost, torch,
matplotlib, seaborn, scipy.

```
.venv\Scripts\Activate.ps1   # PowerShell
```

`mp_api` / `pymatgen` are **not** installed — they're only needed for `Latticmatch.py`, which also needs
a live `MP_API_KEY` and network access to materialsproject.org (not available in this sandbox).

## Running the pipeline

Scripts must be run in this order from the repo root; each stage's output feeds the next by
writing/overwriting `csv/rs_training_data_real_candidates.csv` in place:

```
python Prepare_real_candidates.py   # csv/synthetic_rs_dataset_fixed__1_.csv -> csv/rs_training_data_real_candidates.csv
python Latticmatch.py               # optional: adds lattice_mismatch_pct column (needs MP_API_KEY + network)
python kmeans_xgboost_train.py      # ~20s. -> xgb_model_*.json (root), csv/predictions_xgb_*.csv, csv/cv_scores_xgb.csv
python nn.py                        # ~100s total (3 targets). -> nn_model_*.pt, csv/predictions_nn_*.csv, csv/permutation_importance_nn_*.csv
python nn_permutation_importance.py # computes csv/permutation_importance_nn_*.csv for already-saved nn_model_*.pt without retraining
python plot2.py                     # xgb 4-panel dashboard -> plots/plot_1..4_*.png
python plot3.py                     # nn 4-panel dashboard -> plots/nn_plot_1..4_*.png
python plot4.py                     # per-target parity/residuals -> plots/{model_type}_{target}_parity.png / _residuals.png
python plot5.py                     # per-target feature importance -> plots/xgb_{target}_importance.png / nn_{target}_importance.png
python plot.py                      # standalone exploratory 2x3 grid, shows interactively (no savefig)
```

There is no build step, linter, or test command in this repo.

Every plot script raises `FileNotFoundError` if its input CSV/model is missing — they never fall back to
synthetic data. All plotted numbers are computed: XGBoost importances come from `feature_importances_` on
the saved model JSON, NN importances from permutation importance (mean drop in test R², 10 repeats) in
`csv/permutation_importance_nn_*.csv`, and CV bars from `csv/cv_scores_xgb.csv`.

**Shared modules:** `split_utils.py` holds `load_dataset()` and `candidate_split()` (the candidate-grouped
split, with the zero-overlap assert) used by both training scripts and `nn_permutation_importance.py`.
`nn_model.py` holds `TabularResNet`, the NN feature engineering, and `permutation_importance()`. Keep the
split logic only in `split_utils.py` — `nn_permutation_importance.py` depends on reproducing the exact
test set of a previous `nn.py` run and verifies this against `csv/predictions_nn_*.csv` before computing.

**NN training speed:** `nn.py` uses batch size 128, `torch.set_num_threads(4)` and manual index batching
(no DataLoader). The original batch-32 / 16-thread / DataLoader setup took ~2.7 min per target because
per-step overhead and thread contention dominated for a model this small — 4 threads beat 16 by ~2x. The
larger batch also improved held-out R² on all three targets, not just speed. Don't "optimize" back to
small batches or all cores without re-measuring.

## Architecture

**Upstream (not in this repo):** a Materials Project screening step selecting Type I band-aligned
core/shell candidate pairs. Its output is what `synthetic_rs_dataset_fixed__1_.csv` derives from.

**`Prepare_real_candidates.py`** — adapter/harmonization layer. Renames the real MP-screened dataset's
columns (`dEc_eV`, `dEv_eV`, `shell_thickness_nm`, ...) into the schema the training scripts expect
(`dE_LUMO_eV`, `dE_HOMO_eV`, `shell_thick_nm`, ...), builds `candidate_id = formula_core__formula_shell`,
derives `log10_on_off_ratio`, one-hot encodes `band_alignment` into `band_align_*` columns. Does not
compute `confinement_score_eV` (not available for real candidates) or `lattice_mismatch_pct` (needs
`Latticmatch.py`) — downstream scripts guard every optional column with `if c in df.columns`, so the
feature set used for training silently shrinks when these are absent.

**`Latticmatch.py`** — computes `lattice_mismatch_pct` between core and shell via a pseudo-cubic lattice
constant `a_pc = (V_cell/Z)^(1/3)` (standard approximation for halide perovskites/layered structures that
have no single natural lattice parameter). Requires live MP API access; merges its result back onto
`rs_training_data_real_candidates.csv` by `candidate_id`.

**`kmeans_xgboost_train.py` / `nn.py`** — the training scripts, both using `split_utils.candidate_split()`:
- Build a candidate-level table (one row per material pair) from material features (`dE_LUMO_eV`,
  `dE_HOMO_eV`, `confinement_score_eV`, `lattice_mismatch_pct`, whichever exist) plus each candidate's
  mean `log10_retention_tau_s`, binned into 3 retention tiers via `qcut`.
- KMeans-cluster candidates by material features *within* each retention tier, then hold out whole
  clusters (~25% target) as test candidates. This is the **candidate-grouped, retention-stratified split**
  — it guarantees no `candidate_id` (and its ~4 device-parameter sweep rows) is split across train/test.
  The greedy whole-cluster loop overshoots the target: the actual split is 758/2000 candidates (38%) in
  test, not 25%.
- The same split (stratified only by retention) is reused to train/evaluate all three targets.
- XGBoost additionally runs 5-fold `GroupKFold` CV (grouped by `candidate_id`) before the final fit.
- The NN (`TabularResNet` in `nn.py`) is a custom residual MLP: input projection →
  `BatchNorm`/`SiLU`/`Dropout` residual blocks → linear head, trained with AdamW + cosine annealing.
  It uses a feature set that isn't identical to XGBoost's — it adds `exp(dE_LUMO_eV)`, `exp(dE_HOMO_eV)`,
  `log10(Nt_cm3)` — so R²/MAE between the two model families aren't directly comparable.
- Held-out R² (same 758-candidate test set): hysteresis XGB 0.925 / NN 0.916; on/off ratio XGB 0.976 /
  NN 0.978; retention XGB 0.815 / NN 0.942. Retention is where the models diverge most — XGB's GroupKFold
  CV R² for it is 0.977, so its 0.815 test score is a CV-vs-test gap the other targets don't show.
- Permutation importance shows the NN relies on different features per target: trap density (`log10_Nt`)
  dominates hysteresis and on/off ratio, but retention is driven by the band offsets (`dE_HOMO_eV`,
  `dE_LUMO_eV` and their `exp_` transforms), with trap density barely registering.

**`plot*.py`** — all read `rs_training_data_real_candidates.csv` and the `predictions_{xgb,nn}_*.csv`
files; none retrain anything. `plot2.py`/`plot3.py` produce the 4-panel xgb/nn dashboards, `plot4.py`/
`plot5.py` produce the same content split out per-target-per-model, and `plot.py` is a standalone
exploratory feature-vs-feature scatter.

## Project goal: publication readiness, not just predictive accuracy

Per `Publication possibilities_figures.docx` (validation/publication-strategy notes checked into this
repo), the target is a short paper for a Q1 venue (candidates by framing: Applied Physics Letters or IEEE
Electron Device Letters for a device-physics framing; Digital Discovery or npj Computational Materials for
an ML-for-materials framing; J. Mater. Chem. C or Nanoscale for a materials-chemistry framing). Its
"essential" figure set doesn't exist yet in `plot*.py`: (1) a parity plot on a **grouped or
leave-one-material-out** held-out set specifically (not the current candidate-pair split framed as such),
(2) a SHAP/feature-importance plot, (3) a model-vs-real-experimental-data benchmark. That third figure
requires an actual measured core-shell RS dataset, which does not currently exist anywhere in this repo —
everything here traces back to the LHS-sampled simulator output (`synthetic_rs_dataset_fixed__1_.csv`).

## Verified data/pipeline caveats (do not assume these are fixed without re-checking)

These were confirmed against the actual data, not just asserted by the doc — re-verify if the underlying
CSVs change:

- **The train/test split is grouped by material *pair* (`candidate_id`), not by individual material.**
  A single material can appear in training (paired with material A) and in test (paired with a different
  material B). This tests generalization to unseen *combinations* of known materials, not to a genuinely
  unseen material — don't conflate the two when describing what the split validates.
- **`dE_LUMO_eV`, `dE_HOMO_eV`, and the `band_align_*` one-hots are redundant, not independent.**
  Confirmed to floating-point precision: `dEc_eV = shell_chi_eV - core_chi_eV` and
  `dEv_eV = (core_chi_eV + core_Eg_eV) - (shell_chi_eV + shell_Eg_eV)`; `band_alignment` is a categorical
  bucketing of those same two values. All three feature groups encode one underlying signal.
  `core_material_id`/`shell_material_id`/`core_material`/`shell_material` are, however, correctly excluded
  from `feature_cols` in both training scripts — the "model memorizes material identity" failure mode does
  not currently apply.
- **A `chi_is_estimated`-based ablation is not feasible on this dataset as-is.** 0 of 8000 rows have both
  `core_chi_is_estimated` and `shell_chi_is_estimated` False (only 64 and 88 rows respectively, never
  overlapping) — there is no "fully real" subset to compare against.
- **Band-alignment vs. hysteresis ordering currently contradicts band-engineering theory.** Mean
  `hysteresis_window_V` by class: Type I (core confines shell) 0.52 < Type I (shell confines core) 0.60 <
  Type II 0.61 < Type III 0.65 — increasing toward Type III, the opposite of what confinement theory
  predicts (Type III should confine least and show the least hysteresis). Root-cause this (generator
  formula vs. a confound like trap density/voltage correlating with band-alignment class) before treating
  any physics-trend claim from this data as validated.
- **`retention_time_s`'s bimodality is a real property of the raw data, not a split-logic artifact** — 25th
  percentile ~8.6×10⁵ s (~10 days) vs. median ~9.6×10⁹ s (~304 years). This is what
  `kmeans_xgboost_train.py`'s candidate-clustering split was built to work around.
