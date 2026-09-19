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
python split_strategy_comparison.py # trains XGBoost under 3 split strategies -> csv/split_strategy_comparison.csv, plots/split_strategy_comparison.png
python retention_extrapolation_check.py # confirms/quantifies WHY retention drops under the grouped split -> csv/retention_extrapolation_check.csv, plots/retention_extrapolation_check.png
python repeated_split_evaluation.py # re-evaluates the fixed XGBoost hyperparams under 10 unseen seeds -> csv/repeated_split_evaluation.csv, plots/repeated_split_evaluation.png (~35s)
```

There is no build step, linter, or test command in this repo.

Every plot script raises `FileNotFoundError` if its input CSV/model is missing — they never fall back to
synthetic data. All plotted numbers are computed: XGBoost importances come from `feature_importances_` on
the saved model JSON, NN importances from permutation importance (mean drop in test R², 10 repeats) in
`csv/permutation_importance_nn_*.csv`, and CV bars from `csv/cv_scores_xgb.csv`.

**Shared modules:** `split_utils.py` holds `load_dataset()` and three split strategies, all with signature
`(df, test_frac=0.25, random_state=...) -> (train_df, test_df)`:
- `candidate_split()` — the pipeline's default (candidate-pair-grouped, retention-stratified via `qcut`
  into 3 bins), used by `kmeans_xgboost_train.py` and `nn.py`. Has a zero-overlap assert on `candidate_id`.
- `random_row_split()` — deliberately leaky row-level random split, comparison baseline only.
- `leave_materials_out_split()` — holds out entire materials (union of `formula_core`/`formula_shell`;
  840 of 1444 distinct materials appear in both roles) so a held-out material never appears in training at
  all, in either role. Stricter than `candidate_split()`, which only guarantees an unseen *combination* of
  otherwise-seen materials. Test fraction is a side effect of how densely held-out materials are reused
  across pairs, not a tunable target.

`nn_model.py` holds `TabularResNet`, the NN feature engineering, and `permutation_importance()`. Keep the
split logic only in `split_utils.py` — `nn_permutation_importance.py` depends on reproducing the exact
test set of a previous `nn.py` run and verifies this against `csv/predictions_nn_*.csv` before computing.

**`split_strategy_comparison.py`** trains one XGBoost model per (strategy × target) — 9 fits — and plots
test R² grouped by target. `hysteresis_window_V` and `log10_on_off_ratio` are stable across all three
strategies (0.92–0.99). `log10_retention_tau_s` is not: random 0.992, grouped-by-pair 0.815,
leave-materials-out 0.982.

**Why retention drops under the grouped split — confirmed, not guessed (`retention_extrapolation_check.py`):**
it's feature-space extrapolation, not primarily the target-distribution skew an earlier version of this
note claimed. `candidate_split()` clusters candidates by `(dE_LUMO_eV, dE_HOMO_eV)` — the only 2 of the 4
intended clustering columns present in this dataset — and holds out whole clusters. Permutation importance
(`csv/permutation_importance_nn_log10_retention_tau_s.csv`) shows retention is driven almost entirely by
those same two features, while hysteresis/on-off ratio are driven by trap density, which is independent of
cluster membership. So holding out whole offset-space clusters is a real extrapolation test for retention
and a near-irrelevant one for the other two targets — matching exactly which target the comparison figure
flags. Verified directly: grouped-by-pair test candidates sit ~2x farther from their nearest training
candidate in scaled offset-space than leave-materials-out's (mean distance 0.111 vs 0.058), and within the
grouped test set, distance-to-nearest-train correlates with retention error (Pearson r=0.25, Spearman
r=0.31) — the farthest half of test candidates has 2.7x the mean absolute error of the nearest half (1.62
vs 0.59). The test-set retention distribution is also mildly skewed toward the harder short-retention mode
(26.8% vs the population's 22.2%), which contributes but is not the primary driver.
**Do not read cross-strategy R² differences as a pure leakage signal — check feature-space distance from
train first**, especially for any target whose top permutation-importance features overlap with the
clustering columns.

**`repeated_split_evaluation.py`** re-runs `candidate_split()`'s exact fixed XGBoost hyperparameters
(`max_depth=6`, `colsample_bytree=0.7`, etc. — chosen, per this file's own original docstring, by observing
prediction behavior on the seed-42 test set) under 10 seeds that were never involved in choosing those
hyperparameters. **Seed 42 (the one reported everywhere else in this repo) scores above the 10-seed mean on
all three targets**: hysteresis +1.25σ, on/off ratio +1.37σ, retention +0.43σ. This is direct evidence that
the commonly-quoted 0.925/0.976/0.815 numbers are on the optimistic side of what an untried split gives —
a real, if partial, contamination effect from hyperparameters having been shaped by looking at this
specific split. It also surfaces something more important than the seed-42 bias: **retention's test R² is
extremely seed-dependent** (10-seed range 0.355–0.909, std 0.203) versus hysteresis (std 0.018) and on/off
ratio (std 0.028) being an order of magnitude tighter. A single retention R² number, from any seed, should
not be quoted without this spread alongside it.

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
- XGBoost runs 5-fold `StratifiedGroupKFold` CV (grouped by `candidate_id`, stratified by
  `band_alignment` so the ~50/15/15/20% class split is respected in every fold) before the final fit,
  recording both R² and MAE per fold to `csv/cv_scores_xgb.csv`.
- **The CV number and the test number measure different things for retention, and the CV doesn't warn
  you.** CV interpolates (folds are drawn from the same candidate pool with no feature-space holdout);
  the outer test set extrapolates (whole offset-space clusters held out). For hysteresis/on-off ratio,
  whose relevant features aren't the clustering columns, CV and test agree closely. For retention, whose
  relevant features *are* the clustering columns, CV R²=0.975 vs test R²=0.815 — a gap the tight CV
  fold-to-fold spread (±0.005) gives no indication of. See `retention_extrapolation_check.py` above for
  why. **Do not treat a tight CV spread as evidence the test-set gap is noise — for this pipeline it can
  mean the CV isn't measuring the same regime as the test set.**
- The NN (`TabularResNet` in `nn.py`) is a custom residual MLP: input projection →
  `BatchNorm`/`SiLU`/`Dropout` residual blocks → linear head, trained with AdamW + cosine annealing, with
  early stopping (see below). It uses a feature set that isn't identical to XGBoost's — it adds
  `exp(dE_LUMO_eV)`, `exp(dE_HOMO_eV)`, `log10(Nt_cm3)` — so R²/MAE between the two model families aren't
  directly comparable.
- **NN early stopping:** `nn.py` carves 15% off `train_df` (via `GroupShuffleSplit` on `candidate_id`, not
  `candidate_split()` — that function's cluster-overshoot is far worse at this smaller scale, e.g. it gave
  51.6% to validation instead of 15% when tried here) for a validation fold used only for early stopping.
  Patience (30 epochs) doesn't start counting until after a 40-epoch warmup, because `CosineAnnealingLR`
  hasn't annealed yet before that and validation R² is dominated by LR-driven noise, not convergence — an
  earlier version without warmup stopped retention's training at epoch 21 with test R² = **-0.41**.
  Per-target best epoch/val-R² is logged to `csv/nn_early_stopping_log.csv`.
- Held-out R² (same 758-candidate test set): hysteresis XGB 0.925 / NN 0.918; on/off ratio XGB 0.976 /
  NN 0.982; retention XGB 0.815 / NN 0.938. Don't read the NN's higher retention score as "the NN
  generalizes better" without checking `repeated_split_evaluation.py`'s spread first — a single-seed
  comparison between two models on the single hardest-to-generalize target is not a reliable ranking.
- Permutation importance shows the NN relies on different features per target: trap density (`log10_Nt`)
  dominates hysteresis and on/off ratio, but retention is driven by the band offsets (`dE_HOMO_eV`,
  `dE_LUMO_eV` and their `exp_` transforms), with trap density barely registering. This is *why* retention
  is the target sensitive to the clustering-based split — see `retention_extrapolation_check.py` above.

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
- **Test-set contamination is mitigated for past decisions, not prevented for future ones.**
  `repeated_split_evaluation.py` shows seed 42's fixed hyperparameters aren't wildly cherry-picked (they
  sit inside the 10-seed spread, just on the high side), but that check only covers hyperparameters that
  are already frozen. If `max_depth`, `colsample_bytree`, or the split design are changed again based on
  what improves the seed-42 test score, the same contamination reappears immediately. Any future tuning
  should pick a seed, average over multiple seeds, or use nested CV — and validate the result against
  seeds not used to choose it, the way `repeated_split_evaluation.py` does now for the current values.
