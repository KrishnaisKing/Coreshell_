# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An ML pipeline that predicts resistive-switching (memristor-like) device metrics — `hysteresis_window_V`,
`log10_on_off_ratio`, `log10_retention_tau_s` — for core/shell nanocrystal candidates, from
material/device descriptors (the project is framed around Type I pairs, but only 29% of the 2000 candidates
are Type I — 51% are Type II, 20% Type III; nothing filters to Type I, `band_alignment` is just a feature) (band offsets, shell thickness, trap density, dielectric constant, lattice
mismatch, etc.). There is no package manifest (no `requirements.txt`/`pyproject.toml`) and no test suite;
it's a set of standalone scripts run in sequence, each reading/writing by hardcoded relative path (no CLI
args) — paths are relative to the repo root regardless of which subfolder the script itself lives in, so
**scripts must always be invoked from the repo root** (e.g. `python scripts/training/nn.py`, never `cd
scripts/training && python nn.py`). Layout: all `.py` files live under `scripts/`, organized by purpose
(see "Script organization" below); model artifacts (`xgb_model_*.json`, `nn_model_*.pt`) live in the repo
root; all CSVs live in `csv/`; all figures in `plots/`.

### Script organization

```
scripts/
├── split_utils.py      # shared: load_dataset(), candidate_split(), random_row_split(), leave_materials_out_split()
├── nn_model.py          # shared: TabularResNet, NN feature engineering, permutation_importance()
├── pipeline/            # Prepare_real_candidates.py, Latticmatch.py
├── training/            # kmeans_xgboost_train.py, nn.py, nn_permutation_importance.py
├── plotting/            # plot.py, plot2.py, plot3.py, plot4.py, plot5.py
└── validation/          # split_strategy_comparison.py, retention_extrapolation_check.py,
                         # repeated_split_evaluation.py, physics_sanity_checks.py,
                         # multicollinearity_check.py, uncertainty_quantification.py
```

`split_utils.py`/`nn_model.py` stay directly under `scripts/` rather than in their own subfolder because
every other script imports them as siblings. Python only auto-adds a *script's own directory* to its import
path, not the repo root or sibling folders — so every script under `pipeline/`, `training/`, or
`validation/` that needs them starts with:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import ...
```
This is boilerplate, not a design choice to admire — if this project ever gains a package manifest, replace
it with a proper installed package and absolute imports instead of copying this pattern further. `plot*.py`
and the two `pipeline/` scripts need no such fix; they don't import either shared module.

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

All paths below are relative to the repo root — run every command from there, regardless of the script's
own subfolder:

```
python scripts/pipeline/Prepare_real_candidates.py   # csv/synthetic_rs_dataset_fixed__1_.csv -> csv/rs_training_data_real_candidates.csv
python scripts/pipeline/Latticmatch.py               # optional: adds lattice_mismatch_pct column (needs MP_API_KEY + network)
python scripts/training/kmeans_xgboost_train.py      # ~20s. -> xgb_model_*.json (root), csv/predictions_xgb_*.csv, csv/cv_scores_xgb.csv
python scripts/training/nn.py                        # ~90s total (3 targets, early-stopped). -> nn_model_*.pt, csv/predictions_nn_*.csv, csv/permutation_importance_nn_*.csv, csv/nn_early_stopping_log.csv
python scripts/training/nn_permutation_importance.py # computes csv/permutation_importance_nn_*.csv for already-saved nn_model_*.pt without retraining
python scripts/plotting/plot2.py                     # xgb 4-panel dashboard -> plots/plot_1..4_*.png
python scripts/plotting/plot3.py                     # nn 4-panel dashboard -> plots/nn_plot_1..4_*.png
python scripts/plotting/plot4.py                     # per-target parity/residuals -> plots/{model_type}_{target}_parity.png / _residuals.png
python scripts/plotting/plot5.py                     # per-target feature importance -> plots/xgb_{target}_importance.png / nn_{target}_importance.png
python scripts/plotting/plot.py                      # standalone exploratory 2x3 grid, shows interactively (no savefig)
python scripts/validation/split_strategy_comparison.py     # trains XGBoost under 3 split strategies -> csv/split_strategy_comparison.csv, plots/split_strategy_comparison.png
python scripts/validation/retention_extrapolation_check.py # confirms/quantifies WHY retention drops under the grouped split -> csv/retention_extrapolation_check.csv, plots/retention_extrapolation_check.png
python scripts/validation/repeated_split_evaluation.py     # re-evaluates the fixed XGBoost hyperparams under 10 unseen seeds -> csv/repeated_split_evaluation.csv, plots/repeated_split_evaluation.png (~35s)
python scripts/validation/physics_sanity_checks.py         # monotonicity checks (hysteresis vs Nt/barrier height, on-off/retention vs shell thickness) -> csv/physics_sanity_checks.csv, plots/physics_sanity_checks.png
python scripts/validation/multicollinearity_check.py       # VIF across the full feature set -> csv/multicollinearity_check.csv, plots/multicollinearity_check.png
python scripts/validation/uncertainty_quantification.py    # 80% prediction intervals via XGBoost quantile regression -> csv/uncertainty_quantification.csv, plots/uncertainty_quantification.png
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

`nn_model.py` holds `TabularResNet`, the NN feature engineering, `permutation_importance()`,
`inner_split_and_scaler()` (the validation carve-out + feature scaler every saved NN was trained with) and
`TORCH_THREADS`. Keep the split logic only in `split_utils.py` and NN preprocessing only in `nn_model.py` —
`nn_permutation_importance.py` depends on reproducing `nn.py`'s preprocessing exactly. Its guard re-predicts
the test set with each saved model and refuses to run unless the result matches
`csv/predictions_nn_*.csv` to 1e-4. **Why the guard is that strict:** an earlier version only compared
candidate IDs, and passed while the script fit its scaler on the full `train_df` after `nn.py` had moved to
fitting on `inner_train_df` — feeding every model shifted inputs (retention predictions off by up to 2.07
log10 units, i.e. ~100x in retention time) and silently computing importances against degraded models.
Thread count is shared too: it changes floating-point summation order, so a mismatch breaks bit-for-bit
reproduction. With matching preprocessing, dtype and threads, the script reproduces `nn.py`'s importance
CSVs bit-for-bit.

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

**NN training speed:** `nn.py` uses batch size 128, `torch.set_num_threads(TORCH_THREADS)` (4, set in `nn_model.py`) and manual index batching
(no DataLoader). The original batch-32 / 16-thread / DataLoader setup took ~2.7 min per target because
per-step overhead and thread contention dominated for a model this small — 4 threads beat 16 by ~2x. The
larger batch also improved held-out R² on all three targets, not just speed. Don't "optimize" back to
small batches or all cores without re-measuring.

## Architecture

**Upstream (not in this repo):** `synthetic_rs_dataset_fixed__1_.csv` was provided by faculty, who
extracted the material properties from Materials Project with AI assistance. The screening did **not**
restrict to Type I pairs (see the class split above). How the device metrics (hysteresis, on/off ratio,
retention) were computed from those material properties is not known — no generator script is in this repo.

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
needs measured device data; only a small, partial seed exists so far (`experimental_benchmark/`, see Tier 3).
All training data traces back to the faculty-provided `synthetic_rs_dataset_fixed__1_.csv`.

**`experimental_benchmark/` is real literature data, not training data** — kept out of `csv/` deliberately
so no script picks it up by accident. No script reads it yet.

## Fix/validation roadmap (tiered by priority)

Update this checklist's status markers as work lands — don't let it go stale. `[x]` done, `[~]` partial,
`[ ]` not started.

**Tier 0 — blocking correctness issues (fix before anything else; on `main`):**
- [x] Hardcoded fake numbers in plot scripts (`plot.py`'s CV bars, NN feature-importance panels) —
  fixed, `657124f`.
- [x] Missing NN retention-time model (run had been manually terminated) — trained, `5ca45bd`.
- [~] Inverted Type I→III `hysteresis_window_V` ordering (contradicts band-confinement theory) —
  **partially answered, not resolved.** Faculty confirmed the underlying material data was extracted from
  Materials Project using AI assistance. This explains *provenance* (where `core_Eg_eV`, `core_chi_eV`,
  `shell_Eg_eV`, `shell_chi_eV` etc. came from) but not the *mechanism* — it doesn't by itself say why
  `hysteresis_window_V` increases toward Type III instead of decreasing. New candidate hypothesis worth
  checking: bulk AI-assisted extraction across ~1444 materials is exactly the kind of process where a
  systematic error (e.g. core/shell role swapped for some fraction of candidates, or a sign-convention
  mistake in χ/Eg extraction) could silently scramble the `band_alignment` classification relative to the
  true physical roles, without affecting `dE_LUMO_eV`/`dE_HOMO_eV`'s magnitude — which would produce
  exactly this kind of "right features, wrong label" pattern. Not confirmed; still can't be fully
  root-caused from the data alone, but now has a concrete, checkable lead instead of an unknown.
  **Second, independent piece of evidence** (from `physics_sanity_checks.py`'s diagnostics): hysteresis
  depends on |dE_LUMO_eV|/|dE_HOMO_eV| just as strongly in Type II/III rows as in Type I rows, although
  those offsets aren't confining barriers outside Type I. That points at the label-generation step
  treating |offset| as a barrier regardless of alignment type — a different mechanism from the
  role-swap hypothesis above, and one that would produce the inverted ordering by itself. Worth asking
  faculty specifically how hysteresis was computed from the offsets.

**Tier 1 — core validation gaps (needed before any accuracy claim is credible; on `core-validation-gaps`,
not yet merged to `main`):**
- [x] Leave-materials-out test (not just leave-one-*pair*-out) — `leave_materials_out_split()`, `05477ff`.
- [x] Split-strategy comparison figure (random vs. grouped vs. LOMO) — `split_strategy_comparison.py`,
  `05477ff`.
- [ ] **Decide what to do about the redundant features** — `dE_LUMO_eV`, `dE_HOMO_eV`, and the
  `band_align_*` one-hots all encode the same underlying signal (confirmed to floating-point precision).
  Not started; needs a decision, not just an implementation: drop the one-hots and keep the continuous
  offsets, drop the offsets and keep the categorical, or keep both deliberately (e.g. to test whether tree
  models "rediscover" the band-alignment rule themselves) and document why.
- [x] Real permutation importance for the NN (was a hardcoded placeholder) — done as part of the Tier 0
  cleanup, `657124f`.
- [~] **Stratify splits by `band_alignment`** — half done, and the other half was tried and reverted.
  The inner 5-fold CV in `kmeans_xgboost_train.py` uses `StratifiedGroupKFold` on `band_alignment`
  (`92780cd`) — that part's solid. The **outer split** (`candidate_split()` in `split_utils.py`, which
  decides which candidates become the held-out test set) is still stratified only by retention bin.
  Attempted fix: stratify by `(retention_bin, band_alignment)` jointly (12 strata instead of 3), using the
  same greedy whole-cluster-until-quota loop. **Result was worse, not better — reverted, do not retry this
  exact approach**: test fraction ballooned from 38% to 58.8% of all candidates, and class balance didn't
  even reliably improve (Type III went from 20.4% of the population to 11.3% of test — now
  *under*-represented — while both Type I subclasses became over-represented). Root cause: with only 6
  global KMeans clusters spread across 12 strata instead of 3, many strata have so few candidates in any
  given cluster that adding "one cluster's worth" can itself exceed 100% of that stratum's quota — the same
  failure mode as the NN validation-split overshoot described above, now hitting the outer split instead.
  A real fix needs either a rebalancing pass on top of the existing split (swap whole
  `(retention_bin, cluster)` groups between train/test to correct class share, which is a nontrivial search
  problem) or a different allocation algorithm entirely (e.g. sampling individual candidates per
  `(retention_bin, band_alignment, cluster)` cell instead of whole clusters) — both are real design changes
  that would also invalidate the current `split_strategy_comparison.py` / `retention_extrapolation_check.py`
  / `repeated_split_evaluation.py` numbers, which all assume the current split's behavior. Deliberately not
  pursued for now — `candidate_split()` is unchanged from `92780cd`.
- Also fixed here, surfaced by a direct CV audit rather than being on the original list (`92780cd`):
  - [x] CV was blind to the retention extrapolation regime (CV interpolates, the outer test set
    extrapolates over offset-space clusters) — quantified in `retention_extrapolation_check.py`.
  - [x] NN had no validation split or early stopping (fixed 200 epochs, no evidence it was the right
    number) — added, with a warmup period so patience doesn't fire during the LR schedule's noisy start.
  - [~] Test-set contamination from hyperparameter development (max_depth etc. were tuned by observing
    the seed-42 test set) — partially mitigated: `repeated_split_evaluation.py` validates the *current*
    frozen hyperparameters against 10 unseen seeds, but any *future* re-tuning reintroduces the same
    contamination unless it's checked the same way.

**Tier 2 — physics-rigor checks from the publication doc:**
- [x] Formalize the monotonicity sanity checks as a saved script — `physics_sanity_checks.py`. Bins each
  driver into deciles, reports Spearman rho (a direct monotonicity measure, unlike Pearson) against the
  theoretically-expected sign. **5/6 pass**: hysteresis increases with trap density (rho=+0.77) and with
  both barrier heights |dE_LUMO_eV|/|dE_HOMO_eV| (rho=+0.10/+0.13, Type I rows only); on/off ratio falls with
  shell thickness (rho=-0.14); retention rises with shell thickness (rho=+0.11). The one failure is the
  already-documented `band_alignment` ordering (see "Verified data/pipeline caveats" below) — not a new
  finding, just now formally tracked alongside the others in `csv/physics_sanity_checks.csv` /
  `plots/physics_sanity_checks.png`.
  **Correction from a later audit:** the barrier-height checks originally ran on all rows, but |offset| is
  only a confining barrier for Type I pairs — for Type II/III it isn't a barrier in that sense, so the
  original check mixed physically different quantities. Now restricted to Type I (the conclusion held:
  +0.12/+0.13 on all rows vs +0.10/+0.13 on Type I). **The restriction surfaced a real finding**, recorded as
  a non-pass/fail diagnostic: Type II/III rows show the same dependence on |offset| as Type I rows (rho=+0.10
  /+0.11 vs +0.10/+0.13), even though physics expects it to matter much less where there's no confining
  barrier. Whatever computed these labels appears to treat |offset| as a barrier regardless of alignment
  type — which on its own would explain why Type III doesn't show *less* hysteresis than Type I. Evidence
  for the Tier 0 question, not proof.
- [x] Multicollinearity check (VIF) across the full feature set — `multicollinearity_check.py` (manual
  VIF via `sklearn.LinearRegression`, no `statsmodels` dependency added). **First version of this check had
  a real numerical bug, caught by the user from an oddly-flat plot, not by inspection of the code**: fitting
  VIF's regressions on unstandardized features gave every single feature a VIF of exactly 1.00, including
  `dE_LUMO_eV`/`dE_HOMO_eV`, which are independently known to correlate at r=-0.75. Cause: `Nt_cm3` spans
  1e15-1e19 while every other feature is O(1)-O(300) — a condition number far beyond float64's ~15-16
  digits of precision, so `LinearRegression`'s solver silently returned garbage (R²≈0.00002) for every
  feature. VIF's true value is scale-invariant, but the least-squares fit used to compute it is not at this
  magnitude spread. Fixed by standardizing features before fitting (doesn't change the true VIF, only the
  numerical stability of computing it).
  **Corrected result:** the 4 `band_align_*` one-hots show VIF ≈ 1×10¹⁰ when all 4 are included together —
  the "dummy variable trap" predicted in this script's own docstring, now actually visible. `dE_LUMO_eV`
  and `dE_HOMO_eV` show a real, moderate VIF ≈ 4 (consistent with r=-0.75) — a genuine *linear* relationship
  between the two continuous offsets, independent of the non-linear band-alignment question below. With the
  one-hot reference dropped, the one-hots' residual VIF is mild (1.1–1.45) — so the earlier claim that
  `band_alignment`'s redundancy is mostly invisible to a linear test still holds for *that specific*
  relationship, but "VIF confirms no linear multicollinearity anywhere" was simply wrong, not just
  imprecise. This adds real quantitative weight to the still-open Tier 1 feature-redundancy decision
  (item 6): there IS a moderate linear reason, not just the non-linear one, to not carry both continuous
  offsets and the full one-hot set together in a coefficient-based model.
- [x] Uncertainty quantification — `uncertainty_quantification.py`, XGBoost's native
  `reg:quantileerror` objective (`quantile_alpha=[0.1, 0.5, 0.9]`, no new dependency, same hyperparameters/
  features/split as `kmeans_xgboost_train.py`) gives an 80% prediction interval per candidate. **Result
  lines up exactly with every other retention finding in this file**: hysteresis empirical coverage 0.793
  and on/off ratio 0.770 are both close to the 0.80 nominal target — well calibrated. Retention's coverage
  is **0.632** — badly under nominal. This is the calibration-diagnostic view of the same extrapolation
  problem `retention_extrapolation_check.py` already quantified: prediction intervals are calibrated from
  in-distribution training residuals, and calibration doesn't transfer to a test set that's
  out-of-distribution in the exact features (`dE_LUMO_eV`/`dE_HOMO_eV`) the split holds out on. Visually,
  `plots/uncertainty_quantification.png` shows the retention median saturating near the target's ceiling
  while true values scatter widely below it, well outside the band, for a large share of test candidates.
  **Do not report retention's prediction intervals as reliable without re-running this against
  `leave_materials_out_split()` first** — the same distinction that matters for R² almost certainly matters
  here too, and hasn't been checked yet.

**Tier 3 — blocked on external input:**
- [~] Model-vs-real-experimental-data benchmark (the doc's essential figure #3) — **partial seed only.**
  `experimental_benchmark/real_measured_data.csv` has 4 measured data points from 4 cited papers (sources
  and caveats in `experimental_benchmark/SOURCES.md`), found in one web-search pass, not a systematic
  review. Only 2 rows (CdSe/ZnS, 2-terminal) match this project's device class; the other 2 are 3-terminal
  floating-gate transistors whose gate-voltage "memory window" (77–140 V) is not the same quantity as
  `hysteresis_window_V` (~0–1.1 V here). No row reports all three targets. Usable today only for a limited
  `on_off_ratio` sanity comparison against the CdSe/ZnS rows — not enough for the actual benchmark figure,
  which still needs a deeper search (full texts, CdSe/ZnS-family papers specifically).

**Tier 4 — doc's optional/"if space allows" items (not started):**
- [ ] Residual-by-input-region plot (flag where error concentrates, e.g. thin shells, high trap density).
- [ ] Learning curves (does performance still climb with more data, or plateau?).
- [ ] Applicability-domain/novelty detection, for screening genuinely new candidates outside the training
  distribution.
- [ ] Confusion matrix for `band_alignment` — not applicable unless it becomes a predicted target itself.

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
