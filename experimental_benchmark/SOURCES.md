# Experimental benchmark — sources

Compiled for Tier 3 of the project roadmap ("Model-vs-real-experimental-data benchmark"). Every row in
`real_measured_data.csv` traces to one of the four papers below; nothing in that file is estimated,
interpolated, or invented. Where a paper didn't report a value this project's schema needs
(`hysteresis_window_V`, `on_off_ratio`, `retention_time_s`), the cell is left blank rather than guessed.

This is a **starting point, not a complete benchmark set** — 4 data points found via web search in one
pass, not an exhaustive literature review. See the honest-assessment note at the end.

## Papers cited

1. **Kannan, V., Rhee, J. K.** (2013). *Robust switching characteristics of CdSe/ZnS quantum dot
   non-volatile memory devices.* Physical Chemistry Chemical Physics, 15(30), 12762–12766.
   DOI: [10.1039/c3cp50216c](https://doi.org/10.1039/c3cp50216c)
   — On/off ratio (~1000) and an observed retention stability duration (20,000 s) for a CdSe/ZnS
   core-shell QD 2-terminal resistive memory. Closest real device to this project's "Type I core-shell"
   framing among the four found.

2. **Kannan, V., Kim, H. S., Park, H. C.** (2016). *High speed switching in quantum Dot/Ti-TiOx
   nonvolatile memory device.* Electronic Materials Letters.
   DOI: [10.1007/s13391-015-5410-5](https://doi.org/10.1007/s13391-015-5410-5)
   — Same CdSe/ZnS QD family, different device stack (adds Ti-TiOx). Contributes on/off ratio (100) only;
   no numeric hysteresis or retention value found in available search excerpts.

3. **Core-shell quantum dot-enabled monolayer MoS2 memories with high endurance.** Matter (Cell Press),
   2025. DOI: [10.1016/j.matt.2025.102488](https://doi.org/10.1016/j.matt.2025.102488)
   — Highest-performance device found (on/off 10^6, 96.5% retention at 10 years), but CdSe/CdS is
   generally a quasi-Type-II system in the literature (not Type I), and it's a 3-terminal floating-gate
   transistor, not a 2-terminal RS device — its "140 V memory window" is not the same physical quantity as
   this project's `hysteresis_window_V`. Full author list not confirmed from available search results.

4. **High-Performance Nanofloating Gate Memory Based on Lead Halide Perovskite Nanocrystals.** ACS Applied
   Materials & Interfaces, 11(27), 24367–24377 (2019).
   DOI: [10.1021/acsami.9b03474](https://doi.org/10.1021/acsami.9b03474)
   — CdS nanoribbon / CH3NH3PbBr3 nanocrystal hybrid. Contributes a directly-measured (not extrapolated)
   retention time (12,000 s) and on/off ratio (7×10^7), but is a nanoribbon/nanocrystal hybrid geometry,
   not the spherical core-radius+shell-thickness structure this project's features model, and is also a
   3-terminal transistor device. Full author list not confirmed from available search results.

## Honest assessment

**This is a partial start, not a benchmark set ready to drop into a `plots/model_vs_experiment.png` figure
as-is.** Specific gaps:

- Only 2 of 4 sources are genuinely "Type I core-shell, 2-terminal, charge-trapping" devices matching this
  project's physical framing (rows 1 and 2, both CdSe/ZnS). Rows 3 and 4 are real, citable data but from a
  different device architecture (3-terminal floating-gate transistors) and, in row 3's case, a different
  band-alignment regime (quasi-Type-II) — comparing them to this project's predictions requires the
  architecture/unit caveats in the CSV's `notes` column to be respected, not glossed over.
- No source reports all three of `hysteresis_window_V`, `on_off_ratio`, and `retention_time_s` together —
  every row has at least one blank cell. A genuinely strong benchmark figure would want papers reporting
  all three for the same physical device.
- This was one focused search session, not a systematic literature review. A proper Tier 3 benchmark likely
  needs a deeper pass — including checking each paper's full text/SI (only abstracts and search-result
  excerpts were available here) for numbers not surfaced by search, and searching more specifically for
  CdSe/ZnS-family papers (the best-matching device class) rather than the broader "core-shell nanocrystal
  memory" query that surfaced rows 3 and 4.
- Two of four author lists could not be confirmed from available search results and are cited by DOI/title
  only — verify before using in any written report.

**Recommendation:** treat this CSV as a seed for a real literature search, not a finished input. The two
CdSe/ZnS rows (1 and 2) are the most defensible direct comparison points for this project's `on_off_ratio`
predictions specifically; nothing here should be used for a `hysteresis_window_V` comparison without first
resolving the 2-terminal-vs-3-terminal unit mismatch.
