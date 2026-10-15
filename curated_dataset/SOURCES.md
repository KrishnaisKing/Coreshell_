# Curated dataset (Option B): sources

This dataset replaces the faculty-provided material values with measured, cited ones. Regenerate it with
`python scripts/dataset/build_curated_dataset.py` (from the repo root).

| File | What it is |
|---|---|
| `material_properties.csv` | Hand-maintained. One row per material × property × quality tier, each with its source. **Edit this file, not the output.** |
| `candidates.csv` | Script output. 19 materials (17 usable as shells) → 306 ordered core/shell pairs × 4 device rows = 1,224 rows. 132 of the pairs use only Ioffe handbook values (`pair_quality` column). Column layout matches `csv/synthetic_rs_dataset_fixed__1_.csv`, with extra provenance columns. |

## Status: inputs only, no labels yet

`hysteresis_window_V`, `on_off_ratio`, `retention_time_s` and `endurance_cycles` are **empty**
(`labels_status = pending`). Nothing in this repo generates device labels from material properties: the
faculty dataset's labels came from an unknown generator. Measured device data exists for almost none of
these pairs. Before this dataset can train anything, someone has to choose a label source: the faculty's
generator (if it can be obtained), or a documented physics simulator built for this purpose.

## Quality tiers

| Tier | Meaning | Used? |
|---|---|---|
| `handbook` | Read directly from the Ioffe Institute NSM archive (pages saved and re-checked against this CSV). | Yes; it wins where a material has both tiers. |
| `literature_measured` | A measured value read from the full text of a peer-reviewed paper. If it was derived from measured quantities (an LST relation or an orientational average), the `notes` column gives the derivation. | Yes |
| `conflicting` | Measured, but the sources disagree too much to choose a value. | No |
| `secondary` | From a simulation-input table, an encyclopedia infobox, a figure read by eye, or a search summary. Kept to show where sources disagree. | No |

A material is used if it has band gap, electron affinity and lattice constant in a used tier. It can also
be a **shell** only if it has a static dielectric constant, because `eps_shell` is the only dielectric
constant the models see. AlSb and ZnO have no sourced dielectric constant yet, so they appear only as
cores.

**Not used, and why:**
- Diamond, 3C-SiC: no electron affinity (it depends on surface termination).
- CdS: its two measured ionization potentials, 6.10 and 7.26 eV, give an electron affinity of either 3.68
  or 4.84 eV (`conflicting`).
- PbS, PbSe: only search-summary band gaps.
- InN: not recorded. Ioffe's 1.97 eV gap is superseded by ~0.7 eV, and Ioffe gives no electron affinity.
- AlP: Hinuma 2014 has its gap and lattice constant, but no measured ionization potential.

## Sources

### IOFFE: Ioffe Institute NSM archive (handbook)

Ioffe Physico-Technical Institute, *New Semiconductor Materials. Characteristics and Properties*:
`http://www.ioffe.ru/SVA/NSM/Semicond/<Material>/basic.html` and `.../bandstr.html`. Used for Si, Ge,
GaAs, GaP, GaSb, InP, InAs, InSb, GaN, AlN, BN, and the excluded Diamond and SiC.

For the nitrides and BN, Ioffe names the primary reference for each value. These are in the
`cited_reference` column: Bougrov et al. (2001), Chow & Ghezzo (1996), Qian et al. (1996), Guo & Yoshida
(1994), Goldberg (2001), Collins et al. (1967), Iwama et al. (1971), Rumyantsev et al. (2001), Gielisse et
al. (1967) and Sohno et al. (1974). Bougrov, Goldberg and Rumyantsev (2001) are chapters of Levinshtein,
Rumyantsev & Shur (eds.), *Properties of Advanced Semiconductor Materials: GaN, AlN, InN, BN, SiC, SiGe*
(Wiley, 2001). For Si, Ge and the other III-Vs, Ioffe gives no per-value reference. Its index page names
Levinshtein, Rumyantsev & Shur (eds.), *Handbook Series on Semiconductor Parameters*, Vols. 1–2 (World
Scientific, 1996/1999) as the database's starting point. The primary references were not read.

**IOFFE_AlGaAs**: AlAs is not a separate Ioffe page. Its values come from the AlₓGa₁₋ₓAs composition
formulas evaluated at x = 1 (`http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/basic.html`, `bandstr.html`):

| Property | Formula at x = 1 | Value |
|---|---|---|
| Band gap | Eg = 1.9 + 0.125x + 0.143x² | 2.168 eV |
| Electron affinity | χ = 3.64 − 0.14x | 3.50 eV |
| Static dielectric constant | ε = 12.90 − 2.84x | 10.06 |
| Lattice constant | a = 5.6533 + 0.0078x | 5.6611 Å |

These agree with Hinuma 2014's Table I (2.16 eV, 5.661 Å).

Where Ioffe gives more than one value:

| Material | Property | Used | Alternative |
|---|---|---|---|
| GaN | band gap | 3.39 eV (Chow & Ghezzo 1996) | 3.44 eV (Madelung 1991) |
| GaN | lattice a | 3.189 Å (Qian et al. 1996) | 3.160–3.190 Å (Lagerstedt et al. 1979) |
| BN | band gap | 6.25 eV (midpoint) | 6.1–6.4 eV (Rumyantsev et al. 2001) |

### HINUMA2014: band gaps, lattice constants and electron affinities (literature_measured)

Hinuma Y., Grüneis A., Kresse G., Oba F. (2014). *Band alignment of semiconductors from density-functional
theory and many-body perturbation theory.* Phys. Rev. B 90, 155405.
[doi:10.1103/PhysRevB.90.155405](https://doi.org/10.1103/PhysRevB.90.155405). Open-access copy at Kyoto
University's repository, [hdl:2433/191210](https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/191210).
Full text read.

Used for ZnS, ZnSe, ZnTe, CdSe, CdTe, AlSb and ZnO, plus CdS (`conflicting`).

- **Band gap and lattice constant: Table I**, the "experimental" columns. The paper compiles these from
  Grüneis et al., Phys. Rev. Lett. 112, 096401 (2014), and the references therein. CdSe and CdS are
  listed in the zinc-blende structure.
- **Electron affinity: Fig. 2.** The figure shows measured ionization potentials as solid bars. The paper
  compiles them from W. Mönch, *Semiconductor Surfaces and Interfaces* (Springer, 2001). It defines the
  experimental electron affinity as measured IP minus the Table I gap.
  - **How the values were read.** The bars are vector graphics, so their positions came from the PDF's
    drawing commands, calibrated on the y-axis ticks. They were not read by eye. The check: for every
    material, extracted IP − EA equals the Table I gap to 0.01 eV.
  - **Cross-check against Ioffe.** The measured III-V electron affinities in the same figure agree with
    Ioffe's handbook values to within ~0.1 eV, which validates the method.

    | Material | This figure | Ioffe |
    |---|---|---|
    | GaAs | 3.88–4.04 | 4.07 |
    | InP | 4.27–4.43 | 4.38 |
    | InSb | 4.53–4.66 | 4.59 |
    | GaSb | 3.89–4.13 | 4.06 |
    | GaP | 3.69–3.75 | 3.80 |

  - **Values used:**

    | Material | IP (eV) | EA (eV) |
    |---|---|---|
    | ZnS | 7.50 | 3.90 |
    | ZnSe | 6.82 | 4.12 |
    | ZnTe | 5.76 | 3.51 |
    | CdSe | 6.62 | 4.89 |
    | CdTe | 5.78 / 5.80 | 4.34 (mean) |
    | AlSb | 5.22 | 3.57 |
    | ZnO, (11-20) and (10-10) surfaces | 7.82 / 8.00 | 4.51 (mean) |
    | CdS | 6.10 / 7.26 | not used (`conflicting`) |

### GOPAL2015: static dielectric constants of ZnS, ZnSe, ZnTe, CdTe (literature_measured, derived)

Gopal P., Fornari M., Curtarolo S., Agapito L.A., Liyanage L.S.I., Buongiorno Nardelli M. (2015). *Improved
predictions of the physical properties of Zn- and Cd-based wide band-gap semiconductors: a validation of
the ACBN0 functional.* [arXiv:1505.05245](https://arxiv.org/abs/1505.05245). Full text read.

The paper reports measured high-frequency dielectric constants ε∞ (Table VIII) and Γ-point TO/LO phonon
frequencies (Table VII) in its "Experiment" columns. Its primary data are Hennion et al., Phys. Lett. A 36,
376 (1971); Vagelatos et al., J. Chem. Phys. 60, 3613 (1974); and Rowe et al., Phys. Rev. B 10, 671
(1974). The static constant follows from the Lyddane–Sachs–Teller relation, ε₀ = ε∞ (ω_LO/ω_TO)²:

| Material | ε∞ | ω_TO (cm⁻¹) | ω_LO (cm⁻¹) | ε₀ | Commonly quoted |
|---|---|---|---|---|---|
| ZnS | 5.13 | 273 | 344 | 8.15 | ~8.0–8.3 |
| ZnSe | 5.70 | 213 | 254 | 8.11 | ~9.1 (**lower here**; `value_high` = 9.2) |
| ZnTe | 7.2 | 175 | 204 | 9.78 | ~9.7–10.1 |
| CdTe | 7.1 | 138 | 167 | 10.40 | ~10.2–10.6 |

### PLANELLES_SI: static dielectric constants of CdSe, CdS (literature_measured)

Planelles J., Rajadell F., Climente J.I. *Electronic Origin of Linearly Polarized Emission in CdSe/CdS
Dot-in-Rod Heterostructures.* [arXiv:1611.00973](https://arxiv.org/abs/1611.00973). The values come from
its supplementary material-parameter table (full text read). That table cites the measurements through
Landolt-Börnstein III/41B:

- CdSe: Geick, Perry & Mitra, J. Appl. Phys. 37, 1994 (1966)
- CdS: Ninomiya & Adachi, J. Appl. Phys. 78, 1183 (1995)

| Material | ε⊥ | ε∥ | Used: (2ε⊥ + ε∥)/3 |
|---|---|---|---|
| CdSe (wurtzite) | 9.29 | 10.16 | 9.58 |
| CdS (wurtzite) | 8.28 | 8.73 | 8.43 |

These are wurtzite values. The dataset uses zinc-blende lattice constants for CdSe and CdS. The static
dielectric constants of the two polytypes differ much less than the sources' own spread.

### Secondary sources (recorded, not used)

| Key | Source | Used for | How much was read |
|---|---|---|---|
| VARLEY2024 | Varley J.B., Weber J.R., Janotti A., Van de Walle C.G. (2024). *Dangling bonds, the charge neutrality level, and band alignment in semiconductors.* J. Appl. Phys. 135, 075703. [doi:10.1063/5.0190043](https://doi.org/10.1063/5.0190043) | Experimental gaps: ZnSe 2.83, ZnS 3.84, CdS 2.50 eV | Full text (OSTI preprint), Table II gap column. Superseded by Hinuma 2014, whose gaps are the ones its electron affinities were derived with. |
| PARAJULI2023 | Parajuli D. et al. (2023). *Numerical assessment of optoelectrical properties of ZnSe-CdSe solar cell-based with ZnO antireflection coating layer.* Sci. Rep. 13. [doi:10.1038/s41598-023-38906-z](https://doi.org/10.1038/s41598-023-38906-z) | χ: CdSe 4.56, ZnSe 4.09 eV; ε: ZnSe 9.2 | Full text, Table 1. These are simulation inputs. |
| WIKI_CdSe | Wikipedia, *Cadmium selenide* (infobox cites Ninomiya & Adachi, J. Appl. Phys. 78, 4681 (1995)) | CdSe gap 1.74 eV | Infobox only |
| WIKI_ZnTe | Wikipedia, *Zinc telluride* | ZnTe gap 2.26 eV, a 6.101 Å | Infobox only |
| SWANK1967 | Swank R.K. (1967). *Surface properties of II-VI compounds.* Phys. Rev. 153, 844. [doi:10.1103/PhysRev.153.844](https://doi.org/10.1103/PhysRev.153.844) | ZnTe χ 3.53 eV | Search summary only. It agrees with the 3.51 eV measured value above. |
| M3, M4, M5 | See `literature_analysis/SOURCES.md` | χ for CdSe/ZnS from a band-edge figure and a SCAPS table; ZnS ε | Search summaries |
| PB_SEARCH | Web search summaries | PbS 0.41 and PbSe 0.28 eV bulk gaps | Search summary only |

## Check against known pairs

With these values, the dataset's band-alignment rule reproduces the well-established cases:

| Pair | dEc / dEv (eV) | Result | Known |
|---|---|---|---|
| CdSe/ZnS | −0.99 / −0.88 | Type I | Type I. Both offsets fall inside the literature ranges in `literature_analysis/literature_pair_offsets.csv`. |
| InP/ZnSe | −0.26 / −1.10 | Type I | Type I. The faculty dataset classified it Type III. |
| CdSe/ZnSe | −0.77 / −0.20 | Type I, small VBO | Photoemission VBO 0.13 ± 0.07 eV (seen in a search summary only) |
| CdTe/CdSe, ZnTe/ZnSe | | Type II | Type II, textbook |
| InAs/GaSb | | Type III (broken gap) | Type III, textbook |

There are two known limitations:

- **The electron-affinity rule fails for GaN/AlN.** Offsets follow Anderson's electron-affinity rule
  (`dEc = shell_χ − core_χ`), which ignores interface dipoles. That rule classifies GaN/AlN as type II, but
  it is type I. AlN's handbook χ = 0.6 eV is a surface measurement: it gives a conduction-band offset of
  3.5 eV, larger than the 2.6 eV gap difference. Treat pairs involving AlN with caution.
- **The InP/ZnSe offsets sit at the edge of the literature range.** The conduction offset (−0.26 eV) is
  smaller than the literature's −0.4…−0.8, and the valence offset (−1.10 eV) slightly larger than its
  −0.57…−1.0. The classification is still right.

A more robust alternative is measured valence-band offsets on a common scale, e.g. Vurgaftman, Meyer &
Ram-Mohan, J. Appl. Phys. 89, 5815 (2001), for III-Vs. That would change how `dEc_eV`/`dEv_eV` are
computed, and it is not done here.

## What would add more materials

- **CdS:** resolve its electron affinity. A third measurement, or a source explaining which surface the
  6.10 vs 7.26 eV IPs belong to, would settle it.
- **AlSb, ZnO as shells:** each needs a measured static dielectric constant (as ε∞ with TO/LO phonons, or
  directly).
- **PbS, PbSe:** need everything except the band gap.
- **AlP:** needs a measured ionization potential or electron affinity.
