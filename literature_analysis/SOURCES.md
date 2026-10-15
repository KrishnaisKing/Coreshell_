# Literature analysis: sources

Every measured value in `measured_devices.csv` and every literature constant in
`literature_material_constants.csv` / `literature_pair_offsets.csv` traces to a source below. Nothing is
estimated or invented; a blank cell means the source didn't report it. Each source is labelled with how
much of it was actually read, because that varies.

## Device papers (`measured_devices.csv`)

| Device id | Source | How much was read |
|---|---|---|
| `Kannan2013_CdSe_ZnS` | Kannan V., Rhee J.K. (2013). *Robust switching characteristics of CdSe/ZnS quantum dot non-volatile memory devices.* Phys. Chem. Chem. Phys. 15, 12762–12766. [doi:10.1039/c3cp50216c](https://doi.org/10.1039/c3cp50216c) | Abstract (Europe PMC). QD diameter 5.2 nm from a search summary only. |
| `Kannan2016_CdSe_ZnS_TiTiOx` | Kannan V., Kim H.S., Park H.C. (2016). *High speed switching in quantum Dot/Ti-TiOx nonvolatile memory device.* Electron. Mater. Lett. [doi:10.1007/s13391-015-5410-5](https://doi.org/10.1007/s13391-015-5410-5) | Search excerpts only (from the first-pass seed). |
| `Yagati2017_Azurin_CdSe_ZnS` | Yagati A.K., Lee T., Choi J.-W. (2017). *Azurin/CdSe-ZnS-based bio-nano hybrid structure for nanoscale resistive memory device.* Materials 10, 803. [doi:10.3390/ma10070803](https://doi.org/10.3390/ma10070803) | Full text (Europe PMC). |
| `Kim2021_CdSe_ZnS_multilevel` | Kim S.-W., Kwon J., Lee J.-S., et al. (2021). *An organic/inorganic nanomaterial and nanocrystal quantum dots-based multi-level resistive memory device.* Nanomaterials 11, 3004. [doi:10.3390/nano11113004](https://doi.org/10.3390/nano11113004) | Full text (Europe PMC). |
| `Kim2018_InP_ZnSe`, `Kim2018_InP_ZnSe_ZnS` | Kim D.H., Wu C., Park D.H., Kim W.K., Seo H.W., Kim S.W., Kim T.W. (2018). *Flexible memristive devices based on InP/ZnSe/ZnS core–multishell quantum dot nanocomposites.* ACS Appl. Mater. Interfaces 10, 14843–14849. [doi:10.1021/acsami.7b18817](https://doi.org/10.1021/acsami.7b18817) | Abstract (Europe PMC). |
| `Matter2025_CdSe_CdS_MoS2` | *Core-shell quantum dot-enabled monolayer MoS2 memories with high endurance.* Matter (Cell Press), 2025. [doi:10.1016/j.matt.2025.102488](https://doi.org/10.1016/j.matt.2025.102488) | Abstract excerpt only; author list not confirmed. |
| `ACSAMI2019_CdS_NR_MAPbBr3` | *High-performance nanofloating gate memory based on lead halide perovskite nanocrystals.* ACS Appl. Mater. Interfaces 11(27), 24367–24377 (2019). [doi:10.1021/acsami.9b03474](https://doi.org/10.1021/acsami.9b03474) | Abstract excerpt only; author list not confirmed. |

**Found but left out** (numbers seen only in search summaries, device details unverified): a write-once
CdSe/ZnS + ZnO nanoparticle memory (Thin Solid Films 2020, on/off 1.43×10⁶; write-once, so it isn't a
rewritable memory anyway) and a PVK/QD multiple-read memory (Applied Surface Science 2019, on/off 3.78×10⁵;
QD material not confirmed).

## Material constants (`literature_material_constants.csv`, `literature_pair_offsets.csv`)

| Key | Source | Used for | How much was read |
|---|---|---|---|
| M1 | *Optical characteristics of ZnS passivated CdSe/CdS quantum dots for high photostability and lasing.* Sci. Rep. (2018). [nature.com/articles/s41598-018-35768-8](https://www.nature.com/articles/s41598-018-35768-8) | Bulk gaps: CdSe 1.75, CdS 2.50, ZnS 3.8 eV | Search summary |
| M2 | Varley J.B., Weber J.R., Janotti A., Van de Walle C.G. (2024). *Dangling bonds, the charge neutrality level, and band alignment in semiconductors.* J. Appl. Phys. 135, 075703. [doi:10.1063/5.0190043](https://doi.org/10.1063/5.0190043) | Experimental gaps (Table II): CdSe 1.90, CdS 2.50, ZnS 3.84, ZnSe 2.83, InP 1.42 eV | Full text (OSTI preprint). Only the gap column was used; the offset columns didn't extract cleanly from the PDF. |
| M3 | Band-edge diagram, *Valence and conduction band alignment for CdSe core quantum dots with ZnS green shell* (figure, [ResearchGate](https://www.researchgate.net/figure/Valence-and-conduction-band-alignment-for-CdSe-core-quantum-dots-with-ZnS-green-shell_fig3_233829213)) | CdSe CB −4.8 / VB −6.8 eV; ZnS CB −3.4 / VB −7.4 eV vs vacuum | Search summary of a figure; originating paper not identified. Weakest source here. |
| M4 | SCAPS-1D material-parameter table ([ResearchGate](https://www.researchgate.net/figure/Material-parameters-used-for-the-simulation-in-SCAPS-1D_tbl2_283664370)) | ZnS Eg 3.52 eV, χ 4.1 eV, ε 10 | Search summary. Simulation inputs, not measurements. |
| M5 | *ZnS: dielectric constants* (Landolt-Börnstein data sheet, [ResearchGate](https://www.researchgate.net/publication/315553540_ZnS_dielectric_constants)) | Cubic ZnS ε∞ 5.1, εs 8.0 | Search summary |
| M6 | *Energy-level structure and band alignment in InP/ZnSe core/shell quantum dots.* ACS Nano (2025). [doi:10.1021/acsnano.5c02258](https://doi.org/10.1021/acsnano.5c02258) | InP/ZnSe is type I; CBO ~0.4–0.8 eV, VBO ~0.57–1.0 eV | Search summary |
| M7 | *Band offsets in InP/ZnSe nanocrystals evaluated using two-photon transitions analysis.* [arXiv:2603.13074](https://arxiv.org/html/2603.13074) | Same offset range as M6 | Search summary |

**How much to trust these constants.** Most came from search summaries, not full texts, and sources
disagree with each other by ~0.5 eV on ZnS — hence ranges rather than single values. That's enough for
the conclusion drawn from them (the dataset's electron affinities sit near 0 eV where every source puts
them at 3.4–4.8 eV, a gap no source disagreement explains), but the exact offset ranges should be
re-checked against full texts before going into a paper.
