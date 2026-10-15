# FAIR Reflection

This project aimed to predict the DFT HOMO-LUMO gap of small organic molecules from their SMILE codes.

The QM9 dataset was explored and statistical learning, gradient-boosted trees, with linear and neural-network (TODO) comparisons were applied to test how well the trained model extroplates to a larger molecule over anyone in QM9 (TODO, aspirin with 13 heavy atoms).

Below, each FAIR principle is dicussed for following objects:
1. **data** input QM9 data and output prediction data.
2. **workflow** code, environment and analysis steps.

| | Data | Workflow |
|---|---|---|
| **F**indable | QM9 cited by paper DOI and dataset DOI; our outputs in `results/` with descriptive names | Public GitHub repository; release archived on Zenodo with a DOI (planned) |
| **A**ccessible | QM9 openly downloadable (CC0); our generated data committed to the repo | Notebook and scripts in the repo, run with free, open-source tools |
| **I**nteroperable | Standard formats (xyz, SMILES, CSV, PNG); units stated (eV) | Python + standard libraries; environment locked with `uv` |
| **R**eusable | Licences stated; cleaning steps and removed molecules documented | MIT licence; fixed random seed; one notebook runs end to end |

---

## F - Findable

**data**
QM9 is the original data that cited in the ge_readme.txt (should be in README) by the article: Artem Fediai, Patrick Reiser, Jorge Enrique Olivares Peña, Pascal Friederich, Wolfgang Wenzel:
Accurate GW frontier orbital energies of 134 kilo molecules, arXiv, 2023, https://doi.org/10.48550/ARXIV.2303.08708
Persistent identifiers mean the exact data version can be found again even if web pages move.
Every molecule keeps its original QM9 index (`dsgdb9nsd_XXXXXX`), so any row in our tables can be
traced back to its source file.

Outcome data is saved in `results/` with ..... 

**workflow**
The code is in a public (or not) Github repository: https://github.com/yuxru/bern02-challenge.git. The README states the questions and instructions.
Data source and how to run the analysis, with keywords QM9, HOMO-LUMO gap, machine learninf, RDKit, scikit-learn. So the repository can be found. 

## A - Accessible

**data**
QM9 is accessed from figshare without registration, released under CC0 1.0.

The raw data '.xyz' files of QM9 was not redistributed, due to they are large and already
permanently hosted at the original source.

Output data were generated is small and committed directly: 1. the aspirin structure; 2. the ORCA input and output file for its DFT calculateion. 

**workflow**
All code needed to reproduce the results is in the repository and uses free, open-source software: Python, RDKit, scikit-learn, SHAP, NumPy, pandas, Matplotlib.

Two tools sit partly outside this:
- **LGV** 
Was used to build the aspirin structure.      
LGV is being developed at Lund University
and is not yet publicly released. To keep the workflow accessible without it, the `.xyz` file was exported, so every later step can be reproduced without LGV.

- **ORCA** 
Used for the aspirin DFT reference is free for academic use but cannot be redistributed. 

## I - Interoperable
**data**
All data uses common, non-proprietart formats that other software can read:
* `.xyz` for 3D structure, **SMILES** for molecular connectivity, **CSV** for tables, and PNG for
figures. Units are stated explicitly: QM9 stores orbital energies in Hartree, and it converted to eV (1 Ha = 27.211386 eV)for gap in a single, documented line of code.

**workflow**
Python libraries were used for analysis with standard interfaces: RDKit molecules,
pandas DataFrames, scikit-learn estimators, so individual steps such as the feature
functions can be reused in other projects. Features are derived from SMILES, a standard format, so the trained model can be applied to any new molecule given as a SMILES string.

## R - Reusable 
**data**
1. **licence:** QM9 is CC0 1.0. output results and figures are shared under CC BY 4.0 
2. **Provenance:** the README and notebook document every cleaning step and how many molecules each step removes: all files read → molecules on the official *uncharacterized* list removed (failed geometry check) → SMILES that RDKit cannot parse removed → duplicate structures (same canonical SMILES) merged by averaging their gap.