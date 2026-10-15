# FAIR Reflection

This project aimed to predict the DFT HOMO-LUMO gap of small organic molecules from SMILE-derived features. Using QM9 dataset, gradient-boosting with a Ridge regression. 
We also test the model on aspirin, which is larger than any QM9 molecule, against an ORCA DFT reference; that calculation is still running at the time of submission.

Below, each FAIR principle is dicussed for following objects:
1. **data** input QM9 data and output prediction data.
2. **workflow** code, environment and analysis steps.

## F - Findable

**data**
QM9 is cited by its article DOI ([10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)) and its figshare collection DOI ([10.6084/m9.figshare.c.978904.v5](https://doi.org/10.6084/m9.figshare.c.978904.v5)).

Every molecule keeps its original QM9 index (`dsgdb9nsd_XXXXXX`), and each analysis run writes to its own folder (`results/gap_x`) with descriptive file names.

**workflow**
the code is in a public GitHub repository, <https://github.com/yuxru/bern02-challenge>. A Zenodo release with a citable DOI is planned, not yet done.

## A - Accessible

**data**
QM9 is openly downloadable from figshare (CC0 1.0). The README gives the download link and the expected layout (`data/qm9/dsgdb9nsd.xyz/`, `data/qm9/uncharacterized.txt`), and QM9's own format description (`data/qm9/qm9_readme.txt`) is kept in the repo. Our own results (tables, figures) are small and committed.

**workflow**
All code needed to reproduce the results is in the repository and uses free, open-source software: Python, RDKit, scikit-learn, SHAP, NumPy, pandas, Matplotlib. Two exceptions: LGV (built at Lund University, not yet released), used to build aspirin, and ORCA (free for academics, not redistributable), used for the DFT reference. We therefore plan to commit the exported .xyz and the ORCA input/output files once the calculation has finished, so later steps can be reproduced or checked without either tool.

## I - Interoperable

**data**
open formats only: .xyz (3D structure), SMILES (connectivity), CSV (tables), PNG (figures). Units are explicit (QM9's Hartree converted to eV in one documented line). Our aspirin reference is set up with the same level of theory as QM9, B3LYP/6-31G(2df,p) (ORCA keyword B3LYP/G, the Gaussian variant), so the gap values will be directly comparable.

**workflow**
standard Python interfaces (RDKit molecules, pandas DataFrames, scikit-learn estimators). Features come from SMILES, so the trained model can be applied to any new molecule given as a SMILES string, as we do for aspirin.

## R - Reusable 

**data**
QM9 is CC0; our generated results and figures are CC BY 4.0. The notebook documents every cleaning step with the number of molecules removed: uncharacterized list → SMILES RDKit cannot parse → duplicate canonical SMILES merged (mean gap), which also prevents train/test leakage. Each feature is described with its expected effect on the gap.

**workflow**
code is MIT-licensed (`LICENSE`). `uv` locks the Python and package versions (`.python-version`, `pyproject.toml`, `uv.lock`), so `uv sync` recreates the environment. The notebook `gap_model.ipynb` runs from raw data to figures with all settings at the top; one seed (`SEED = 42`) controls the train/test split, the models and SHAP sampling. The `results/` folder is created by the code, so a fresh clone runs without manual setup. Git keeps the history.

## Limitations

- The ORCA reference for aspirin is not finished at submission, so the ML-vs-DFT comparison is not yet in the repository.
- No Zenodo DOI yet, so the repository is findable but not yet permanent.
- LGV is unreleased and ORCA cannot be redistributed: outsiders cannot build a new structure the same way or fully re-run the DFT step; inputs and outputs are shared instead.
- The model only knows QM9 (≤ 9 heavy atoms; C, H, O, N, F; neutral molecules). Aspirin (13 heavy atoms) is extrapolation, which we document and quantify rather than hide.
- The gap is a computed orbital-energy difference, not a measured spectrum: the model learns B3LYP's values, including its errors, not experiment.


## References

Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1, 140022. https://doi.org/10.1038/sdata.2014.22

Wilkinson, M. D. et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship.
*Scientific Data*, 3, 160018. https://doi.org/10.1038/sdata.2016.18