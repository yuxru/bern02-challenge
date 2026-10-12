"""
HOMO-LUMO gap of QM9 molecules, culled version
=====================================================================================

Pipeline:
    1. Read every .xyz file in data/dsgdb9nsd.xyz/ (only index, gap, SMILES).
    2. Select molecules: remove the 3054 "uncharacterized" ones, unreadable SMILES,
       and merge duplicate structures.
    3. Compute 17 chemically meaningful variables from the SMILES (structure only).
    4. Split 80 % training / 20 % test, show correlated variable pairs, and drop
       the variables YOU choose (DROP_VARIABLES).
    5. Baseline + HistGradientBoostingRegressor with DEFAULT settings.
    6. Evaluate on the test set (MAE, RMSE, R2, parity plot).
    7. Interpret: permutation importance and direction of each effect.

Not used as input: DFT coordinates, alpha, mu, and every other DFT column
(they come from the same calculation as the gap -> leakage).

One-time setup in the VS Code terminal (from the repo root):
    uv add numpy pandas matplotlib scikit-learn rdkit
    uv add --dev ipykernel           # needed to run the "# %%" cells interactively
Then: Ctrl+Shift+P -> "Python: Select Interpreter" -> .venv
"""

# %% ---------------------------------------------------------------------------------
# 0. Imports and settings
# ------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence

RDLogger.DisableLog("rdApp.*")   # hide RDKit warnings for the few odd molecules


def find_repo_root():
    """Walk up from the current folder until we find pyproject.toml (the uv project root).
    This makes the paths work whether you run from the terminal or cell by cell, where
    VS Code starts in the folder of this file."""
    folder = Path.cwd().resolve()
    while not (folder / "pyproject.toml").exists():
        if folder.parent == folder:   # reached the top of the drive without finding it
            raise FileNotFoundError("pyproject.toml not found - open the bern02-challenge folder")
        folder = folder.parent
    return folder


ROOT = find_repo_root()
XYZ_DIR = ROOT /"data"/"qm9"/"dsgdb9nsd.xyz"
UNCHARACTERIZED_FILE = ROOT /"data"/"qm9"/"uncharacterized.txt"
RESULTS_DIR = ROOT /"results"/"cull"
RESULTS_DIR.mkdir(exist_ok=True)

HARTREE_TO_EV = 27.211386   # QM9 stores orbital energies in Hartree
SEED = 12                   # same split and same model on every run
TEST_FRACTION = 0.20

DROP_VARIABLES = ["n_small_rings", "n_0", "n_nitrile", "n_F", "molar_refractivity", "n_heavy_atoms", "n_ether", "tpsa"]


# %% ---------------------------------------------------------------------------------
# 1. Read the .xyz files (only what we need)
# ------------------------------------------------------------------------------------
# Layout of one QM9 .xyz file:
#   line 1         : number of atoms n
#   line 2         : "gdb <index>" + 15 properties (A, B, C, mu, alpha, homo, lumo, gap, ...)
#   lines 3..n+2   : atoms + DFT coordinates   -> skipped
#   line n+3       : vibrational frequencies   -> skipped
#   line n+4       : SMILES (GDB-17 original, and after DFT relaxation)
#   line n+5       : InChI                     -> skipped

def read_xyz(path):
    """Return index, gap (eV) and relaxed-geometry SMILES of one molecule."""
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0])
    props = lines[1].replace("*^", "e").split()   # "1.2*^-6" (Mathematica style) -> "1.2e-6"
    # props: 0 "gdb", 1 index, 2-4 A B C, 5 mu, 6 alpha, 7 homo, 8 lumo, 9 gap
    return {
        "index": int(props[1]),
        "gap_eV": float(props[9]) * HARTREE_TO_EV,
        "smiles": lines[n_atoms + 3].split()[1],   # [1] = SMILES after DFT relaxation
    }


xyz_files = sorted(XYZ_DIR.glob("*.xyz"))
print(f"Found {len(xyz_files)} .xyz files in {XYZ_DIR}")   # expect 133885
qm9 = pd.DataFrame([read_xyz(p) for p in xyz_files])        # takes a minute or two


# %% ---------------------------------------------------------------------------------
# 2. Select the molecules we can trust
# ------------------------------------------------------------------------------------
n_read = len(qm9)

# 2a. Remove molecules whose bonding changed during the DFT optimisation: their SMILES
#     no longer matches the molecule the gap was computed for. Data lines start with
#     the molecule index; header and footer lines do not.

# uses the QM9 paper list of unchar mols to remove them from our set.
bad_index = set()
for line in UNCHARACTERIZED_FILE.read_text().splitlines():
    parts = line.split()
    if parts and parts[0].isdigit():
        bad_index.add(int(parts[0]))
qm9 = qm9[~qm9["index"].isin(bad_index)]          # "~" means NOT
n_characterized = len(qm9)

# 2b. SMILES text -> RDKit molecule. MolFromSmiles returns None if it cannot read it.
qm9["mol"] = [Chem.MolFromSmiles(s) for s in qm9["smiles"]]
qm9 = qm9[qm9["mol"].notna()]
n_parsed = len(qm9)

# 2c. Duplicate structures: same canonical SMILES = same structure = identical variables,
#     so the model can only give them ONE prediction. Keep one row per structure with
#     the MEAN gap; otherwise a copy could sit in both training and test (leakage).
qm9["canonical"] = [Chem.MolToSmiles(m) for m in qm9["mol"]]
molecules = (qm9.groupby("canonical", sort=False)
                .agg(index=("index", "first"), gap_eV=("gap_eV", "mean"),
                     mol=("mol", "first"), n_copies=("index", "size"))
                .reset_index())

print(f"Read {n_read} -> characterized {n_characterized} -> parsed {n_parsed} "
      f"-> unique structures {len(molecules)}")


# %% ---------------------------------------------------------------------------------
# 3. Chemically meaningful variables (from the structure only)
# ------------------------------------------------------------------------------------
# Physics: the gap shrinks when pi electrons are delocalised over a larger region;
# lone pairs and strained rings raise the HOMO; C=O and C#N have low-lying pi* (lower LUMO).
# Hypotheses written BEFORE training: "-" gap decreases, "+" gap increases, "?" unclear.
EXPECTED_EFFECT = {
    "n_conjugated_bonds": "-", "n_aromatic_rings": "-", "frac_sp3_carbon": "+",
    "n_pi_electrons": "-", "n_carbonyl": "-", "n_nitrile": "-", "n_ether": "-",
    "n_N": "-", "n_O": "-", "n_F": "?", "n_small_rings": "-", "n_rings": "?",
    "n_heavy_atoms": "?", "molar_refractivity": "?", "tpsa": "?",
    "max_partial_charge": "?", "min_partial_charge": "?",
}

# pi electrons per bond. RDKit gives aromatic bonds their own type (benzene has 0 DOUBLE
# bonds); 6 aromatic bonds share 6 pi electrons -> 1 each. Approximation: lone pairs
# inside aromatic rings (pyrrole N-H) are not counted.
PI_ELECTRONS = {Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 4, Chem.BondType.AROMATIC: 1}


def chemistry_variables(mol):
    """The 17 interpretable variables of one molecule."""
    bonds = list(mol.GetBonds())
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {
        # pi-conjugation (an isolated pi bond, e.g. C#N in acetonitrile, is NOT "conjugated")
        "n_conjugated_bonds": sum(b.GetIsConjugated() for b in bonds),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "frac_sp3_carbon": rdMolDescriptors.CalcFractionCSP3(mol),  # saturation
        "n_pi_electrons": sum(PI_ELECTRONS.get(b.GetBondType(), 0) for b in bonds),
        # functional groups and heteroatoms
        "n_carbonyl": Descriptors.fr_C_O(mol),       # C=O, also inside acids/amides
        "n_nitrile": Descriptors.fr_nitrile(mol),    # C#N
        "n_ether": Descriptors.fr_ether(mol),        # C-O-C
        "n_N": atoms.count("N"),
        "n_O": atoms.count("O"),
        "n_F": atoms.count("F"),
        # ring strain (3- and 4-membered rings) and topology
        "n_small_rings": sum(size <= 4 for size in ring_sizes),
        "n_rings": len(ring_sizes),
        # size, polarisability (empirical estimate, NOT the DFT alpha), polarity
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "molar_refractivity": Descriptors.MolMR(mol),
        "tpsa": Descriptors.TPSA(mol),                       # polar surface area, A^2
        "max_partial_charge": Descriptors.MaxPartialCharge(mol),  # Gasteiger; can be NaN,
        "min_partial_charge": Descriptors.MinPartialCharge(mol),  # the model handles NaN
    }


# Floats, because partial_dependence (step 7) refuses integer columns.
X = pd.DataFrame([chemistry_variables(m) for m in molecules["mol"]]).astype(float)
y = molecules["gap_eV"].to_numpy()
print(X.describe().T[["mean", "min", "max"]].round(2))   # are the ranges sensible?


# %% ---------------------------------------------------------------------------------
# 4. Split 80/20, inspect correlations, drop YOUR chosen variables
# ------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, molecules["index"], test_size=TEST_FRACTION, random_state=SEED)
print(f"Training: {len(X_train)}   Test: {len(X_test)}")

# Correlations on the TRAINING set only (using test data to choose variables is leakage).
# Spearman = rank correlation, suitable for counts and non-linear but monotone relations.
rho = X_train.corr(method="spearman")
pairs = (rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1))   # upper triangle only
            .stack().reset_index())                                 # one row per pair
pairs.columns = ["variable_1", "variable_2", "rho"]
pairs = pairs.reindex(pairs["rho"].abs().sort_values(ascending=False).index)
print("Most correlated variable pairs (training set):")
print(pairs.head(10).round(2).to_string(index=False))
rho.round(2).to_csv(RESULTS_DIR / "correlation_train_cull.csv")

# Apply YOUR decision from the settings block.
features = [name for name in X.columns if name not in DROP_VARIABLES]
X_train, X_test = X_train[features], X_test[features]
print(f"Using {len(features)} variables; dropped: {DROP_VARIABLES or 'none'}")


# %% ---------------------------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------------------------------
# 5a. Baseline: always predict the average training gap. The real model must beat this.
baseline_pred = np.full(len(y_test), y_train.mean())

# 5b. Gradient boosting with DEFAULT settings: up to 100 small trees, each correcting
#     the errors of the trees before it. With >10,000 molecules the default switches on
#     early stopping (it holds back 10 % of the training data to decide when to stop).
model = HistGradientBoostingRegressor(random_state=SEED)
model.fit(X_train, y_train)
# If this prints 100, the default limit (max_iter=100) was reached before early stopping
# kicked in: max_iter is then the first setting worth raising later.
print(f"Trees used: {model.n_iter_}")


# %% ---------------------------------------------------------------------------------
# 6. Evaluate on the test set
# ------------------------------------------------------------------------------------
y_pred = model.predict(X_test)

scores = pd.DataFrame({
    "baseline (mean)": [mean_absolute_error(y_test, baseline_pred),
                        root_mean_squared_error(y_test, baseline_pred),
                        r2_score(y_test, baseline_pred)],
    "gradient boosting": [mean_absolute_error(y_test, y_pred),
                          root_mean_squared_error(y_test, y_pred),
                          r2_score(y_test, y_pred)],
}, index=["MAE (eV)", "RMSE (eV)", "R2"]).T
print(scores.round(3))

# Parity plot: ~26k points would be one solid blob as a scatter, so count per hexagon.
fig, ax = plt.subplots(figsize=(5.5, 5))
hb = ax.hexbin(y_test, y_pred, gridsize=60, mincnt=1, bins="log", cmap="Blues")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, color="0.3", linewidth=1)          # perfect prediction
ax.set_xlabel("DFT (B3LYP) HOMO-LUMO gap [eV]")
ax.set_ylabel("Predicted gap [eV]")
ax.set_title(f"Test set, MAE = {scores.loc['gradient boosting', 'MAE (eV)']:.2f} eV")
fig.colorbar(hb, label="molecules per hexagon")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "parity_plot_cull.png", dpi=200)


# %% ---------------------------------------------------------------------------------
# 7. Interpretation
# ------------------------------------------------------------------------------------
# 7a. Permutation importance: shuffle ONE variable in the test set and measure how much
#     the MAE grows. Correlated variables share the credit, which is one reason to
#     remove redundant variables in step 4.
perm = permutation_importance(model, X_test, y_test, scoring="neg_mean_absolute_error",
                              n_repeats=5, random_state=SEED)
importance = pd.Series(perm.importances_mean, index=features)

# 7b. Direction: partial dependence = average prediction as one variable moves from its
#     5th to its 95th percentile. Only read the sign for variables with clear
#     importance; for unimportant ones the sign is noise.
observed = {}
for name in features:
    avg = partial_dependence(model, X_train, features=[name], kind="average")["average"][0]
    observed[name] = "too rare" if len(avg) < 2 else ("+" if avg[-1] > avg[0] else "-")

summary = pd.DataFrame({
    "importance (eV)": importance.round(3),
    "expected": [EXPECTED_EFFECT[n] for n in features],
    "observed": [observed[n] for n in features],
}).sort_values("importance (eV)", ascending=False)
print(summary.to_string())

fig, ax = plt.subplots(figsize=(6, 5))
ordered = importance.sort_values()
ax.barh(ordered.index, ordered.values, color="#3b6ea8")
ax.set_xlabel("Increase in test MAE when shuffled [eV]")
ax.set_title("Which variables does the model rely on?")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "permutation_importance_cull.png", dpi=200)


# %% ---------------------------------------------------------------------------------
# 8. Save results (figures and later comparisons can be redone without retraining)
# ------------------------------------------------------------------------------------
pd.DataFrame({"qm9_index": id_test.to_numpy(), "gap_dft_eV": y_test,
              "gap_pred_eV": y_pred}).to_csv(RESULTS_DIR / "test_predictions_cull.csv", index=False)
scores.to_csv(RESULTS_DIR / "test_scores_cull.csv")
summary.to_csv(RESULTS_DIR / "interpretation_cull.csv")
print("Done. Results in", RESULTS_DIR)
