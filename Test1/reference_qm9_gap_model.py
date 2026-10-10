"""
REFERENCE GUIDE (not for submission) - Predicting the HOMO-LUMO gap of QM9 molecules
=====================================================================================
Version 2: updated with the recommendations from our decision log.

Steps:
    1. Read the QM9 archive, keeping ONLY what we need (molecule id, gap, SMILES).
    2. Select the molecules we can trust; choose the target (B3LYP or GW);
       merge duplicate structures.
    3. Describe each molecule with ~17 chemically meaningful variables (structure only)
       and estimate the error floor of this description.
    4. Split 80/20; remove near-duplicate variables using the TRAINING set only.
    5. Models: baseline -> gradient boosting with defaults -> tuned gradient boosting.
    6. Evaluate once on the untouched test set.
    7. Interpret: permutation importance, direction check, drop-and-refit.
    8. Robustness: repeated random splits, and one held-out chemical family.
    9. Save everything needed to reproduce figures and decisions.

NOT used as input (and why):
    * DFT coordinates: a new molecule would first need a DFT optimisation.
    * alpha, mu and all other DFT columns: same calculation as the gap -> leakage.
    HOMO and LUMO are read only to check that gap = LUMO - HOMO.

Files expected:
    data/dsgdb9nsd.xyz.tar.bz2   QM9 (Figshare)
    data/uncharacterized.txt     3054 molecules that failed the geometry check
    data/gw_gaps.csv             ONLY if TARGET = "GW" (see step 2b for how to make it)

Workflow tips (decision 1):
    * "# %%" lines are cells: run one at a time in VS Code (Shift+Enter).
    * Before every commit or hand-in: restart the kernel and run the WHOLE file top to
      bottom, so no result depends on cells run out of order.
    * For submission: Command Palette -> "Jupyter: Export Current Python File as Jupyter
      Notebook".
"""

# %% ---------------------------------------------------------------------------------
# 0. Imports and settings
# ------------------------------------------------------------------------------------
import os
import json
import tarfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit
import sklearn
from scipy.stats import loguniform, randint

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence

RDLogger.DisableLog("rdApp.*")  # hide RDKit warnings for the few odd molecules

# --- every decision you might change lives here ---------------------------------------
TARGET = "B3LYP"            # "B3LYP" (in QM9) or "GW" (needs data/gw_gaps.csv), decision 2
QM9_ARCHIVE = "data/dsgdb9nsd.xyz.tar.bz2"
UNCHARACTERIZED_FILE = "data/uncharacterized.txt"
GW_TABLE = "data/gw_gaps.csv"
RESULTS_DIR = "results"

HARTREE_TO_EV = 27.211386   # QM9 stores orbital energies in Hartree
SEED = 42                   # main split and model seed
TEST_FRACTION = 0.20
CORR_THRESHOLD = 0.90       # |Spearman rho| above which two variables count as redundant (decision 4)
N_TUNING_ITER = 40          # random hyperparameter settings to try (decision 6)
CV_FOLDS = 5
REPEAT_SEEDS = [1, 2, 3, 4, 5]   # extra random splits to measure split-to-split variation (decision 7)
N_DROP_REFIT = 5            # how many top variables get the drop-and-refit test (decision 5)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Record the software versions: part of the FAIR / reproducibility story.
VERSIONS = {"rdkit": rdkit.__version__, "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
print("Versions:", VERSIONS)


# %% ---------------------------------------------------------------------------------
# 1. Read the QM9 archive (only the parts we need)
# ------------------------------------------------------------------------------------
# Layout of one .xyz file in QM9:
#   line 1           : number of atoms n
#   line 2           : "gdb <index>" + 15 properties (A, B, C, mu, alpha, homo, lumo, gap, ...)
#   lines 3 .. n+2   : atoms with DFT coordinates   -> SKIPPED (not used)
#   line n+3         : vibrational frequencies      -> skipped
#   line n+4         : SMILES: GDB-17 original, and after DFT relaxation
#   line n+5         : InChI                        -> skipped

def read_qm9_record(text):
    """Return index, homo, lumo, gap (Hartree) and the relaxed-geometry SMILES of one molecule."""
    lines = text.splitlines()
    n_atoms = int(lines[0])
    props = lines[1].replace("*^", "e").split()   # "1.2*^-6" (Mathematica) -> "1.2e-6"
    # props positions: 0 "gdb", 1 index, 2-4 A B C, 5 mu, 6 alpha, 7 homo, 8 lumo, 9 gap
    # mu (5) and alpha (6) are deliberately NOT read.
    smiles_line = lines[n_atoms + 3].split()
    return {
        "index": int(props[1]),
        "homo": float(props[7]),
        "lumo": float(props[8]),
        "gap": float(props[9]),
        "smiles": smiles_line[1],   # [1] = SMILES after DFT relaxation
    }


records = []
with tarfile.open(QM9_ARCHIVE, "r:bz2") as tar:   # read straight from the archive
    for member in tar:
        if member.isfile() and member.name.endswith(".xyz"):
            records.append(read_qm9_record(tar.extractfile(member).read().decode()))

qm9 = pd.DataFrame(records).sort_values("index").reset_index(drop=True)
print(f"Read {len(qm9)} molecules")               # expect 133885 for the real archive


# %% ---------------------------------------------------------------------------------
# 2. Select molecules, choose the target, merge duplicates (the "data funnel")
# ------------------------------------------------------------------------------------
funnel = {"QM9 molecules read": len(qm9)}

# 2a. Remove molecules whose bonding changed during the DFT optimisation (their SMILES
#     no longer describes the molecule the gap belongs to). Data lines start with the index.
bad_index = set()
with open(UNCHARACTERIZED_FILE) as f:
    for line in f:
        parts = line.split()
        if parts and parts[0].isdigit():
            bad_index.add(int(parts[0]))
qm9 = qm9[~qm9["index"].isin(bad_index)]
funnel["after removing uncharacterized"] = len(qm9)

# 2b. The target, y (decision 2).
#     B3LYP: in QM9 already; convert Hartree -> eV and check gap = LUMO - HOMO.
#     GW   : from Fediai et al. (2023). The paper names the eigenvalue-self-consistent
#            GW HOMO/LUMO "occ_scf" / "vir_scf". Inspect the downloaded file FIRST (open it,
#            look at one entry: are values numbers or lists? which unit?), then write a
#            small script that saves data/gw_gaps.csv with two columns:
#                index, gap_gw_eV   (gap = LUMO - HOMO, in eV)
#            Tip: yaml.load(f, Loader=yaml.CSafeLoader) is much faster than the default.
mismatch = (qm9["gap"] - (qm9["lumo"] - qm9["homo"])).abs().max() * HARTREE_TO_EV
print(f"Largest |gap - (LUMO - HOMO)| in QM9 = {mismatch:.4f} eV (should be ~0)")

if TARGET == "B3LYP":
    qm9["target_eV"] = qm9["gap"] * HARTREE_TO_EV
    target_label = "B3LYP HOMO-LUMO gap [eV]"
elif TARGET == "GW":
    gw = pd.read_csv(GW_TABLE)
    # inner join = keep only molecules present in BOTH tables;
    # validate="one_to_one" raises an error if an index appears twice on either side.
    qm9 = pd.merge(qm9, gw, on="index", how="inner", validate="one_to_one")
    qm9 = qm9.dropna(subset=["gap_gw_eV"])
    qm9["target_eV"] = qm9["gap_gw_eV"]
    target_label = "GW HOMO-LUMO gap [eV]"
    funnel["after keeping molecules with a GW value"] = len(qm9)
else:
    raise ValueError("TARGET must be 'B3LYP' or 'GW'")

# 2c. SMILES -> RDKit molecule; MolFromSmiles returns None when it cannot read one.
qm9["mol"] = [Chem.MolFromSmiles(s) for s in qm9["smiles"]]
qm9 = qm9[qm9["mol"].notna()]
funnel["after RDKit parsing"] = len(qm9)

# 2d. Duplicate structures (decision 3). Canonical SMILES = one unique text per structure.
#     Duplicates get identical input variables, so the model can only ever predict ONE
#     value for them; the best single value is their mean. We therefore keep one row per
#     structure with the mean target, and record how much the copies disagreed.
qm9["canonical_smiles"] = [Chem.MolToSmiles(m) for m in qm9["mol"]]
groups = qm9.groupby("canonical_smiles", sort=False)
molecules = groups.agg(
    index=("index", "first"),              # QM9 index of the first copy (our molecule id)
    n_copies=("index", "size"),
    target_eV=("target_eV", "mean"),
    target_spread_eV=("target_eV", lambda v: v.max() - v.min()),
).reset_index()
molecules["mol"] = groups["mol"].first().to_numpy()   # same order as the agg above

dup = molecules[molecules["n_copies"] > 1]
print(f"Structures with copies: {len(dup)}; "
      f"median spread of their targets: {dup['target_spread_eV'].median() if len(dup) else 0:.3f} eV")
funnel["unique structures (duplicates averaged)"] = len(molecules)

for step, count in funnel.items():
    print(f"  {step:45s} {count:7d}")


# %% ---------------------------------------------------------------------------------
# 3. Chemically meaningful variables (structure only)
# ------------------------------------------------------------------------------------
# Physics: the gap shrinks when pi electrons are delocalised over a larger region; lone
# pairs and strained rings raise the HOMO; low-lying pi* orbitals (C=O, C#N) lower the LUMO.
#
# EXPECTED_EFFECT = hypothesis written BEFORE training ("-" gap decreases, "+" increases,
# "?" no clear expectation). Evidence notes are in the decision log.
# The ORDER of this dict is also our interpretability PRIORITY for step 4: when two
# variables are redundant, the one listed first is kept.
EXPECTED_EFFECT = {
    # pi-conjugation (strong evidence)
    "n_conjugated_bonds": "-",
    "n_aromatic_rings": "-",
    "frac_sp3_carbon": "+",
    "n_pi_electrons": "-",
    # functional groups / heteroatoms
    "n_carbonyl": "-",          # strong evidence (aldehydes ~6.5 eV vs alcohols 7.2-7.8 eV)
    "n_nitrile": "-",
    "n_ether": "-",
    "n_N": "-",
    "n_O": "-",
    "n_F": "?",
    # ring strain and topology
    "n_small_rings": "-",
    "n_rings": "?",
    # size, polarisability, polarity
    "n_heavy_atoms": "?",
    "molar_refractivity": "?",  # changed from "-": polarisability vs gap R^2 = 0.13 in QM7-X
    "tpsa": "?",
    "max_partial_charge": "?",
    "min_partial_charge": "?",
}

# pi electrons per bond: double 2, triple 4. RDKit gives aromatic bonds their own type
# (benzene has 0 DOUBLE bonds); 6 aromatic bonds share 6 pi electrons -> 1 per bond.
# Approximation: lone pairs inside aromatic rings (pyrrole N-H) are not counted.
PI_ELECTRONS_PER_BOND = {Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 4,
                         Chem.BondType.AROMATIC: 1}


def chemistry_variables(mol):
    """Compute the interpretable variables for one RDKit molecule."""
    bonds = list(mol.GetBonds())
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
    return {
        # pi-conjugation. NOTE: an isolated pi bond (C#N in acetonitrile) is not
        # "conjugated" in RDKit; only pi bonds next to other pi bonds are.
        "n_conjugated_bonds": sum(b.GetIsConjugated() for b in bonds),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "frac_sp3_carbon": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_pi_electrons": sum(PI_ELECTRONS_PER_BOND.get(b.GetBondType(), 0) for b in bonds),
        # functional groups and heteroatoms
        "n_carbonyl": Descriptors.fr_C_O(mol),
        "n_nitrile": Descriptors.fr_nitrile(mol),
        "n_ether": Descriptors.fr_ether(mol),
        "n_N": atoms.count("N"),
        "n_O": atoms.count("O"),
        "n_F": atoms.count("F"),
        # ring strain (3- and 4-membered rings) and topology
        "n_small_rings": sum(size <= 4 for size in ring_sizes),
        "n_rings": len(ring_sizes),
        # size, polarisability (empirical, NOT the DFT alpha), polarity
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "molar_refractivity": Descriptors.MolMR(mol),
        "tpsa": Descriptors.TPSA(mol),
        "max_partial_charge": Descriptors.MaxPartialCharge(mol),   # Gasteiger; may be NaN
        "min_partial_charge": Descriptors.MinPartialCharge(mol),
    }


# Floats, because partial_dependence (step 7) refuses integer columns.
X = pd.DataFrame([chemistry_variables(m) for m in molecules["mol"]]).astype(float)
y = molecules["target_eV"].to_numpy()
print(f"Feature table: {X.shape[0]} molecules x {X.shape[1]} variables")
print(X.describe().T[["mean", "min", "max"]].round(2))

# 3b. Error floor of this description (decision 3, extension).
#     Different molecules can have IDENTICAL values for all variables. The model must give
#     them one prediction, so the spread of their targets is an error no model using these
#     variables can remove. Descriptive statistic on all data - not used for training.
identical_group = X.round(6).groupby(list(X.columns), dropna=False).ngroup()
group_mean = pd.Series(y).groupby(identical_group.to_numpy()).transform("mean")
error_floor_mae = np.mean(np.abs(y - group_mean.to_numpy()))
share_shared = (identical_group.map(identical_group.value_counts()) > 1).mean()
print(f"{share_shared:.1%} of molecules share their variable values with another molecule; "
      f"MAE floor of this description = {error_floor_mae:.3f} eV")


# %% ---------------------------------------------------------------------------------
# 4. Split 80/20 and remove redundant variables (training data only!)
# ------------------------------------------------------------------------------------
def prune_correlated(X_train, priority, threshold):
    """Walk through variables in priority order; drop any variable whose |Spearman rho|
    with an already-kept variable exceeds the threshold. Uses TRAINING data only, because
    choosing variables with test data is itself leakage."""
    rho = X_train.corr(method="spearman").abs()
    kept, dropped = [], []
    for name in priority:
        partners = [k for k in kept if rho.loc[name, k] > threshold]   # NaN compares False
        if partners:
            dropped.append({"dropped": name, "kept_instead": partners[0],
                            "abs_rho": round(rho.loc[name, partners[0]], 3)})
        else:
            kept.append(name)
    return kept, pd.DataFrame(dropped)


ids = molecules["index"].to_numpy()
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=TEST_FRACTION, random_state=SEED)

features, dropped_table = prune_correlated(X_train, list(EXPECTED_EFFECT), CORR_THRESHOLD)
print(f"Training {len(X_train)} / test {len(X_test)} molecules")
print("Variables removed as redundant:\n", dropped_table if len(dropped_table) else "none")
X_train, X_test = X_train[features], X_test[features]


# %% ---------------------------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------------------------------
def reference_model(seed):
    """Gradient boosting with scikit-learn defaults, plus early stopping on 10 % of the
    training data. Small trees are added one by one, each correcting the previous ones."""
    return HistGradientBoostingRegressor(max_iter=1000, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20,
                                         random_state=seed)


# 5a. Baseline: always predict the mean training gap.
baseline_pred = np.full(len(y_test), y_train.mean())

# 5b. Reference model (defaults) - the point of comparison for tuning.
ref_model = reference_model(SEED).fit(X_train, y_train)

# 5c. Tuning (decision 6): random search, 5-fold cross-validation on the TRAINING set.
#     The test set is never seen here; tuning on it would make the score optimistic.
#     loguniform = values spread evenly on a log scale (good for rates and penalties).
search_space = {
    "learning_rate": loguniform(0.02, 0.3),
    "max_leaf_nodes": randint(15, 128),        # tree size
    "min_samples_leaf": randint(10, 201),      # minimum molecules per leaf (smoothing)
    "l2_regularization": loguniform(1e-3, 10), # penalty on large leaf values
}
search = RandomizedSearchCV(
    reference_model(SEED), search_space, n_iter=N_TUNING_ITER,
    scoring="neg_mean_absolute_error",          # scikit-learn maximises, so MAE is negated
    cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
    random_state=SEED, refit=True)              # refit=True: best setting retrained on all training data
search.fit(X_train, y_train)
best_model = search.best_estimator_

cv_scores = -search.cv_results_["mean_test_score"]   # MAE of every tried setting
best_i = search.best_index_
cv_mae = cv_scores[best_i]
cv_sd = search.cv_results_["std_test_score"][best_i]
print(f"Best CV MAE: {cv_mae:.3f} +/- {cv_sd:.3f} eV with {search.best_params_}")
# Option to discuss: loss="absolute_error" optimises MAE directly (default = squared error).
# Option NOT used here: monotonic_cst would FORCE our expected directions into the model;
# we want to TEST them, so it is at most a side experiment.


# %% ---------------------------------------------------------------------------------
# 6. Evaluate ONCE on the untouched test set
# ------------------------------------------------------------------------------------
def scores_for(y_true, y_hat):
    return {"MAE (eV)": mean_absolute_error(y_true, y_hat),
            "RMSE (eV)": root_mean_squared_error(y_true, y_hat),
            "R2": r2_score(y_true, y_hat)}


y_pred = best_model.predict(X_test)
scores = pd.DataFrame({
    "baseline (mean)": scores_for(y_test, baseline_pred),
    "boosting, defaults": scores_for(y_test, ref_model.predict(X_test)),
    "boosting, tuned": scores_for(y_test, y_pred),
}).T
print(scores.round(3))
print(f"(Error floor of this description: {error_floor_mae:.3f} eV)")

# Parity plot. ~26k points would be one blob as a scatter, so count per hexagon.
fig, ax = plt.subplots(figsize=(5.5, 5))
hb = ax.hexbin(y_test, y_pred, gridsize=60, mincnt=1, bins="log", cmap="Blues")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, color="0.3", linewidth=1)          # perfect prediction
ax.set_xlabel(f"Reference {target_label}")
ax.set_ylabel("Predicted gap [eV]")
ax.set_title(f"Test set, MAE = {scores.loc['boosting, tuned', 'MAE (eV)']:.2f} eV")
fig.colorbar(hb, label="molecules per hexagon")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "parity_plot.png"), dpi=200)


# %% ---------------------------------------------------------------------------------
# 7. Interpretation (decision 5)
# ------------------------------------------------------------------------------------
# 7a. Permutation importance: shuffle one variable in the test set, measure how much the
#     MAE grows. Caveat: shuffling can create impossible molecules when variables are
#     correlated - step 4 reduced this, 7c checks it without shuffling.
perm = permutation_importance(best_model, X_test, y_test, scoring="neg_mean_absolute_error",
                              n_repeats=5, random_state=SEED)
importance = pd.Series(perm.importances_mean, index=features).sort_values()

fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(importance.index, importance.values, color="#3b6ea8")
ax.set_xlabel("Increase in test MAE when shuffled [eV]")
ax.set_title("Which variables does the model rely on?")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "permutation_importance.png"), dpi=200)

# 7b. Direction: partial dependence = average prediction as ONE variable moves from its
#     5th to its 95th percentile. Sign of that change vs. our hypothesis.
#     Only read the sign for variables with clear importance: for a variable the model
#     barely uses, the change is tiny and its sign is just noise.
# 7c. Drop-and-refit for the most important variables: retrain WITHOUT the variable
#     (same hyperparameters) and see how much the test MAE grows. No shuffling, so no
#     impossible molecules (Hooker et al. 2021).
tuned_mae = scores.loc["boosting, tuned", "MAE (eV)"]
top_vars = list(importance.sort_values(ascending=False).index[:N_DROP_REFIT])

rows = []
for name in features:
    avg = partial_dependence(best_model, X_train, features=[name], kind="average")["average"][0]
    observed = "too rare to tell" if len(avg) < 2 else ("+" if avg[-1] > avg[0] else "-")
    drop_refit = np.nan
    if name in top_vars:
        refit = clone(best_model).fit(X_train.drop(columns=[name]), y_train)
        drop_refit = mean_absolute_error(y_test, refit.predict(X_test.drop(columns=[name]))) - tuned_mae
    rows.append({"variable": name, "expected": EXPECTED_EFFECT[name], "observed": observed,
                 "permutation (eV)": round(importance[name], 3),
                 "drop-and-refit (eV)": round(drop_refit, 3)})
direction_table = pd.DataFrame(rows).sort_values("permutation (eV)", ascending=False)
print(direction_table.to_string(index=False))


# %% ---------------------------------------------------------------------------------
# 8. Robustness (decision 7)
# ------------------------------------------------------------------------------------
# 8a. Repeated random splits: how much does the test MAE move with a different split?
#     Each repeat redoes the variable pruning on its own training set. To keep it cheap
#     and leak-free we use the DEFAULT model here (the tuned settings were chosen on the
#     seed-42 training set, which overlaps the other seeds' test sets).
repeat_mae = []
for s in [SEED] + REPEAT_SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRACTION, random_state=s)
    kept, _ = prune_correlated(Xtr, list(EXPECTED_EFFECT), CORR_THRESHOLD)
    m = reference_model(s).fit(Xtr[kept], ytr)
    repeat_mae.append(mean_absolute_error(yte, m.predict(Xte[kept])))
print(f"Default model over {len(repeat_mae)} random splits: "
      f"MAE = {np.mean(repeat_mae):.3f} +/- {np.std(repeat_mae, ddof=1):.3f} eV")

# 8b. Held-out chemical family: train WITHOUT any aromatic molecule, test on all aromatic
#     molecules. Compare with aromatic molecules in the ordinary random test set, where the
#     model HAD seen aromatics during training.
is_aromatic = X["n_aromatic_rings"] > 0
Xfam_tr, yfam_tr = X[~is_aromatic], y[~is_aromatic.to_numpy()]
Xfam_te, yfam_te = X[is_aromatic], y[is_aromatic.to_numpy()]
kept_fam, _ = prune_correlated(Xfam_tr, list(EXPECTED_EFFECT), CORR_THRESHOLD)
fam_model = reference_model(SEED).fit(Xfam_tr[kept_fam], yfam_tr)
mae_unseen = mean_absolute_error(yfam_te, fam_model.predict(Xfam_te[kept_fam]))

arom_in_test = X_test.index.isin(X.index[is_aromatic])
mae_seen = mean_absolute_error(y_test[arom_in_test], ref_model.predict(X_test[arom_in_test]))
print(f"Aromatic molecules: MAE {mae_seen:.3f} eV when the family was seen in training, "
      f"{mae_unseen:.3f} eV when it was held out ({is_aromatic.sum()} molecules)")


# %% ---------------------------------------------------------------------------------
# 9. Save results and decisions (FAIR: anyone can redo figures without retraining)
# ------------------------------------------------------------------------------------
pd.DataFrame({"qm9_index": id_test, "target_eV": y_test, "predicted_eV": y_pred}).to_csv(
    os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
scores.to_csv(os.path.join(RESULTS_DIR, "test_scores.csv"))
direction_table.to_csv(os.path.join(RESULTS_DIR, "interpretation.csv"), index=False)

run_record = {
    "target": TARGET, "seed": SEED, "test_fraction": TEST_FRACTION, "versions": VERSIONS,
    "funnel": funnel, "error_floor_mae_eV": error_floor_mae,
    "features_used": features, "features_dropped": dropped_table.to_dict("records"),
    "best_params": {k: float(v) for k, v in search.best_params_.items()},
    "cv_mae_eV": [cv_mae, cv_sd], "repeat_split_mae_eV": repeat_mae,
    "aromatic_mae_seen_vs_unseen_eV": [mae_seen, mae_unseen],
}
with open(os.path.join(RESULTS_DIR, "run_record.json"), "w") as f:
    json.dump(run_record, f, indent=2)
print("Done. Outputs are in", RESULTS_DIR)
