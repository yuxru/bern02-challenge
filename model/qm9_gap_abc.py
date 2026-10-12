"""
REFERENCE GUIDE (not for submission) - HOMO-LUMO gap of QM9 molecules
=====================================================================
Version 3: chemistry variables + Morgan COUNT fingerprints.

Pipeline:
    1. Read every .xyz file in data/dsgdb9nsd.xyz/ (only index, gap, SMILES).
    2. Select molecules: remove "uncharacterized" ones, unreadable SMILES, merge duplicates.
    3. Inputs per molecule:
         a. 17 chemically meaningful variables (structure only)
         b. Morgan count fingerprint, UNFOLDED: {substructure id: how often it occurs}
    4. Split 80/20. Descriptor correlations -> YOUR choice of DROP_VARIABLES.
       Fingerprint columns = the most frequent substructures in the TRAINING set
       ("Sort & Slice", Dablander et al. 2024) -> one column = one substructure, no collisions.
    5. Four models, all HistGradientBoostingRegressor with DEFAULT settings:
         baseline (mean) | A chemistry variables | B Morgan counts | C A + B combined
    6. Test-set scores + side-by-side parity plots (A vs C).
    7. Learning curves: test MAE vs training-set size, log-log.
    8. SHAP for model C: direction table for the chemistry variables and a drawn
       gallery of the most important substructures.
    9. Error by chemical family (A vs C).
   10. Final comparison: 5x5 repeated cross-validation, paired A-vs-C difference with 95 % CI
       (Ash et al. 2025: do not compare methods on a single split). The ONLY CV in the guide.
   11. Save results.

Not used as input: DFT coordinates, alpha, mu, and every other DFT column (leakage).

Repository layout (uv project):
    bern02-challenge/
    |-- pyproject.toml, uv.lock, .python-version, .venv/
    |-- data/dsgdb9nsd.xyz/  (dsgdb9nsd_000001.xyz ...),  data/uncharacterized.txt
    `-- <this file anywhere inside the repo, e.g. guide/>

One-time setup (VS Code terminal, repo root):
    uv add numpy pandas matplotlib scikit-learn rdkit scipy shap
    uv add --dev ipykernel
Run:  uv run python guide/qm9_gap_guide.py   (or cell by cell with Shift+Enter)
Expected run time on a laptop: roughly 15-20 min, most of it in step 10 (RUN_FINAL_CV).
Before trusting results: restart and run the whole file top to bottom once.
"""

# %% ---------------------------------------------------------------------------------
# 0. Imports and settings
# ------------------------------------------------------------------------------------
import os
# TROUBLESHOOTING: on CPUs with many cores, HistGradientBoosting's parallel threads can
# slow each other down. If model fits are very slow, remove the "#" on the next line.
# It must run BEFORE numpy/sklearn are imported; restart the kernel after changing it.
# os.environ["OMP_NUM_THREADS"] = "4"

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy import stats

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors, rdFingerprintGenerator

from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")   # hide RDKit warnings for the few odd molecules


def find_repo_root():
    """Walk up from the current folder to the folder containing pyproject.toml."""
    folder = Path.cwd().resolve()
    while not (folder / "pyproject.toml").exists():
        if folder.parent == folder:
            raise FileNotFoundError("pyproject.toml not found - open the bern02-challenge folder")
        folder = folder.parent
    return folder


ROOT = find_repo_root()
XYZ_DIR = ROOT /"data"/"qm9"/"dsgdb9nsd.xyz"
UNCHARACTERIZED_FILE = ROOT /"data"/"qm9"/"uncharacterized.txt"
RESULTS_DIR = ROOT /"results"/"abc"
RESULTS_DIR.mkdir(exist_ok=True)

HARTREE_TO_EV = 27.211386
SEED = 42
TEST_FRACTION = 0.20

DROP_VARIABLES = []          # YOUR choice after reading the correlation table (step 4)

FP_RADIUS = 2                # Morgan radius 2 = ECFP4: atom environments up to 2 bonds away
FP_VOCAB_SIZE = 256          # keep the 256 most frequent TRAINING-set substructures
LEARNING_CURVE_SIZES = [1000, 2000, 4000, 8000, 16000, 32000, 64000]   # + full training set
N_SHAP = 2000                # test molecules explained with SHAP (enough for stable averages)
N_GALLERY = 8                # substructures drawn in the gallery
RUN_FINAL_CV = True          # step 10 takes most of the run time; set False while developing

# One fixed colour per model in EVERY figure (colour-blind safe: Okabe-Ito hues).
MODEL_NAMES = {"A": "A: chemistry variables", "B": "B: Morgan counts", "C": "C: combined"}
MODEL_COLORS = {"baseline": "#8c8c8c", "A": "#0072B2", "B": "#E69F00", "C": "#009E73"}

# Runtime notices: every message shows the total time since the script started.
# flush=True prints immediately (otherwise Python may hold output back while it works).
START = time.perf_counter()


def notice(message):
    elapsed = time.perf_counter() - START
    print(f"[{int(elapsed // 60):3d} min {elapsed % 60:4.1f} s] {message}", flush=True)


notice(f"Settings loaded. CPU cores seen by Python: {os.cpu_count()}; "
       f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'not set (all cores)')}")


# %% ---------------------------------------------------------------------------------
# 1. Read the .xyz files (only index, gap and SMILES)
# ------------------------------------------------------------------------------------
def read_xyz(path):
    """Line 1 = number of atoms n; line 2 = properties; line n+4 = SMILES."""
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0])
    props = lines[1].replace("*^", "e").split()       # "1.2*^-6" -> "1.2e-6"
    return {"index": int(props[1]),                   # props[9] = gap in Hartree
            "gap_eV": float(props[9]) * HARTREE_TO_EV,
            "smiles": lines[n_atoms + 3].split()[1]}  # SMILES after DFT relaxation


xyz_files = sorted(XYZ_DIR.glob("*.xyz"))
notice(f"Step 1: reading {len(xyz_files)} .xyz files")          # expect 133885
qm9 = pd.DataFrame([read_xyz(p) for p in xyz_files])
notice("Step 1 finished")


# %% ---------------------------------------------------------------------------------
# 2. Select the molecules we can trust
# ------------------------------------------------------------------------------------
n_read = len(qm9)
bad_index = {int(line.split()[0]) for line in UNCHARACTERIZED_FILE.read_text().splitlines()
             if line.split() and line.split()[0].isdigit()}
qm9 = qm9[~qm9["index"].isin(bad_index)]                     # 2a. failed geometry check
n_characterized = len(qm9)

qm9["mol"] = [Chem.MolFromSmiles(s) for s in qm9["smiles"]]  # 2b. readable SMILES only
qm9 = qm9[qm9["mol"].notna()]
n_parsed = len(qm9)

qm9["canonical"] = [Chem.MolToSmiles(m) for m in qm9["mol"]] # 2c. one row per structure
molecules = (qm9.groupby("canonical", sort=False)
                .agg(index=("index", "first"), gap_eV=("gap_eV", "mean"),
                     mol=("mol", "first"), n_copies=("index", "size"))
                .reset_index())
y = molecules["gap_eV"].to_numpy()
del qm9          # no longer needed; frees the memory of a second copy of every molecule
notice(f"Step 2 finished: read {n_read} -> characterized {n_characterized} -> parsed {n_parsed} "
       f"-> unique structures {len(molecules)}")


# %% ---------------------------------------------------------------------------------
# 3a. Chemically meaningful variables (unchanged from version 2)
# ------------------------------------------------------------------------------------
EXPECTED_EFFECT = {
    "n_conjugated_bonds": "-", "n_aromatic_rings": "-", "frac_sp3_carbon": "+",
    "n_pi_electrons": "-", "n_carbonyl": "-", "n_nitrile": "-", "n_ether": "-",
    "n_N": "-", "n_O": "-", "n_F": "?", "n_small_rings": "-", "n_rings": "?",
    "n_heavy_atoms": "?", "molar_refractivity": "?", "tpsa": "?",
    "max_partial_charge": "?", "min_partial_charge": "?",
}
PI_ELECTRONS = {Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 4, Chem.BondType.AROMATIC: 1}


def chemistry_variables(mol):
    bonds = list(mol.GetBonds())
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {
        "n_conjugated_bonds": sum(b.GetIsConjugated() for b in bonds),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "frac_sp3_carbon": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_pi_electrons": sum(PI_ELECTRONS.get(b.GetBondType(), 0) for b in bonds),
        "n_carbonyl": Descriptors.fr_C_O(mol),
        "n_nitrile": Descriptors.fr_nitrile(mol),
        "n_ether": Descriptors.fr_ether(mol),
        "n_N": atoms.count("N"), "n_O": atoms.count("O"), "n_F": atoms.count("F"),
        "n_small_rings": sum(size <= 4 for size in ring_sizes),
        "n_rings": len(ring_sizes),
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "molar_refractivity": Descriptors.MolMR(mol),
        "tpsa": Descriptors.TPSA(mol),
        "max_partial_charge": Descriptors.MaxPartialCharge(mol),
        "min_partial_charge": Descriptors.MinPartialCharge(mol),
    }


notice("Step 3a: computing the 17 chemistry variables")
X_chem_all = pd.DataFrame([chemistry_variables(m) for m in molecules["mol"]]).astype(float)
notice("Step 3a finished")


# %% ---------------------------------------------------------------------------------
# 3b. Morgan COUNT fingerprints, unfolded
# ------------------------------------------------------------------------------------
# For every atom, the Morgan algorithm describes its neighbourhood out to FP_RADIUS bonds
# and turns that neighbourhood into an integer id. "Unfolded" = we keep the real ids
# instead of hashing them into a fixed number of columns, so two different substructures
# can never share a column (no "bit collisions"). "Count" = how many times each
# substructure occurs (two C=C-C=O units count 2), not just whether it occurs.
notice("Step 3b: computing Morgan count fingerprints")
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS)
fp_dicts = [morgan.GetSparseCountFingerprint(m).GetNonzeroElements() for m in molecules["mol"]]
notice(f"Step 3b finished: {len({k for d in fp_dicts for k in d})} unique substructures in the dataset")


def build_vocabulary(train_positions):
    """Sort & Slice: rank substructures by how many TRAINING molecules contain them and keep
    the FP_VOCAB_SIZE most common. Only training molecules are used, so the test set cannot
    influence which columns exist (that would be leakage)."""
    frequency = Counter(sub_id for i in train_positions for sub_id in fp_dicts[i])
    ranked = sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))  # ties: smaller id first
    return [sub_id for sub_id, _ in ranked[:FP_VOCAB_SIZE]]


def fingerprint_table(vocabulary):
    """Count matrix (molecules x vocabulary) with one named column per substructure."""
    column_of = {sub_id: j for j, sub_id in enumerate(vocabulary)}
    counts = np.zeros((len(fp_dicts), len(vocabulary)), dtype=np.float32)
    for i, d in enumerate(fp_dicts):
        for sub_id, count in d.items():
            j = column_of.get(sub_id)          # None if this substructure is not a column
            if j is not None:
                counts[i, j] = count
    return pd.DataFrame(counts, columns=[f"fp_{sub_id}" for sub_id in vocabulary])


# %% ---------------------------------------------------------------------------------
# 4. Split, descriptor correlations, YOUR variable choice
# ------------------------------------------------------------------------------------
positions = np.arange(len(molecules))      # row numbers; the split works on these
train_pos, test_pos = train_test_split(positions, test_size=TEST_FRACTION, random_state=SEED)
print(f"Training: {len(train_pos)}   Test: {len(test_pos)}")

rho = X_chem_all.iloc[train_pos].corr(method="spearman")      # training data only
pairs = (rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1))
            .stack().dropna().reset_index())
pairs.columns = ["variable_1", "variable_2", "rho"]
pairs = pairs.reindex(pairs["rho"].abs().sort_values(ascending=False).index)
print(pairs.head(10).round(2).to_string(index=False))

features = [c for c in X_chem_all.columns if c not in DROP_VARIABLES]
X_chem = X_chem_all[features]
print(f"Chemistry variables used: {len(features)}; dropped: {DROP_VARIABLES or 'none'}")


# %% ---------------------------------------------------------------------------------
# 5. The four models (all with default settings)
# ------------------------------------------------------------------------------------
def model_inputs(vocabulary):
    """Input tables for models A, B, C (all molecules; rows are selected later)."""
    fp = fingerprint_table(vocabulary)
    return {"A": X_chem, "B": fp, "C": pd.concat([X_chem, fp], axis=1)}


def fit_and_test(train, test, which=("A", "B", "C"), verbose=False):
    """Build the vocabulary from `train`, fit the chosen models, score them on `test`.
    verbose=True prints a timed notice before and after every slow sub-step."""
    t = time.perf_counter()
    vocabulary = build_vocabulary(train)
    if verbose:
        notice(f"  vocabulary built from {len(train)} training molecules "
               f"({time.perf_counter() - t:.1f} s)")

    t = time.perf_counter()
    inputs = model_inputs(vocabulary)
    if verbose:
        notice(f"  input tables built: A {inputs['A'].shape}, B {inputs['B'].shape}, "
               f"C {inputs['C'].shape} ({time.perf_counter() - t:.1f} s)")

    results = {"baseline": {"mae": mean_absolute_error(y[test], np.full(len(test), y[train].mean()))}}
    for name in which:
        if verbose:
            notice(f"  fitting model {name} on {len(train)} x {inputs[name].shape[1]} inputs ...")
        t = time.perf_counter()
        model = HistGradientBoostingRegressor(random_state=SEED)
        model.fit(inputs[name].iloc[train], y[train])
        fit_seconds = time.perf_counter() - t
        pred = model.predict(inputs[name].iloc[test])
        results[name] = {"model": model, "pred": pred, "mae": mean_absolute_error(y[test], pred)}
        if verbose:
            # Expect seconds per model, not minutes. Much slower -> see OMP_NUM_THREADS in step 0.
            notice(f"  model {name} done: fit {fit_seconds:.1f} s, {model.n_iter_} trees, "
                   f"test MAE {results[name]['mae']:.3f} eV")
    return results, inputs


notice("Step 5: fitting models A, B, C on the main 80/20 split")
main, main_inputs = fit_and_test(train_pos, test_pos, verbose=True)
notice("Step 5 finished")


# %% ---------------------------------------------------------------------------------
# 6. Test-set scores and parity plots
# ------------------------------------------------------------------------------------
notice("Step 6: computing test-set scores")
y_test = y[test_pos]
baseline_pred = np.full(len(test_pos), y[train_pos].mean())
rows = {"baseline (mean)": baseline_pred}
rows.update({MODEL_NAMES[k]: main[k]["pred"] for k in "ABC"})
scores = pd.DataFrame({name: [mean_absolute_error(y_test, p), root_mean_squared_error(y_test, p),
                              r2_score(y_test, p)] for name, p in rows.items()},
                      index=["MAE (eV)", "RMSE (eV)", "R2"]).T
scores["RMSE/MAE"] = scores["RMSE (eV)"] / scores["MAE (eV)"]   # ~1.25 for Gaussian errors
print(scores.round(3))

# Figure 4: A and C side by side, same axes and same colour scale -> a fair before/after.
notice("Step 6: drawing parity plots (Figure 4)")
lims = [y_test.min(), y_test.max()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for ax, key in zip(axes, ["A", "C"]):
    hb = ax.hexbin(y_test, main[key]["pred"], gridsize=60, mincnt=1, bins="log",
                   cmap="Blues", extent=(*lims, *lims), vmin=1, vmax=len(y_test) / 50)
    ax.plot(lims, lims, color="0.3", linewidth=1)
    ax.set_title(f"{MODEL_NAMES[key]}\nMAE = {main[key]['mae']:.3f} eV")
    ax.set_xlabel("DFT (B3LYP) HOMO-LUMO gap [eV]")
axes[0].set_ylabel("Predicted gap [eV]")
fig.colorbar(hb, ax=axes, label="molecules per hexagon")
fig.savefig(RESULTS_DIR / "fig4_parity_A_vs_C.png", dpi=200, bbox_inches="tight")
notice("Step 6 finished: saved fig4_parity_A_vs_C.png")


# %% ---------------------------------------------------------------------------------
# 7. Learning curves (Figure 2)
# ------------------------------------------------------------------------------------
# Huang & von Lilienfeld (2021): log(error) falls linearly with log(training size). The
# OFFSET shows how much relevant physics a representation captures; a curve that FLATTENS
# means the representation cannot tell some different molecules apart.
# Same test set for every point; nested training subsets (each contains the smaller one).
shuffled_train = np.random.default_rng(SEED).permutation(train_pos)
sizes = [n for n in LEARNING_CURVE_SIZES if n < len(train_pos)] + [len(train_pos)]
curve = []
notice(f"Step 7: learning curves for {len(sizes)} training sizes")
for n in sizes:
    res, _ = fit_and_test(shuffled_train[:n], test_pos)   # vocabulary rebuilt from these n
    curve.append({"n_train": n, **{k: res[k]["mae"] for k in "ABC"}})
    notice(f"  n = {n:6d}: " + "  ".join(f"{k} {res[k]['mae']:.3f}" for k in "ABC"))
curve = pd.DataFrame(curve)

fig, ax = plt.subplots(figsize=(6, 4.5))
for k in "ABC":
    ax.plot(curve["n_train"], curve[k], "o-", color=MODEL_COLORS[k], label=MODEL_NAMES[k])
ax.set_xscale("log")
ax.set_yscale("log")
# Plain numbers instead of matplotlib's default "2 x 10^-1" labels on log axes.
plain = plt.FuncFormatter(lambda value, _: f"{value:g}")
ax.set_xticks(curve["n_train"])                 # one tick per training size used
ax.xaxis.set_major_formatter(plain)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.tick_params(axis="x", labelrotation=45)
ax.yaxis.set_major_formatter(plain)
ax.yaxis.set_minor_formatter(plain)     # MAE often spans less than one decade
ax.set_xlabel("Training molecules")
ax.set_ylabel("Test MAE [eV]")
ax.set_title("Learning curves (same test set)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig2_learning_curves.png", dpi=200)


# %% ---------------------------------------------------------------------------------
# 8. SHAP for model C (Figure 3 + direction table)
# ------------------------------------------------------------------------------------
# SHAP splits each prediction into one contribution per input (in eV) that add up to
# (prediction - average prediction). TreeExplainer computes these exactly for tree models.
model_C = main["C"]["model"]
X_C = main_inputs["C"]
shap_rows = np.random.default_rng(SEED).choice(test_pos, size=min(N_SHAP, len(test_pos)),
                                               replace=False)
X_shap = X_C.iloc[shap_rows]
shap_values = shap.TreeExplainer(model_C).shap_values(X_shap)        # (molecules x inputs)
shap_df = pd.DataFrame(shap_values, columns=X_C.columns, index=X_shap.index)
mean_abs = shap_df.abs().mean().sort_values(ascending=False)          # overall importance

# 8a. Chemistry variables: observed direction = sign of the rank correlation between a
#     variable's value and its SHAP contribution. Only meaningful for important variables.
def observed_direction(name):
    if X_shap[name].nunique() < 2:          # constant in this sample (e.g. a rare group)
        return "too rare"
    if shap_df[name].nunique() < 2:         # the model never splits on this variable
        return "not used"
    return "+" if stats.spearmanr(X_shap[name], shap_df[name]).statistic > 0 else "-"


direction = pd.DataFrame({
    "mean |SHAP| (eV)": mean_abs[features].round(3),
    "expected": [EXPECTED_EFFECT[f] for f in features],
    "observed": [observed_direction(f) for f in features],
}).sort_values("mean |SHAP| (eV)", ascending=False)
print(direction.to_string())

# 8b. Substructure gallery: the N_GALLERY most important fingerprint columns, each drawn
#     in an example molecule with its atoms highlighted, labelled with the average SHAP
#     value when present (negative = lowers the predicted gap).
fp_cols = [c for c in mean_abs.index if c.startswith("fp_")][:N_GALLERY]
gallery_mols, gallery_atoms, gallery_labels = [], [], []
for col in fp_cols:
    sub_id = int(col[3:])
    present = X_shap[col] > 0
    effect = shap_df.loc[present, col].mean()
    n_containing = int((X_C[col] > 0).sum())
    example = molecules["mol"].iloc[X_shap.index[present][0]]
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    morgan.GetSparseCountFingerprint(example, additionalOutput=info)
    center, radius = info.GetBitInfoMap()[sub_id][0]           # where the substructure sits
    bond_ids = Chem.FindAtomEnvironmentOfRadiusN(example, radius, center)
    atoms = {center} | {a for b in bond_ids for a in (example.GetBondWithIdx(b).GetBeginAtomIdx(),
                                                      example.GetBondWithIdx(b).GetEndAtomIdx())}
    gallery_mols.append(example)
    gallery_atoms.append(sorted(atoms))
    gallery_labels.append(f"{effect:+.2f} eV | in {n_containing} molecules")

gallery = Draw.MolsToGridImage(gallery_mols, molsPerRow=4, subImgSize=(260, 200),
                               legends=gallery_labels, highlightAtomLists=gallery_atoms)
if isinstance(gallery, str):                     # some setups return SVG text
    (RESULTS_DIR / "fig3_substructure_gallery.svg").write_text(gallery)
else:
    gallery.save(RESULTS_DIR / "fig3_substructure_gallery.png")


# %% ---------------------------------------------------------------------------------
# 9. Error by chemical family (Figure 5) - where do fingerprints help?
# ------------------------------------------------------------------------------------
# Families overlap (a molecule can be aromatic AND contain C=O); they are views, not classes.
test_chem = X_chem_all.iloc[test_pos].reset_index(drop=True)
families = {
    "all": np.ones(len(test_pos), dtype=bool),
    "saturated\n(no pi bonds)": (test_chem["n_pi_electrons"] == 0).to_numpy(),
    "aromatic": (test_chem["n_aromatic_rings"] > 0).to_numpy(),
    "carbonyl": (test_chem["n_carbonyl"] > 0).to_numpy(),
    "nitrile": (test_chem["n_nitrile"] > 0).to_numpy(),
    "3/4-ring": (test_chem["n_small_rings"] > 0).to_numpy(),
}
abs_err = {k: np.abs(main[k]["pred"] - y_test) for k in "AC"}

fig, ax = plt.subplots(figsize=(10.5, 4.5))
for i, (fam, mask) in enumerate(families.items()):
    for shift, k in [(-0.18, "A"), (0.18, "C")]:
        box = ax.boxplot(abs_err[k][mask], positions=[i + shift], widths=0.3, showfliers=False,
                         patch_artist=True, medianprops={"color": "black"})
        box["boxes"][0].set_facecolor(MODEL_COLORS[k])
# Group size under each label (small groups = noisier boxes); whiskers = 1.5 x IQR.
ax.set_xticks(range(len(families)), [f"{fam}\n(n={mask.sum()})" for fam, mask in families.items()])
ax.set_ylabel("Absolute error [eV]")
ax.set_title("Test error by chemical family (outliers not shown)")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[k]) for k in "AC"],
          [MODEL_NAMES[k] for k in "AC"], loc="upper left", bbox_to_anchor=(1.01, 1.0),
          frameon=False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig5_error_by_family.png", dpi=200)


# %% ---------------------------------------------------------------------------------
# 10. Final comparison: 5x5 repeated cross-validation (Figure 1)
# ------------------------------------------------------------------------------------
# Ash et al. (2025): never compare methods on one split. 5 folds x 5 repeats = 25 paired
# results; the paired t-test on the per-fold differences (A - C) is their recommendation
# for two methods. Vocabulary and models are rebuilt inside every fold (no leakage).
if RUN_FINAL_CV:
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv_rows = []
    for fold, (tr, te) in enumerate(cv.split(positions)):
        res, _ = fit_and_test(tr, te)
        cv_rows.append({"fold": fold, **{k: res[k]["mae"] for k in ["baseline", "A", "B", "C"]}})
        notice(f"  fold {fold + 1:2d}/25 done")
    cv_mae = pd.DataFrame(cv_rows)

    diff = cv_mae["A"] - cv_mae["C"]                       # > 0 means C is better
    t_crit = stats.t.ppf(0.975, df=len(diff) - 1)
    half_width = t_crit * diff.std(ddof=1) / np.sqrt(len(diff))
    p_value = stats.ttest_rel(cv_mae["A"], cv_mae["C"]).pvalue
    print(f"MAE(A) - MAE(C) = {diff.mean():.3f} eV, 95% CI [{diff.mean() - half_width:.3f}, "
          f"{diff.mean() + half_width:.3f}], paired t-test p = {p_value:.1e}")

    # Figure 1: every fold as a dot, mean +/- 95 % CI as a bar; log axis so the physical
    # reference lines (chemical accuracy and B3LYP-vs-experiment error) fit on one scale.
    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = ["baseline", "A", "B", "C"]
    jitter = np.random.default_rng(SEED).uniform(-0.12, 0.12, len(cv_mae))
    for row, k in enumerate(order):
        values = cv_mae[k]
        ci = t_crit * values.std(ddof=1) / np.sqrt(len(values))
        ax.scatter(values, row + jitter, s=14, alpha=0.5, color=MODEL_COLORS[k])
        ax.errorbar(values.mean(), row, xerr=ci, fmt="D", color="black", markersize=5, capsize=4)
    for x, label in [(0.043, "chemical accuracy\n(0.043 eV)"),
                     (1.2, "B3LYP vs experiment\n(~1.2 eV, Faber 2017)")]:
        ax.axvline(x, color="0.5", linestyle="--", linewidth=1)
        ax.text(x * 1.06, len(order) - 0.45, label, ha="left", va="bottom", fontsize=8,
                color="0.3")                       # text just right of its line
    ax.set_yticks(range(len(order)), ["baseline (mean)"] + [MODEL_NAMES[k] for k in "ABC"])
    ax.set_xscale("log")
    ax.set_xlabel("MAE on held-out fold [eV] (25 folds; diamond = mean, bar = 95 % CI)")
    ax.set_ylim(-0.6, len(order) + 0.4)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_model_comparison.png", dpi=200)


# %% ---------------------------------------------------------------------------------
# 11. Save results
# ------------------------------------------------------------------------------------
pd.DataFrame({"qm9_index": molecules["index"].iloc[test_pos].to_numpy(), "gap_dft_eV": y_test,
              **{f"pred_{k}_eV": main[k]["pred"] for k in "ABC"}}).to_csv(
    RESULTS_DIR / "test_predictions.csv", index=False)
scores.to_csv(RESULTS_DIR / "test_scores.csv")
curve.to_csv(RESULTS_DIR / "learning_curves.csv", index=False)
direction.to_csv(RESULTS_DIR / "shap_directions.csv")
mean_abs.to_csv(RESULTS_DIR / "shap_mean_abs.csv", header=["mean |SHAP| (eV)"])
rho.round(3).to_csv(RESULTS_DIR / "correlation_train.csv")
if RUN_FINAL_CV:
    cv_mae.to_csv(RESULTS_DIR / "cv_mae_5x5.csv", index=False)
notice(f"Done. Results in {RESULTS_DIR}")
