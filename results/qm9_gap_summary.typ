// ============================================================
// QM9 HOMO-LUMO Gap: Model Summary & Analysis
// ============================================================
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#show heading: it => [#v(0.5em) #it #v(0.3em)]

#align(center)[
  #text(size: 16pt, weight: "bold")[QM9 HOMO--LUMO Gap: ML Model Summary]
  #linebreak()
  #text(size: 11pt)[BERN02 Challenge --- October 2026]
]
#v(1em)

= Model Overview

The pipeline fits a *HistGradientBoostingRegressor* (scikit-learn's histogram-
based variant of XGBoost / LightGBM) on 17 chemoinformatic features derived
exclusively from molecular structure (SMILES strings). No DFT geometry,
molecular coordinates, or orbital energies are used as inputs --- those would
constitute label leakage, since the gap itself is a DFT output.

#block(fill: luma(235), inset: 0.7em, radius: 3pt)[
  *Dataset after cleaning:* ~130 k QM9 molecules, minus 3 054 "uncharacterised"
  (bonding changed during DFT optimisation), minus unparseable SMILES, minus
  duplicate canonical structures (gap averaged) --- leaving $≈$124 k unique
  structures. 80/20 train/test split, `random_state=42`.
]

= Model Performance

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    table.header[*Model*][*MAE (eV)*][*RMSE (eV)*][*R²*],
    [Baseline (predict mean)], [1.074], [1.281], [≈ 0.000],
    [HistGradientBoosting (default)], [*0.261*], [*0.351*], [*0.925*],
  ),
  caption: [Test-set scores, 17 features, default hyperparameters.]
)

The model reduces MAE by a factor of *4.1×* relative to the mean-prediction
baseline and explains 92.5 % of variance. This already represents a dramatic
compression of computational cost: a full B3LYP/6-31G\* single-point for one
molecule takes seconds to minutes on a single CPU; predicting 26 k test
molecules with the trained model takes milliseconds.

= How Histogram Gradient Boosting Works

Gradient boosting builds an *ensemble of shallow decision trees* iteratively.
Each new tree is trained on the *pseudo-residuals* --- the gradient of the
loss (MAE here) of the current ensemble's predictions. In pseudocode:

```
F₀(x) = mean(y_train)
for m = 1 … n_iter:
    rₘ = -∂L/∂F(xᵢ) evaluated at F_{m-1}   # negative gradient = residual
    hₘ = fit a shallow tree to (X, rₘ)
    F_m(x) = F_{m-1}(x) + learning_rate · hₘ(x)
```

The *histogram* trick (used by LightGBM and this sklearn implementation)
bins each continuous feature into at most 255 integer buckets before
finding splits. This makes each tree fit $O(n \cdot B)$ instead of
$O(n log n)$, where $B = 255$. For 100 k molecules and 17 features,
training with default settings (`max_iter=100`, `max_leaf_nodes=31`,
`learning_rate=0.1`, `max_depth=None`) completes in seconds rather than
minutes.

*Early stopping* activates automatically when the training set exceeds
10 000 samples: 10 % of training data is held as an internal validation
set, and boosting stops when the score does not improve for
`n_iter_no_change=10` rounds. If `n_iter_` printed equals 100, the cap
`max_iter` was the binding constraint --- the first worthwhile tuning
knob.

*NaN handling:* `HistGradientBoostingRegressor` natively handles missing
values (it learns which branch to send NaN values down). This is why
`Descriptors.MaxPartialCharge` / `MinPartialCharge` (which can fail on
some molecules and return NaN) do not require imputation.

= Variable Descriptions and Permutation Importances

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, right, center, center),
    table.header[*Variable*][*How RDKit computes it*][*Imp. (eV)*][*Exp.*][*Obs.*],
    [`frac_sp3_carbon`],
      [`rdMolDescriptors.CalcFractionCSP3(mol)` — sp³ C atoms / total C. An sp³ centre has no p-orbital available for conjugation.],
      [*0.830*], [+], [+],
    [`n_conjugated_bonds`],
      [`b.GetIsConjugated()` — true when the bond participates in a π system (alternating single/double or aromatic). Counted over `mol.GetBonds()`.],
      [0.222], [−], [−],
    [`n_carbonyl`],
      [`Descriptors.fr_C_O(mol)` — SMARTS substructure count for C=O (ketones, aldehydes, acids, amides).],
      [0.218], [−], [−],
    [`n_pi_electrons`],
      [Custom loop over bonds: DOUBLE→2, TRIPLE→4, AROMATIC→1 per bond, summed. Approximates total π-electron count.],
      [0.129], [−], [+],
    [`max_partial_charge`],
      [`Descriptors.MaxPartialCharge(mol)` — largest Gasteiger partial charge across all atoms. Reflects polar pull toward electron-withdrawing groups.],
      [0.120], [?], [+],
    [`n_aromatic_rings`],
      [`rdMolDescriptors.CalcNumAromaticRings(mol)` — rings satisfying Hückel aromaticity (4n+2 π electrons, planar, cyclic conjugation).],
      [0.076], [−], [+],
    [`n_N`],
      [`atoms.count("N")` after `a.GetSymbol()` over `mol.GetAtoms()`. Total nitrogen count.],
      [0.063], [−], [−],
    [`n_rings`],
      [`mol.GetRingInfo().AtomRings()` — total number of SSSR rings (smallest set of smallest rings).],
      [0.045], [?], [−],
    [`min_partial_charge`],
      [`Descriptors.MinPartialCharge(mol)` — most negative Gasteiger charge (electron-rich atom).],
      [0.031], [?], [+],
    [`tpsa`],
      [`Descriptors.TPSA(mol)` — topological polar surface area (Å²), sum of O and N atom contributions.],
      [0.024], [?], [+],
    [`n_ether`],
      [`Descriptors.fr_ether(mol)` — SMARTS count for C-O-C linkages (excludes C=O and hydroxyl).],
      [0.014], [−], [+],
    [`molar_refractivity`],
      [`Descriptors.MolMR(mol)` — Crippen molar refractivity; empirical polarisability proxy, atom-additive.],
      [0.014], [?], [+],
    [`n_heavy_atoms`],
      [`mol.GetNumHeavyAtoms()` — total non-hydrogen atoms; proxy for molecular size.],
      [0.013], [?], [−],
    [`n_small_rings`],
      [`sum(size <= 4 for size in ring_sizes)` — 3- and 4-membered rings (cyclopropane, azetidine etc.).],
      [0.006], [−], [−],
    [`n_O`],
      [`atoms.count("O")` — total oxygen count.],
      [0.005], [−], [−],
    [`n_nitrile`],
      [`Descriptors.fr_nitrile(mol)` — SMARTS count for C≡N.],
      [0.002], [−], [+],
    [`n_F`],
      [`atoms.count("F")` — total fluorine count.],
      [0.001], [?], [+],
  ),
  caption: [All 17 features sorted by permutation importance (mean MAE increase over 5
  shuffling repeats on the test set). "Exp." = prior chemical expectation; "Obs." =
  partial-dependence direction (+ = gap rises as feature increases).]
)

= Variable Culling Recommendations

== Variables to Drop (Redundant or Near-Noise)

The decision to drop is based on three criteria simultaneously:
(i) permutation importance < 0.01 eV (shuffling barely hurts the model),
(ii) high Spearman correlation with a retained, more-informative variable
(the redundant variable "free-rides" on its partner and shares permutation
credit), and (iii) the chemical information it carries is already captured
by higher-ranked variables.

#block(fill: luma(235), inset: 0.7em, radius: 3pt)[
*Recommended to drop:* `n_small_rings`, `n_O`, `n_nitrile`, `n_F`

*Consider dropping (borderline):* `molar_refractivity`, `n_heavy_atoms`,
`n_ether`, `tpsa`
]

*`n_small_rings` (imp. 0.006):* Spearman ρ = 0.79 with `n_rings`. Nearly all
information about ring strain is absorbed by the total ring count. The
gradient booster can split on `n_rings` to distinguish strained vs. unstrained
situations without needing the separate counter. Dropping frees one split
dimension and reduces chance over-splitting on a near-constant column (most
QM9 molecules have 0 small rings).

*`n_O` (imp. 0.005):* Chemically, oxygen contributes to the gap either through
lone-pair donation (HOMO raise) or C=O lowering the LUMO. Both effects are
already encoded by `n_carbonyl` (imp. 0.218) and `n_ether` (imp. 0.014), which
capture the *type* of C--O bonding rather than just a headcount. The Spearman
correlation `n_O`↔`n_carbonyl` is ρ = 0.42 and `n_O`↔`n_ether` is ρ = 0.48,
confirming substantial overlap.

*`n_nitrile` (imp. 0.002):* Despite C≡N having a very low-lying π\* (strong
LUMO pull → expected −), the observed partial-dependence sign is + and
importance is negligible. Nitriles are rare in QM9 (~few % of molecules), so
the tree ensemble never sees enough examples to learn a reliable split; the
variable contributes pure noise. The π-electron count in `n_pi_electrons` also
accounts for C≡N bonds (TRIPLE → 4 in the `PI_ELECTRONS` dictionary).

*`n_F` (imp. 0.001):* Fluorine's inductive withdrawal modulates partial charges,
but `max_partial_charge` (imp. 0.120) already captures that polarisation with
far higher resolution (Gasteiger charges are computed atom-by-atom and take into
account the full connectivity). Like nitriles, F is sparse in QM9.

*Borderline --- `molar_refractivity` and `n_heavy_atoms` (both imp. ≈ 0.013--0.014):*
These encode molecular size and polarisability. They are moderately correlated
(ρ = 0.43) and `molar_refractivity` is itself an atom-additive Crippen
estimate already subsumed by knowing atom counts. However, because the gap
does correlate weakly with system size (larger conjugated systems tend to
smaller gaps), these might still contribute marginal signal; test by dropping
one at a time and monitoring validation MAE.

*Borderline --- `tpsa` (imp. 0.024):* Correlated with `max_partial_charge`
(ρ = 0.61) and with `n_N` (ρ = 0.58). TPSA is a polar-surface proxy
dominated by N--H and O--H contributions, but `max_partial_charge` is a
strictly richer descriptor (atom-resolved). After dropping, the model may
lose 0.02--0.03 eV MAE; acceptable if interpretability is the goal.

== Variables to Retain and Why

`frac_sp3_carbon` is the single most important feature by a factor of *3.7×*
over the next best. This aligns with the fundamental physics: an sp³ carbon
has its p-orbital hybridised into σ bonds, breaking conjugation. A higher sp³
fraction means a more saturated molecule, which raises the HOMO--LUMO gap.
The partial dependence is monotonically increasing (+), confirming the
expected direction.

`n_conjugated_bonds` and `n_carbonyl` are complementary: the former counts the
spatial extent of the π network (more bonds = broader delocalisation = smaller
gap), the latter counts electron-withdrawing groups that independently lower
the LUMO. Their Spearman correlation is only ρ = 0.21, so they encode
largely independent information.

`n_pi_electrons` (imp. 0.129) shows an *unexpected positive* partial-dependence
direction despite the prior expectation of −. This is likely a *collinearity
artefact*: `n_pi_electrons` and `frac_sp3_carbon` have ρ = −0.89. When
`frac_sp3_carbon` absorbs the dominant delocalisation signal, the residual
variation in `n_pi_electrons` becomes positively associated with features like
triple bonds (C≡N, C≡C), which introduce antibonding but also compact π\*
orbitals that can *raise* the LUMO. This sign reversal is a known symptom of
correlated features sharing permutation credit; it is not a physical result.
*Retain* because the importance is real (0.129 eV), but interpret with caution.

= Improving the Model: Roadmap to GW-Quality Predictions

The target is GW-corrected gaps (typically ~1--2 eV higher than B3LYP for
organic molecules). Two distinct goals arise:

+ *Reduce MAE towards ~0.1 eV (DFT quality):* Still predicting B3LYP gaps,
  but more accurately.
+ *Predict GW gaps directly* (~0.1--0.2 eV MAE vs. GW reference), which
  would demonstrate that ML beats DFT at DFT cost.

== Step 1: Better Features (Without GW Labels)

#block(fill: luma(235), inset: 0.7em, radius: 3pt)[
*Expected MAE improvement: −0.05 to −0.10 eV* relative to current 0.261 eV.
]

Replace hand-crafted counts with richer descriptors:

- *Morgan fingerprints (ECFP4):* `rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)` --- a 2048-bit binary vector encoding circular
  chemical neighbourhoods up to radius 2. The boosted trees can split on
  individual bits; this captures functional-group context that single-atom
  counts miss.

- *RDKit topological descriptors:* Wiener index (path sum), Balaban J
  (connectivity), Kier/Hall κ indices --- available via `Chem.GraphDescriptors`
  --- encode graph topology beyond ring counts.

- *Coulomb matrix eigenvalues* (from xyz coordinates in the QM9 files):
  `CM_i = Z_i²·0.5`, off-diagonal `CM_ij = Z_i·Z_j / |r_i - r_j|`.
  Sorted eigenvalues are rotationally invariant and encode 3-D geometry.
  This step *requires* the DFT coordinates but NOT the gap, so it is not
  leakage.

== Step 2: Hyperparameter Tuning

With the current feature set, the default (`max_iter=100`) was the binding
limit (the printed `n_iter_` == 100 indicates early stopping did not terminate
first). Tuning plan:

```python
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    "max_iter":        [300, 500, 800],
    "max_leaf_nodes":  [63, 127, 255],
    "learning_rate":   [0.05, 0.02, 0.01],
    "l2_regularization": [0.0, 0.1, 1.0],
    "min_samples_leaf": [20, 50, 100],
}
```

Lower `learning_rate` (0.01--0.05) with higher `max_iter` (500--1000) is the
single highest-return tuning move for gradient boosting: it widens the
ensemble so individual trees need not overfit.

== Step 3: Add GW Labels via Δ-Learning

The most principled path to GW-quality predictions is *Δ-ML* (delta
machine learning):

$
Delta_"GW-DFT" = epsilon_"gap"^"GW" - epsilon_"gap"^"DFT"
$

Train a *second* boosted model on the correction $Delta$ using the same or
richer descriptors. At inference, use:

$
hat(epsilon)_"gap"^"GW"(x) = hat(epsilon)_"gap"^"DFT"(x) + hat(Delta)(x)
$

Because corrections are smaller in magnitude and smoother than absolute gaps,
a second model trained on even 1 000--5 000 GW-calculated molecules (a
feasible subset given GW is ~100× more expensive than DFT) can interpolate
accurate corrections across the full 130 k dataset. This has been demonstrated
for atomisation energies in the SchNet/ANI literature and for HSE06 corrections
by Ramakrishnan et al.

== Step 4: Graph Neural Networks

GNNs natively operate on molecular graphs and learn both atom- and bond-level
representations end-to-end. Pre-trained models (e.g., DimeNet++, GemNet, NequIP)
fine-tuned on QM9 gaps achieve MAE ~0.05--0.08 eV on B3LYP targets --- roughly
3--5× better than the current hand-feature model. With Δ-learning on top of
GNN DFT predictions, sub-0.1 eV MAE vs. GW is realistic.

= Demonstrating that ML Beats DFT at Lower Cost

The claim "ML beats DFT" requires a clear experimental design:

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    table.header[*Method*][*Typical wall-time per molecule*][*MAE vs. GW (eV)*],
    [B3LYP/6-31G\* (DFT, QM9)], [10--300 s (CPU)], [~0.8--1.5],
    [GW\@B3LYP (target)], [10--100× DFT], [0.0 (reference)],
    [Current ML (hand features)], [< 1 ms (inference)], [~1.0--1.5 (inherited DFT error + 0.26)],
    [Tuned ML + Morgan FP], [< 5 ms], [estimated ~0.8--1.2],
    [Δ-ML (DFT + correction)], [DFT + < 1 ms], [projected ~0.15--0.30],
    [GNN + Δ-ML (no DFT at inference)], [< 50 ms], [projected ~0.10--0.20],
  ),
  caption: [Schematic cost--accuracy comparison. "ML beats DFT" means the bottom row
  beats row 1 on GW-referenced accuracy at a tiny fraction of GW cost.]
)

The key experiment:

+ Train GNN on QM9 B3LYP gaps → get ~0.05 eV MAE (vs. B3LYP).
+ Train Δ-ML model on a GW subset (N~1000 molecules, GW labels).
+ At inference: apply GNN + Δ predictor in < 50 ms per molecule.
+ Compare to: *running B3LYP alone* for the same molecules → slower
  *and* further from GW.

This demonstrates that the ML pipeline is simultaneously (a) faster than DFT
by 3--5 orders of magnitude and (b) closer to the GW reference than DFT itself.
The critical caveat is that the GW subset used for Δ-training must be
representative (diverse in functional group space) and held entirely out of the
feature-engineering decisions.

= Conclusion

The default `HistGradientBoostingRegressor` with 17 hand-crafted SMILES
features achieves *MAE = 0.261 eV, R² = 0.925* on the QM9 test set ---
a strong baseline given the simplicity of the inputs.

Key takeaways:
- *Drop `n_small_rings`, `n_O`, `n_nitrile`, `n_F`* immediately; their combined
  importance is < 0.015 eV and their information is absorbed by retained features.
- *`frac_sp3_carbon` dominates* (0.830 eV importance) because sp³ hybridisation
  is the primary gap-opening mechanism in this dataset.
- *Beware of sign reversals* (`n_pi_electrons`, `n_aromatic_rings`): high
  inter-feature correlation (|ρ| ≥ 0.53) causes permutation credit to split,
  making partial-dependence directions unreliable for those variables.
- The clearest path to GW-quality predictions is *Δ-learning*: train on
  a small GW-computed subset to correct the systematic DFT underestimation,
  then apply the correction to ML DFT predictions across the full dataset.

