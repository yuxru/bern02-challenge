Accurate frontier orbital energies from GW for the QM9 dataset
==============================================================

Related publications:

* Artem Fediai, Patrick Reiser, Jorge Enrique Olivares Peña, Pascal Friederich, Wolfgang Wenzel:
  Accurate GW frontier orbital energies of 134 kilo molecules, arXiv, 2023, https://doi.org/10.48550/ARXIV.2303.08708


Files
-----

* db_new_qm9_gw.csv
* db_new_qm9_gw.yaml


Description
-----------


This dataset presents the HOMO/LUMO energies of molecules from the QM9 dataset, calculated using the GW method.

In particular, the following data is provided:

*     Eigenvalue self-consistent quasiparticle HOMO/LUMO energies (GW@PBE).
*     Non-self-consistent or one-shot GW HOMO/LUMO energies (G0W0@PBE).
*     HOMO and LUMO energies computed using the DFT method with the PBE functional, which are used as starting guesses for the GW calculations.

The dataset includes data generated using two basis sets: aug-cc-DZVP and aug-cc-TZVP, and their corresponding
extrapolations to the infinite basis set limit.


Motivation:

It is widely acknowledged that the accuracy of HOMO/LUMO energies calculated using density functional theory (DFT),
including hybrid methods such as B3LYP, is often insufficient for many application areas.
The error margin can exceed 1 eV. In contrast, the mean unsigned errors of the GW method may be as low as 100 meV,
as demonstrated by the GW100 dataset [4].

Applications of this Dataset: Where Can It Be Used?
This database can be used as a benchmark for HOMO/LUMO prediction, delta-learning, and transfer learning, even for
larger organic molecules where the GW method is the most accurate but still computationally feasible approach.



References
----------

[1] Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, O. Anatole von Lilienfeld:
    Quantum chemistry structures and properties of 134 kilo molecules
    Scientific Data (2014)
[2] Lorenz C. Blum, Jean-Louis Reymond: 970 Million Druglike Small Molecules
    for Virtual Screening in the Chemical Universe Database GDB-13, Journal of
    the American Chemical Society 131(25): 8732-8733, 2009. DOI: 10.1021/ja902302h
[3] Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. : 
    Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach. 
    J. Chem. Theory Comput. 2015, 11 (5), 2087–2096.     https://doi.org/10.1021/acs.jctc.5b00099.
[4] van Setten, M. J. et al. GW100: Benchmarking G0W0 for Molecular Systems.
    J. Chem. Theory Comput. 11, 5665–5687 (2015).