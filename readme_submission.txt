Gap Project readme

/model folder contains all models used, inclusing early drafts.
/data folder contains QM9 134k molecule data set, lists, readmes, and GM gap set.
/results folder contains plots, tables, model outputs.

uv manages dependencies and versions.

Fair Principles:
Data:
134k molecules source .xyz files, characterization list, and readme.
citation
Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014).
Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection.
https://doi.org/10.6084/m9.figshare.978904

SemVer: 
git project release for submission tagged v1.0.0

MAJOR: changes that alter the results, dataset, or target definition.
MINOR: new models added without changing existing results.
PATCH: bug fixes, documentation, or cleanup that leave results unchanged.

from Specification 4: "Major version zero (0.y.z) is for initial development. 
Anything MAY change at any time. The public API SHOULD NOT be considered stable."