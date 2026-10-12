
Data set dsgdb9nsd
==================

Thermochemical properties for 133885 small organic molecules at the DFT/B3LYP level of theory.

Please cite this publication if you use this data set:
* Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, O. Anatole von Lilienfeld:
  Quantum chemistry structures and properties of 134 kilo molecules
  Scientific Data (2014)

Related publications:
* Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, O. Anatole von Lilienfeld:
  Learning the error: Augmenting legacy quantum chemistry with machine learning. 
  submitted (2014)

* Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert Mueller, O. Anatole von 
  Lilienfeld: Fast and Accurate Modeling of Molecular Atomization Energies with 
  Machine Learning, Physical Review Letters, 108(5): 058301, 2012. 
  DOI: 10.1103/PhysRevLett.108.058301

This data set is publicly available at 
* http://dx.doi.org/10.6084/m9.figshare.XXXX

Files
-----

dsgdb9nsd.xyz.tar.bz2    - 133885 molecules with properties in XYZ-like format
dsC7O2H10nsd.xyz.tar.bz2 - 6095 isomers of C7O2H10 with properties in XYZ-like format
validation.txt           - 100 randomly drawn molecules from the 133885 set with enthalpies of formation
uncharacterized.txt      - 3054 molecules from the 133885 set that failed a consistency check
atomref.txt              - Atomic reference data
readme.txt               - Documentation

Molecules
---------

For a subset of the GDB-9 database [1] consisting of 133885 neutral organic 
molecules composed from elements H,C,N,O,F, molecular geometries were relaxed 
and properties calculated at the DFT/B3LYP/6-31G(2df,p) level of theory. 

For a subset of 6095 isomers of C7O2H10, energetics were calculated 
at the G4MP2 [2] level of theory.

For a validation set of 100 randomly drawn molecules from the 133885 molecules set,
enthalpies of formation were additionally calculated at the 
DFT/B3LYP/6-31G(2df,p), G4MP2, G4 and CBS-QB3 levels of theory.

3054 molecules from the 133885 GDB9 molecules failed a consistency check where the Corina generated
Cartesian coordinates and the B3LYP/6-31G(2df,p) equilibrium geometry lead to different SMILES strings.

Format
------

Each molecule is stored in its own file, ending in ".xyz".
The format is an ad hoc extension of the XYZ format [3].

Line       Content
----       -------
1          Number of atoms na
2          Properties 1-17 (see below)
3,...,na+2 Element type, coordinate (x,y,z) (Angstrom), and Mulliken partial charge (e) of atom
na+3       Frequencies (3na-5 or 3na-6)
na+4       SMILES from GDB9 and for relaxed geometry
na+5       InChI for GDB9 and for relaxed geometry

The properties stored in the second line of each file:

I.  Property  Unit         Description
--  --------  -----------  --------------
 1  tag       -            "gdb9"; string constant to ease extraction via grep
 2  index     -            Consecutive, 1-based integer identifier of molecule
 3  A         GHz          Rotational constant A
 4  B         GHz          Rotational constant B
 5  C         GHz          Rotational constant C
 6  mu        Debye        Dipole moment
 7  alpha     Bohr^3       Isotropic polarizability
 8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
 9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
10  gap       Hartree      Gap, difference between LUMO and HOMO
11  r2        Bohr^2       Electronic spatial extent
12  zpve      Hartree      Zero point vibrational energy
13  U0        Hartree      Internal energy at 0 K
14  U         Hartree      Internal energy at 298.15 K
15  H         Hartree      Enthalpy at 298.15 K
16  G         Hartree      Free energy at 298.15 K
17  Cv        cal/(mol K)  Heat capacity at 298.15 K

I. = Property index (properties are given in this order)
For the 6095 isomers, properties 12-16 were calculated at the G4MP2 level of theory.
All other calculations were done at the DFT/B3LYP/6-31G(2df,p) level of theory.

Notes
-----

Out of the 133885 molecules, geometries of the 11 molecules with indices 
21725, 87037, 59827, 117523, 128113, 129053, 129152, 129158, 130535, 6620, 59818 
were difficult to converge.
Low threshold convergence was possible for 21725, 59827, 128113, 129053, 129152, 130535.
Molecules 6620 and 59818 converged to very low-lying saddlepoints, with lowest frequency < 10i cm^-1.

References
----------

[1] Lorenz C. Blum, Jean-Louis Reymond: 970 Million Druglike Small Molecules
    for Virtual Screening in the Chemical Universe Database GDB-13, Journal of
    the American Chemical Society 131(25): 8732-8733, 2009. DOI: 10.1021/ja902302h
[2] Larry A. Curtiss, Paul C. Redfern, Krishnan Raghavachari: Gaussian-4 theory 
    using reduced order perturbation theory, Journal of Chemical Physics 127(12):
    124105, 2007. DOI: 10.1063/1.2770701
[3] The XYZ format, originally developed for the XMol program by the Minnesota
    Supercomputer Center, is a widespread plain text format for exchange of molecules
    (atomic coordinates and annotation). There is no formal specification. See, e.g.,
    http://openbabel.org/wiki/XYZ, or, http://wiki.jmol.org/index.php/File_formats/Formats/XYZ
