# Changelog

All notable changes to Universal Dendrimer Builder are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) ·
Versioning: [Semantic Versioning](https://semver.org).

## [Unreleased]

## [1.0.1] - 2026-10-11

### Fixed
- Parameterization GUI, Extract mol2 tab: the conversion worker thread was never
  started, so **Convert mol2 / Extract mol2** did nothing. It now runs.

### Documentation
- README links the new companion repository,
  [UDB Characterization Tools](https://github.com/HirushaW/UDB-Characterization)
  (UDB Characteriser and the parameterization / MOL2 converter).

## [1.0.0] - 2026-10-10

First public stable release. It consolidates the placement engine, force-field
parameterization, NAMD/AMBER simulation setup and structural characterisation into a
single GUI application, after several internal development versions.

### Placement engine
- Seven-stage pipeline: cone subdivision, Fibonacci-sphere sampling, labile-H grafting,
  simulated annealing over SO(3), collective minimisation, intergenerational
  minimisation and fragment-level AM1-BCC charges.
- KD-tree spatial pre-filter for placement (about 3.6x faster whole builds).
- Flexible leaving-atom and binding-atom selection for non-hydrogen chemistries
  (Monomer Terminal Leaving Atom, Terminal Group Binding Atom, Terminal Group Leaving
  Atom) with automatic hydrogen detection and manual override.

### Force fields
- GAFF2 default pathway (antechamber, parmchk2, tleap), fully automated.
- Fragment BCC charges that avoid antechamber's whole-molecule MAXBOND limit.
- CGenFF parameter routing for heteroatom-rich fragments.
- Gasteiger charges as a fallback.

### Simulation
- AMBER `.prmtop`/`.inpcrd` output and sander-compatible minimisation.
- Direct CHARMM PSF export.
- Complete NAMD 3.x pipelines: TIP3P solvation, ions, minimisation, heating, NVT/NPT
  equilibration, production and continuation scripts.

### Characterisation
- Radius of gyration and full Rg tensor (asphericity, acylindricity, relative shape
  anisotropy), SASA, van der Waals volume, hydrodynamic radius and density.

### Families
- PAMAM, PPI, Triazine, Carbosilane, PLL, Phosphorus, Polyester, Polyglycerol and
  Custom, generations G1–G10.

### Also in this release
- Released as open source under the MIT License, with a security policy (`SECURITY.md`).
- Companion parameterization GUI (`dendrimer_param_gui_v2.py`) for AMBER, NAMD/CHARMM
  and GROMACS topologies.
- Example structures for eight families (G1–G6) in `examples/Structures.tar.gz`.
- The NAMD binary is found from `UDB_NAMD_BIN`, then `namd3` on `PATH`, then
  `/opt/namd/...`; it can still be changed in the NAMD tab.

[Unreleased]: https://github.com/HirushaW/universal-dendrimer-builder/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/HirushaW/universal-dendrimer-builder/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/HirushaW/universal-dendrimer-builder/releases/tag/v1.0.0
