# Universal Dendrimer Builder (UDB)

**Version 1.0.0** · Python desktop application (Tkinter) · Linux · [MIT License](LICENSE)

UDB builds, force-field parameterizes and sets up molecular-dynamics simulations of
dendrimer nanostructures. It covers nine chemical families from generation G1 to G10
and drives AmberTools25 (GAFF2) and NAMD 3.x for you.

**Website and documentation:** <https://universal-dendrimer-builder-showcas.vercel.app>
(installation guide, user manual, benchmarks and a 3D gallery of UDB-built structures).

---

## Features

- **Nine dendrimer families:** PAMAM, PPI, Triazine, Carbosilane, PLL, Phosphorus,
  Polyester, Polyglycerol and Custom (your own core and monomer).
- **Seven-stage placement engine:** cone subdivision, Fibonacci-sphere sampling,
  labile-H grafting, simulated annealing over SO(3), collective minimisation,
  intergenerational minimisation and fragment-level AM1-BCC charges.
- **Force fields:** GAFF2 by default (antechamber → parmchk2 → tleap), CGenFF routing for
  heteroatom-rich fragments, Gasteiger fallback when BCC fails.
- **Simulation setup:** AMBER `.prmtop`/`.inpcrd`, CHARMM/PSF export and complete NAMD 3.x
  pipelines (solvation, ions, minimisation, heating, equilibration, production).
- **Structural characterisation:** radius of gyration, Rg tensor and shape descriptors,
  SASA, van der Waals volume, hydrodynamic radius and density.
- **Output formats:** MOL2, PDB, XYZ, GRO and PSF.

## Repository contents

| Path | What it is |
|---|---|
| `universal_dendrimer_builder_v1_0.py` | The UDB application (build, parameterize, simulate). |
| `dendrimer_param_gui_v2.py` | Companion parameterization GUI: converts finished structures to AMBER, NAMD/CHARMM and GROMACS topologies and validates atom types, charges and penalty scores. |
| `examples/Structures.tar.gz` | Example UDB builds for eight families (G1–G6) with AMBER/NAMD inputs, minimisation logs and build logs. |
| `requirements.txt` | Python packages. |
| `CHANGELOG.md` | What changed in each version. |
| `CITATION.cff` | How to cite UDB. |
| `LICENSE` | MIT License. |
| `SECURITY.md` | How to report a security problem. |

## Requirements

- Linux (Ubuntu 20.04 or newer), x86_64, with a display for the GUI
- Python 3.8 or newer with Tkinter (`sudo apt-get install python3-tk`)
- [AmberTools25](https://ambermd.org/AmberTools.php) (conda-forge)
- [NAMD 3.x](https://www.ks.uiuc.edu/Research/namd/) for MD runs
- Open Babel (optional, extra MOL2 conversions)

## Installation

```bash
# 1. Get the code
git clone https://github.com/HirushaW/universal-dendrimer-builder.git
cd universal-dendrimer-builder

# 2. Create the AmberTools25 environment and install the Python packages
conda create -n AmberTools25 -c conda-forge ambertools=25 -y
conda activate AmberTools25
pip install -r requirements.txt

# 3. Launch UDB
python3 universal_dendrimer_builder_v1_0.py
```

The full step-by-step guide, including NAMD installation and known issues, is on the
[Installation page](https://universal-dendrimer-builder-showcas.vercel.app/tools/udb/install/).

### Telling UDB where NAMD is

UDB looks for the NAMD binary in this order:

1. the `UDB_NAMD_BIN` environment variable, e.g.
   `export UDB_NAMD_BIN=$HOME/NAMD_3.0.2/namd3` in `~/.bashrc`;
2. `namd3` on your `PATH`;
3. `/opt/namd/NAMD_3.0.x_Linux-x86_64-multicore/namd3`.

You can also change it at any time in the NAMD tab of the application.

### Example structures

```bash
tar xzf examples/Structures.tar.gz
```

## Usage

1. Choose a dendrimer family (or Custom) and load or accept the core and monomer.
2. Set the target generation and placement parameters, then build.
3. Parameterize with GAFF2 / AM1-BCC (or CGenFF), then export AMBER, PSF or GRO files.
4. Generate the NAMD pipeline and run it.

The [User Manual](https://universal-dendrimer-builder-showcas.vercel.app/tools/udb/manual/)
walks through every tab and parameter.

## Versions and branches

| Branch | Purpose |
|---|---|
| `main` | Stable code. Every release is tagged here (`v1.0.0`, `v1.1.0`, …). |
| `develop` | Ongoing work. Changes go into `develop` and are merged into `main` for the next release. |

Versions follow [Semantic Versioning](https://semver.org): new features raise the middle
number (1.1.0), fixes raise the last (1.0.1). See [CHANGELOG.md](CHANGELOG.md).

## Citation

A paper describing UDB is in preparation. Until it is published, please cite the software
(see [CITATION.cff](CITATION.cff), or use **Cite this repository** on GitHub):

```bibtex
@software{walgampaya2025udb,
  author  = {Walgampaya, Hirusha},
  title   = {{Universal Dendrimer Builder (UDB): An Open-Source Python
             Tool for Building, Parameterizing, and Simulating
             Dendrimer Nanostructures}},
  year    = {2025},
  version = {1.0},
  url     = {https://github.com/HirushaW/universal-dendrimer-builder},
  note    = {Citation pending peer-reviewed publication}
}
```

## License

UDB is released under the [MIT License](LICENSE): you may use, modify and share it,
including commercially, as long as the copyright notice is kept. AmberTools, NAMD and
the other programs UDB calls are separate software under their own licences.

## Author

**Hirusha Walgampaya**. Questions and bug reports:
[open an issue](https://github.com/HirushaW/universal-dendrimer-builder/issues).

UDB is an independent tool and is not affiliated with AmberTools, NAMD or CHARMM-GUI;
it calls these packages as external dependencies.
