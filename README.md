# Universal Dendrimer Builder (UDB) v1.0

> A GUI-driven Python tool for building, visualising, and characterising dendrimer molecular structures — from PAMAM to custom topologies.

**Designed by Hirusha Walgampaya**

---

## Overview

The Universal Dendrimer Builder (UDB) is an all-in-one scientific tool that generates atomistic 3-D dendrimer structures ready for molecular dynamics (MD) simulation pipelines (AmberTools25 / NAMD). It supports nine dendrimer types, an extended 43-property characterisation engine, and a Tkinter GUI — no command-line expertise required.

### Supported Dendrimer Types

| Type | Chemistry | Core atom | Common application |
|---|---|---|---|
| **PAMAM** | Polyamidoamine | N | Drug delivery, biomedical |
| **PPI** | Poly(propyleneimine) | N | Gene delivery |
| **Triazine** | Aromatic triazine | N | 3-way branching |
| **Carbosilane** | Si-C backbone | Si | Antiviral |
| **PLL** | Poly-L-Lysine | N | Biocompatible peptide |
| **Phosphorus** | P-N backbone | P | Anti-inflammatory (Majoral) |
| **Polyester** | Ester linkages | O | Biodegradable drug release |
| **Polyglycerol** | Hydroxyl surface | O | Highly biocompatible |
| **Custom** | User-defined | any | Research / novel dendrimers |

---

## Key Features

- **Placement pipeline**: Fibonacci sphere (Stage 2) + Simulated Annealing over SO(3) (Stage 7) + post-generation Collective Minimisation, with 7 placement refinement stages
- **43-property characterisation** module (see [Characterisation](#characterisation))
- **Live 3-D visualisation** with matplotlib (bond rendering, atom colouring, per-generation views)
- **AmberTools25 integration**: automatic GAFF2 atom typing, charge assignment (AM1-BCC), AMBER topology (`.prmtop` / `.inpcrd`)
- **NAMD integration**: writes `.namd` config files for energy minimisation and MD
- **Multi-format export**: PDB, PSF, MOL2, SDF, XYZ; optional OpenBabel conversion
- **Build log**: timestamped, copy-paste-ready summary per generation

---

## File Structure

```
.
├── universal_dendrimer_builder_v1_0.py   # Main GUI application (7 245 lines)
├── udb_characterization.py               # Extended characterisation module
├── requirements.txt                      # pip dependencies
├── environment.yml                       # Conda environment spec
└── README.md
```

---

## Installation

### Option A — Conda (recommended, includes AmberTools25)

```bash
conda env create -f environment.yml
conda activate udb

# Install AmberTools25 (required for GAFF2/AM1-BCC)
conda install -c conda-forge ambertools=25

# Optional: install OpenBabel for extra format support
conda install -c conda-forge openbabel
```

### Option B — pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# python3-tk must be installed via your system package manager:
#   Ubuntu/Debian: sudo apt install python3-tk
```

### NAMD (optional, for MD)

Download NAMD 3.x from https://www.ks.uiuc.edu/Research/namd/ and install to `/opt/namd/`. Update the path constant at the top of `universal_dendrimer_builder_v1_0.py` if you use a different location:

```python
NAMD_DEFAULT_BIN = '/opt/namd/NAMD_3.0.1_Linux-x86_64-multicore/namd3'
```

---

## Running UDB

```bash
conda activate udb   # or source .venv/bin/activate
python3 universal_dendrimer_builder_v1_0.py
```

The GUI will open. Select a dendrimer preset (or Custom), set the target generation, and click **Build**.

---

## Characterisation

`udb_characterization.py` is a standalone module (NumPy only) that computes 43 properties after each generation is built:

| Category | Properties |
|---|---|
| **Composition & topology** | N, N_heavy, N_H, formula, DB, g_max, strain risk |
| **Molecular mass** | MW (Da / kDa), MW per terminal group |
| **Geometry** | Rg (all / heavy), R_sphere, Rh, diameter, end-to-end distances |
| **Shape** | Asphericity (δ), acylindricity (c), κ², shape tensor eigenvalues, Maiti 2004 criterion |
| **Volumetric** | V_sphere, V_vdW (MC / GAFF2), void fraction, V_enc, Vh |
| **Surface** | SASA (Shrake–Rupley), SASA per terminal, terminal density |
| **Physical (298 K)** | ρ, D (Stokes–Einstein), s (Svedbergs), [η] (Flory–Fox) |

The module can also be used independently:

```python
from udb_characterization import compute_full_characterization, format_characterization_log

report = compute_full_characterization(
    atoms, generation=3, mass_amu=14234.5,
    n_term=32, branching_factor=2, den_type='PAMAM', P=17
)
print(format_characterization_log(report))
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| NumPy | ≥ 1.24 | Geometry, characterisation |
| SciPy | ≥ 1.10 | Simulated annealing, minimisation |
| Matplotlib | ≥ 3.7 | 3-D visualisation |
| Tkinter | stdlib | GUI |
| AmberTools25 | 25 | GAFF2 typing, AM1-BCC charges (recommended) |
| NAMD | 3.x | MD simulation runner (optional) |
| OpenBabel | any | Extra format conversion (optional) |

---

## References

- de Gennes & Hervet (1983) — dendrimer generation limit g_max
- Maiti et al. (2004) — shape criterion (δ < 0.05 for G3+)
- Shrake & Rupley (1973) — SASA algorithm
- Bondi (1964) — vdW radii fallback
- AmberTools25 gaff2.dat — GAFF2 radii and atom types

---

## License

This software was developed by **Hirusha Walgampaya** for academic and research use.
