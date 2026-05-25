#!/usr/bin/env python3
"""
udb_characterization.py
========================
Universal Dendrimer Builder  v1.0  —  Extended Characterisation Module
Designed by Hirusha Walgampaya

Standalone module: import and call compute_full_characterization() after
each generation is built.  No GUI dependency.  Requires numpy only.

Usage (inside build_dendrimer or any external script):
    from udb_characterization import compute_full_characterization, format_characterization_log
    report = compute_full_characterization(atoms, gen, mass_amu, n_term,
                                           branching_factor, den_type, P, g_max,
                                           strain_risk, area_per_term)
    log_text = format_characterization_log(report)
    print(log_text)
"""

import math
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
kB      = 1.380649e-23    # J/K
NA      = 6.02214076e23   # mol⁻¹
ETA_H2O = 8.9e-4          # Pa·s  (water viscosity at 298 K)
T_STD   = 298.0           # K

# GAFF2 vdW radii (Å) — AmberTools25 gaff2.dat MOD4 RE block
GAFF2_RADII = {
    # Hydrogens
    'h1':1.387,'h2':1.287,'h3':1.187,'h4':1.409,'h5':1.359,
    'ha':1.459,'hc':1.487,'hn':0.600,'ho':0.000,'hp':0.600,
    'hs':0.600,'hw':0.000,'hx':1.100,
    # Carbons
    'c':1.908,'c1':1.908,'c2':1.908,'c3':1.908,'ca':1.908,
    'cc':1.908,'cd':1.908,'ce':1.908,'cf':1.908,'cg':1.908,
    'ch':1.908,'cp':1.908,'cq':1.908,'cu':1.908,'cv':1.908,
    'cx':1.908,'cy':1.908,'cz':1.908,
    # Nitrogens
    'n':1.824,'n1':1.824,'n2':1.824,'n3':1.824,'n4':1.824,
    'na':1.824,'nb':1.824,'nc':1.824,'nd':1.824,'ne':1.824,
    'nf':1.824,'nh':1.824,'no':1.824,
    # Oxygens
    'o':1.661,'oh':1.721,'os':1.684,'ow':1.768,
    # Sulfurs
    's':2.000,'s2':2.000,'s4':2.000,'s6':2.000,
    'sh':2.000,'ss':2.000,'sx':2.000,'sy':2.000,
    # Phosphorus
    'p2':2.100,'p3':2.100,'p4':2.100,'p5':2.100,'pb':2.100,
    'pc':2.100,'pd':2.100,'pe':2.100,'pf':2.100,'px':2.100,'py':2.100,
    # Halogens
    'f':1.750,'cl':1.948,'br':2.220,'i':2.350,
    # Silicon
    'si':2.290,'Si':2.290,
    # Element fallbacks
    'C':1.908,'N':1.824,'O':1.683,'H':1.487,
    'S':2.000,'P':2.100,'SI':2.290,
    'F':1.750,'CL':1.948,'BR':2.220,'I':2.350,
}

# Bondi 1964 fallback radii (Å)
BONDI = {
    'H':1.20,'C':1.70,'N':1.55,'O':1.52,'F':1.47,
    'P':1.80,'S':1.80,'CL':1.75,'BR':1.85,'I':1.98,'SI':2.10,
}

# MW lookup table (Da)
MW_TABLE = {
    'C':12.011,'H':1.008,'N':14.007,'O':15.999,'S':32.06,
    'P':30.974,'SI':28.086,'F':18.998,'CL':35.45,'BR':79.904,'I':126.90,
}


# ── Atom helpers ───────────────────────────────────────────────────────────────
def _elem(atom):
    """Extract element symbol (1-2 chars, uppercase) from atom name."""
    e = ''.join(c for c in atom.get('name','C') if c.isalpha())[:2].upper()
    return e if e else 'C'

def _is_H(atom):
    return (atom.get('name','').upper().startswith('H') or
            atom.get('type','').lower().startswith('h'))

def _vdw_r(atom):
    t = atom.get('type','').strip()
    if t in GAFF2_RADII:
        return GAFF2_RADII[t]
    tl = t.lower()
    if tl in GAFF2_RADII:
        return GAFF2_RADII[tl]
    e = _elem(atom)
    if e in BONDI:
        return BONDI[e]
    if e[:1] in BONDI:
        return BONDI[e[:1]]
    return 1.70


# ══════════════════════════════════════════════════════════════════════════════
# 1. STRUCTURAL GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def radius_of_gyration(atoms):
    """Rg over all atoms."""
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    com = coords.mean(axis=0)
    return float(np.sqrt(((coords - com)**2).sum(axis=1).mean()))

def radius_of_gyration_heavy(atoms):
    """Rg over heavy atoms only (Maiti 2004 convention)."""
    heavy = [a for a in atoms if not _is_H(a)]
    if len(heavy) < 2:
        return radius_of_gyration(atoms)
    coords = np.array([[a['x'],a['y'],a['z']] for a in heavy])
    com = coords.mean(axis=0)
    return float(np.sqrt(((coords - com)**2).sum(axis=1).mean()))

def bounding_sphere_radius(rg_heavy):
    """R = sqrt(5/3) * Rg_heavy."""
    return math.sqrt(5.0/3.0) * rg_heavy

def hydrodynamic_radius(rg_heavy):
    """Rh = sqrt(5/3) * Rg_heavy  (hard-sphere approximation)."""
    return math.sqrt(5.0/3.0) * rg_heavy

def end_to_end_distances(atoms, core_centre, terminal_names):
    """
    Distances from each terminal atom to the core centre.
    Returns (max_ee, mean_ee, n_terminals_found).
    """
    if terminal_names is None:
        terminal_names = []
    term_atoms = [a for a in atoms if a.get('name','') in terminal_names]
    if not term_atoms:
        # fallback: outermost 10% of atoms by distance from centroid
        coords_all = np.array([[a['x'],a['y'],a['z']] for a in atoms])
        dists_all = np.linalg.norm(coords_all - core_centre, axis=1)
        cutoff = np.percentile(dists_all, 90)
        term_atoms = [atoms[i] for i,d in enumerate(dists_all) if d >= cutoff]
    if not term_atoms:
        return 0.0, 0.0, 0
    dists = [
        float(np.linalg.norm(
            np.array([a['x'],a['y'],a['z']]) - core_centre))
        for a in term_atoms
    ]
    return max(dists), sum(dists)/len(dists), len(dists)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAPE DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════
def shape_tensor_and_eigenvalues(atoms):
    """
    Compute shape tensor G over heavy atoms.
    Returns (eigenvalues sorted ascending, G matrix).
    """
    heavy = [a for a in atoms if not _is_H(a)]
    if len(heavy) < 4:
        heavy = atoms
    coords = np.array([[a['x'],a['y'],a['z']] for a in heavy])
    com = coords.mean(axis=0)
    r = coords - com
    N = len(heavy)
    G = np.array([[np.dot(r[:,i],r[:,j]) for j in range(3)]
                  for i in range(3)]) / N
    eigs = sorted(np.linalg.eigvalsh(G))
    return eigs, G

def asphericity(eigs):
    """delta = 1 - 3*I2/I1^2; 0=sphere, 1=rod."""
    I1 = sum(eigs)
    I2 = eigs[0]*eigs[1] + eigs[1]*eigs[2] + eigs[0]*eigs[2]
    if abs(I1) < 1e-12:
        return 0.0
    return float(1.0 - 3.0*I2/(I1**2))

def acylindricity(eigs):
    """c = lambda2 - lambda1."""
    return float(eigs[1] - eigs[0])

def relative_shape_anisotropy(eigs):
    """kappa^2 = (3/2)*Tr(G^2)/Tr(G)^2 - 1/2."""
    I1 = sum(eigs)
    I2 = eigs[0]*eigs[1] + eigs[1]*eigs[2] + eigs[0]*eigs[2]
    if abs(I1) < 1e-12:
        return 0.0
    return float(1.0 - 3.0*I2/(I1**2))   # same as asphericity for this defn

def shape_descriptor_label(delta):
    """Human-readable shape label from asphericity value."""
    if delta < 0.02:
        return 'near-perfect sphere'
    elif delta < 0.05:
        return 'slightly aspherical  (typical G3+, Maiti 2004)'
    elif delta < 0.15:
        return 'moderately aspherical'
    else:
        return 'highly aspherical / elongated'

def fractal_dimension_note():
    """
    Fractal dimension Df from log(N) vs log(Rg) across generations.
    Cannot be computed from a single generation — needs multi-gen data.
    Returns a note string.
    """
    return 'Requires Rg values across multiple generations: Df = d[log N]/d[log Rg]'


# ══════════════════════════════════════════════════════════════════════════════
# 3. VOLUMETRIC PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
def bounding_sphere_volume(R_sphere):
    """V_sphere = (4/3)*pi*R^3  (Å³)."""
    return (4.0/3.0) * math.pi * R_sphere**3

def vdw_volume_mc(atoms, n_samples=100000, seed=42):
    """
    Monte Carlo van der Waals volume (Å³).
    Uses GAFF2 radii. Chunked for memory efficiency.
    """
    if not atoms:
        return 0.0
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    radii  = np.array([_vdw_r(a) for a in atoms])
    rmax   = float(radii.max())
    lo = coords.min(axis=0) - rmax
    hi = coords.max(axis=0) + rmax
    box_vol = float(np.prod(hi - lo))
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    inside = np.zeros(n_samples, dtype=bool)
    chunk = 2000
    r2 = radii**2
    for i in range(0, n_samples, chunk):
        d2 = ((pts[i:i+chunk,None,:] - coords[None,:,:])**2).sum(axis=2)
        inside[i:i+chunk] = (d2 < r2).any(axis=1)
    return float(box_vol * inside.sum() / n_samples)

def void_fraction(V_vdw, V_sphere):
    """f_void = 1 - V_vdW / V_sphere."""
    if V_sphere <= 0:
        return 0.0
    return max(0.0, 1.0 - V_vdw / V_sphere)

def encapsulation_volume(V_sphere, V_vdw):
    """V_enc ≈ V_sphere - V_vdW  (Å³) — internal cavity estimate."""
    return max(0.0, V_sphere - V_vdw)

def hydrodynamic_volume(Rh):
    """Vh = (4/3)*pi*Rh^3  (nm³)."""
    return (4.0/3.0) * math.pi * (Rh / 10.0)**3


# ══════════════════════════════════════════════════════════════════════════════
# 4. SURFACE PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
def sasa_shrake_rupley(atoms, probe=1.4, n_points=960):
    """
    Solvent-accessible surface area (Å²) — Shrake & Rupley (1973).
    Fibonacci-lattice sphere, n_points test points per atom.
    probe = 1.4 Å (water radius).
    """
    if not atoms:
        return 0.0
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    radii  = np.array([_vdw_r(a) for a in atoms])
    # Fibonacci sphere unit vectors
    idx = np.arange(n_points)
    phi_f = np.arccos(1 - 2*(idx+0.5)/n_points)
    th_f  = np.pi*(1 + 5**0.5)*idx
    unit  = np.column_stack([np.sin(phi_f)*np.cos(th_f),
                             np.sin(phi_f)*np.sin(th_f),
                             np.cos(phi_f)])
    rmax = float(radii.max())
    total = 0.0
    for i in range(len(atoms)):
        Ri = radii[i] + probe
        pts = coords[i] + Ri * unit
        dij = np.linalg.norm(coords - coords[i], axis=1)
        nbr = (dij > 0) & (dij < Ri + rmax + probe)
        if not nbr.any():
            total += 4 * math.pi * Ri * Ri
            continue
        nb_c  = coords[nbr]
        nb_r2 = (radii[nbr] + probe)**2
        d2    = ((pts[:,None,:] - nb_c[None,:,:])**2).sum(axis=2)
        acc   = (d2 >= nb_r2).all(axis=1)
        total += 4 * math.pi * Ri * Ri * acc.sum() / n_points
    return float(total)

def sasa_per_terminal(sasa, n_term):
    if n_term and n_term > 0:
        return sasa / n_term
    return None

def terminal_surface_density(n_term, R_sphere):
    """terminals per Å² on bounding sphere surface."""
    A = 4 * math.pi * R_sphere**2
    if A <= 0:
        return 0.0
    return n_term / A


# ══════════════════════════════════════════════════════════════════════════════
# 5. MASS & COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
def atom_composition(atoms):
    """Return dict of {element: count}."""
    ec = {}
    for a in atoms:
        e = _elem(a)
        if e not in MW_TABLE:
            e = e[:1]
        ec[e] = ec.get(e, 0) + 1
    return ec

def molecular_weight(atoms):
    """MW in Da from atom list."""
    ec = atom_composition(atoms)
    return sum(MW_TABLE.get(e, 0) * n for e, n in ec.items())

def degree_of_branching(actual_n_term, theoretical_n_term):
    """DB = actual / theoretical terminal groups (1.0 = perfect)."""
    if theoretical_n_term <= 0:
        return 1.0
    return actual_n_term / theoretical_n_term


# ══════════════════════════════════════════════════════════════════════════════
# 6. PHYSICAL PROPERTIES (ESTIMATED, solution at 298 K)
# ══════════════════════════════════════════════════════════════════════════════
def diffusion_coefficient(Rh_ang, T=T_STD, eta=ETA_H2O):
    """D = kT / (6*pi*eta*Rh)  [m²/s]  — Stokes–Einstein."""
    Rh_m = Rh_ang * 1e-10
    return kB * T / (6 * math.pi * eta * Rh_m)

def sedimentation_coefficient(MW_Da, Rh_ang, v_bar=0.73e-6, rho=1.0e6,
                               T=T_STD, eta=ETA_H2O):
    """
    s = M(1 - v_bar*rho) / (NA * 6*pi*eta*Rh)   [Svedbergs]
    v_bar = 0.73 cm³/g = 0.73e-6 m³/kg (typical polymer partial specific volume)
    rho   = 1000 kg/m³ (water)
    """
    Rh_m = Rh_ang * 1e-10
    M_kg = MW_Da * 1e-3 / NA    # kg per molecule
    f    = 6 * math.pi * eta * Rh_m
    s_si = M_kg * (1 - v_bar * rho) / f
    return s_si / 1e-13           # convert to Svedbergs

def intrinsic_viscosity(Vh_nm3, MW_Da):
    """
    [eta] = 2.5 * Vh * NA / M   [dL/g]  — Flory–Fox approximation.
    Vh in nm³, MW in Da.
    """
    Vh_L  = Vh_nm3 * 1e-24     # nm³ → L
    M_g   = MW_Da               # g/mol
    return 2.5 * Vh_L * NA / M_g

def molecular_density(MW_Da, V_sphere_ang3):
    """rho = (M / V_sphere) * 1.66054  [g/cm³]."""
    if V_sphere_ang3 <= 0:
        return 0.0
    return (MW_Da / V_sphere_ang3) * 1.66053906660


# ══════════════════════════════════════════════════════════════════════════════
# 7. TOPOLOGY & BRANCHING
# ══════════════════════════════════════════════════════════════════════════════
def theoretical_n_term(f, m, G):
    """N_term = f * m^G."""
    return f * (m ** G)

def g_max_de_gennes(P, const=4.0):
    """g_max = 2.88*ln(P) + const  (de Gennes & Hervet 1983)."""
    return 2.88 * math.log(max(P, 1)) + const

def surface_area_per_terminal(Rg, n_term, probe=1.4):
    """A_term = 4*pi*(R+probe)^2 / N_term  [Å²]."""
    R = math.sqrt(5.0/3.0) * Rg
    A = 4 * math.pi * (R + probe)**2
    return A / n_term if n_term > 0 else 9999.0


# ══════════════════════════════════════════════════════════════════════════════
# MASTER COMPUTE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def compute_full_characterization(
        atoms,
        generation,
        mass_amu,
        n_term,
        branching_factor=2,
        den_type='PAMAM',
        P=17,
        g_max=None,
        strain_risk='NONE',
        area_per_term=None,
        terminal_names=None,
        core_centre=None,
        mc_samples=100000,
        sasa_points=960):
    """
    Compute all 43 characterisation properties for a built dendrimer generation.

    Parameters
    ----------
    atoms           : list of atom dicts  (keys: name, type, x, y, z, charge)
    generation      : int  — current generation number
    mass_amu        : float — molecular weight in Da (pre-computed)
    n_term          : int  — actual terminal group count from build
    branching_factor: int  — m (children per junction)
    den_type        : str  — dendrimer type label
    P               : int  — spacer length (from preset)
    g_max           : float or None — if None, computed from P
    strain_risk     : str  — 'NONE'|'LOW'|'MODERATE'|'HIGH'
    area_per_term   : float or None — pre-computed surface area per terminal
    terminal_names  : list of str — terminal atom names for end-to-end calc
    core_centre     : np.array shape (3,) — core centroid
    mc_samples      : int — MC samples for vdW volume
    sasa_points     : int — Fibonacci sphere points for SASA

    Returns
    -------
    dict of all computed properties (ready for format_characterization_log)
    """
    if core_centre is None:
        coords_all = np.array([[a['x'],a['y'],a['z']] for a in atoms])
        core_centre = coords_all.mean(axis=0)

    # ── Counts & composition ────────────────────────────────────────────────
    N        = len(atoms)
    N_heavy  = sum(1 for a in atoms if not _is_H(a))
    N_H      = N - N_heavy
    comp     = atom_composition(atoms)

    # ── Geometry ─────────────────────────────────────────────────────────────
    Rg_all   = radius_of_gyration(atoms)
    Rg_heavy = radius_of_gyration_heavy(atoms)
    R_sph    = bounding_sphere_radius(Rg_heavy)
    Rh       = hydrodynamic_radius(Rg_heavy)
    diam     = 2.0 * R_sph

    # ── Shape ────────────────────────────────────────────────────────────────
    eigs, G_mat = shape_tensor_and_eigenvalues(atoms)
    delta    = asphericity(eigs)
    acyl     = acylindricity(eigs)
    kappa2   = relative_shape_anisotropy(eigs)
    shape_lbl= shape_descriptor_label(delta)

    # ── Volumetric ───────────────────────────────────────────────────────────
    V_sph    = bounding_sphere_volume(R_sph)
    V_vdw    = vdw_volume_mc(atoms, n_samples=mc_samples)
    f_void   = void_fraction(V_vdw, V_sph)
    V_enc    = encapsulation_volume(V_sph, V_vdw)
    Vh       = hydrodynamic_volume(Rh)

    # ── Surface ──────────────────────────────────────────────────────────────
    sasa     = sasa_shrake_rupley(atoms, n_points=sasa_points)
    sasa_t   = sasa_per_terminal(sasa, n_term)
    term_dens= terminal_surface_density(n_term, R_sph)

    # ── Physical ─────────────────────────────────────────────────────────────
    rho      = molecular_density(mass_amu, V_sph)
    D        = diffusion_coefficient(Rh)
    s_Sv     = sedimentation_coefficient(mass_amu, Rh)
    eta_iv   = intrinsic_viscosity(Vh, mass_amu)

    # ── End-to-end ───────────────────────────────────────────────────────────
    ee_max, ee_mean, n_ee = end_to_end_distances(atoms, core_centre, terminal_names)

    # ── Topology ─────────────────────────────────────────────────────────────
    if g_max is None:
        g_max = g_max_de_gennes(P)
    n_term_theo = theoretical_n_term(
        4 if den_type == 'PAMAM' else branching_factor,
        branching_factor, generation)
    DB = degree_of_branching(n_term, n_term_theo)
    if area_per_term is None:
        area_per_term = surface_area_per_terminal(Rg_heavy, n_term)

    # ── Predicted Rg (scaling) ────────────────────────────────────────────────
    Rg_pred  = 3.0 * (N / 84.0)**(1.0/3.0) if N > 0 else 0.0
    Rg_dev   = abs(Rg_heavy - Rg_pred) / Rg_pred * 100.0 if Rg_pred > 0 else 0.0

    return {
        # metadata
        'den_type':    den_type,
        'generation':  generation,
        # counts
        'N':           N,
        'N_heavy':     N_heavy,
        'N_H':         N_H,
        'composition': comp,
        'n_term':      n_term,
        'n_term_theo': n_term_theo,
        'DB':          round(DB, 4),
        # mass
        'MW_Da':       round(mass_amu, 2),
        'MW_kDa':      round(mass_amu / 1000, 3),
        'MW_per_term': round(mass_amu / n_term, 1) if n_term else None,
        # geometry
        'Rg_all':      round(Rg_all,  3),
        'Rg_heavy':    round(Rg_heavy,3),
        'Rg_pred':     round(Rg_pred, 3),
        'Rg_dev_pct':  round(Rg_dev,  2),
        'R_sphere':    round(R_sph,   3),
        'Rh':          round(Rh,      3),
        'diameter_A':  round(diam,    2),
        'diameter_nm': round(diam/10, 3),
        # shape
        'asphericity': round(delta,   4),
        'acylindricity':round(acyl,   4),
        'kappa2':      round(kappa2,  4),
        'shape_label': shape_lbl,
        'eigenvalues': [round(e,4) for e in eigs],
        # volumetric
        'V_sphere_A3': round(V_sph,  1),
        'V_sphere_nm3':round(V_sph/1000, 3),
        'V_vdW_A3':    round(V_vdw,  1),
        'V_vdW_nm3':   round(V_vdw/1000, 3),
        'void_pct':    round(f_void*100, 2),
        'V_enc_A3':    round(V_enc,  1),
        'Vh_nm3':      round(Vh,     3),
        # surface
        'SASA_A2':     round(sasa,   1),
        'SASA_nm2':    round(sasa/100,3),
        'SASA_per_term':round(sasa_t,2) if sasa_t else None,
        'term_density_per_A2': round(term_dens, 6),
        # physical
        'density_g_cm3': round(rho,    4),
        'D_m2s':         D,
        's_Svedberg':    round(max(0,s_Sv), 3),
        'eta_iv_dLg':    round(eta_iv,  4),
        # end-to-end
        'ee_max_A':    round(ee_max,  2),
        'ee_mean_A':   round(ee_mean, 2),
        'n_ee_atoms':  n_ee,
        # topology
        'g_max':       round(g_max,   2),
        'strain_risk': strain_risk,
        'area_per_term_A2': round(area_per_term, 2),
        'MIN_AREA_OK': area_per_term >= 28.4,
        'gen_frac':    round(generation / g_max, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOG FORMATTER  — produces the build log block
# ══════════════════════════════════════════════════════════════════════════════
def format_characterization_log(rep, label='AS-BUILT'):
    """
    Format the full characterisation dict as a fixed-width text block
    suitable for the UDB build log.  Copy-paste ready for reports.

    Returns a single string with embedded newlines.
    """
    W = 66    # total width
    BAR  = '\u2550' * W
    bar2 = '\u2500' * W
    SEP  = '\u2508' * W

    def row(left, right, width=W):
        gap = width - len(left) - len(str(right))
        return f"  {left}{' '*max(1,gap)}{right}"

    lines = [
        '',
        BAR,
        f"  DENDRIMER CHARACTERISATION  ({label})".center(W),
        f"  {rep['den_type']}  |  Generation G{rep['generation']}  |  Universal Dendrimer Builder v1.0".center(W),
        BAR,

        '',
        f"  {'─'*20} COMPOSITION & TOPOLOGY {'─'*20}",
        row('Total atoms (N)',             f"{rep['N']:,}"),
        row('  Heavy atoms',               f"{rep['N_heavy']:,}"),
        row('  Hydrogen atoms',            f"{rep['N_H']:,}  ({rep['N_H']/max(rep['N'],1)*100:.1f}%)"),
        row('Molecular formula',
            '  '.join(f"{e}{n}" for e,n in sorted(rep['composition'].items()))),
        row('Terminal groups (actual)',    f"{rep['n_term']}"),
        row('Terminal groups (theoretical)',f"{rep['n_term_theo']}"),
        row('Degree of branching (DB)',    f"{rep['DB']:.4f}  (1.000 = perfect)"),
        row('Max generation  g_max',       f"{rep['g_max']:.2f}  (de Gennes 1983)"),
        row('G / g_max  ratio',            f"{rep['gen_frac']:.3f}"),
        row('Strain risk',                 f"{rep['strain_risk']}"),

        '',
        f"  {'─'*22} MOLECULAR MASS {'─'*23}",
        row('Molecular weight  (MW)',       f"{rep['MW_Da']:,.2f} Da  =  {rep['MW_kDa']:.3f} kDa"),
        *([ row('MW per terminal group',
                f"{rep['MW_per_term']:.1f} Da") ] if rep['MW_per_term'] else []),

        '',
        f"  {'─'*22} GEOMETRY {'─'*29}",
        row('Rg  (all atoms)',             f"{rep['Rg_all']:.3f} Å"),
        row('Rg  (heavy atoms, primary)',  f"{rep['Rg_heavy']:.3f} Å"),
        row('Rg  predicted  (N^1/3 law)',  f"{rep['Rg_pred']:.3f} Å  (dev {rep['Rg_dev_pct']:.1f}%)"),
        row('Bounding sphere radius  R',   f"{rep['R_sphere']:.3f} Å  =  {rep['R_sphere']/10:.3f} nm"),
        row('Hydrodynamic radius  Rh',     f"{rep['Rh']:.3f} Å  =  {rep['Rh']/10:.3f} nm"),
        row('Diameter  (2R)',              f"{rep['diameter_A']:.2f} Å  =  {rep['diameter_nm']:.3f} nm"),
        row('End-to-end max',              f"{rep['ee_max_A']:.2f} Å"),
        row('End-to-end mean',             f"{rep['ee_mean_A']:.2f} Å  (over {rep['n_ee_atoms']} terminal atoms)"),

        '',
        f"  {'─'*22} SHAPE {'─'*32}",
        row('Asphericity  (\u03b4)',             f"{rep['asphericity']:.4f}  \u2192  {rep['shape_label']}"),
        row('Acylindricity  (c)',           f"{rep['acylindricity']:.4f}"),
        row('Rel. shape anisotropy  (\u03ba\u00b2)',   f"{rep['kappa2']:.4f}"),
        row('Shape tensor eigenvalues',    f"\u03bb\u2081={rep['eigenvalues'][0]:.3f}  \u03bb\u2082={rep['eigenvalues'][1]:.3f}  \u03bb\u2083={rep['eigenvalues'][2]:.3f}"),
        row('Maiti 2004 criterion  (\u03b4<0.05)', '\u2714 PASS' if rep['asphericity'] < 0.05 else '\u2718 FAIL'),

        '',
        f"  {'─'*22} VOLUMETRIC {'─'*27}",
        row('Bounding sphere volume  (V_sph)', f"{rep['V_sphere_A3']:.1f} \u00c5\u00b3  =  {rep['V_sphere_nm3']:.3f} nm\u00b3"),
        row('vdW volume  (MC, GAFF2)', f"{rep['V_vdW_A3']:.1f} \u00c5\u00b3  =  {rep['V_vdW_nm3']:.3f} nm\u00b3"),
        row('Void fraction  (f_void)', f"{rep['void_pct']:.2f} %"),
        row('Encapsulation volume  (V_enc)', f"{rep['V_enc_A3']:.1f} \u00c5\u00b3"),
        row('Hydrodynamic volume  (Vh)', f"{rep['Vh_nm3']:.3f} nm\u00b3"),

        '',
        f"  {'─'*22} SURFACE {'─'*30}",
        row('SASA  (Shrake\u2013Rupley, probe=1.4\u00c5)', f"{rep['SASA_A2']:.1f} \u00c5\u00b2  =  {rep['SASA_nm2']:.3f} nm\u00b2"),
        *([ row('SASA per terminal group',
                f"{rep['SASA_per_term']:.2f} \u00c5\u00b2") ] if rep['SASA_per_term'] else []),
        row('Terminal group density', f"{rep['term_density_per_A2']:.6f} groups/\u00c5\u00b2"),
        row('Surface area per terminal', f"{rep['area_per_term_A2']:.1f} \u00c5\u00b2  " +
            ('\u2714 \u226528.4' if rep['MIN_AREA_OK'] else '\u2718 <28.4 \u00c5\u00b2 \u2014 crowded')),

        '',
        f"  {'─'*22} PHYSICAL PROPERTIES  (estimated, 298 K) {'─'*4}",
        row('Density  \u03c1  (M / V_sphere)', f"{rep['density_g_cm3']:.4f} g/cm\u00b3"),
        row('Diffusion coeff  D  (Stokes\u2013Einstein)', f"{rep['D_m2s']:.3e} m\u00b2/s"),
        row('Sedimentation coeff  s', f"{rep['s_Svedberg']:.3f} S  (Svedbergs)"),
        row('Intrinsic viscosity  [\u03b7]  (Flory\u2013Fox)', f"{rep['eta_iv_dLg']:.4f} dL/g"),

        '',
        SEP,
        '  References: de Gennes & Hervet 1983; Maiti et al. 2004; Shrake & Rupley 1973',
        '  Bondi 1964 (vdW radii fallback); AmberTools25 gaff2.dat (GAFF2 radii)',
        BAR,
        '',
    ]
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import json

    # Minimal test: generate a fake 84-atom PAMAM G1-like structure
    rng = np.random.default_rng(0)
    test_atoms = []
    elements = ['C','C','N','O','H','H','H']
    types    = ['c3','c3','n3','oh','hc','hc','hn']
    for i in range(84):
        e = elements[i % len(elements)]
        t = types[i % len(types)]
        r = 5.0 + rng.normal(0,2,3)
        test_atoms.append({
            'idx': i+1, 'name': f"{e}{i+1}", 'type': t,
            'x': float(r[0]), 'y': float(r[1]), 'z': float(r[2]),
            'charge': '0.0', 'res_idx': '1', 'res_name': 'UNL',
        })

    rep = compute_full_characterization(
        test_atoms, generation=1, mass_amu=1129.5,
        n_term=8, branching_factor=2, den_type='PAMAM',
        P=17, terminal_names=['N1','N2','N3','N4','N5','N6','N7','N8'],
    )
    print(format_characterization_log(rep, label='TEST RUN'))
    print(json.dumps({k:v for k,v in rep.items() if k != 'composition'}, indent=2))
