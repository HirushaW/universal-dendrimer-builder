#!/usr/bin/env python3
"""
===============================================================================
UNIVERSAL DENDRIMER BUILDER  v1.0
===============================================================================
Chemistries : PAMAM, PPI, Triazine, Carbosilane, PLL, Phosphorus, Polyester,
              Polyglycerol, Custom
Designed by : Hirusha Walgampaya

Placement pipeline: Fibonacci sphere (Stage 2) + Simulated Annealing over SO(3)
(Stage 7) + post-generation Collective Minimisation, with 7 w11-era placement
modifications.

Highlights (see CHANGELOG.md):
  * Flexible leaving-atom / binding-atom selection for non-hydrogen chemistries
    (Monomer Terminal Leaving Atom, Terminal Group Binding Atom, Terminal Group
    Leaving Atom) with automatic-H detection and optional manual override.
  * KD-tree spatial pre-filter for placement (~3.6x whole-build speedup).
  * NAMD 3.x / AMBER-GAFF2 pipelines, intergenerational minimisation and
    CHARMM/PSF export.

Usage:
  conda activate AmberTools25
  python3 universal_dendrimer_builder_v1_0.py

Requires: python3-tk  numpy  scipy  matplotlib  openbabel (optional)

-------------------------------------------------------------------------------
Module contents (source sections)
-------------------------------------------------------------------------------
   1.  Imports & runtime configuration
   2.  Chemistry presets & reference data tables
   3.  Theme palettes & shared UI helpers
   4.  Chemistry & geometry helpers
   5.  Cone placement engine
   6.  Growth equations
   7.  Van der Waals radii & structural characterization
   8.  Structure file writers (MOL2 / PDB / XYZ / GRO / min script)
   9.  Antechamber / AM1-BCC fragment parameterization
  10.  Structure validation & CHARMM/PSF export
  11.  NAMD / AMBER simulation pipelines
  12.  Intergenerational minimization helpers
  13.  Dendrimer build engine
  14.  3D molecular viewer & fragment manager widgets
  15.  CGenFF parameter support
  16.  Main application GUI
  17.  Application entry point
===============================================================================
"""

__version__ = "1.0.0"


# =============================================================================
# SECTION 1.  IMPORTS & RUNTIME CONFIGURATION
# =============================================================================

# NAMD binary — update this path when upgrading NAMD.
# Resolution order: $UDB_NAMD_BIN, then `namd3` on PATH, then the standard
# install location. It can also be changed per session in the NAMD tab.
import os as _os, shutil as _shutil
NAMD_DEFAULT_BIN = (_os.environ.get('UDB_NAMD_BIN')
                    or _shutil.which('namd3')
                    or '/opt/namd/NAMD_3.0.x_Linux-x86_64-multicore/namd3')

# Extended characterisation module (UDB v1.0).
try:
    from udb_characterization import (
        compute_full_characterization,
        format_characterization_log,
    )
    _CHAR_MODULE_AVAILABLE = True
except ImportError:
    _CHAR_MODULE_AVAILABLE = False
    def compute_full_characterization(*a, **kw): return {}
    def format_characterization_log(*a, **kw): return ''

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import os, json, threading, copy, traceback, subprocess, shlex, math, stat as _stat
import hashlib

# Optional acceleration kernels (Numba/CuPy). Off by default; enabled via env
# flags (see udb_accel). Import is guarded so a missing/broken numba never breaks
# the default NumPy build path.
try:
    import udb_accel as _accel
except Exception:  # pragma: no cover
    _accel = None

# Placement spatial pre-filter (#1): restrict the O(N*M) azimuthal clash reduction
# in position_monomer to atoms within SPATIAL_FILTER_ANG of the attachment point.
# On by default; opt out with UDB_PLACE_NEARBY=0 to restore full-molecule scanning.
_PLACE_NEARBY = os.environ.get('UDB_PLACE_NEARBY', '1') != '0'


def _monomer_seed(*key_parts):
    """Deterministic 64-bit seed from a per-monomer key.

    Uses hashlib (NOT the builtin hash(), which is per-process randomised via
    PYTHONHASHSEED) so the same monomer always seeds the same SA RNG stream,
    independent of process or placement order — required before any build
    parallelism can be trusted."""
    key = "|".join(str(p) for p in key_parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


# =============================================================================
# SECTION 2.  CHEMISTRY PRESETS & REFERENCE DATA TABLES
# =============================================================================

PRESETS = {
    "PAMAM":{
        "description":"Polyamidoamine — most common, biomedical & drug delivery",
        "bond_length_auto":True,"bond_length":1.47,
        "bond_angle_auto":True,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"N","mon_input_hint":"C","mon_terminal_hint":"N",
        "P":17,"f":4},
    "PPI":{
        "description":"Poly(propyleneimine) — amine-terminated, gene delivery",
        "bond_length_auto":True,"bond_length":1.47,
        "bond_angle_auto":True,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"N","mon_input_hint":"C","mon_terminal_hint":"N",
        "P":11,"f":4},
    "Triazine":{
        "description":"Triazine-based — aromatic core, 3-way branching",
        "bond_length_auto":False,"bond_length":1.33,
        "bond_angle_auto":False,"bond_angle":120.0,"branching_factor":2,
        "core_atom_hint":"N","mon_input_hint":"C","mon_terminal_hint":"N",
        "P":6,"f":3},
    "Carbosilane":{
        "description":"Carbosilane — Si-C backbone, antiviral applications",
        "bond_length_auto":False,"bond_length":1.89,
        "bond_angle_auto":False,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"Si","mon_input_hint":"C","mon_terminal_hint":"Si",
        "P":30,"f":4},
    "PLL":{
        "description":"Poly-L-Lysine — peptide backbone, biocompatible",
        "bond_length_auto":True,"bond_length":1.47,
        "bond_angle_auto":True,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"N","mon_input_hint":"C","mon_terminal_hint":"N",
        "P":20,"f":2},
    "Phosphorus":{
        "description":"Phosphorus — P-N backbone, anti-inflammatory (Majoral type)",
        "bond_length_auto":False,"bond_length":1.61,
        "bond_angle_auto":False,"bond_angle":120.0,"branching_factor":2,
        "core_atom_hint":"P","mon_input_hint":"C","mon_terminal_hint":"N",
        "P":12,"f":3},
    "Polyester":{
        "description":"Polyester — biodegradable, ester linkages, drug release (Hawker-Frechet)",
        "bond_length_auto":False,"bond_length":1.43,
        "bond_angle_auto":True,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"O","mon_input_hint":"C","mon_terminal_hint":"O",
        "P":18,"f":4},
    "Polyglycerol":{
        "description":"Polyglycerol — highly biocompatible, hydroxyl surface groups",
        "bond_length_auto":False,"bond_length":1.43,
        "bond_angle_auto":True,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"O","mon_input_hint":"C","mon_terminal_hint":"O",
        "P":11,"f":3},
    "Custom":{
        "description":"Custom — set all parameters manually",
        "bond_length_auto":False,"bond_length":1.47,
        "bond_angle_auto":False,"bond_angle":109.5,"branching_factor":2,
        "core_atom_hint":"","mon_input_hint":"","mon_terminal_hint":"",
        "P":4,"f":4},
}

BOND_LENGTH_TABLE = {
    frozenset(['N','C']):1.47, frozenset(['C','C']):1.54,
    frozenset(['C','O']):1.43, frozenset(['C','S']):1.82,
    frozenset(['Si','C']):1.89,frozenset(['Si','O']):1.63,
    frozenset(['N','N']):1.45, frozenset(['P','C']):1.84,
    frozenset(['P','O']):1.61,
}

ELEMENT_COLORS = {
    'C':'#adb5bd','N':'#4dabf7','O':'#ff6b6b','H':'#6c757d',
    'S':'#ffd43b','P':'#ff922b','SI':'#74c0fc','F':'#69db7c',
    'CL':'#51cf66','BR':'#c0392b','I':'#be4bdb',
    'DEFAULT':'#8b949e',
}

ROLE_COLORS = {
    'core_binding':'#56d364',   # green
    'core_h':      '#79c0ff',   # cyan
    'mon_input':   '#bc8cff',   # purple
    'mon_h':       '#f85149',   # red
    'mon_terminal':'#ffd43b',   # yellow
    'default':     None,        # use element colour
}
ROLE_LABELS = {
    'core_binding':'Core binding site',
    'core_h':      'Core strip-H atom',
    'mon_input':   'Monomer input atom',
    'mon_h':       'Monomer strip-H atom',
    'mon_terminal':'Terminal atom',
}

MW_TABLE = {
    'C':12.011,'H':1.008,'N':14.007,'O':15.999,'S':32.06,
    'P':30.974,'SI':28.086,'F':18.998,'CL':35.45,'BR':79.904,'I':126.90,
}


# =============================================================================
# SECTION 3.  THEME PALETTES & SHARED UI HELPERS
# =============================================================================

DARK = {
    'bg':      '#080c14',   # deep navy black
    'panel':   '#0f1623',   # dark navy panel
    'border':  '#1c2840',   # navy border
    'cyan':    '#38bdf8',   # sky blue accent
    'green':   '#06d6a0',   # emerald teal
    'yellow':  '#f4a261',   # warm amber
    'red':     '#ef233c',   # vivid red
    'white':   '#edf2f4',   # off-white text
    'muted':   '#6b7c93',   # steel blue muted
    'log_bg':  '#050810',   # near-black log
    'log_fg':  '#06d6a0',   # emerald log text
    'entry':   '#0a0f1e',   # deep entry bg
    'sel':     '#162035',   # selection highlight
    'sbtn':    '#131d30',   # subtle button
    'prog':    '#38bdf8',   # progress bar fill
    'prog_tr': '#080c14',   # progress trough
    'toggle_icon':'☀','toggle_tip':'Light mode',
}
LIGHT = {
    'bg':      '#f4f7fb',   # cool off-white
    'panel':   '#ffffff',   # pure white panel
    'border':  '#c4cfe0',   # steel border
    'cyan':    '#0077b6',   # rich navy blue
    'green':   '#028a58',   # deep emerald
    'yellow':  '#c47a00',   # rich amber
    'red':     '#d62839',   # deep red
    'white':   '#1a1f2e',   # dark navy text
    'muted':   '#5a6880',   # slate muted
    'log_bg':  '#eef3fa',   # light log bg
    'log_fg':  '#028a58',   # emerald log text
    'entry':   '#ffffff',
    'sel':     '#dbeafe',
    'sbtn':    '#e8eef7',
    'prog':    '#0077b6',
    'prog_tr': '#c4cfe0',
    'toggle_icon':'🌙','toggle_tip':'Dark mode',
}

FONT = ("Courier New",11)
BOLD = ("Courier New",11,"bold")

def _mk_entry(parent,var,T,width=28):
    return tk.Entry(parent,textvariable=var,font=FONT,
                    bg=T['entry'],fg=T['cyan'],
                    insertbackground=T['cyan'],
                    relief='flat',bd=4,width=width)

def _mk_btn(parent,text,cmd,T,bg_key='sbtn',fg_key='cyan',**kw):
    return tk.Button(parent,text=text,command=cmd,font=BOLD,
                     bg=T[bg_key],fg=T[fg_key],
                     relief='flat',padx=10,pady=4,cursor='hand2',
                     activebackground=T[bg_key],
                     activeforeground=T[fg_key],**kw)


# =============================================================================
# SECTION 4.  CHEMISTRY & GEOMETRY HELPERS
# =============================================================================

def auto_bond_length(a1,a2):
    e1=''.join(c for c in a1 if c.isalpha()).capitalize()[:2]
    e2=''.join(c for c in a2 if c.isalpha()).capitalize()[:2]
    for x,y in [(e1,e2),(e1[:1],e2[:1])]:
        k=frozenset([x,y])
        if k in BOND_LENGTH_TABLE: return BOND_LENGTH_TABLE[k]
    return 1.47

def auto_bond_angle(atom_type):
    sp2={'n2','na','nb','nc','nd','ne','nf','ca','cc','cd','ce','cf',
         'cg','ch','cp','cq','cu','cv','cx','cy'}
    return 120.0 if atom_type.lower() in sp2 else 109.5

def parse_mol2(filename):
    atoms,bonds=[],[]
    in_atom=in_bond=False
    with open(filename) as f:
        for line in f:
            line=line.rstrip()
            if '@<TRIPOS>ATOM' in line:
                in_atom,in_bond=True,False; continue
            elif '@<TRIPOS>BOND' in line:
                in_bond,in_atom=True,False; continue
            elif '@<TRIPOS>' in line:
                in_atom=in_bond=False; continue
            if in_atom and line.strip():
                p=line.split()
                if len(p)>=9:
                    atoms.append({'idx':int(p[0]),'name':p[1],
                                  'x':float(p[2]),'y':float(p[3]),'z':float(p[4]),
                                  'type':p[5],'res_idx':p[6],'res_name':p[7],
                                  'charge':p[8]})
            if in_bond and line.strip():
                p=line.split()
                if len(p)>=4:
                    bonds.append((int(p[1]),int(p[2]),p[3]))
    return atoms,bonds

def is_hydrogen(atom):
    return (atom['name'].upper().startswith('H') or
            atom['type'].lower().startswith('h'))

def build_adj(bonds):
    adj={}
    for b in bonds:
        adj.setdefault(b[0],[]).append(b[1])
        adj.setdefault(b[1],[]).append(b[0])
    return adj

def get_h_neighbors(atoms,bonds,atom_idx):
    adj=build_adj(bonds)
    amap={a['idx']:a for a in atoms}
    return [amap[nb] for nb in adj.get(atom_idx,[])
            if amap.get(nb) and is_hydrogen(amap[nb])]

def deduplicate_bonds(bonds):
    seen,unique=set(),[]
    for b in bonds:
        key=(min(b[0],b[1]),max(b[0],b[1]))
        if key not in seen:
            seen.add(key); unique.append(b)
    return unique

def remove_atoms(atoms,bonds,remove_set):
    mapping={}; new_idx=1
    for a in atoms:
        mapping[a['idx']]=None if a['idx'] in remove_set else new_idx
        if a['idx'] not in remove_set: new_idx+=1
    new_atoms=[dict(a)|{'idx':mapping[a['idx']]}
               for a in atoms if a['idx'] not in remove_set]
    new_bonds=[(mapping[b[0]],mapping[b[1]],b[2])
               for b in bonds
               if b[0] not in remove_set and b[1] not in remove_set
               and mapping.get(b[0]) and mapping.get(b[1])]
    return new_atoms,new_bonds,mapping

def rotate_around_axis(points,axis,angle,origin):
    # Vectorised Rodrigues rotation. Broadcasts the identical per-point algebra
    # over all points at once instead of a Python for-loop (profiling: this loop
    # was ~65% of total build CPU). Operation order matches the original scalar
    # form term-for-term, so results are bit-identical.
    axis=axis/np.linalg.norm(axis)
    c,s=np.cos(angle),np.sin(angle)
    V=np.asarray(points,dtype=float)-origin              # (M,3)
    cross=np.cross(axis,V)                                # (M,3)
    # Explicit elementwise dot matches the scalar np.dot(axis,v) summation order
    # exactly (a0*v0+a1*v1+a2*v2); using V@axis instead routes through BLAS/FMA,
    # whose different rounding cascades through the chaotic greedy placement and
    # breaks bit-identity at high generations.
    dot=V[:,0]*axis[0]+V[:,1]*axis[1]+V[:,2]*axis[2]      # (M,)
    return V*c+cross*s+axis*dot[:,None]*(1-c)+origin

def get_perpendicular(v):
    v=v/np.linalg.norm(v)
    ref=np.array([1,0,0]) if abs(v[0])<0.9 else np.array([0,1,0])
    perp=ref-np.dot(ref,v)*v
    return perp/np.linalg.norm(perp)

def position_monomer(n_pos,bond_dir,mon_atoms,existing_arr,bond_length,input_atom_name,
                     filter_ang=None):
    inp=next((a for a in mon_atoms if a['name']==input_atom_name),mon_atoms[0])
    inp_pos=np.array([inp['x'],inp['y'],inp['z']])
    target=n_pos+bond_dir*bond_length
    t=target-inp_pos
    mon_coords=np.array([[a['x']+t[0],a['y']+t[1],a['z']+t[2]] for a in mon_atoms])
    best_dist=-1; best_coords=mon_coords.copy()
    # in Stage 7 → filtered=[]) — previously raised ValueError → silent skip of ALL
    # monomers at G1 for small/compact cores (e.g. Polyester, Polyglycerol).
    if existing_arr is None or len(existing_arr)==0:
        return mon_coords, 999.0   # no neighbours → treat as clash-free
    # #1 spatial pre-filter: only atoms within filter_ang of the attachment point
    # n_pos can be the nearest neighbour to any rotation of this monomer (whose atoms
    # all sit within ~bond_length+span of n_pos), so the O(N*M) azimuthal clash scan
    # below only needs them. Same neighbourhood the vdW clash check already trusts.
    # N collapses from the whole molecule (thousands at high gen) to a local shell —
    # the dominant build cost per profiling. Bit-identical to the full scan: min-over-
    # subset == min-over-all whenever the nearest atom is retained (verified vs the
    # golden master). opt out with UDB_PLACE_NEARBY=0.
    if filter_ang is not None and _PLACE_NEARBY:
        d2=((existing_arr-n_pos)**2).sum(axis=1)
        existing_arr=existing_arr[d2 < filter_ang*filter_ang]
        if len(existing_arr)==0:
            return mon_coords, 999.0
    for deg in range(0,360,5):
        rotated=np.array(rotate_around_axis(mon_coords,bond_dir,np.radians(deg),target))
        diffs=existing_arr[:,None,:]-rotated[None,:,:]
        d=float(np.sqrt((diffs**2).sum(axis=2)).min())
        if d>best_dist:
            best_dist=d; best_coords=rotated
    return best_coords,best_dist

def atom_element(atom):
    e=''.join(c for c in atom['name'] if c.isalpha())[:2].upper()
    return e if e else 'C'

def atom_mw(atoms):
    ec={}
    for a in atoms:
        e=atom_element(a)
        if e not in MW_TABLE: e=e[:1]
        ec[e]=ec.get(e,0)+1
    return sum(MW_TABLE.get(e,0)*n for e,n in ec.items())


# Valid antechamber -c flag values. Maps any legacy/invalid GUI string to
# a working value. RESP is rejected because it requires a precomputed
# Gaussian/GAMESS ESP file which UDB doesn't produce.
_VALID_CHARGE_METHODS = {'bcc', 'gas', 'none'}
_LEGACY_CHARGE_MAP = {
    'am1bcc':    ('bcc', 'am1bcc is not a valid antechamber flag; using bcc (which IS AM1-BCC).'),
    'gasteiger': ('gas', 'gasteiger is not a valid antechamber flag; using gas.'),
    'resp':      ('bcc', 'RESP charges require a Gaussian/GAMESS ESP file which UDB cannot generate; falling back to bcc.'),
}

def sanitise_charge_method(cm):
    """Single source of truth for charge-method validation.
    Returns (clean_cm, was_changed, reason).
    Called once at the boundary where charge_method enters the pipeline."""
    cm = (cm or 'bcc').strip().lower()
    if cm in _VALID_CHARGE_METHODS:
        return cm, False, ''
    if cm in _LEGACY_CHARGE_MAP:
        new_cm, reason = _LEGACY_CHARGE_MAP[cm]
        return new_cm, True, reason
    return 'bcc', True, f"unrecognised charge_method '{cm}'; falling back to bcc."


def detect_rotatable_bonds(atoms, bonds, input_atom_name=None):
    """Identify rotatable single bonds in a monomer.
    A bond is rotatable if: order==1, both atoms have >=2 heavy neighbors,
    not in a ring, not amide/ester planar (sp2 N/C-N or C(=O)-N).
    The bond connecting input_atom_name to its neighbor is excluded — it's
    the grafted bond and rotation around it is handled by SA Stage 7.
    Returns list of (a_idx, b_idx) tuples (1-indexed)."""
    atom_by_idx = {a['idx']: a for a in atoms}
    nbrs = {a['idx']: [] for a in atoms}
    for b in bonds:
        nbrs[b[0]].append((b[1], b[2]))
        nbrs[b[1]].append((b[0], b[2]))

    # Ring detection via DFS — mark atoms on cycles
    in_ring = set()
    visited = set()
    parent  = {}
    for start in atom_by_idx:
        if start in visited: continue
        stack = [(start, None)]
        while stack:
            v, p = stack.pop()
            if v in visited:
                continue
            visited.add(v); parent[v] = p
            for (n, _) in nbrs[v]:
                if n == p: continue
                if n in visited:
                    cur = v
                    while cur is not None and cur != n:
                        in_ring.add(cur); cur = parent.get(cur)
                    if cur == n: in_ring.add(n)
                else:
                    stack.append((n, v))

    # Find input atom idx for grafted-bond exclusion
    input_idx = None
    if input_atom_name:
        for a in atoms:
            if a.get('name') == input_atom_name:
                input_idx = a['idx']; break
    grafted_bond = None
    if input_idx is not None and len(nbrs.get(input_idx, [])) > 0:
        grafted_bond = tuple(sorted([input_idx, nbrs[input_idx][0][0]]))

    rotatable = []
    seen = set()
    for b in bonds:
        a, c, order = b[0], b[1], b[2]
        try: order_num = int(order) if str(order).isdigit() else 1
        except Exception: order_num = 1
        if order_num != 1: continue
        if a in in_ring and c in in_ring: continue
        # Both atoms must have >= 2 heavy neighbors (else terminal — methyl, etc.)
        a_heavy = sum(1 for (n, _) in nbrs[a] if atom_element(atom_by_idx[n]) != 'H')
        c_heavy = sum(1 for (n, _) in nbrs[c] if atom_element(atom_by_idx[n]) != 'H')
        if a_heavy < 2 or c_heavy < 2: continue
        # Skip amide-like: planar C-N where C is carbonyl
        ta = atom_by_idx[a].get('type', '').lower()
        tc = atom_by_idx[c].get('type', '').lower()
        if (ta in ('c','c2') and tc in ('n','na')) or (tc in ('c','c2') and ta in ('n','na')):
            continue
        key = tuple(sorted([a, c]))
        if key == grafted_bond: continue
        if key in seen: continue
        seen.add(key)
        rotatable.append(key)
    return rotatable


def _rotate_subtree(coords, atoms, bonds, pivot_a, pivot_b, angle):
    """Rotate the subtree on the b-side of bond (pivot_a, pivot_b) by angle radians
    around the bond axis. coords is an Nx3 numpy array indexed by atom idx-1.
    Returns a new array with rotated subtree atoms."""
    nbrs = {a['idx']: set() for a in atoms}
    for b in bonds:
        nbrs[b[0]].add(b[1]); nbrs[b[1]].add(b[0])

    # BFS from pivot_b, blocking pivot_a, to collect subtree
    subtree = set()
    queue = [pivot_b]
    while queue:
        v = queue.pop()
        if v in subtree or v == pivot_a: continue
        subtree.add(v)
        for n in nbrs[v]:
            if n != pivot_a and n not in subtree:
                queue.append(n)

    p_a = coords[pivot_a - 1]
    p_b = coords[pivot_b - 1]
    axis = p_b - p_a
    norm = np.linalg.norm(axis)
    if norm < 1e-10: return coords.copy()
    axis = axis / norm

    cos_a = math.cos(angle); sin_a = math.sin(angle)
    out = coords.copy()
    for atom_idx in subtree:
        if atom_idx == pivot_b: continue
        v = coords[atom_idx - 1] - p_a
        # Rodrigues rotation
        v_rot = (v * cos_a
                 + np.cross(axis, v) * sin_a
                 + axis * np.dot(axis, v) * (1.0 - cos_a))
        out[atom_idx - 1] = p_a + v_rot
    return out


# =============================================================================
# SECTION 5.  CONE PLACEMENT ENGINE
# =============================================================================

class ConePlacementEngine:
    """
    Stage 1  compute_cone_axis       — radial unit vector parent→outward [Fix1]
    Stage 2  partition_azimuthal     — full 2π equally among n children  [Fix4]
    Stage 3  spread_angle            — fixed 20° sibling spread           [Fix2]
    Stage 4  compute_child_directions— d=cos(θ)axis+sin(θ)(cos(φ)p1+sin(φ)p2)
    Stage 5  (mol2 geometry)         — position_monomer() with cone dir
    Stage 6  has_clash_vdw           — d<vdw_scale*(ri+rj), spatial filter [Fix6,Mod1,Mod4]
    Stage 7  find_clash_free_rotation— adaptive-step scan, max clearance   [Fix3,5,Mod2]
    """
    VDW_RADII_GAFF2 = {
        # Carbon
        'c3':1.908, 'c2':1.908, 'ca':1.908, 'c':1.908,
        'c1':1.908, 'cc':1.908, 'cd':1.908, 'ce':1.908,
        'cf':1.908, 'cg':1.908, 'ch':1.908, 'cp':1.908,
        'cq':1.908, 'cu':1.908, 'cv':1.908, 'cx':1.908, 'cy':1.908,
        # Nitrogen
        'n3':1.824, 'n2':1.824, 'n4':1.824, 'n':1.824,
        'na':1.824, 'nb':1.824, 'nc':1.824, 'nd':1.824,
        'ne':1.824, 'nf':1.824, 'nh':1.824, 'no':1.824, 'n1':1.824,
        # Oxygen
        'o':1.661,  'oh':1.721, 'os':1.683, 'op':1.683, 'ow':1.768,
        # Hydrogen — near-zero for polar H (ho, hn) to avoid false clashes
        'hc':1.487, 'h1':1.387, 'h2':1.287, 'h3':1.187,
        'h4':1.409, 'h5':1.359, 'ha':1.459,
        'hn':0.600, 'ho':0.000, 'hs':0.600, 'hp':1.100,
        'hw':0.000, 'hx':1.100,
        # Sulfur
        'sh':2.000, 'ss':2.000, 's2':2.000, 's':2.000,
        's4':2.000, 's6':2.000, 'sx':2.000, 'sy':2.000,
        # Phosphorus
        'p3':2.100, 'p4':2.100, 'p5':2.100, 'p2':2.100,
        'pb':2.100, 'pc':2.100, 'pd':2.100, 'pe':2.100,
        'pf':2.100, 'px':2.100, 'py':2.100,
        # Silicon
        'si':2.290, 'Si':2.290,
        # Halogens
        'f':1.750, 'cl':1.948, 'br':2.220, 'i':2.350,
        # Element fallbacks (used when GAFF type not found)
        'C':1.908, 'N':1.824, 'O':1.683, 'H':1.487,
        'S':2.000, 'P':2.100, 'SI':2.290,
        'F':1.750, 'CL':1.948, 'BR':2.220, 'I':2.350,
    }
    # Keep legacy VDW_RADII for any external callers
    VDW_RADII = {
        'C':1.908,'N':1.824,'O':1.683,'H':1.487,'SI':2.290,
        'P':2.100,'S':2.000,'F':1.750,'CL':1.948,'BR':2.220,'I':2.350,
    }
    # SPATIAL_FILTER_ANG: Stage 6/7 pre-filter — only check atoms within this
    # distance. 12 A covers 2 full monomer lengths; fast enough at G5.
    SPATIAL_FILTER_ANG = 12.0
    KDTREE_REBUILD_INTERVAL = 20

    def __init__(self, vdw_scale=0.75, max_retries=36):
        self.vdw_scale    = vdw_scale
        self.max_retries  = max_retries
        self.core_centre  = None   # set by build_dendrimer before use
    def _vdw_radius(self, gaff_type, elem_fallback):
        """Look up vdW radius: GAFF2 type first, then element, then default."""
        r = self.VDW_RADII_GAFF2.get(gaff_type)
        if r is not None: return r
        r = self.VDW_RADII_GAFF2.get(gaff_type.lower())
        if r is not None: return r
        eu = elem_fallback.upper()[:2]
        r = self.VDW_RADII_GAFF2.get(eu)
        if r is not None: return r
        r = self.VDW_RADII_GAFF2.get(eu[:1])
        return r if r is not None else 1.70

    # ── Stage 1 ──────────────────────────────────────────────────
    def compute_cone_axis(self, parent_pos, ref_pos=None):
        """Stage 1 cone axis. G1: axis = parent_pos - ref_pos (core-bond direction).
        G2+: axis = parent_pos - core_centre (pure radial)."""
        if ref_pos is not None:
            axis = parent_pos - np.asarray(ref_pos, dtype=float)
        elif self.core_centre is not None:
            axis = parent_pos - self.core_centre
        else:
            axis = parent_pos.copy()
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return axis / norm

    def partition_azimuthal(self, n_children, base_azimuth=0.0,
                             gen=1, max_gen=1, site_idx=0, n_sites=1):
        """Fibonacci sphere (golden-ratio) azimuth distribution across sites.
        Global index i = site_idx*n_children + local_i; phi = base + 2*pi*i/phi_golden."""
        PHI_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0   # ≈ 1.6180
        N_total = max(n_sites * n_children, n_children)
        azimuths = []
        for local_i in range(n_children):
            # Global index into the Fibonacci lattice
            global_i = site_idx * n_children + local_i
            # Fibonacci azimuth — golden ratio step avoids rational recurrences
            phi = base_azimuth + 2.0 * math.pi * global_i / PHI_GOLDEN
            azimuths.append(phi % (2.0 * math.pi))
        return azimuths

    # ── Stage 3  [CORRECTED + Mod-5 w11.6] ──────────────────────
    def branching_half_angle(self, branch_angle_deg, n_children,
                              gen=1, max_gen=1):
        """Stage 3: half-angle of the branching cone; widens up to 15 deg over gens,
        capped at 70 deg."""
        if n_children <= 2:
            base = math.radians(branch_angle_deg / 2.0)
        elif n_children == 3:
            base = math.radians(54.74)
        else:
            base = math.radians(min(branch_angle_deg / 2.0, 60.0))
        t = (gen - 1) / max(max_gen - 1, 1)
        extra = math.radians(15.0) * t
        return min(base + extra, math.radians(70.0))

    # ── Stage 4 ───────────────────────────────────────────────────
    def compute_child_directions(self, cone_axis, azimuthal_angles, theta):
        """Stage 4: d_i = cos(theta)*axis + sin(theta)*(cos(phi_i)*p1 + sin(phi_i)*p2).
        Reference vector chosen as axis-of-smallest component for stability."""
        abs_c = np.abs(cone_axis)
        min_idx = int(np.argmin(abs_c))   # index of smallest component
        ref = np.zeros(3); ref[min_idx] = 1.0
        p1 = np.cross(cone_axis, ref)
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(cone_axis, p1)
        p2 /= np.linalg.norm(p2)
        dirs = []
        for phi in azimuthal_angles:
            d = (math.cos(theta) * cone_axis
                 + math.sin(theta) * (math.cos(phi) * p1 + math.sin(phi) * p2))
            norm = np.linalg.norm(d)
            dirs.append(d / norm if norm > 1e-10 else cone_axis.copy())
        return dirs

    # ── Stage 6  [Fix6 + w11.3 KDTree] ──────────────────────────
    def _build_kdtree(self, existing_atoms):
        """[w11.3] Build a KDTree from existing atom coordinates.
        Call once per generation and store as self._kdtree +
        self._kdtree_atoms. Falls back to linear scan if scipy unavailable."""
        try:
            from scipy.spatial import KDTree as _KDTree
            if not existing_atoms:
                self._kdtree = None
                self._kdtree_atoms = []
                return
            coords = np.array([[a['x'],a['y'],a['z']] for a in existing_atoms])
            self._kdtree = _KDTree(coords)
            self._kdtree_atoms = existing_atoms
        except ImportError:
            self._kdtree = None
            self._kdtree_atoms = []

    def _nearby_atoms(self, centre_pos, existing_atoms):
        """
        [w11.3] KDTree-accelerated spatial pre-filter.
        If a tree is cached and covers existing_atoms, query it in O(log N).
        Falls back to O(N) linear scan when tree not available.
        """
        if (hasattr(self, '_kdtree') and self._kdtree is not None
                and self._kdtree_atoms is existing_atoms):
            # KDTree path — O(log N)
            idxs = self._kdtree.query_ball_point(centre_pos,
                                                  self.SPATIAL_FILTER_ANG)
            return [self._kdtree_atoms[i] for i in idxs]
        # Fallback linear scan
        r2_cut = self.SPATIAL_FILTER_ANG ** 2
        nearby = []
        for a in existing_atoms:
            dx = centre_pos[0]-a['x']
            dy = centre_pos[1]-a['y']
            dz = centre_pos[2]-a['z']
            if dx*dx + dy*dy + dz*dz < r2_cut:
                nearby.append(a)
        return nearby

    def has_clash_vdw(self, new_coords, new_elements, existing_atoms,
                      centre_pos=None, exclude_radius=0.0,
                      new_types=None):
        """Stage 6 vdW clash: d < vdw_scale*(r_i + r_j). Uses GAFF2 radii, spatial
        filter from centre_pos, and an optional bond-geometry exclusion radius."""
        if centre_pos is not None:
            nearby = self._nearby_atoms(centre_pos, existing_atoms)
            if exclude_radius > 0.0:
                r2_ex = exclude_radius ** 2
                check_atoms = []
                for a in nearby:
                    dx = a['x']-centre_pos[0]
                    dy = a['y']-centre_pos[1]
                    dz = a['z']-centre_pos[2]
                    if dx*dx+dy*dy+dz*dz >= r2_ex:
                        check_atoms.append(a)
            else:
                check_atoms = nearby
        else:
            check_atoms = existing_atoms
        for i, (pos, elem) in enumerate(zip(new_coords, new_elements)):
            eu = elem.upper()[:2]
            gaff_t = new_types[i] if new_types and i < len(new_types) else ''
            r_new = self._vdw_radius(gaff_t, eu)
            for a in check_atoms:
                ae = atom_element(a).upper()
                if eu == 'H' and ae == 'H':
                    continue
                a_gaff = a.get('type','')
                r_ex = self._vdw_radius(a_gaff, ae)
                thr  = self.vdw_scale * (r_new + r_ex)
                dx = pos[0]-a['x']; dy = pos[1]-a['y']; dz = pos[2]-a['z']
                if math.sqrt(dx*dx + dy*dy + dz*dz) < thr:
                    return True
        return False

    # ── Stage 7  [Simulated Annealing over SO(3)] ──
    def find_clash_free_rotation(self, initial_dir, radial_axis,
                                  mon_atoms, cur_inp_name,
                                  n_pos, bond_length, existing_atoms,
                                  exclude_radius=0.0,
                                  mon_bonds=None, flexible=False,
                                  rng=None):
        """Simulated Annealing over SO(3) with Rodrigues 1D warm-start, plus
        deterministic Euler-grid fallback and optional flexible-rotamer fallback.
        Returns (coords, dir, found_clean, clearance).

        `rng` is a numpy Generator supplying the SA stochasticity.  Passing a
        per-monomer seeded Generator (see build_dendrimer) makes placement
        reproducible and order-independent — a prerequisite for trusting any
        parallelism.  If None, a fresh non-deterministic Generator is used
        (legacy behaviour)."""
        if rng is None:
            rng = np.random.default_rng()

        if exclude_radius > 0.0:
            r2_ex = exclude_radius ** 2
            filtered = []
            for a in existing_atoms:
                dx=a['x']-n_pos[0]; dy=a['y']-n_pos[1]; dz=a['z']-n_pos[2]
                if dx*dx+dy*dy+dz*dz >= r2_ex:
                    filtered.append(a)
        else:
            filtered = existing_atoms
        existing_arr = (np.array([[a['x'],a['y'],a['z']] for a in filtered])
                        if filtered else None)
        nearby      = self._nearby_atoms(n_pos, filtered) if filtered else []
        nearby_arr  = (np.array([[a['x'],a['y'],a['z']] for a in nearby])
                       if nearby else np.zeros((0,3)))

        _use_numba_sa = (_accel is not None and _accel.numba_sa_enabled())

        def _clearance(direction):
            coords, _ = position_monomer(n_pos, direction, mon_atoms,
                                         existing_arr, bond_length, cur_inp_name,
                                         filter_ang=self.SPATIAL_FILTER_ANG)
            if len(nearby_arr) == 0:
                return 999.0, coords
            pts   = np.array([list(c) for c in coords])
            if _use_numba_sa:   # T6: Numba clash-scoring kernel (opt-in)
                return _accel.sa_min_clearance(pts, nearby_arr), coords
            diffs = nearby_arr[:, None, :] - pts[None, :, :]
            d     = float(np.sqrt((diffs**2).sum(axis=2)).min())
            return d, coords

        def _rotate_dir_by_quaternion(direction, qw, qx, qy, qz):
            v  = direction
            uq = np.array([qx, qy, qz])
            t  = 2.0 * np.cross(uq, v)
            return v + qw * t + np.cross(uq, t)

        def _random_small_quaternion(sigma):
            axis = rng.standard_normal(3)
            n    = np.linalg.norm(axis)
            if n < 1e-10:
                return 1.0, 0.0, 0.0, 0.0
            axis /= n
            angle = rng.normal(0.0, sigma)
            s     = math.sin(angle / 2.0)
            return math.cos(angle / 2.0), axis[0]*s, axis[1]*s, axis[2]*s

        # Warm-start: Rodrigues 1D scan
        k         = radial_axis / np.linalg.norm(radial_axis)
        n_warm    = max(self.max_retries, min(60, len(mon_atoms) * 2))
        warm_step = 2.0 * math.pi / n_warm
        best_dir     = initial_dir.copy()
        best_min_d, best_coords = _clearance(best_dir)
        found_clean  = best_min_d > self.vdw_scale * 3.0

        for i in range(1, n_warm):
            angle   = i * warm_step
            rot_dir = (initial_dir * math.cos(angle)
                       + np.cross(k, initial_dir) * math.sin(angle)
                       + k * np.dot(k, initial_dir) * (1.0 - math.cos(angle)))
            norm = np.linalg.norm(rot_dir)
            if norm < 1e-10: continue
            rot_dir /= norm
            d, coords = _clearance(rot_dir)
            if d > best_min_d:
                best_min_d = d; best_coords = coords; best_dir = rot_dir
                if d > self.vdw_scale * 3.0:
                    found_clean = True

        if found_clean:
            return best_coords, best_dir, found_clean, best_min_d

        # SA over SO(3)
        SA_STEPS  = max(60, min(200, len(mon_atoms) * 4))
        T         = 1.0
        T_END     = 0.01
        COOLING   = (T_END / T) ** (1.0 / SA_STEPS)
        SIGMA_MAX = math.pi / 4.0
        SIGMA_MIN = math.pi / 36.0

        cur_dir   = best_dir.copy()
        cur_d     = best_min_d

        for step in range(SA_STEPS):
            sigma = SIGMA_MAX * (T - T_END) / (1.0 - T_END) + SIGMA_MIN
            qw, qx, qy, qz = _random_small_quaternion(sigma)
            cand_dir = _rotate_dir_by_quaternion(cur_dir, qw, qx, qy, qz)
            norm = np.linalg.norm(cand_dir)
            if norm < 1e-10:
                T *= COOLING; continue
            cand_dir /= norm

            cand_d, cand_coords = _clearance(cand_dir)
            delta = cand_d - cur_d

            if delta > 0:
                cur_dir = cand_dir; cur_d = cand_d
                if cand_d > best_min_d:
                    best_min_d = cand_d
                    best_coords = cand_coords
                    best_dir    = cand_dir
                    if cand_d > self.vdw_scale * 3.0:
                        found_clean = True
                        break
            else:
                if T > 1e-6 and rng.random() < math.exp(delta / T):
                    cur_dir = cand_dir; cur_d = cand_d

            T *= COOLING

        # Deterministic Euler-grid fallback when SA fails
        if not found_clean:
            GRID = 10
            step_a = 2.0 * math.pi / GRID
            for i in range(GRID):
                a = i * step_a
                rd = (initial_dir * math.cos(a)
                      + np.cross(k, initial_dir) * math.sin(a)
                      + k * np.dot(k, initial_dir) * (1.0 - math.cos(a)))
                n = np.linalg.norm(rd)
                if n < 1e-10: continue
                rd /= n
                for j in range(GRID):
                    b = j * step_a
                    axis2 = np.cross(k, rd)
                    n2 = np.linalg.norm(axis2)
                    if n2 < 1e-10: continue
                    axis2 /= n2
                    rd2 = (rd * math.cos(b)
                           + np.cross(axis2, rd) * math.sin(b)
                           + axis2 * np.dot(axis2, rd) * (1.0 - math.cos(b)))
                    n3 = np.linalg.norm(rd2)
                    if n3 < 1e-10: continue
                    rd2 /= n3
                    d, coords = _clearance(rd2)
                    if d > best_min_d:
                        best_min_d = d
                        best_coords = coords
                        best_dir = rd2
                        if d > self.vdw_scale * 3.0:
                            found_clean = True
                            break
                if found_clean: break

        # Flexible-rotamer fallback: try alternate internal conformers when
        # rigid placement still fails. Sample N_ROT angles per rotatable bond
        # and re-run a quick warm-scan + clearance test on each combination.
        if not found_clean and flexible and mon_bonds is not None and len(mon_bonds) > 0:
            try:
                rotatable = detect_rotatable_bonds(mon_atoms, mon_bonds, cur_inp_name)
            except Exception:
                rotatable = []
            if rotatable:
                N_ROT = 5
                MAX_TORS = 3
                rotatable = rotatable[:MAX_TORS]
                base_coords_arr = np.array([[a['x'], a['y'], a['z']] for a in mon_atoms])

                def _clearance_with_rotamer(direction, rot_coords_arr):
                    """Position rotated monomer along direction and measure clearance."""
                    flex_atoms = []
                    for i, a in enumerate(mon_atoms):
                        na = dict(a)
                        na['x'] = float(rot_coords_arr[i][0])
                        na['y'] = float(rot_coords_arr[i][1])
                        na['z'] = float(rot_coords_arr[i][2])
                        flex_atoms.append(na)
                    coords, _ = position_monomer(n_pos, direction, flex_atoms,
                                                 existing_arr, bond_length, cur_inp_name,
                                                 filter_ang=self.SPATIAL_FILTER_ANG)
                    if len(nearby_arr) == 0:
                        return 999.0, coords
                    pts   = np.array([list(c) for c in coords])
                    diffs = nearby_arr[:, None, :] - pts[None, :, :]
                    d     = float(np.sqrt((diffs**2).sum(axis=2)).min())
                    return d, coords

                step_t = 2.0 * math.pi / N_ROT
                from itertools import product as _prod
                combos = list(_prod(range(N_ROT), repeat=len(rotatable)))
                for combo in combos:
                    rot_coords = base_coords_arr.copy()
                    for ti, ri in enumerate(combo):
                        if ri == 0: continue
                        a_idx, b_idx = rotatable[ti]
                        rot_coords = _rotate_subtree(rot_coords, mon_atoms, mon_bonds,
                                                      a_idx, b_idx, ri * step_t)
                    # Quick warm-scan over grafted-bond rotation for this rotamer
                    for i in range(self.max_retries):
                        a_g = i * (2.0 * math.pi / self.max_retries)
                        rd = (initial_dir * math.cos(a_g)
                              + np.cross(k, initial_dir) * math.sin(a_g)
                              + k * np.dot(k, initial_dir) * (1.0 - math.cos(a_g)))
                        nrm = np.linalg.norm(rd)
                        if nrm < 1e-10: continue
                        rd /= nrm
                        d, coords = _clearance_with_rotamer(rd, rot_coords)
                        if d > best_min_d:
                            best_min_d = d
                            best_coords = coords
                            best_dir = rd
                            if d > self.vdw_scale * 3.0:
                                found_clean = True
                                break
                    if found_clean: break

        if best_coords is None:
            best_coords, _ = position_monomer(n_pos, initial_dir, mon_atoms,
                                               existing_arr, bond_length, cur_inp_name,
                                               filter_ang=self.SPATIAL_FILTER_ANG)
            best_dir = initial_dir

        return best_coords, best_dir, found_clean, best_min_d


# =============================================================================
# SECTION 6.  GROWTH EQUATIONS
# =============================================================================

class GrowthEquations:
    @staticmethod
    def atom_count_pamam(G, n_core=8, f=4, m=2, M_inner=17, M_terminal=19):
        """Exact PAMAM count. Validated: G1=84 G3=516 G6=4548 (Maiti 2004 Table 1)."""
        if G == 0: return n_core
        inner    = f*M_inner*(m**(G-1)-1)//(m-1) if G >= 2 else 0
        terminal = f*(m**(G-1))*M_terminal
        return n_core + inner + terminal

    @staticmethod
    def n_terminal(f, m, G):
        return f*(m**G)

    @staticmethod
    def g_max_de_gennes(P, const=4.0):
        """g_max = 2.88*ln(P)+const  (de Gennes & Hervet 1983).
        const=4.0: P=7 (PAMAM) gives g_max~9.5, consistent with
        published practical limit G10-11 (Maiti 2004 Table 1).
        Old value 2.5 gave g_max=8.0 which was too conservative."""
        return 2.88*math.log(max(P,1)) + const

    @staticmethod
    def rg_predicted(N, rg_ref=3.0, N_ref=84.0):
        """Rg ~ N^(1/3)  (Maiti 2004)."""
        return rg_ref*(N/N_ref)**(1.0/3.0)

    @staticmethod
    def rg_to_radius(rg):
        return math.sqrt(5.0/3.0)*rg

    @staticmethod
    def surface_area_per_terminal(rg, n_terminal, probe=1.4):
        R = GrowthEquations.rg_to_radius(rg)
        SASA = 4.0*math.pi*(R+probe)**2
        return SASA/n_terminal if n_terminal > 0 else 9999.0

# =============================================================================
# SECTION 7.  VAN DER WAALS RADII & STRUCTURAL CHARACTERIZATION
# =============================================================================
# vdW radii used by extended structural characterization (volume, SASA):
#   Primary  — GAFF2 R* values (gaff2.dat MOD4 RE block, AmberTools25)
#   Fallback — Bondi 1964 by element when the GAFF type is unknown

GAFF_VDW_RADII = {
    # hydrogens
    'h1': 1.387, 'h2': 1.287, 'h3': 1.187, 'h4': 1.409, 'h5': 1.359,
    'ha': 1.459, 'hc': 1.487, 'hn': 0.600, 'ho': 0.000, 'hp': 0.600,
    'hs': 0.600, 'hw': 0.000, 'hx': 1.100,
    # carbons (all 1.908 in gaff2)
    'c':  1.908, 'c1': 1.908, 'c2': 1.908, 'c3': 1.908, 'ca': 1.908,
    'cc': 1.908, 'cd': 1.908, 'ce': 1.908, 'cf': 1.908, 'cg': 1.908,
    'ch': 1.908, 'cp': 1.908, 'cq': 1.908, 'cu': 1.908, 'cv': 1.908,
    'cx': 1.908, 'cy': 1.908, 'cz': 1.908,
    # nitrogens (all 1.824 in gaff2)
    'n':  1.824, 'n1': 1.824, 'n2': 1.824, 'n3': 1.824, 'n4': 1.824,
    'na': 1.824, 'nb': 1.824, 'nc': 1.824, 'nd': 1.824, 'ne': 1.824,
    'nf': 1.824, 'nh': 1.824, 'no': 1.824,
    # oxygens
    'o':  1.661, 'oh': 1.721, 'os': 1.6837, 'ow': 1.7683,
    # sulfurs (all 2.000 in gaff2)
    's':  2.000, 's2': 2.000, 's4': 2.000, 's6': 2.000, 'sh': 2.000,
    'ss': 2.000, 'sx': 2.000, 'sy': 2.000,
    # phosphorus (all 2.100 in gaff2)
    'p2': 2.100, 'p3': 2.100, 'p4': 2.100, 'p5': 2.100, 'pb': 2.100,
    'pc': 2.100, 'pd': 2.100, 'pe': 2.100, 'pf': 2.100,
    'px': 2.100, 'py': 2.100,
    # halogens
    'f':  1.75,  'cl': 1.948, 'br': 2.22,  'i':  2.35,
    # silicon
    'Si': 2.220, 'si': 2.220,
}

BONDI_FALLBACK = {
    'H':  1.20, 'C':  1.70, 'N':  1.55, 'O':  1.52, 'F':  1.47,
    'P':  1.80, 'S':  1.80, 'CL': 1.75, 'BR': 1.85, 'I':  1.98,
    'SI': 2.10,
}

def get_vdw_radius(atom):
    """Return vdW radius (Å) for an atom dict.
    Tries GAFF type first, falls back to Bondi by element symbol.
    Final fallback: 1.70 Å (carbon)."""
    t = (atom.get('type') or '').strip()
    if t in GAFF_VDW_RADII:
        return GAFF_VDW_RADII[t]
    # element from atom name (strip digits, take first 1-2 letters)
    nm = (atom.get('name') or 'C').strip()
    elem = ''.join(c for c in nm if c.isalpha())[:2].upper()
    if elem in BONDI_FALLBACK:
        return BONDI_FALLBACK[elem]
    if elem and elem[0] in BONDI_FALLBACK:
        return BONDI_FALLBACK[elem[0]]
    return 1.70

def read_pdb_coords(pdb_path, ref_atoms):
    """Read coordinates from a PDB file (ATOM/HETATM records) and return a
    list of atom dicts matching ref_atoms 1:1, with x/y/z replaced by the
    PDB values. Used to re-characterize structures after minimization while
    preserving GAFF types from the as-built atom list.

    Raises ValueError if atom counts don't match."""
    new_coords = []
    with open(pdb_path, 'r') as f:
        for ln in f:
            rec = ln[0:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except (ValueError, IndexError):
                continue
            new_coords.append((x, y, z))
    if len(new_coords) != len(ref_atoms):
        raise ValueError(
            f"PDB atom count ({len(new_coords)}) != reference ({len(ref_atoms)})")
    out = []
    for ref, (x, y, z) in zip(ref_atoms, new_coords):
        a2 = dict(ref)
        a2['x'] = x; a2['y'] = y; a2['z'] = z
        out.append(a2)
    return out

class StructuralAnalysis:
    @staticmethod
    def radius_of_gyration(atoms):
        coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
        com = coords.mean(axis=0)
        return float(np.sqrt(((coords-com)**2).sum(axis=1).mean()))

    @staticmethod
    def shape_tensor(atoms):
        coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
        com = coords.mean(axis=0); r = coords-com; N = len(atoms)
        return np.array([[np.dot(r[:,i],r[:,j]) for j in range(3)]
                         for i in range(3)])/N

    @staticmethod
    def asphericity(atoms):
        """delta=1-3*I2/I1^2. 0=sphere ->1=rod. G3+ < 0.05 (Maiti 2004).
        [C1] Uses heavy atoms only — Maiti 2004 reference values are
        computed on heavy atoms. Including H dilutes the structural
        signal with the more isotropic H cloud."""
        heavy = [a for a in atoms if not is_hydrogen(a)]
        if len(heavy) < 4:
            heavy = atoms   # fallback for tiny fragments
        G = StructuralAnalysis.shape_tensor(heavy)
        eigs = sorted(np.linalg.eigvalsh(G))
        Ix,Iy,Iz = eigs; I1 = Ix+Iy+Iz; I2 = Ix*Iy+Iy*Iz+Iz*Ix
        if abs(I1) < 1e-12: return 0.0
        return float(1.0 - 3.0*I2/(I1**2))

    @staticmethod
    def full_report(atoms, generation):
        N = len(atoms)
        rg   = StructuralAnalysis.radius_of_gyration(atoms)
        rg_p = GrowthEquations.rg_predicted(N)
        solv = 1.20 if generation <= 6 else 1.10
        asp  = StructuralAnalysis.asphericity(atoms)
        dev  = abs(rg-rg_p)/rg_p*100.0 if rg_p > 0 else 0.0
        return {
            'N':N, 'Rg':round(rg,3), 'Rg_predicted':round(rg_p,3),
            'Rg_solvent':round(rg*solv,3),
            'R_sphere':round(GrowthEquations.rg_to_radius(rg),3),
            'asphericity':round(asp,4), 'deviation_%':round(dev,2),
        }

    # ─── Extended characterization (UDB w12.5+) ─────────────────────────────
    # Reference convention: bounding sphere of R_sphere = sqrt(5/3)*Rg_heavy,
    # following Maiti & Goddard (Macromolecules 2004) and Wu et al. (NIST 2008).

    @staticmethod
    def radius_of_gyration_heavy(atoms):
        """Rg over heavy atoms only — primary literature value (Maiti 2004)."""
        heavy = [a for a in atoms if not is_hydrogen(a)]
        if len(heavy) < 2:
            return StructuralAnalysis.radius_of_gyration(atoms)
        coords = np.array([[a['x'],a['y'],a['z']] for a in heavy])
        com = coords.mean(axis=0)
        return float(np.sqrt(((coords-com)**2).sum(axis=1).mean()))

    @staticmethod
    def vdw_volume_mc(atoms, n_samples=100000, seed=42):
        """Monte Carlo van der Waals volume in Å³ using GAFF radii.
        Samples uniform random points in the atom bounding box, counts those
        inside any vdW sphere, scales by box volume."""
        if len(atoms) < 1:
            return 0.0
        coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
        radii  = np.array([get_vdw_radius(a) for a in atoms])
        rmax   = float(radii.max())
        lo = coords.min(axis=0) - rmax
        hi = coords.max(axis=0) + rmax
        box_vol = float(np.prod(hi - lo))
        rng = np.random.default_rng(seed)
        pts = rng.uniform(lo, hi, size=(n_samples, 3))
        inside = np.zeros(n_samples, dtype=bool)
        # chunked vectorization — avoid building (n_samples × n_atoms) at once
        chunk = 2000
        r2 = radii**2
        for i in range(0, n_samples, chunk):
            d2 = ((pts[i:i+chunk,None,:] - coords[None,:,:])**2).sum(axis=2)
            inside[i:i+chunk] = (d2 < r2).any(axis=1)
        return float(box_vol * inside.sum() / n_samples)

    @staticmethod
    def sasa_shrake_rupley(atoms, probe=1.4, n_points=960):
        """Solvent-accessible surface area in Å² via Shrake & Rupley (1973).
        Generates n_points on each atom's expanded sphere using a Fibonacci
        lattice (same golden-ratio sampling as UDB w12.0 placement), counts
        points not occluded by neighbour atoms.  probe=1.4 Å is the water
        radius standard."""
        if len(atoms) < 1:
            return 0.0
        coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
        radii  = np.array([get_vdw_radius(a) for a in atoms])
        # Fibonacci sphere
        idx = np.arange(n_points)
        phi = np.arccos(1 - 2*(idx+0.5)/n_points)
        th  = np.pi*(1 + 5**0.5)*idx
        unit = np.column_stack([np.sin(phi)*np.cos(th),
                                np.sin(phi)*np.sin(th),
                                np.cos(phi)])
        rmax = float(radii.max())
        total_sasa = 0.0
        for i in range(len(atoms)):
            R_i = radii[i] + probe
            pts = coords[i] + R_i*unit
            dij = np.linalg.norm(coords - coords[i], axis=1)
            nbr = (dij > 0) & (dij < (R_i + rmax + probe))
            if not nbr.any():
                total_sasa += 4*math.pi*R_i*R_i
                continue
            nb_c = coords[nbr]
            nb_r2 = (radii[nbr] + probe)**2
            d2 = ((pts[:,None,:] - nb_c[None,:,:])**2).sum(axis=2)
            accessible = (d2 >= nb_r2).all(axis=1)
            total_sasa += 4*math.pi*R_i*R_i * accessible.sum() / n_points
        return float(total_sasa)

    @staticmethod
    def hydrodynamic_radius(rg):
        """Hard-sphere approximation: Rh = sqrt(5/3) * Rg."""
        return math.sqrt(5.0/3.0) * rg

    @staticmethod
    def extended_report(atoms, generation, mass_amu, n_terminal=None,
                        mc_samples=100000, sasa_points=960):
        """Full structural characterization. Returns dict ready for
        format_extended_table()."""
        N = len(atoms)
        rg_all   = StructuralAnalysis.radius_of_gyration(atoms)
        rg_heavy = StructuralAnalysis.radius_of_gyration_heavy(atoms)
        R_sph    = math.sqrt(5.0/3.0) * rg_heavy
        V_sph    = (4.0/3.0)*math.pi*R_sph**3
        V_vdw    = StructuralAnalysis.vdw_volume_mc(atoms, n_samples=mc_samples)
        f_void   = (1.0 - V_vdw/V_sph) if V_sph > 0 else 0.0
        sasa     = StructuralAnalysis.sasa_shrake_rupley(atoms, n_points=sasa_points)
        asp      = StructuralAnalysis.asphericity(atoms)
        Rh       = StructuralAnalysis.hydrodynamic_radius(rg_heavy)
        # density (g/cm³) from MW (Da) and bounding-sphere volume (Å³)
        # 1 Da/atom × atom / Å³ = 1.66054 g/cm³
        density = (mass_amu / V_sph) * 1.66053906660 if V_sph > 0 else 0.0
        sasa_per_term = (sasa/n_terminal) if (n_terminal and n_terminal>0) else None
        return {
            'N': N, 'M_amu': float(mass_amu), 'gen': int(generation),
            'Rg_all':   round(rg_all,3),
            'Rg_heavy': round(rg_heavy,3),
            'R_sphere': round(R_sph,3),
            'Rh':       round(Rh,3),
            'asphericity': round(asp,4),
            'V_sphere': round(V_sph,1),
            'V_vdW':    round(V_vdw,1),
            'void_fraction_pct': round(f_void*100.0, 2),
            'SASA':     round(sasa,1),
            'SASA_per_terminal': (round(sasa_per_term,2)
                                  if sasa_per_term is not None else None),
            'density_g_per_cm3': round(density,4),
            'n_terminal': n_terminal,
        }

    @staticmethod
    def format_extended_table(rep, label='AS-BUILT'):
        """Format the extended_report dict as a fixed-width text table."""
        lines = []
        bar = "═" * 60
        lines.append("")
        lines.append(bar)
        lines.append(f"  STRUCTURAL CHARACTERIZATION  ({label}, G{rep['gen']})")
        lines.append(bar)
        lines.append(f"  Atom count                   N = {rep['N']}")
        lines.append(f"  Molecular mass               M = {rep['M_amu']:.2f} Da")
        lines.append(f"  Rg (all atoms)              Rg = {rep['Rg_all']:.3f} Å")
        lines.append(f"  Rg (heavy atoms, primary)   Rg = {rep['Rg_heavy']:.3f} Å")
        lines.append(f"  Sphere radius     R = √(5/3)Rg = {rep['R_sphere']:.3f} Å")
        lines.append(f"  Hydrodynamic radius         Rh = {rep['Rh']:.3f} Å")
        lines.append(f"  Asphericity                  δ = {rep['asphericity']:.4f}")
        lines.append(f"  Bounding-sphere volume   V_sph = {rep['V_sphere']:.1f} Å³")
        lines.append(f"  vdW volume (GAFF, MC)    V_vdW = {rep['V_vdW']:.1f} Å³")
        lines.append(f"  Void fraction          f_void = {rep['void_fraction_pct']:.2f} %")
        lines.append(f"  SASA (Shrake–Rupley)         A = {rep['SASA']:.1f} Å²")
        if rep.get('SASA_per_terminal') is not None:
            lines.append(f"  SASA per terminal          a_t = {rep['SASA_per_terminal']:.2f} Å²")
        lines.append(f"  Density  (M / V_sph)         ρ = {rep['density_g_per_cm3']:.4f} g/cm³")
        lines.append(bar)
        return "\n".join(lines)
class StericValidator:
    MIN_AREA = 28.4   # Ang^2/terminal (Maiti 2004 Sec 14.3)

    @staticmethod
    def check(den_type, generation):
        p = PRESETS.get(den_type, PRESETS['Custom'])
        P = p.get('P',4); f = p.get('f',4); m = p.get('branching_factor',2)
        g_max  = GrowthEquations.g_max_de_gennes(P)
        n_term = GrowthEquations.n_terminal(f, m, generation)
        N_est  = GrowthEquations.atom_count_pamam(generation) if den_type=='PAMAM' else max(100*n_term,100)
        rg_est = GrowthEquations.rg_predicted(N_est)
        area   = GrowthEquations.surface_area_per_terminal(rg_est, n_term)
        warns  = []
        if generation >= g_max:
            risk = 'HIGH'
            warns.append(f"G={generation} exceeds de Gennes g_max={g_max:.1f} (P={P}). "
                         "Branching defects and steric strain expected (Maiti 2004).")
        elif generation >= g_max*0.9:
            risk = 'MODERATE'
            warns.append(f"G={generation} near g_max={g_max:.1f}. Strain in outer shells expected.")
        elif generation >= 7: risk = 'LOW'
        else:                 risk = 'NONE'
        if area < StericValidator.MIN_AREA:
            warns.append(f"Surface area/terminal ({area:.1f} A^2) < {StericValidator.MIN_AREA} A^2 "
                         "— terminals will be sterically crowded.")
        if generation >= 9:
            warns.append("G>=9: back-folding of terminals expected (Lescanec & Muthukumar 1990).")
        return {'g_max':round(g_max,2),'allowed':generation<g_max,'strain_risk':risk,
                'n_terminal':n_term,'area_per_terminal':round(area,2),'P':P,'f':f,'warnings':warns}


# =============================================================================
# SECTION 8.  STRUCTURE FILE WRITERS (MOL2 / PDB / XYZ / GRO / MIN SCRIPT)
# =============================================================================

def detect_bond_type(a1,a2,amap,adj):
    t1,t2=a1['type'].lower(),a2['type'].lower()
    if ({t1,t2}&{'c','c2'}) and ({t1,t2}&{'o','o2'}): return '2'
    AR={'ca','cp','cq','cc','cd','ce','cf','nc','nd','ne','nf','pb','pc','pe','pf'}
    if t1 in AR and t2 in AR: return 'ar'
    if {t1,t2}=={'c','na'} or {t1,t2}=={'c','n2'} or {t1,t2}=={'c','nb'}: return 'am'
    return '1'

def write_mol2(filename,atoms,bonds,gen_num,den_type):
    amap={a['idx']:a for a in atoms}
    adj=build_adj(bonds)
    with open(filename,'w') as f:
        f.write("@<TRIPOS>MOLECULE\nUNL\n")
        f.write(f"   {len(atoms)}    {len(bonds)}     1     0     0\n")
        f.write("SMALL\nbcc\n\n\n@<TRIPOS>ATOM\n")
        for i,a in enumerate(atoms):
            f.write(f"{i+1:7d} {a['name']:<8s} {a['x']:10.4f} {a['y']:10.4f} "
                    f"{a['z']:10.4f} {a['type']:<10s} {a['res_idx']:>2s} "
                    f"{a['res_name']:<8s} {a['charge']:>12s}\n")
        f.write("@<TRIPOS>BOND\n")
        for i,b in enumerate(bonds):
            a1=amap.get(b[0]); a2=amap.get(b[1])
            btype=b[2] if b[2]!='1' else (
                detect_bond_type(a1,a2,amap,adj) if a1 and a2 else '1')
            f.write(f"{i+1:6d} {b[0]:5d} {b[1]:5d} {btype}\n")
        f.write("@<TRIPOS>SUBSTRUCTURE\n"
                "     1 UNL         1 TEMP              0 ****  ****    0 ROOT\n")

def write_pdb(filename,atoms,bonds,gen_num,den_type):
    connect={}
    for b in bonds:
        for a1,a2 in [(b[0],b[1]),(b[1],b[0])]:
            connect.setdefault(a1,[]).append(a2)
    with open(filename,'w') as f:
        f.write(f"REMARK  {den_type} Dendrimer Generation {gen_num}\n")
        for i,a in enumerate(atoms):
            elem=''.join(c for c in a['name'] if c.isalpha())[:2].capitalize()
            pname=f" {a['name']:<3s}" if len(a['name'])<4 else f"{a['name']:<4s}"
            f.write(f"HETATM{i+1:5d} {pname} UNL A   1    "
                    f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
                    f"  1.00  0.00          {elem:>2s}  \n")
        for ai in sorted(connect):
            bonded=connect[ai]
            for j in range(0,len(bonded),4):
                chunk=bonded[j:j+4]
                f.write(f"CONECT{ai:5d}"+"".join(f"{b:5d}" for b in chunk)+"\n")
        f.write("END\n")

def write_xyz(filename,atoms,gen_num,den_type):
    with open(filename,'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{den_type} Dendrimer Generation {gen_num}\n")
        for a in atoms:
            elem=''.join(c for c in a['name'] if c.isalpha())[:2].capitalize()
            f.write(f"{elem:<4s} {a['x']:12.6f} {a['y']:12.6f} {a['z']:12.6f}\n")

def write_gro(filename,atoms,bonds,gen_num,den_type):
    """GROMACS .gro format (nm units).
    [D1] Box size computed from atom coordinates + 2 nm padding each side.
    Old hardcoded 10x10x10 nm was wrong for both small (G1) and large (G6+)
    structures."""
    with open(filename,'w') as f:
        f.write(f"{den_type} G{gen_num}\n")
        f.write(f"{len(atoms):5d}\n")
        for i,a in enumerate(atoms):
            elem=''.join(c for c in a['name'] if c.isalpha())[:2].capitalize()
            f.write(f"{1:5d}{'UNL':<5s}{a['name']:>5s}{i+1:5d}"
                    f"{a['x']/10:8.3f}{a['y']/10:8.3f}{a['z']/10:8.3f}\n")
        if atoms:
            xs=[a['x'] for a in atoms]; ys=[a['y'] for a in atoms]; zs=[a['z'] for a in atoms]
            pad=20.0  # 2 nm padding in Angstrom
            bx=(max(xs)-min(xs)+2*pad)/10
            by=(max(ys)-min(ys)+2*pad)/10
            bz=(max(zs)-min(zs)+2*pad)/10
            # minimum 5 nm box
            bx=max(bx,5.0); by=max(by,5.0); bz=max(bz,5.0)
        else:
            bx=by=bz=10.0
        f.write(f"   {bx:.5f}   {by:.5f}   {bz:.5f}\n")

def write_min_script(path,gen_num,atom_count,den_type,
                     ff='gaff',charge_method='bcc',net_charge=0,
                     param_method='full_antechamber',
                     core_file='',monomer_file='',terminal_file='',
                     ncpus=1,
                     conda_path='',conda_env='AmberTools25'):
    """[A3] Generates correct shell script depending on param_method.
    fragment_bcc: antechamber on small fragments → transfer → parmchk2 → tleap → sander
    full_antechamber: antechamber on entire structure → parmchk2 → tleap → sander"""
    prefix=f"dendrimer_{den_type.lower()}_gen{gen_num}"
    maxcyc=max(10000,atom_count*6); ncyc=int(maxcyc*0.8)
    with open(path,'w') as f:
        f.write(
            f"#!/bin/bash\n"
            f"# {'='*56}\n"
            f"# {den_type} G{gen_num} — AMBER Minimization Script\n"
            f"# Atoms        : {atom_count:,}\n"
            f"# Force field  : {ff}\n"
            f"# Charges      : {charge_method}  (net charge = {net_charge})\n"
            f"# Param method : {param_method}\n"
            f"# Generated by Universal Dendrimer Builder v1.0\n"
            f"# {'='*56}\n\nset -e\n\n"
            f"# AmberTools activation\n"
            f"source {conda_path or os.path.expanduser('~/miniconda3')}/etc/profile.d/conda.sh\n"
            f"conda activate {conda_env or 'AmberTools25'}\n\n"
        )
        if param_method == 'fragment_bcc':
            # Step 1 — antechamber on each fragment
            frags = [(core_file,'core_typed.mol2','core'),
                     (monomer_file,'monomer_typed.mol2','monomer')]
            if terminal_file:
                frags.append((terminal_file,'terminal_typed.mol2','terminal'))
            for frag_in,frag_out,label in frags:
                if not frag_in: continue
                bn=os.path.basename(frag_in)
                f.write(
                    f"# Fragment: {label}\n"
                    f"antechamber \\\n"
                    f"  -i  {bn} -fi mol2 \\\n"
                    f"  -o  {frag_out} -fo mol2 \\\n"
                    f"  -c  {charge_method} \\\n"
                    f"  -at {ff} \\\n"
                    f"  -nc {net_charge} \\\n"
                    f"  -j  5\n\n"
                )
            f.write(
                f"# Transfer fragment types/charges to full structure\n"
                f"# (use run_fragment_bcc() in the live pipeline for this step)\n"
                f"# The full typed mol2 is: {prefix}_typed.mol2\n\n"
            )
        else:
            # Full antechamber on the entire built structure
            # to avoid antechamber reading and writing the same file.
            f.write(
                f"antechamber \\\n"
                f"  -i  {prefix}_typed.mol2 -fi mol2 \\\n"
                f"  -o  {prefix}_typed_params.mol2 -fo mol2 \\\n"
                f"  -c  {charge_method} \\\n"
                f"  -at {ff} \\\n"
                f"  -nc {net_charge} \\\n"
                f"  -j  5\n\n"
            )
        f.write(
            f"parmchk2 \\\n"
            f"  -i {prefix}_typed.mol2 -f mol2 \\\n"
            f"  -o {prefix}.frcmod\n\n"
            f"tleap -f - <<'TLEAP_EOF'\n"
            f"source leaprc.{ff}\n"
            f"loadamberparams {prefix}.frcmod\n"
            f"mol = loadmol2 {prefix}_typed.mol2\n"
            f"check mol\n"
            f"saveamberparm mol {prefix}.prmtop {prefix}.inpcrd\n"
            f"quit\nTLEAP_EOF\n\n"
            f"cat > min.in <<'SANDER_EOF'\n"
            f"Energy minimization {den_type} G{gen_num}\n"
            f" &cntrl\n  imin=1, maxcyc={maxcyc}, ncyc={ncyc},\n"
            f"  ntb=0, igb=0, cut=999.0,\n /\nSANDER_EOF\n\n"
        )
        sander_exe = f"mpirun -np {ncpus} sander.MPI" if ncpus > 1 else "sander"
        f.write(
            f"{sander_exe} -O -i min.in -o min_g{gen_num}.out "
            f"-p {prefix}.prmtop -c {prefix}.inpcrd -r {prefix}_min.rst\n\n"
            f"ambpdb -p {prefix}.prmtop -c {prefix}_min.rst > {prefix}_min.pdb\n\n"
            f"# Minimized GAFF mol2 (atom types + charges preserved from prmtop)\n"
            f"cat > cpptraj_min_g{gen_num}.in <<'CPPTRAJ_EOF'\n"
            f"parm {prefix}.prmtop\n"
            f"trajin {prefix}_min.rst\n"
            f"trajout {prefix}_min_gaff.mol2 mol2\n"
            f"run\n"
            f"CPPTRAJ_EOF\n"
            f"cpptraj -i cpptraj_min_g{gen_num}.in\n\n"
            f"echo 'Minimization complete'\n"
        )
    os.chmod(path,os.stat(path).st_mode|_stat.S_IEXEC|_stat.S_IXGRP|_stat.S_IXOTH)


# =============================================================================
# SECTION 9.  ANTECHAMBER / AM1-BCC FRAGMENT PARAMETERIZATION
# =============================================================================

def run_fragment_bcc(core_file, monomer_file, terminal_file,
                     built_mol2, output_dir, ff_name, log_cb,
                     charge_method='bcc', net_charge=0,
                     cancel_flag=None,
                     conda_path='', conda_env='AmberTools25'):
    """Fragment BCC parameterisation: antechamber on each fragment, then fingerprint-
    based type/charge transfer to the full dendrimer. Returns path to typed mol2,
    or None on failure."""
    import os, subprocess

    # Map legacy/invalid GUI values to actual antechamber -c flags.
    _cm_map = {'am1bcc':'bcc', 'gasteiger':'gas', 'resp':'bcc'}
    if charge_method in _cm_map:
        if charge_method == 'resp':
            log_cb(f"  [WARN] RESP charges require Gaussian/GAMESS input. Falling back to bcc (AM1-BCC).", 'warn')
        charge_method = _cm_map[charge_method]

    def run(cmd, step, cwd=None):
        if cancel_flag and cancel_flag[0]: return False
        log_cb(f"\n  [FragBCC:{step}]  $ {cmd}", 'head')
        try:
            _cmd = shlex.split(cmd) if isinstance(cmd, str) else cmd
            _conda_path = conda_path or os.path.expanduser('~/miniconda3')
            _conda_env  = conda_env or 'AmberTools25'
            _env = os.environ.copy()
            _env['PATH'] = (f"{_conda_path}/envs/{_conda_env}/bin:"
                            f"{_conda_path}/bin:{_env.get('PATH','')}")
            r = subprocess.run(_cmd, shell=False, capture_output=True,
                               text=True, cwd=cwd or output_dir, timeout=28800,
                               env=_env)
            if r.returncode == 0:
                log_cb(f"  [FragBCC:{step}] done", 'ok'); return True
            log_cb(f"  [FragBCC:{step}] FAILED (rc={r.returncode})", 'err')
            if r.stderr: log_cb(r.stderr[:400], 'err')
            return False
        except subprocess.TimeoutExpired:
            log_cb(f"  [FragBCC:{step}] TIMEOUT", 'err'); return False
        except Exception as e:
            log_cb(f"  [FragBCC:{step}] ERROR: {e}", 'err'); return False

    log_cb(f"\n{'='*56}", 'head')
    log_cb(f"  FRAGMENT BCC PARAMETERIZATION"
           f"  (-c {charge_method}  -nc {net_charge})", 'head')
    log_cb(f"{'='*56}", 'head')

    # ── Step 1: antechamber on each fragment ──────────────────
    fragments = [
        (core_file,     'core_typed.mol2',     'core'),
        (monomer_file,  'monomer_typed.mol2',  'monomer'),
    ]
    if terminal_file and os.path.exists(terminal_file):
        fragments.append((terminal_file, 'terminal_typed.mol2', 'terminal'))

    # T8: content-hash cache + concurrent antechamber. The same monomer/core is
    # re-parameterised on every generation and every rebuild; AM1-BCC via sqm is
    # the pipeline's most expensive CPU step, so cache the typed .mol2 by
    # sha256(fragment_contents + charge_method + net_charge + ff) and reuse it.
    # Cache misses run concurrently, each in its own scratch dir so antechamber's
    # fixed-name temp files (ANTECHAMBER_*.AC) cannot collide.
    import hashlib as _hl, shutil as _sh, tempfile as _tf
    from concurrent.futures import ThreadPoolExecutor as _TPE

    cache_dir = os.environ.get(
        "UDB_FRAGBCC_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "udb_fragbcc"))
    os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(frag_in):
        h = _hl.sha256()
        with open(frag_in, "rb") as _f:
            h.update(_f.read())
        h.update(f"|{charge_method}|{net_charge}|{ff_name}".encode())
        return h.hexdigest()

    def _param_one(frag):
        frag_in, frag_out, label = frag
        abs_in = os.path.abspath(frag_in)
        dest = os.path.join(output_dir, frag_out)
        cached = os.path.join(cache_dir, _cache_key(abs_in) + ".mol2")
        if os.path.exists(cached):
            _sh.copyfile(cached, dest)
            log_cb(f"  [FragBCC:{label}] cache hit — antechamber skipped", "ok")
            return True
        tmp = _tf.mkdtemp(prefix=f"fragbcc_{label}_")
        try:
            ok = run(f"antechamber -i {abs_in} -fi mol2 -o {frag_out} -fo mol2 "
                     f"-c {charge_method} -at {ff_name} -nc {net_charge} -j 5",
                     label, cwd=tmp)
            produced = os.path.join(tmp, frag_out)
            if not ok or not os.path.exists(produced):
                return False
            _sh.copyfile(produced, dest)
            _sh.copyfile(produced, cached)   # populate cache for next time
            return True
        finally:
            _sh.rmtree(tmp, ignore_errors=True)

    with _TPE(max_workers=len(fragments)) as _ex:
        if not all(_ex.map(_param_one, fragments)):
            return None

    # ── Step 2: build lookup tables from typed fragments ──────
    def _fingerprint(atom, mol2_atoms, mol2_bonds):
        """[w11.3] Environment fingerprint: element + sorted bonded elements + GAFF type.
        Example: amide C → 'C[C,N,O]_c'   methylene C → 'C[C,C,H,H]_c3'
        Prevents name-collision misassignment when core and monomer share
        atom names in different chemical environments."""
        adj = build_adj(mol2_bonds)
        amap = {a['idx']: a for a in mol2_atoms}
        elem = ''.join(c for c in atom['name'] if c.isalpha())[:2].upper()
        bonded = []
        for nb_idx in adj.get(atom['idx'], []):
            nb = amap.get(nb_idx)
            if nb:
                bonded.append(''.join(c for c in nb['name'] if c.isalpha())[:2].upper())
        bonded.sort()
        return f"{elem}[{','.join(bonded)}]_{atom['type']}"

    def load_typed_with_fp(path, source_mol2_path):
        """Load fingerprint -> (type, charge) and name -> (type, charge) tables."""
        fp_table   = {}   # fingerprint key — primary lookup
        name_table = {}   # name key — fallback
        source_atoms, source_bonds = parse_mol2(source_mol2_path)
        in_atom = False
        full = os.path.join(output_dir, path)
        if not os.path.exists(full): return fp_table, name_table
        typed_atoms = []
        with open(full) as f:
            for line in f:
                line = line.rstrip()
                if '@<TRIPOS>ATOM' in line: in_atom = True; continue
                elif '@<TRIPOS>' in line:   in_atom = False; continue
                if in_atom and line.strip():
                    p = line.split()
                    if len(p) >= 9:
                        typed_atoms.append({'idx':int(p[0]),'name':p[1],
                                            'type':p[5],'charge':p[8]})
        # match typed atoms back to source by index order
        for ta in typed_atoms:
            src = next((a for a in source_atoms if a['idx'] == ta['idx']), None)
            if src:
                fp = _fingerprint(src, source_atoms, source_bonds)
                fp_table[fp]         = (ta['type'], ta['charge'])
                name_table[ta['name']] = (ta['type'], ta['charge'])
        return fp_table, name_table

    core_fp,     core_name     = load_typed_with_fp('core_typed.mol2',     core_file)
    monomer_fp,  monomer_name  = load_typed_with_fp('monomer_typed.mol2',  monomer_file)
    terminal_fp, terminal_name = ({}, {})
    if os.path.exists(os.path.join(output_dir, 'terminal_typed.mol2')) and terminal_file:
        terminal_fp, terminal_name = load_typed_with_fp('terminal_typed.mol2', terminal_file)

    log_cb(f"  Fragment fingerprint tables: "
           f"core={len(core_fp)} monomer={len(monomer_fp)} "
           f"terminal={len(terminal_fp)} entries", 'ok')

    # ── Step 3: read full built mol2 and transfer types/charges ─
    full_atoms, full_bonds = parse_mol2(built_mol2)
    if not full_atoms:
        log_cb("  ERROR: built mol2 has no atoms", 'err'); return None

    transferred = 0; fp_hits = 0; name_hits = 0; unmatched = []

    for a in full_atoms:
        name = a['name']
        fp   = _fingerprint(a, full_atoms, full_bonds)
        matched = False
        # Priority: terminal > monomer > core; fingerprint > name fallback
        for fp_tbl, name_tbl, label in [
            (terminal_fp, terminal_name, 'terminal'),
            (monomer_fp,  monomer_name,  'monomer'),
            (core_fp,     core_name,     'core'),
        ]:
            if fp in fp_tbl:
                a['type'], a['charge'] = fp_tbl[fp]
                transferred += 1; fp_hits += 1; matched = True; break
            elif name in name_tbl:
                a['type'], a['charge'] = name_tbl[name]
                transferred += 1; name_hits += 1; matched = True; break
        if not matched:
            unmatched.append(name)

    log_cb(f"  Transferred: {transferred}/{len(full_atoms)} atoms  "
           f"(fingerprint: {fp_hits}  name-fallback: {name_hits})",
           'ok' if not unmatched else 'warn')
    if unmatched:
        log_cb(f"  Unmatched atom names: {list(set(unmatched))[:10]}", 'warn')

    try:
        total_q  = sum(float(a['charge']) for a in full_atoms)
        residual = -total_q
        heavy    = [a for a in full_atoms if not is_hydrogen(a)]
        if heavy:
            per_heavy = residual / len(heavy)
            for a in heavy:
                a['charge'] = f"{float(a['charge']) + per_heavy:.6f}"
        else:
            per_atom = residual / len(full_atoms)
            for a in full_atoms:
                a['charge'] = f"{float(a['charge']) + per_atom:.6f}"
        final_q = sum(float(a['charge']) for a in full_atoms)
        log_cb(f"  Charge residual: {total_q:+.4f} e  →  distributed to "
               f"{len(heavy)} heavy atoms  (final net: {final_q:+.4f} e)", 'ok')
    except Exception as e:
        log_cb(f"  Charge correction warning: {e}", 'warn')

    # ── Step 5: write typed mol2 ──────────────────────────────
    # types from input fragments). Guard against _typed_typed double suffix.
    base_name = os.path.basename(built_mol2)
    if base_name.endswith('_typed.mol2'):
        out_name = base_name          # already correctly named — overwrite in place
    else:
        out_name = base_name.replace('.mol2', '_typed.mol2')
    out_path = os.path.join(output_dir, out_name)
    write_mol2(out_path, full_atoms, full_bonds,
               0, os.path.basename(built_mol2).split('_gen')[0])
    log_cb(f"  Fragment BCC typed mol2: {out_name}", 'ok')
    return out_path


# =============================================================================
# SECTION 10.  STRUCTURE VALIDATION & CHARMM/PSF EXPORT
# =============================================================================

def validate_structure(atoms,bonds,terminal_names,clash_threshold=1.0):
    if isinstance(terminal_names,str): terminal_names=[terminal_names]
    adj=build_adj(bonds)
    amap={a['idx']:a for a in atoms}
    results=[]

    results.append(("Total atoms > 0", len(atoms)>0, str(len(atoms)),"> 0"))
    # Old ±2 tolerance masked silently skipped branches.
    bond_ok = len(bonds) == (len(atoms)-1)
    results.append(("Bond count (tree topology)",bond_ok,str(len(bonds)),str(len(atoms)-1)))
    if atoms:
        start = atoms[0]['idx']
        visited = {start}
        queue  = [start]
        while queue:
            cur = queue.pop()
            for nb in adj.get(cur,[]):
                if nb not in visited:
                    visited.add(nb); queue.append(nb)
        n_disconnected = len(atoms) - len(visited)
        connected_ok   = n_disconnected == 0
        results.append(("Single connected component (BFS)",connected_ok,
                        f"{len(visited)}/{len(atoms)}",str(len(atoms))))
    else:
        results.append(("Single connected component (BFS)",False,"0","0"))

    term_atoms=[a for a in atoms if a['name'] in terminal_names]
    terminal_N=[]
    branch_N=[]
    for a in term_atoms:
        nbs=adj.get(a['idx'],[])
        h_nbs=[nb for nb in nbs if amap.get(nb) and is_hydrogen(amap[nb])]
        if h_nbs: terminal_N.append((a['idx'],len(h_nbs)))
        else: branch_N.append(a['idx'])
    results.append((f"Peripheral Groups (surface) detected",len(terminal_N)>0,str(len(terminal_N)),"> 0"))
    results.append((f"Branching Junctions (inner layers) detected",len(branch_N)>0,str(len(branch_N)),"> 0"))
    # passes if every terminal has at least 1 H neighbour.
    # Old check assumed h==2 (NH2, PAMAM-only) — wrong for all other types.
    complete = sum(1 for _,h in terminal_N if h >= 1)
    incomplete = len(terminal_N) - complete
    results.append(("Terminal H neighbors detected (>= 1 per site)", incomplete==0,
                    f"{complete}/{len(terminal_N)}", str(len(terminal_N))))

    # Bond lengths
    lengths=[]
    for b in bonds:
        a1=amap.get(b[0]); a2=amap.get(b[1])
        if a1 and a2:
            d=math.sqrt((a1['x']-a2['x'])**2+(a1['y']-a2['y'])**2+(a1['z']-a2['z'])**2)
            lengths.append(d)
    severe=sum(1 for l in lengths if l<0.5)
    clashes=sum(1 for l in lengths if l<clash_threshold)
    results.append(("No Steric Overlap  (d > 0.5 A)",severe==0,str(severe),"0"))
    results.append((f"No close contacts (< {clash_threshold:.1f} Å)",clashes==0,str(clashes),"0"))

    mw=atom_mw(atoms)
    results.append(("MW > 0 Da",mw>0,f"{mw:,.0f} Da","> 0"))

    coords=np.array([[a['x'],a['y'],a['z']] for a in atoms])
    com=coords.mean(axis=0)
    rg=float(np.sqrt(((coords-com)**2).sum(axis=1).mean()))
    results.append(("Radius of gyration > 0 Å",rg>0.0,f"{rg:.2f} Å","> 0 Å"))

    passed=sum(1 for _,ok,_,_ in results if ok)
    return results,passed,len(results),mw,rg,len(terminal_N),len(branch_N)

GAFF_TO_CHARMM = {
    # Carbon sp3
    'c3': ('CT3',  12.011), 'c2': ('CE2',  12.011),
    'c' : ('C',    12.011), 'ca': ('CA',   12.011),
    'cc': ('CG2R61',12.011),'cd': ('CG2R61',12.011),
    'ce': ('CE1',  12.011), 'cf': ('CE1',  12.011),
    'cp': ('CG2R61',12.011),'cq': ('CG2R61',12.011),
    'cx': ('CT3',  12.011), 'cy': ('CT3',  12.011),
    # Nitrogen
    'n' : ('NH1',  14.007), 'n1': ('NG1T1',14.007),
    'n2': ('NG2D1',14.007), 'n3': ('NH3',  14.007),
    'n4': ('NP',   14.007), 'na': ('NH1',  14.007),
    'nb': ('NR2',  14.007), 'nh': ('NH2',  14.007),
    'no': ('NG2D1',14.007),
    # Oxygen
    'o' : ('O',    15.999), 'oh': ('OH1',  15.999),
    'os': ('OS',   15.999), 'o2': ('OG2P1',15.999),
    # Hydrogen
    'hc': ('HA3',   1.008), 'ha': ('HA2',   1.008),
    'hn': ('H',     1.008), 'ho': ('H',     1.008),
    'hp': ('H',     1.008), 'hs': ('H',     1.008),
    'h1': ('HA1',   1.008), 'h2': ('HA2',   1.008),
    'h3': ('HA3',   1.008), 'h4': ('HA3',   1.008),
    'h5': ('HA3',   1.008),
    # Silicon (Carbosilane)
    'si': ('SIG1',  28.086),
    # Phosphorus
    'p2': ('PG2',   30.974), 'p3': ('PG1',  30.974),
    'p4': ('PG2',   30.974), 'p5': ('PG1',  30.974),
    'pb': ('PG2',   30.974), 'pc': ('PG2',  30.974),
    'pe': ('PG2',   30.974), 'pf': ('PG2',  30.974),
    # Sulphur
    's' : ('S',     32.060), 's2': ('SM',   32.060),
    's4': ('SM',    32.060), 's6': ('SM',   32.060),
    'sh': ('SH1',   32.060), 'ss': ('SM',   32.060),
    # Halogens
    'f' : ('F',     18.998), 'cl': ('CLGA1',35.450),
    'br': ('BRGA',  79.904), 'i' : ('I',   126.900),
}
CHARMM_DEFAULT = ('C', 12.011)   # fallback for unmapped types

def write_psf(filename, atoms, bonds, gen_num, den_type):
    """
    [w11.3] Write a CHARMM PSF file directly from UDB atom/bond lists.
    No external tools needed. Derives angles and dihedrals from connectivity.
    GAFF types are mapped to approximate CHARMM types via GAFF_TO_CHARMM.
    Bonds are 3-tuples (idx1, idx2, bond_type) — bond_type is ignored here.
    """
    from collections import defaultdict

    # normalise bonds — accept both (b1,b2) and (b1,b2,btype) tuples
    def _b12(b):
        return b[0], b[1]

    adj = defaultdict(list)
    for b in bonds:
        b1, b2 = _b12(b)
        adj[b1].append(b2)
        adj[b2].append(b1)

    # ── atom type and mass lookup ────────────────────────────
    def charmm_type_mass(gaff_type):
        return GAFF_TO_CHARMM.get(gaff_type.lower(), CHARMM_DEFAULT)

    # ── derive angles from adjacency ────────────────────────
    angles = []
    seen_ang = set()
    for b_atom in atoms:
        bi = b_atom['idx']
        nbs = adj[bi]
        for i in range(len(nbs)):
            for j in range(i+1, len(nbs)):
                a, c = nbs[i], nbs[j]
                key = (min(a,c), bi, max(a,c))
                if key not in seen_ang:
                    seen_ang.add(key)
                    angles.append((a, bi, c))

    # ── derive dihedrals from adjacent bonds ─────────────────
    dihedrals = []
    seen_dih = set()
    for b in bonds:
        b1, b2 = _b12(b)
        for a in adj[b1]:
            if a == b2: continue
            for d in adj[b2]:
                if d == b1 or d == a: continue
                key = tuple(sorted([a, b1, b2, d]))
                if key not in seen_dih:
                    seen_dih.add(key)
                    dihedrals.append((a, b1, b2, d))

    SP2_TYPES = {
        'c','c2','ca','cc','cd','ce','cf','cp','cq','cu','cv',
        'n','na','nb','nc','nd','ne','nf','n2',
        'p4','pb','pc','pe','pf',
    }
    impropers = []
    seen_imp = set()
    atype_map = {a['idx']: a.get('type','').lower() for a in atoms}
    for b_atom in atoms:
        bi = b_atom['idx']
        if atype_map.get(bi,'') not in SP2_TYPES:
            continue
        heavy_nbs = [nb for nb in adj[bi]
                     if not atype_map.get(nb,'').startswith('h')]
        if len(heavy_nbs) < 3:
            continue
        # canonical improper: centre atom second (I, J, K, L) → (nb0, bi, nb1, nb2)
        a0, a1, a2 = heavy_nbs[0], heavy_nbs[1], heavy_nbs[2]
        key = (bi, min(a0,a1,a2), max(a0,a1,a2))
        if key not in seen_imp:
            seen_imp.add(key)
            impropers.append((a0, bi, a1, a2))

    N   = len(atoms)
    NB  = len(bonds)
    NAN = len(angles)
    NDI = len(dihedrals)
    NIM = len(impropers)

    amap = {a['idx']: i+1 for i, a in enumerate(atoms)}  # idx → 1-based serial

    with open(filename, 'w') as f:
        # PSF header
        f.write('PSF\n\n')
        f.write(f'       4 !NTITLE\n')
        f.write(f' REMARKS CHARMM PSF file\n')
        f.write(f' REMARKS {den_type} Dendrimer Generation {gen_num}\n')
        f.write(f' REMARKS Generated by Universal Dendrimer Builder v1.0\n')
        f.write(f' REMARKS GAFF types mapped to approximate CHARMM types\n\n')

        # ATOM section
        f.write(f'{N:>8d} !NATOM\n')
        for i, a in enumerate(atoms):
            serial   = i + 1
            segname  = 'DEND'
            resid    = 1
            resname  = 'UNL'
            aname    = a['name'][:4]
            ctype, mass = charmm_type_mass(a['type'])
            charge   = float(a['charge'])
            f.write(f'{serial:>8d} {segname:<4s} {resid:<4d} {resname:<4s} '
                    f'{aname:<4s} {ctype:<6s} {charge:>10.6f} '
                    f'{mass:>13.4f}           0\n')

        # BOND section
        f.write(f'\n{NB:>8d} !NBOND: bonds\n')
        bond_items = []
        for b in bonds:
            b1, b2 = _b12(b)
            s1 = amap.get(b1, 0)
            s2 = amap.get(b2, 0)
            if s1 and s2:
                bond_items += [s1, s2]
        for k in range(0, len(bond_items), 8):
            f.write(''.join(f'{v:>8d}' for v in bond_items[k:k+8]) + '\n')

        # ANGLE section
        f.write(f'\n{NAN:>8d} !NTHETA: angles\n')
        ang_items = []
        for a, b, c in angles:
            sa, sb, sc = amap.get(a,0), amap.get(b,0), amap.get(c,0)
            if sa and sb and sc:
                ang_items += [sa, sb, sc]
        for k in range(0, len(ang_items), 9):
            f.write(''.join(f'{v:>8d}' for v in ang_items[k:k+9]) + '\n')

        # DIHEDRAL section
        f.write(f'\n{NDI:>8d} !NPHI: dihedrals\n')
        dih_items = []
        for a, b, c, d in dihedrals:
            sa,sb,sc,sd = amap.get(a,0),amap.get(b,0),amap.get(c,0),amap.get(d,0)
            if sa and sb and sc and sd:
                dih_items += [sa, sb, sc, sd]
        for k in range(0, len(dih_items), 8):
            f.write(''.join(f'{v:>8d}' for v in dih_items[k:k+8]) + '\n')

        f.write(f'\n{NIM:>8d} !NIMPHI: impropers\n')
        imp_items = []
        for a, b, c, d in impropers:
            sa,sb,sc,sd = amap.get(a,0),amap.get(b,0),amap.get(c,0),amap.get(d,0)
            if sa and sb and sc and sd:
                imp_items += [sa, sb, sc, sd]
        for k in range(0, len(imp_items), 8):
            f.write(''.join(f'{v:>8d}' for v in imp_items[k:k+8]) + '\n')
        f.write('\n')

        # DONOR / ACCEPTOR sections (empty)
        f.write(f'       0 !NDON: donors\n\n')
        f.write(f'       0 !NACC: acceptors\n\n')

        # NNB (non-bonded exclusions) — empty placeholder
        f.write(f'       0 !NNB\n\n')
        for k in range(0, N, 8):
            chunk = atoms[k:k+8]
            f.write(''.join('       0' for _ in chunk) + '\n')
        f.write('\n')

        # NGRP (group section) — one group for whole molecule
        f.write(f'       1       0 !NGRP\n')
        f.write(f'       0       0       0\n\n')


# =============================================================================
# SECTION 11.  NAMD / AMBER SIMULATION PIPELINES
# =============================================================================

def run_namd_pipeline(output_dir, den_type, gen_num, log_cb,
                      steps, cancel_flag=None):
    """Execute the NAMD 3.x AMBER/GAFF2 pipeline (tLeap → NAMD stages).
    Generates all input files then runs them sequentially."""
    pre      = f"dendrimer_{den_type.lower()}_gen{gen_num}"
    namd_bin = steps.get('namd_binary', NAMD_DEFAULT_BIN)
    ncores   = steps.get('namd_cores', 8)
    namd_cmd = f"{namd_bin} +p{ncores}"

    def run(cmd, step_name):
        if cancel_flag and cancel_flag[0]: return False
        log_cb(f"\n  [NAMD:{step_name}]  $ {cmd}", 'head')
        try:
            _cmd = shlex.split(cmd) if isinstance(cmd, str) else cmd
            r = subprocess.run(_cmd, shell=False, capture_output=True,
                               text=True, cwd=output_dir, timeout=86400)
            if r.returncode == 0:
                log_cb(f"  [NAMD:{step_name}] done", 'ok'); return True
            log_cb(f"  [NAMD:{step_name}] FAILED (rc={r.returncode})", 'err')
            for ln in (r.stdout + r.stderr).splitlines()[-30:]:
                log_cb(f"    {ln}", 'err')
            return False
        except subprocess.TimeoutExpired:
            log_cb(f"  [NAMD:{step_name}] TIMEOUT (>24hr)", 'err'); return False
        except Exception as e:
            log_cb(f"  [NAMD:{step_name}] ERROR: {e}", 'err'); return False

    log_cb(f"\n{'='*56}", 'head')
    log_cb(f"  NAMD 3.x PIPELINE  G{gen_num}  ({den_type})", 'head')
    log_cb(f"{'='*56}", 'head')

    # Step 0: tLeap solvation (only if prmtop not yet present)
    prmtop = os.path.join(output_dir, f"{pre}_solvated.prmtop")
    inpcrd = os.path.join(output_dir, f"{pre}_solvated.inpcrd")
    if steps.get('namd_solvate') and not os.path.exists(prmtop):
        tleap_f = os.path.join(output_dir, "00_solvate.tleap")
        if not os.path.exists(tleap_f):
            log_cb(f"  [NAMD] 00_solvate.tleap not found — generate files first", 'err')
            return False
        ok = run("tleap -f 00_solvate.tleap", "tLeap")
        if not ok: return False
    elif os.path.exists(prmtop):
        log_cb(f"  [NAMD] prmtop found — skipping tLeap", 'ok')

    if not os.path.exists(inpcrd):
        log_cb(f"  [NAMD] ERROR: {pre}_solvated.inpcrd not found after tLeap", 'err')
        return False

    try:
        with open(inpcrd) as _f: _last = _f.readlines()[-1].split()
        if len(_last) >= 3:
            bx, by, bz = float(_last[0]), float(_last[1]), float(_last[2])
            # centre: tLeap places box at (bx/2, by/2, bz/2)
            cx, cy, cz = bx / 2, by / 2, bz / 2
            min_conf = os.path.join(output_dir, '01_minimize.conf')
            if os.path.exists(min_conf):
                with open(min_conf) as _f: _mc = _f.read()
                import re as _re
                _mc = _re.sub(r'cellBasisVector1\s+[\d.]+.*',
                              f'cellBasisVector1    {bx:.4f}  0.0    0.0', _mc)
                _mc = _re.sub(r'cellBasisVector2\s+.*',
                              f'cellBasisVector2    0.0    {by:.4f}  0.0', _mc)
                _mc = _re.sub(r'cellBasisVector3\s+.*',
                              f'cellBasisVector3    0.0    0.0    {bz:.4f}', _mc)
                _mc = _re.sub(r'cellOrigin\s+.*',
                              f'cellOrigin          {cx:.4f}  {cy:.4f}  {cz:.4f}', _mc)
                with open(min_conf, 'w') as _f: _f.write(_mc)
                log_cb(f"  [NAMD] cellBasisVector auto-patched: {bx:.2f} x {by:.2f} x {bz:.2f} A", 'ok')
                log_cb(f"  [NAMD] cellOrigin: ({cx:.2f}, {cy:.2f}, {cz:.2f})", 'ok')
    except Exception as _e:
        log_cb(f"  [NAMD] WARN: could not auto-patch cellBasisVector ({_e}) — update manually", 'warn')

    stage_map = [
        ('namd_minimize', '01_minimize'),
        ('namd_heat',     '02_heat'),
        ('namd_equil',    '03_equil'),
        ('namd_prod',     '04_production'),
    ]
    for step_key, tag in stage_map:
        if cancel_flag and cancel_flag[0]: return False
        if not steps.get(step_key): continue
        conf = f"{tag}.conf"
        if not os.path.exists(os.path.join(output_dir, conf)):
            log_cb(f"  [NAMD] {conf} not found — skipping", 'warn'); continue
        ok = run(f"{namd_cmd} {conf} > logs/{tag}.log", tag)
        if not ok: return False

    log_cb(f"  NAMD pipeline complete: G{gen_num}", 'ok')
    return True

def run_full_pipeline(mol2_path,output_dir,den_type,gen_num,
                      net_charge,log_cb,steps,cancel_flag=None):
    atoms,_=parse_mol2(mol2_path)
    natoms=len(atoms)
    maxcyc=max(10000,natoms*6); ncyc=int(maxcyc*0.8)
    prefix=f"dendrimer_{den_type.lower()}_gen{gen_num}"
    charge_method=steps.get('charge_method','bcc')
    # Map legacy/invalid GUI values to the actual antechamber -c flags.
    # 'resp' is rejected — antechamber cannot generate RESP without QM input.
    _cm_map = {'am1bcc':'bcc', 'gasteiger':'gas', 'resp':'bcc'}
    if charge_method in _cm_map:
        if charge_method == 'resp':
            log_cb(f"  [WARN] RESP charges require Gaussian/GAMESS input. Falling back to bcc (AM1-BCC).", 'warn')
        charge_method = _cm_map[charge_method]
    ff_name=steps.get('ff','gaff')

    typed_mol2=f"{prefix}_typed.mol2"; frcmod=f"{prefix}.frcmod"
    prmtop=f"{prefix}.prmtop";         inpcrd=f"{prefix}.inpcrd"
    min_rst=f"{prefix}_min.rst";       min_pdb=f"{prefix}_min.pdb"
    solv_prmtop=f"{prefix}_solv.prmtop"; solv_inpcrd=f"{prefix}_solv.inpcrd"
    leap_f=f"leap_{prefix}.in";        leap_solv=f"leap_{prefix}_solv.in"
    bn=os.path.basename(mol2_path)
    # antechamber reads bn and must write to a different name to avoid
    # reading and writing the same file — use _typed_params.mol2.
    if bn.endswith('_typed.mol2'):
        typed_mol2_ac = f"{prefix}_typed_params.mol2"
    else:
        typed_mol2_ac = typed_mol2

    # Wall-clock cap for each AMBER step. parmchk2 on large (G6+) dendrimers
    # can run for hours, so allow up to 8 hr. Single source of truth: the
    # TIMEOUT message below derives its hour count from this value.
    STEP_TIMEOUT = 28800  # 8 hr

    def run(cmd,step_name):
        if cancel_flag and cancel_flag[0]: return False
        log_cb(f"\n  [{step_name}]  $ {cmd}",'head')
        try:
            _cmd = shlex.split(cmd) if isinstance(cmd, str) else cmd
            _conda_path = steps.get('conda_path','') or os.path.expanduser('~/miniconda3')
            _conda_env  = steps.get('conda_env','AmberTools25')
            _env = os.environ.copy()
            _env['PATH'] = (f"{_conda_path}/envs/{_conda_env}/bin:"
                            f"{_conda_path}/bin:{_env.get('PATH','')}")
            r=subprocess.run(_cmd,shell=False,capture_output=True,
                             text=True,cwd=output_dir,timeout=STEP_TIMEOUT,env=_env)
            if r.returncode==0:
                log_cb(f"  [{step_name}] done",'ok'); return True
            log_cb(f"  [{step_name}] FAILED (rc={r.returncode})",'err')
            if r.stderr: log_cb(r.stderr[:500],'err')
            return False
        except subprocess.TimeoutExpired:
            log_cb(f"  [{step_name}] TIMEOUT (>{STEP_TIMEOUT//3600}hr)",'err'); return False
        except Exception as e:
            log_cb(f"  [{step_name}] ERROR: {e}",'err'); return False

    log_cb(f"\n{'='*56}",'head')
    log_cb(f"  AMBER PIPELINE  G{gen_num}  ({natoms} atoms)",'head')
    log_cb(f"{'='*56}",'head')

    param_method = steps.get('param_method', 'full_antechamber')

    ANTECHAMBER_WARN_LIMIT  = 150   # warn above this
    ANTECHAMBER_BLOCK_LIMIT = 400   # hard block above this
    if param_method == 'full_antechamber' and steps.get('antechamber'):
        if natoms > ANTECHAMBER_BLOCK_LIMIT:
            log_cb(f"\n  [gate] BLOCKED: {natoms} atoms > {ANTECHAMBER_BLOCK_LIMIT} limit "
                   f"for full antechamber (sqm MAXBOND will fail).",'warn')
            log_cb(f"  [gate] Auto-switching to Fragment BCC for this generation.",'warn')
            param_method = 'fragment_bcc'
        elif natoms > ANTECHAMBER_WARN_LIMIT:
            log_cb(f"\n  [gate] WARNING: {natoms} atoms — full antechamber may be slow "
                   f"or hit sqm limits. Consider Fragment BCC for G3+.",'warn')
    # Full antechamber only runs when the checkbox is ticked AND
    # param_method == 'full_antechamber'.
    if param_method == 'fragment_bcc':
        # ── Fragment BCC: antechamber on small fragments only ──
        core_file_     = steps.get('core_file', '')
        monomer_file_  = steps.get('monomer_file', '')
        terminal_file_ = steps.get('terminal_file', '')
        typed_mol2_path = run_fragment_bcc(
            core_file_, monomer_file_, terminal_file_,
            mol2_path, output_dir, ff_name, log_cb,
            charge_method=charge_method,
            net_charge=net_charge,
            cancel_flag=cancel_flag,
            conda_path=steps.get('conda_path', ''),
            conda_env=steps.get('conda_env', 'AmberTools25'))
        if typed_mol2_path is None:
            log_cb("  Fragment BCC failed — aborting pipeline", 'err')
            return False
        typed_mol2 = os.path.basename(typed_mol2_path)
    elif steps.get('antechamber'):
        # ── Full antechamber: run on entire built dendrimer ──
        ok=run(f"antechamber -i {bn} -fi mol2 "
               f"-o {typed_mol2_ac} -fo mol2 "
               f"-c {charge_method} -at {ff_name} -nc {net_charge} -j 5",
               "antechamber")
        if not ok: return False
        typed_mol2 = typed_mol2_ac
    else:
        typed_mol2 = bn

    if steps.get('parmchk2'):
        ok=run(f"parmchk2 -i {typed_mol2} -f mol2 -o {frcmod}","parmchk2")
        if not ok: return False

    if steps.get('tleap'):
        with open(os.path.join(output_dir,leap_f),'w') as lf:
            lf.write(f"source leaprc.{ff_name}\n"
                     f"loadamberparams {frcmod}\n"
                     f"mol = loadmol2 {typed_mol2}\ncheck mol\n"
                     f"saveamberparm mol {prmtop} {inpcrd}\nquit\n")
        ok=run(f"tleap -f {leap_f}","tLeap")
        if not ok: return False

    if steps.get('minimize'):
        with open(os.path.join(output_dir,"min.in"),'w') as mf:
            mf.write(f"Energy minimization {den_type} G{gen_num}\n"
                     f" &cntrl\n  imin=1, maxcyc={maxcyc}, ncyc={ncyc},\n"
                     f"  ntb=0, igb=0, cut=999.0,\n /\n")
        ncpus = steps.get('ncpus', 1)
        sander_cmd = (f"mpirun -np {ncpus} sander.MPI"
                      if ncpus > 1 else "sander")
        ok=run(f"{sander_cmd} -O -i min.in -o min_g{gen_num}.out "
               f"-p {prmtop} -c {inpcrd} -r {min_rst}","sander (vacuum min)")
        if not ok: return False

    if steps.get('ambpdb'):
        src=min_rst if os.path.exists(os.path.join(output_dir,min_rst)) else inpcrd
        # ambpdb has no '-o' flag — it writes PDB to stdout. The run() helper
        # uses shell=False and shlex.split, so a '>' in the command would be
        # passed as a literal argv element (and ambpdb would reject it).
        # Handle this one tool inline: capture stdout, write to the PDB file.
        log_cb(f"\n  [ambpdb]  $ ambpdb -p {prmtop} -c {src} > {min_pdb}",'head')
        if cancel_flag and cancel_flag[0]:
            pass
        else:
            try:
                _conda_path = steps.get('conda_path','') or os.path.expanduser('~/miniconda3')
                _conda_env  = steps.get('conda_env','AmberTools25')
                _env = os.environ.copy()
                _env['PATH'] = (f"{_conda_path}/envs/{_conda_env}/bin:"
                                f"{_conda_path}/bin:{_env.get('PATH','')}")
                _cmd = ['ambpdb', '-p', prmtop, '-c', src]
                with open(os.path.join(output_dir, min_pdb), 'w') as pdb_fh:
                    r = subprocess.run(_cmd, shell=False, stdout=pdb_fh,
                                       stderr=subprocess.PIPE, text=True,
                                       cwd=output_dir, timeout=300, env=_env)
                if r.returncode == 0:
                    log_cb(f"  [ambpdb] done", 'ok')
                else:
                    log_cb(f"  [ambpdb] FAILED (rc={r.returncode})", 'err')
                    if r.stderr: log_cb(r.stderr[:500], 'err')
            except subprocess.TimeoutExpired:
                log_cb(f"  [ambpdb] TIMEOUT", 'err')
            except Exception as e:
                log_cb(f"  [ambpdb] ERROR: {e}", 'err')

    # ── cpptraj: minimized AMBER restart → GAFF mol2 ──────────────────────
    # Produces {prefix}_min_gaff.mol2 — GAFF types + charges from prmtop,
    # minimized coordinates from rst. Used by intergenerational minimization
    # and available as a standalone output regardless of intergen setting.
    if steps.get('minimize') and steps.get('intergen_save_gaff', False):
        _rst_path = os.path.join(output_dir, min_rst)
        if os.path.exists(_rst_path):
            _gaff_mol2  = f"{prefix}_min_gaff.mol2"
            _cpptraj_in = f"cpptraj_min_g{gen_num}.in"
            _script = (f"parm {prmtop}\n"
                       f"trajin {min_rst}\n"
                       f"trajout {_gaff_mol2} mol2\n"
                       f"run\n")
            with open(os.path.join(output_dir, _cpptraj_in), 'w') as _cf:
                _cf.write(_script)
            ok = run(f"cpptraj -i {_cpptraj_in}", "cpptraj (rst→mol2)")
            if ok:
                log_cb(f"  [cpptraj] GAFF mol2 written: {_gaff_mol2}", 'ok')

    if steps.get('solvate'):
        box_size=steps.get('box_size',12.0)
        with open(os.path.join(output_dir,leap_solv),'w') as lf:
            lf.write(f"source leaprc.{ff_name}\nsource leaprc.water.tip3p\n"
                     f"loadamberparams {frcmod}\nmol = loadmol2 {typed_mol2}\n"
                     f"solvatebox mol TIP3PBOX {box_size}\n"
                     f"saveamberparm mol {solv_prmtop} {solv_inpcrd}\nquit\n")
        ok=run(f"tleap -f {leap_solv}","tLeap (solvate)")
        if not ok: return False

    if steps.get('add_ions'):
        leap_ions=f"leap_{prefix}_ions.in"
        with open(os.path.join(output_dir,leap_ions),'w') as lf:
            lf.write(f"source leaprc.{ff_name}\nsource leaprc.water.tip3p\n"
                     f"loadamberparams {frcmod}\nmol = loadmol2 {typed_mol2}\n"
                     f"solvatebox mol TIP3PBOX {steps.get('box_size',12.0)}\n"
                     f"addions mol Na+ 0\naddions mol Cl- 0\n"
                     f"saveamberparm mol {prefix}_ions.prmtop {prefix}_ions.inpcrd\nquit\n")
        ok=run(f"tleap -f {leap_ions}","tLeap (ions)")
        if not ok: return False

    if steps.get('md'):
        prod_ns=steps.get('prod_ns',5.0)
        prod_nstep=int(prod_ns*1e6/2)
        ion_prmtop=f"{prefix}_ions.prmtop"; ion_inpcrd=f"{prefix}_ions.inpcrd"
        pref=ion_prmtop if os.path.exists(os.path.join(output_dir,ion_prmtop)) else solv_prmtop
        md_stages=[
            ("md1_solv_min",ion_inpcrd if pref==ion_prmtop else solv_inpcrd,
             f"{prefix}_md1.rst","md1.in",
             "Solvent minimization\n &cntrl\n  imin=1, maxcyc=5000, ncyc=2500,\n"
             f"  ntb=1, ntr=1, restraint_wt=50.0,\n  restraintmask='!:WAT,Na+,Cl-',\n  cut=10.0,\n /\n"),
            ("md2_all_min",f"{prefix}_md1.rst",f"{prefix}_md2.rst","md2.in",
             "Full minimization\n &cntrl\n  imin=1, maxcyc=10000, ncyc=5000,\n  ntb=1, cut=10.0,\n /\n"),
            ("md3_heat",f"{prefix}_md2.rst",f"{prefix}_md3.rst","md3.in",
             "Heating 0→300K\n &cntrl\n  imin=0, irest=0, ntx=1,\n  nstlim=50000, dt=0.002,\n"
             f"  temp0=300.0, tempi=0.0,\n  ntt=3, gamma_ln=1.0,\n  ntb=1, ntp=0, cut=10.0,\n  ntpr=1000, ntwx=1000,\n /\n"),
            ("md4_equil",f"{prefix}_md3.rst",f"{prefix}_md4.rst","md4.in",
             "NPT Equilibration\n &cntrl\n  imin=0, irest=1, ntx=5,\n  nstlim=500000, dt=0.002,\n"
             f"  temp0=300.0, ntt=3, gamma_ln=1.0,\n  ntp=1, barostat=2, pres0=1.0, taup=2.0,\n"
             f"  ntb=2, cut=10.0,\n  ntpr=5000, ntwx=5000,\n /\n"),
            ("md5_prod",f"{prefix}_md4.rst",f"{prefix}_md5.rst","md5.in",
             f"Production MD {prod_ns} ns\n &cntrl\n  imin=0, irest=1, ntx=5,\n"
             f"  nstlim={prod_nstep}, dt=0.002,\n  temp0=300.0, ntt=3, gamma_ln=1.0,\n"
             f"  ntp=1, barostat=2, pres0=1.0, taup=2.0,\n  ntb=2, cut=10.0,\n"
             f"  ntpr=10000, ntwx=10000, ntwr=50000,\n /\n"),
        ]
        for (stage,in_r,out_r,inf,content) in md_stages:
            if cancel_flag and cancel_flag[0]: return False
            with open(os.path.join(output_dir,inf),'w') as mf: mf.write(content)
            trj=f"-x {prefix}_{stage}.nc " if 'prod' not in stage else f"-x {prefix}_production.nc "
            ncpus = steps.get('ncpus', 1)
            sander_cmd = (f"mpirun -np {ncpus} sander.MPI"
                          if ncpus > 1 else "sander")
            ok=run(f"{sander_cmd} -O -i {inf} -o {prefix}_{stage}.out -p {pref} -c {in_r} -r {out_r} {trj}",
                   f"MD {stage}")
            if not ok: return False

    log_cb(f"\n  Pipeline complete: G{gen_num}",'ok')

    if steps.get('obabel_minimize'):
        ff_ob    = steps.get('obabel_ff', 'MMFF94')
        steps_ob = steps.get('obabel_steps', 2500)
        ob_out   = f"{prefix}_obmin.mol2"
        log_cb(f"\n  [OpenBabel] obabel {bn} -O {ob_out} --minimize "
               f"--ff {ff_ob} --steps {steps_ob}", 'head')
        ok = run(f"obabel {bn} -O {ob_out} --minimize "
                 f"--ff {ff_ob} --steps {steps_ob}", "obabel")
        if ok:
            log_cb(f"  [OpenBabel] optimized mol2: {ob_out}", 'ok')

    any_namd_pipe = (steps.get('namd_solvate') or steps.get('namd_minimize') or
                     steps.get('namd_heat') or steps.get('namd_equil') or
                     steps.get('namd_prod'))
    if any_namd_pipe:
        log_cb(f"\n  [NAMD] Generating input files then running pipeline", 'head')
        run_namd_pipeline(output_dir, den_type, gen_num, log_cb,
                          steps, cancel_flag)

    return True

# =============================================================================
# SECTION 12.  INTERGENERATIONAL MINIMIZATION HELPERS
# =============================================================================

def run_intergen_amber_min(prefix, output_dir, all_atoms, log):
    """Read AMBER-minimized coords from {prefix}_min.pdb into all_atoms.
    Returns updated atom list preserving all GAFF types/charges, or None."""
    min_pdb = os.path.join(output_dir, f"{prefix}_min.pdb")
    if not os.path.exists(min_pdb):
        log(f"  [IntergenMin/AMBER] {prefix}_min.pdb not found — "
            f"ensure sander + ambpdb steps are enabled in the AMBER tab.", 'warn')
        return None
    try:
        updated = read_pdb_coords(min_pdb, all_atoms)
        log(f"  [IntergenMin/AMBER] Loaded {len(updated)} minimized atoms "
            f"from {prefix}_min.pdb", 'ok')
        return updated
    except ValueError as e:
        log(f"  [IntergenMin/AMBER] Atom-count mismatch: {e}", 'warn')
        return None
    except Exception as e:
        log(f"  [IntergenMin/AMBER] Error reading PDB: {e}", 'warn')
        return None


def run_intergen_namd_min(prefix, output_dir, all_atoms, log, steps, gen,
                          cancel_flag=None):
    """Run a NAMD energy minimization for intergenerational coordinate update.

    namd_mode='vacuum'  : no PBC, uses {prefix}.prmtop + {prefix}.inpcrd.
                          Requires tLeap step in the AMBER tab.
    namd_mode='solvated': PBC water box, uses {prefix}_solvated.prmtop.
                          Requires namd_solvate step in the NAMD tab.
                          Solvent is stripped by cpptraj before coord read-back.

    Returns updated atom list or None on any failure.
    """
    namd_bin   = steps.get('namd_binary', NAMD_DEFAULT_BIN)
    ncores     = steps.get('namd_cores', 8)
    namd_cmd   = f"{namd_bin} +p{ncores}"
    namd_mode  = steps.get('intergen_namd_mode', 'vacuum')
    conda_path = steps.get('conda_path', '') or os.path.expanduser('~/miniconda3')
    conda_env  = steps.get('conda_env', 'AmberTools25')
    tag        = f"intergen_{namd_mode}_g{gen}"

    def run(cmd, step_name, timeout=28800):
        if cancel_flag and cancel_flag[0]: return False
        log(f"\n  [IntergenMin/NAMD:{step_name}]  $ {cmd}", 'head')
        try:
            _cmd = shlex.split(cmd) if isinstance(cmd, str) else cmd
            _env = os.environ.copy()
            _env['PATH'] = (f"{conda_path}/envs/{conda_env}/bin:"
                            f"{conda_path}/bin:{_env.get('PATH','')}")
            r = subprocess.run(_cmd, shell=False, capture_output=True,
                               text=True, cwd=output_dir, timeout=timeout)
            if r.returncode == 0:
                log(f"  [IntergenMin/NAMD:{step_name}] done", 'ok'); return True
            log(f"  [IntergenMin/NAMD:{step_name}] FAILED (rc={r.returncode})", 'err')
            for ln in (r.stdout + r.stderr).splitlines()[-20:]:
                log(f"    {ln}", 'err')
            return False
        except subprocess.TimeoutExpired:
            log(f"  [IntergenMin/NAMD:{step_name}] TIMEOUT (>{timeout}s)", 'err')
            return False
        except Exception as e:
            log(f"  [IntergenMin/NAMD:{step_name}] ERROR: {e}", 'err')
            return False

    if namd_mode == 'vacuum':
        # ── Vacuum (no PBC) ──────────────────────────────────────────────────
        prmtop = f"{prefix}.prmtop"
        inpcrd = f"{prefix}.inpcrd"
        if not os.path.exists(os.path.join(output_dir, prmtop)):
            log(f"  [IntergenMin/NAMD] {prmtop} not found — "
                f"tLeap must run in the AMBER tab first.", 'warn')
            return None
        conf_name = f"{tag}.conf"
        coor_file = f"{tag}.restart.coor"
        mol2_out  = f"{prefix}_namd_vac_intergen_g{gen}.mol2"
        conf_text = (
            f"# NAMD vacuum energy minimisation — G{gen} intergenerational (v1.0)\n"
            f"amber               on\n"
            f"parmfile            {prmtop}\n"
            f"ambercoor           {inpcrd}\n"
            f"outputName          {tag}\n"
            f"binaryoutput        yes\n"
            f"binaryrestart       yes\n"
            f"outputEnergies      100\n"
            f"restartfreq         100\n"
            f"exclude             scaled1-4\n"
            f"oneFourScaling      1.0\n"
            f"cutoff              20.0\n"
            f"switching           on\n"
            f"switchdist          18.0\n"
            f"pairlistdist        22.0\n"
            f"stepspercycle       20\n"
            f"minimize            2000\n"
        )
        with open(os.path.join(output_dir, conf_name), 'w') as fh:
            fh.write(conf_text)
        if not run(f"{namd_cmd} {conf_name}", "vacuum_min"): return None
        if not os.path.exists(os.path.join(output_dir, coor_file)):
            log(f"  [IntergenMin/NAMD] {coor_file} missing after minimize.", 'err')
            return None
        cpptraj_script = (f"parm {prmtop}\n"
                          f"trajin {coor_file} namdbin\n"
                          f"trajout {mol2_out} mol2\n"
                          f"run\n")

    else:
        # ── Solvated + cpptraj strip ─────────────────────────────────────────
        solv_prmtop = f"{prefix}_solvated.prmtop"
        solv_inpcrd = f"{prefix}_solvated.inpcrd"
        if not os.path.exists(os.path.join(output_dir, solv_prmtop)):
            log(f"  [IntergenMin/NAMD] {solv_prmtop} not found — "
                f"namd_solvate must run first.", 'warn')
            return None
        # Read box vectors from inpcrd last line
        try:
            with open(os.path.join(output_dir, solv_inpcrd)) as fh:
                bx = float(fh.readlines()[-1].split()[0])
        except Exception:
            bx = 60.0
        pme_g     = 1 << (int(bx) - 1).bit_length()
        conf_name = f"{tag}.conf"
        coor_file = f"{tag}.restart.coor"
        mol2_out  = f"{prefix}_namd_solv_intergen_g{gen}.mol2"
        conf_text = (
            f"# NAMD solvated energy minimisation — G{gen} intergenerational (v1.0)\n"
            f"amber               on\n"
            f"parmfile            {solv_prmtop}\n"
            f"ambercoor           {solv_inpcrd}\n"
            f"outputName          {tag}\n"
            f"binaryoutput        yes\n"
            f"binaryrestart       yes\n"
            f"outputEnergies      100\n"
            f"restartfreq         100\n"
            f"exclude             scaled1-4\n"
            f"oneFourScaling      1.0\n"
            f"cutoff              12.0\n"
            f"switching           on\n"
            f"switchdist          10.0\n"
            f"pairlistdist        14.0\n"
            f"stepspercycle       20\n"
            f"PME                 yes\n"
            f"PMEGridSizeX        {pme_g}\n"
            f"PMEGridSizeY        {pme_g}\n"
            f"PMEGridSizeZ        {pme_g}\n"
            f"wrapAll             on\n"
            f"wrapWater           on\n"
            f"rigidBonds          none\n"
            f"cellBasisVector1    {bx:.4f}  0.0       0.0\n"
            f"cellBasisVector2    0.0       {bx:.4f}  0.0\n"
            f"cellBasisVector3    0.0       0.0       {bx:.4f}\n"
            f"cellOrigin          {bx/2:.4f}  {bx/2:.4f}  {bx/2:.4f}\n"
            f"temperature         0\n"
            f"minimize            2000\n"
        )
        with open(os.path.join(output_dir, conf_name), 'w') as fh:
            fh.write(conf_text)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        if not run(f"{namd_cmd} {conf_name} > logs/{tag}.log", "solvated_min"):
            return None
        if not os.path.exists(os.path.join(output_dir, coor_file)):
            log(f"  [IntergenMin/NAMD] {coor_file} missing after minimize.", 'err')
            return None
        cpptraj_script = (f"parm {solv_prmtop}\n"
                          f"trajin {coor_file} namdbin\n"
                          f"strip :WAT,Na+,Cl-\n"
                          f"trajout {mol2_out} mol2\n"
                          f"run\n")

    # ── cpptraj: binary restart → mol2 ──────────────────────────────────────
    cpptraj_in = f"cpptraj_{tag}.in"
    with open(os.path.join(output_dir, cpptraj_in), 'w') as fh:
        fh.write(cpptraj_script)
    if not run(f"cpptraj -i {cpptraj_in}", "cpptraj (coor→mol2)"): return None

    mol2_path = os.path.join(output_dir, mol2_out)
    if not os.path.exists(mol2_path):
        log(f"  [IntergenMin/NAMD] {mol2_out} not found after cpptraj.", 'err')
        return None
    try:
        namd_atoms, _ = parse_mol2(mol2_path)
        if len(namd_atoms) != len(all_atoms):
            raise ValueError(
                f"mol2 count ({len(namd_atoms)}) != reference ({len(all_atoms)})")
        updated = []
        for ref, na in zip(all_atoms, namd_atoms):
            a2 = dict(ref)
            a2['x'] = na['x']; a2['y'] = na['y']; a2['z'] = na['z']
            updated.append(a2)
        log(f"  [IntergenMin/NAMD] Loaded {len(updated)} NAMD-minimized atoms "
            f"({namd_mode} mode)", 'ok')
        return updated
    except ValueError as e:
        log(f"  [IntergenMin/NAMD] Atom-count mismatch: {e}", 'warn')
        return None
    except Exception as e:
        log(f"  [IntergenMin/NAMD] Error reading mol2: {e}", 'warn')
        return None


# =============================================================================
# SECTION 13.  DENDRIMER BUILD ENGINE
# =============================================================================

def build_dendrimer(config,log,progress,gen_progress,
                    run_min=False,net_charge=0,pipeline_steps=None,
                    cancel_flag=None):
    core_file         = config['core_file']
    monomer_file      = config['monomer_file']
    terminal_file     = config.get('terminal_file','')
    core_binding_idxs = config['core_binding_idxs']
    mon_input_name    = config['mon_input_name']
    mon_input_h_name  = config['mon_input_h_name']
    mon_terminal_names= config['mon_terminal_names']
    mon_terminal_idxs = config.get('mon_terminal_idxs', [])   
    labile_mode       = config.get('labile_mode',  'single')
    labile_names      = config.get('labile_names', [mon_input_h_name] if mon_input_h_name else [])
    labile_idxs_raw   = config.get('labile_idxs',  [])
    term_input_name   = config.get('term_input_name', mon_input_name)
    term_input_h_name = config.get('term_input_h_name', mon_input_h_name)
    # w12.2 atom-selection additions. All default to None/auto → the pre-w12.2
    # code paths run unchanged (auto-detect the leaving H).
    # #1 accept a list of terminal-leaving atoms (w12.3); fall back to the legacy
    #    single-name key so old configs keep working.
    mon_terminal_leave_names = config.get('mon_terminal_leave_names')  # #1 (multi)
    if not mon_terminal_leave_names and config.get('mon_terminal_leave_name'):
        mon_terminal_leave_names = [config.get('mon_terminal_leave_name')]
    tgroup_bind_name        = config.get('tgroup_bind_name')          # #2
    tgroup_leave_name       = config.get('tgroup_leave_name')         # #3
    # #2/#3 take precedence over the Terminal Unit fields when provided
    if tgroup_bind_name:
        term_input_name = tgroup_bind_name
    if tgroup_leave_name:
        term_input_h_name = tgroup_leave_name
    bond_length       = float(config['bond_length'])
    bond_angle        = float(config['bond_angle'])
    branching_factor  = int(config['branching_factor'])
    gens_to_build     = sorted(config['generations_to_build'])
    output_dir        = config['output_dir']
    den_type          = config['den_type']
    out_formats       = config.get('out_formats', ['mol2','pdb'])
    max_gen           = max(gens_to_build)

    os.makedirs(output_dir,exist_ok=True)
    # initial value 0.75 is overridden each generation in the loop.
    cone_engine = ConePlacementEngine(vdw_scale=0.75, max_retries=36)
    steric = StericValidator.check(den_type, max_gen)
    for w in steric['warnings']:
        log(f"  [Steric] {w}",'warn')

    log(f"Loading core:    {os.path.basename(core_file)}")
    core_atoms,core_bonds=parse_mol2(core_file)
    core_bonds=deduplicate_bonds(core_bonds)
    log(f"  -> {len(core_atoms)} atoms,  {len(core_bonds)} bonds")

    log(f"Loading monomer: {os.path.basename(monomer_file)}")
    mon_atoms_orig,mon_bonds_orig=parse_mol2(monomer_file)
    mon_bonds_orig=deduplicate_bonds(mon_bonds_orig)
    log(f"  -> {len(mon_atoms_orig)} atoms,  {len(mon_bonds_orig)} bonds")

    # #1 resolve the manual terminal-leaving atom(s) to their element(s) (matched
    # per terminal site during growth). Unknown names are dropped; if none resolve,
    # fall back to auto-H. Matching by element means each terminal site sheds the
    # same species regardless of how many distinct names were listed.
    _mon_term_leave_elems = set()
    if mon_terminal_leave_names:
        _resolved = []
        for _nm in mon_terminal_leave_names:
            _mtl = next((a for a in mon_atoms_orig if a['name']==_nm), None)
            _el  = atom_element(_mtl) if _mtl else None
            if _el is None:
                log(f"  [w12.3] terminal leaving atom '{_nm}' not found in "
                    f"monomer — skipped",'warn')
            else:
                _mon_term_leave_elems.add(_el)
                _resolved.append(f"{_nm} ({_el})")
        if not _mon_term_leave_elems:
            log(f"  [w12.3] no terminal leaving atoms resolved — using auto-H",'warn')
            mon_terminal_leave_names = None
        else:
            log(f"  [w12.3] terminal leaving atoms: {', '.join(_resolved)} "
                f"removed at each terminal site")

    use_terminal=bool(terminal_file and os.path.exists(terminal_file))
    if use_terminal:
        log(f"Loading terminal: {os.path.basename(terminal_file)}")
        term_atoms_orig,term_bonds_orig=parse_mol2(terminal_file)
        term_bonds_orig=deduplicate_bonds(term_bonds_orig)
        log(f"  -> {len(term_atoms_orig)} atoms,  {len(term_bonds_orig)} bonds")

    # Strip H from core binding atoms
    core_h_remove=set()
    for idx in core_binding_idxs:
        for h in get_h_neighbors(core_atoms,core_bonds,idx):
            core_h_remove.add(h['idx'])
    # e.g. Cl on Triazine core — without this, Cl stays as orphaned atom
    if labile_names:
        core_adj_tmp = build_adj(core_bonds)
        core_amap_tmp = {a['idx']:a for a in core_atoms}
        for b_idx in core_binding_idxs:
            for nb_idx in core_adj_tmp.get(b_idx, []):
                nb = core_amap_tmp.get(nb_idx)
                if nb and not is_hydrogen(nb) and nb['name'] in labile_names:
                    core_h_remove.add(nb_idx)
                    log(f"  Core non-H labile: {nb['name']} (idx {nb_idx}) removed",'ok')
    log(f"\nRemoving {len(core_h_remove)} atom(s) from {len(core_binding_idxs)} core sites...")
    curr_atoms,curr_bonds,c_map=remove_atoms(core_atoms,core_bonds,core_h_remove)
    binding_remapped=[c_map[i] for i in core_binding_idxs]
    log(f"  Core ready: {len(curr_atoms)} atoms")
    # always points radially outward — no grandparent tracking needed
    core_centre = np.array([[a['x'],a['y'],a['z']] for a in curr_atoms]).mean(axis=0)
    cone_engine.core_centre = core_centre

    # Single mode: one atom (H, Cl, Br, O, F, or any element).
    # Group  mode: multiple atoms removed together (OH, NH2, OCH3 …).
    # Strategy:
    #   1. Build labile_remove_set from labile_idxs_raw (GUI-supplied indices).
    #   2. Fallback: if no indices given, find atoms by name match in labile_names.
    #   3. Fallback: if still empty and single mode, auto-detect first H on focal atom.
    # All atoms in labile_remove_set are stripped at once via remove_atoms().
    # mon_strip_map is used to remap terminal idxs into use_atoms space.

    focal_a = next((a for a in mon_atoms_orig if a['name']==mon_input_name), None)
    if focal_a is None:
        raise ValueError(f"Cannot find focal atom '{mon_input_name}' in monomer")

    labile_remove_set = set()

    # Priority 1: use GUI-supplied indices directly
    if labile_idxs_raw:
        for li in labile_idxs_raw:
            if any(a['idx']==li for a in mon_atoms_orig):
                labile_remove_set.add(li)
            else:
                log(f"  [w11.5] Labile idx {li} not found in monomer — skipping",'warn')

    # Priority 2: match by name
    if not labile_remove_set and labile_names:
        for ln in labile_names:
            for a in mon_atoms_orig:
                if a['name']==ln:
                    labile_remove_set.add(a['idx']); break

    # Priority 3: single mode auto-detect first H on focal atom
    if not labile_remove_set and labile_mode == 'single':
        hn = get_h_neighbors(mon_atoms_orig, mon_bonds_orig, focal_a['idx'])
        if hn:
            labile_remove_set.add(hn[0]['idx'])
            log(f"  [w11.5] Auto-detected labile H: {hn[0]['name']} (idx {hn[0]['idx']})")

    if not labile_remove_set:
        raise ValueError(
            f"Cannot find labile atom(s) to remove on focal atom '{mon_input_name}'.\n"
            f"  labile_names={labile_names}  labile_idxs_raw={labile_idxs_raw}")

    log(f"  Labile mode: {labile_mode.upper()}  "
        f"removing {len(labile_remove_set)} atom(s): "
        f"{sorted(labile_remove_set)}")

    mon_atoms_clean, mon_bonds_clean, mon_strip_map = remove_atoms(
        mon_atoms_orig, mon_bonds_orig, labile_remove_set)

    # match the indices in mon_atoms_clean / use_atoms during placement.
    mon_terminal_idxs_clean = [mon_strip_map[t] for t in mon_terminal_idxs
                                if t in mon_strip_map and mon_strip_map[t] is not None]
    log(f"  Monomer ready: {len(mon_atoms_clean)} atoms "
        f"({len(labile_remove_set)} labile atom(s) removed)")
    log(f"  Terminal idxs (remapped): {mon_terminal_idxs_clean}")

    prev_gen_boundary=0; summary=[]
    next_gen_sites=[]   

    for gen in range(1,max_gen+1):
        if cancel_flag and cancel_flag[0]:
            log(f"\n  Build CANCELLED at Generation {gen}",'warn')
            return summary

        log(f"\n{'='*52}",'head')
        log(f"  GENERATION {gen}  |  {den_type}  [v1.0 Fibonacci+SA+CollMin]",'head')
        log(f"{'='*52}",'head')

        # Build site list with grandparent info
        if gen==1:
            core_adj = build_adj(curr_bonds)
            core_amap = {a['idx']:a for a in curr_atoms}
            g1_sites = []
            for idx in binding_remapped:
                # find the heavy-atom core neighbour of this binding site
                core_nbs = [nb for nb in core_adj.get(idx,[])
                            if nb in core_amap and not is_hydrogen(core_amap[nb])]
                ref_pos = None
                if core_nbs:
                    nb = core_amap[core_nbs[0]]
                    ref_pos = np.array([nb['x'],nb['y'],nb['z']])
                g1_sites.append({'n_idx':idx,'h_list':[],'ref_pos':ref_pos})
            sites = g1_sites
            log(f"  G1: {len(sites)} core sites x {branching_factor} = "
                f"{len(sites)*branching_factor} monomers")
        else:
            # No name scan — only the exact global indices recorded via idx_map.
            sites = list(next_gen_sites)
            next_gen_sites = []   # reset — will be repopulated during this gen
            log(f"  G{gen}: {len(sites)} index-tracked terminal sites")

            expected_sites=summary[-1]['sites']*branching_factor if summary else None
            if expected_sites and len(sites)!=expected_sites:
                log(f"  !! Site count mismatch: got {len(sites)}, expected {expected_sites}",'warn')

            path_a=[s for s in sites if s['h_list']]
            path_b=[s for s in sites if not s['h_list']]

            remove_set=set()
            for s in path_a:
                for h_idx in s['h_list']: remove_set.add(h_idx)
            for s in path_b:
                remove_set.add(s['n_idx'])
            # e.g. Cl neighbours on Triazine terminal N atoms
            if labile_names:
                _cadj = build_adj(curr_bonds)
                _camap = {a['idx']:a for a in curr_atoms}
                for s in path_a:
                    for nb_idx in _cadj.get(s['n_idx'], []):
                        nb = _camap.get(nb_idx)
                        if (nb and not is_hydrogen(nb)
                                and nb['name'] in labile_names
                                and nb_idx not in remove_set):
                            remove_set.add(nb_idx)
                            log(f"  Terminal non-H labile: {nb['name']} (idx {nb_idx}) removed",'ok')

            if remove_set:
                below=[i for i in remove_set if i<=prev_gen_boundary]
                if below:
                    log(f"  !! {len(below)} atoms below prev_gen_boundary — adjusting",'warn')
                curr_atoms,curr_bonds,h_map=remove_atoms(curr_atoms,curr_bonds,remove_set)
                new_sites=[]
                for s in path_a:
                    new_n=h_map.get(s['n_idx'])
                    if new_n: new_sites.append({'n_idx':new_n,'h_list':[],'grandparent_idx':None,'ref_pos':None})
                for s in path_b:
                    gp=s.get('grandparent_idx')
                    new_gp=h_map.get(gp) if gp else None
                    if new_gp:
                        new_sites.append({'n_idx':new_gp,'h_list':[],'grandparent_idx':None,'ref_pos':None})
                    else:
                        log(f"  !! PATH B n_idx={s['n_idx']}: grandparent lost — skipping",'warn')
                sites=new_sites
                log(f"  PATH A={len(path_a)}  PATH B={len(path_b)}  "
                    f"removed={len(remove_set)}  bond sites={len(sites)}")
                if below: prev_gen_boundary-=len(below)

        # Select monomer or terminal unit
        is_last=(gen==max_gen)
        if use_terminal and is_last:
            use_atoms=copy.deepcopy(term_atoms_orig)
            use_bonds=copy.deepcopy(term_bonds_orig)
            inp_a=next((a for a in use_atoms if a['name']==term_input_name),None)
            if inp_a is None:
                inp_a=next((a for a in use_atoms if a['name']==mon_input_name),None)
            if inp_a is None:
                raise ValueError(f"Terminal: cannot find input atom '{term_input_name}'")
            if term_input_h_name:
                hn=next((a for a in use_atoms if a['name']==term_input_h_name),None)
                if hn: use_atoms,use_bonds,_=remove_atoms(use_atoms,use_bonds,{hn['idx']})
                else:
                    hn_list=get_h_neighbors(use_atoms,use_bonds,inp_a['idx'])
                    if hn_list: use_atoms,use_bonds,_=remove_atoms(use_atoms,use_bonds,{hn_list[0]['idx']})
            else:
                hn_list=get_h_neighbors(use_atoms,use_bonds,inp_a['idx'])
                if hn_list: use_atoms,use_bonds,_=remove_atoms(use_atoms,use_bonds,{hn_list[0]['idx']})
            cur_inp_name=inp_a['name']
            cur_terminal_idxs=[]   
            log(f"  Using terminal unit for G{gen}  (input: {cur_inp_name})")
        else:
            use_atoms=copy.deepcopy(mon_atoms_clean)
            use_bonds=copy.deepcopy(mon_bonds_clean)
            cur_inp_name=mon_input_name
            cur_terminal_idxs=list(mon_terminal_idxs_clean)   

        if use_atoms:
            idxs=[a['idx'] for a in use_atoms]
            if idxs!=list(range(1,len(use_atoms)+1)):
                remap={old:new+1 for new,old in enumerate(sorted(idxs))}
                for a in use_atoms: a['idx']=remap[a['idx']]
                use_bonds=[(remap[b[0]],remap[b[1]],b[2]) for b in use_bonds
                           if b[0] in remap and b[1] in remap]
                cur_terminal_idxs=[remap[t] for t in cur_terminal_idxs if t in remap]

        adj=build_adj(curr_bonds)
        atom_map={a['idx']:a for a in curr_atoms}
        all_atoms=list(curr_atoms); all_bonds=list(curr_bonds)
        new_bonds_to_add=[]; existing_coords=[[a['x'],a['y'],a['z']] for a in curr_atoms]
        monomer_count=0; skipped=0; skipped_reasons=[]
        total_monomers=len(sites)*branching_factor
        # generations where the outer shell is more densely packed.
        # Ramps from 0.75 (G1) to 0.85 (G_max) linearly.
        t_gen = (gen - 1) / max(max_gen - 1, 1)
        cone_engine.vdw_scale = 0.75 + t_gen * (0.85 - 0.75)
        # by Δr Å per generation to pre-clear space in the crowded outer shell.
        radial_bias = 0.15 * (gen - 1)   # Å beyond bond_length

        cone_engine._build_kdtree(curr_atoms)
        kdtree_placement_count = 0   
        if len(sites) >= branching_factor and len(sites) % branching_factor == 0:
            n_groups = len(sites) // branching_factor
            interleaved = []
            for bf_round in range(branching_factor):
                for g in range(n_groups):
                    interleaved.append((g * branching_factor + bf_round,
                                        sites[g * branching_factor + bf_round]))
        else:
            # Fallback: original order (odd site counts)
            interleaved = list(enumerate(sites))

        for site_idx, site in interleaved:
            if cancel_flag and cancel_flag[0]:
                log(f"\n  Build CANCELLED at G{gen} mid-placement",'warn')
                return summary

            n_idx=site['n_idx']

            # Stages 1-4: cone subdivision with v11 fixes
            try:
                n_pos=None
                for a in all_atoms:
                    if a['idx']==n_idx: n_pos=np.array([a['x'],a['y'],a['z']]); break
                if n_pos is None:
                    raise ValueError(f"Binding atom idx={n_idx} not found")

                # Stage 1: cone axis
                # G1 uses core bond direction; G2+ uses radial from core_centre
                ref_pos = site.get('ref_pos', None)
                cone_axis = cone_engine.compute_cone_axis(n_pos, ref_pos)
                azimuths   = cone_engine.partition_azimuthal(
                                branching_factor, 0.0,
                                gen=gen, max_gen=max_gen,
                                site_idx=site_idx, n_sites=len(sites))

                # Correct half-angle: branch_angle/2 (54.75° for PAMAM)
                theta      = cone_engine.branching_half_angle(
                                bond_angle, branching_factor, gen, max_gen)
                cone_dirs  = cone_engine.compute_child_directions(cone_axis, azimuths, theta)

            except Exception as e:
                skipped+=branching_factor
                skipped_reasons.append(f"n_idx={n_idx}: cone setup: {e}")
                log(f"  !! Skipped (cone setup) n_idx={n_idx}: {e}",'warn')
                continue

            for b_num in range(branching_factor):
                placed_atoms_start=len(all_atoms); placed_bonds_start=len(all_bonds)
                try:
                    bd = cone_dirs[b_num % len(cone_dirs)]
                    existing_arr=np.array(existing_coords)
                    # so it reflects recently placed monomers (tree becomes stale
                    # after the first few placements of a generation).
                    kdtree_placement_count += 1
                    if kdtree_placement_count % cone_engine.KDTREE_REBUILD_INTERVAL == 0:
                        cone_engine._build_kdtree(all_atoms)

                    # Stage 5: place using mol2 geometry along cone direction
                    # radial_bias Å beyond bond_length to pre-clear crowded shell.
                    best_coords,_=position_monomer(n_pos, bd, use_atoms,
                                                   existing_arr,
                                                   bond_length + radial_bias,
                                                   cur_inp_name,
                                                   filter_ang=cone_engine.SPATIAL_FILTER_ANG)

                    excl_r = bond_length * 1.8

                    _inp_atom_for_phantom = next(
                        (a for a in use_atoms if a['name'] == cur_inp_name),
                        None)
                    _phantom_type = (_inp_atom_for_phantom['type']
                                     if _inp_atom_for_phantom else 'c3')
                    _phantom_elem = (atom_element(_inp_atom_for_phantom)
                                     if _inp_atom_for_phantom else 'C')
                    phantom_atoms = []
                    for sib in range(b_num + 1, branching_factor):
                        sib_dir = cone_dirs[sib % len(cone_dirs)]
                        sib_target = n_pos + sib_dir * bond_length
                        phantom_atoms.append({
                            'x': float(sib_target[0]),
                            'y': float(sib_target[1]),
                            'z': float(sib_target[2]),
                            'name': _phantom_elem, 'type': _phantom_type
                        })
                    check_atoms = all_atoms[:placed_atoms_start] + phantom_atoms

                    # Stage 6: vdW clash check [Fix6 + exclude_radius + phantoms]
                    elems=[atom_element(a) for a in use_atoms]
                    gaff_types=[a.get('type','') for a in use_atoms]
                    coord_list=[list(c) for c in best_coords]
                    if cone_engine.has_clash_vdw(coord_list, elems,
                                                  check_atoms,
                                                  centre_pos=n_pos,
                                                  exclude_radius=excl_r,
                                                  new_types=gaff_types):
                        # Stage 7 + adaptive escalation.
                        # Try base vdw_scale, then +0.05, then +0.10 (max 3 attempts).
                        STRICT_THRESH = 1.0
                        ACCEPT_THRESH = 1.5
                        base_scale = cone_engine.vdw_scale
                        clearance = 0.0
                        resolved = False
                        for retry in range(3):
                            cone_engine.vdw_scale = base_scale + 0.05 * retry
                            # Deterministic per-monomer RNG: keyed on stable
                            # identifiers (family, generation, parent binding
                            # atom, branch, retry) — reproducible and
                            # order-independent across serial or parallel builds.
                            sa_rng = np.random.default_rng(
                                _monomer_seed(den_type, gen, n_idx, b_num, retry))
                            best_coords, bd, resolved, clearance = cone_engine.find_clash_free_rotation(
                                bd, cone_axis, use_atoms, cur_inp_name,
                                n_pos, bond_length + radial_bias, check_atoms,
                                exclude_radius=excl_r,
                                mon_bonds=use_bonds, flexible=True,
                                rng=sa_rng)
                            if resolved or clearance >= ACCEPT_THRESH:
                                break
                        cone_engine.vdw_scale = base_scale

                        # Adaptive policy: strict on intermediate gens, lenient on final gen.
                        if not resolved and clearance < STRICT_THRESH:
                            if not is_last:
                                del all_atoms[placed_atoms_start:]; del all_bonds[placed_bonds_start:]
                                skipped += 1
                                reason = f"n_idx={n_idx} b={b_num}: clearance={clearance:.2f}A < {STRICT_THRESH}"
                                skipped_reasons.append(reason)
                                log(f"  ✗ Skipped (clash): {reason}",'warn')
                                continue
                            else:
                                log(f"  !! final-gen accept n_idx={n_idx} b={b_num} (clearance={clearance:.2f}A)",'warn')
                        elif not resolved:
                            log(f"  ~ tight placement n_idx={n_idx} b={b_num} (clearance={clearance:.2f}A)",'warn')

                    n_ex=len(all_atoms); inp_new_idx=None; idx_map={}
                    for a in use_atoms:
                        na=dict(a)
                        na['x']=float(best_coords[a['idx']-1][0])
                        na['y']=float(best_coords[a['idx']-1][1])
                        na['z']=float(best_coords[a['idx']-1][2])
                        na['idx']=n_ex+a['idx']
                        idx_map[a['idx']]=na['idx']
                        if a['name']==cur_inp_name: inp_new_idx=na['idx']
                        all_atoms.append(na)

                    if inp_new_idx is None:
                        del all_atoms[placed_atoms_start:]
                        raise ValueError(f"Input atom '{cur_inp_name}' not found in placed monomer")

                    for bnd in use_bonds:
                        g1=idx_map.get(bnd[0]); g2=idx_map.get(bnd[1])
                        if g1 is None or g2 is None:
                            raise ValueError(f"Bond ({bnd[0]},{bnd[1]}) references missing atom")
                        all_bonds.append((g1,g2,bnd[2]))

                    amap_local={a['idx']:a for a in all_atoms}
                    gen_bond_type=detect_bond_type(amap_local[n_idx],amap_local[inp_new_idx],amap_local,{})
                    new_bonds_to_add.append((n_idx,inp_new_idx,gen_bond_type))

                    for coord in best_coords: existing_coords.append(list(coord))

                    for t_local in cur_terminal_idxs:
                        if t_local not in idx_map: continue
                        global_t = idx_map[t_local]
                        # find neighbours of terminal atom in monomer topology
                        t_nbs = [b[1] if b[0]==t_local else b[0]
                                 for b in use_bonds if t_local in (b[0], b[1])]
                        h_global=[]; gp_global=None
                        for nb_local in t_nbs:
                            nb_a=next((a for a in use_atoms if a['idx']==nb_local),None)
                            if nb_a is None: continue
                            # #1 Monomer Terminal Leaving Atom(s): when set, remove the
                            # neighbours whose element is in the SELECTED ELEMENT SET (a
                            # terminal site sheds branching_factor leaving atoms — one per
                            # child — so we match by element, not by exact name; listing
                            # several atoms just unions their elements). Picking H alone
                            # reproduces auto-H exactly; add Cl/Br/O to shed those too.
                            # Unset → auto-H detection (pre-w12.2 behaviour, unchanged).
                            if mon_terminal_leave_names:
                                is_leave = (atom_element(nb_a) in _mon_term_leave_elems)
                            else:
                                is_leave = is_hydrogen(nb_a)
                            if is_leave and len(h_global) < branching_factor:
                                mapped=idx_map.get(nb_local)
                                if mapped: h_global.append(mapped)
                            else:
                                if gp_global is None:
                                    gp_global=idx_map.get(nb_local)
                        next_gen_sites.append({
                            'n_idx':         global_t,
                            'h_list':        h_global,
                            'grandparent_idx':gp_global,
                            'ref_pos':       None,
                        })

                    monomer_count+=1

                except (ValueError,ZeroDivisionError) as e:
                    del all_atoms[placed_atoms_start:]; del all_bonds[placed_bonds_start:]
                    skipped+=1; reason=f"n_idx={n_idx} b={b_num}: {e}"
                    skipped_reasons.append(reason)
                    log(f"  !! Skipped (geometry): {reason}",'warn')
                except np.linalg.LinAlgError as e:
                    del all_atoms[placed_atoms_start:]; del all_bonds[placed_bonds_start:]
                    skipped+=1; reason=f"n_idx={n_idx} b={b_num}: LinAlg: {e}"
                    skipped_reasons.append(reason)
                    log(f"  !! Skipped (linalg): {reason}",'warn')

                done=monomer_count+skipped
                if total_monomers>0 and progress:
                    gen_pct=int(done/total_monomers*100)
                    overall_pct=int(((gen-1)+gen_pct/100)/max_gen*100)
                    progress(overall_pct, gen, gen_pct, len(all_atoms))

        all_bonds.extend(new_bonds_to_add)
        all_bonds=deduplicate_bonds(all_bonds)
        IMPROVE_THRESHOLD = 0.30   # Å — only apply if meaningful gain
        PASS2_STEPS       = 36     # rotations to try (10° spacing)
        gen_start_idx     = len(curr_atoms)  # first atom index of this generation

        # Group atoms added this generation by their monomer block
        # Each monomer adds exactly len(use_atoms) atoms consecutively
        mon_size = len(use_atoms)
        if mon_size > 0 and monomer_count > 0:
            cone_engine._build_kdtree(all_atoms)   # final tree for pass 2
            improved = 0
            for m in range(monomer_count):
                # Slice of all_atoms for this monomer copy
                m_start = gen_start_idx + m * mon_size
                m_end   = m_start + mon_size
                if m_end > len(all_atoms):
                    break
                mon_copy = all_atoms[m_start:m_end]
                # Compute radial axis from monomer centroid to core_centre
                cx = sum(a['x'] for a in mon_copy) / mon_size
                cy = sum(a['y'] for a in mon_copy) / mon_size
                cz = sum(a['z'] for a in mon_copy) / mon_size
                mon_cen = np.array([cx, cy, cz])
                raw_axis = mon_cen - cone_engine.core_centre
                raw_norm = np.linalg.norm(raw_axis)
                if raw_norm < 1e-6:
                    continue
                k2 = raw_axis / raw_norm

                # Atoms outside this monomer to check clearance against
                outer = all_atoms[:m_start] + all_atoms[m_end:]
                nearby2 = cone_engine._nearby_atoms(mon_cen, outer)
                if not nearby2:
                    continue
                nearby2_arr = np.array([[a['x'],a['y'],a['z']] for a in nearby2])

                # Current minimum clearance
                cur_coords = np.array([[a['x'],a['y'],a['z']] for a in mon_copy])
                diffs0 = nearby2_arr[:,None,:] - cur_coords[None,:,:]
                cur_clearance = float(np.sqrt((diffs0**2).sum(axis=2)).min())

                best_clearance = cur_clearance
                best_coords_p2 = None
                step2 = 2.0 * math.pi / PASS2_STEPS
                for i2 in range(1, PASS2_STEPS):   # skip i2=0 (identity)
                    angle2 = i2 * step2
                    cos_a, sin_a = math.cos(angle2), math.sin(angle2)
                    # Rodrigues rotation of each atom around k2 about mon_cen
                    rotated_coords = []
                    for ac in cur_coords:
                        v = ac - mon_cen
                        rv = (v * cos_a
                              + np.cross(k2, v) * sin_a
                              + k2 * np.dot(k2, v) * (1.0 - cos_a))
                        rotated_coords.append(rv + mon_cen)
                    rc_arr = np.array(rotated_coords)
                    diffs2 = nearby2_arr[:,None,:] - rc_arr[None,:,:]
                    cand_clearance = float(np.sqrt((diffs2**2).sum(axis=2)).min())
                    if cand_clearance > best_clearance:
                        best_clearance = cand_clearance
                        best_coords_p2 = rc_arr

                if best_coords_p2 is not None and \
                        (best_clearance - cur_clearance) >= IMPROVE_THRESHOLD:
                    for j, a in enumerate(all_atoms[m_start:m_end]):
                        a['x'] = float(best_coords_p2[j][0])
                        a['y'] = float(best_coords_p2[j][1])
                        a['z'] = float(best_coords_p2[j][2])
                    improved += 1

            if improved:
                log(f"  [Mod-7] Pass-2 improved {improved}/{monomer_count} "
                    f"monomers (threshold {IMPROVE_THRESHOLD} A)",'ok')
        CM_STEPS           = 300
        CM_LR              = 0.05    # Å per gradient step
        CM_DAMPING         = 0.995   # LR multiplied by this each step
        CM_CUTOFF          = 8.0     # Å — vdW interaction cutoff
        CM_REBUILD_INTERVAL= 30      # KDTree rebuild interval (steps)
        K_BOND             = 5.0     # bond spring constant (>> vdW)

        gen_end_idx = len(all_atoms)
        n_gen_atoms = gen_end_idx - gen_start_idx

        if n_gen_atoms > 0 and monomer_count > 0:
            try:
                from scipy.spatial import KDTree as _CMKDTree
            except ImportError:
                _CMKDTree = None

            # ── Extract coordinates ────────────────────────────────
            mobile = np.array([[a['x'],a['y'],a['z']]
                               for a in all_atoms[gen_start_idx:gen_end_idx]],
                              dtype=float)
            frozen = (np.array([[a['x'],a['y'],a['z']]
                                for a in all_atoms[:gen_start_idx]], dtype=float)
                      if gen_start_idx > 0 else np.zeros((0,3)))

            # ── Pre-compute vdW radii ──────────────────────────────
            mobile_r = np.array([
                cone_engine._vdw_radius(
                    all_atoms[gen_start_idx+i].get('type',''),
                    atom_element(all_atoms[gen_start_idx+i]))
                for i in range(n_gen_atoms)], dtype=float)
            mobile_types = [all_atoms[gen_start_idx+i].get('type','')
                            for i in range(n_gen_atoms)]
            # H mask — True for hydrogen atoms (skip H-H pairs)
            mobile_h = np.array([t.startswith('h') for t in mobile_types])

            frozen_r = (np.array([
                cone_engine._vdw_radius(
                    all_atoms[i].get('type',''),
                    atom_element(all_atoms[i]))
                for i in range(gen_start_idx)], dtype=float)
                if gen_start_idx > 0 else np.zeros(0))
            frozen_h = (np.array([all_atoms[i].get('type','').startswith('h')
                                  for i in range(gen_start_idx)])
                        if gen_start_idx > 0 else np.zeros(0, dtype=bool))

            mobile_global_idxs = {all_atoms[gen_start_idx+i]['idx']: i
                                   for i in range(n_gen_atoms)}
            frozen_global_idxs = {all_atoms[i]['idx']: i
                                   for i in range(gen_start_idx)}
            bond_constraints = []  # list of (mobile_i, frozen_k, target_d)
            for b in all_bonds:
                b1, b2 = b[0], b[1]
                if b1 in mobile_global_idxs and b2 in frozen_global_idxs:
                    mi = mobile_global_idxs[b1]
                    fk = frozen_global_idxs[b2]
                    d0 = float(np.linalg.norm(mobile[mi] - frozen[fk]))
                    if d0 > 0.1:   # sanity check — skip zero-length
                        bond_constraints.append((mi, fk, d0))
                elif b2 in mobile_global_idxs and b1 in frozen_global_idxs:
                    mi = mobile_global_idxs[b2]
                    fk = frozen_global_idxs[b1]
                    d0 = float(np.linalg.norm(mobile[mi] - frozen[fk]))
                    if d0 > 0.1:
                        bond_constraints.append((mi, fk, d0))

            # ── Gradient descent loop ──────────────────────────────
            lr             = CM_LR
            mob_tree       = None
            frz_tree       = (_CMKDTree(frozen)
                              if _CMKDTree and len(frozen) > 0 else None)

            # T6: optional Numba force kernel. Off unless UDB_NUMBA_CM is set.
            # T7 cell-list requires the Numba CM path (it consumes array pairs).
            _use_numba_cm = (_accel is not None and _accel.numba_cm_enabled())
            _use_cell_list = (_accel is not None and _accel.cell_list_enabled())
            if _use_cell_list:
                _use_numba_cm = True
            if _use_numba_cm:
                _bond_mi = np.array([b[0] for b in bond_constraints], dtype=np.int64)
                _bond_fk = np.array([b[1] for b in bond_constraints], dtype=np.int64)
                _bond_d0 = np.array([b[2] for b in bond_constraints], dtype=float)

            # T12: device-resident GPU collective min (whole 300-step loop on the
            # GPU, one transfer). Gated by UDB_GPU_CM + n_atoms >= GPU_MIN_ATOMS,
            # CPU auto-fallback on any CUDA error. Reorders float reductions vs the
            # CPU path (cf. T7), so off by default.
            _gpu_done = False
            if (_accel is not None and _accel.gpu_cm_enabled()
                    and n_gen_atoms >= _accel.gpu_min_atoms()):
                try:
                    _res = _accel.gpu_collective_min(
                        mobile, mobile_r, mobile_h, frozen, frozen_r, frozen_h,
                        np.array([b[0] for b in bond_constraints], dtype=np.int64),
                        np.array([b[1] for b in bond_constraints], dtype=np.int64),
                        np.array([b[2] for b in bond_constraints], dtype=float),
                        cone_engine.vdw_scale, K_BOND,
                        CM_STEPS, CM_LR, CM_DAMPING, CM_CUTOFF)
                    if _res is not None:
                        mobile = _res
                        _gpu_done = True
                        log("  [Alt-1] Collective min: GPU device-resident (T12)", 'ok')
                except Exception as _gerr:
                    log(f"  [Alt-1] GPU CM failed ({_gerr}); CPU fallback", 'warn')

            for cm_step in range(0 if _gpu_done else CM_STEPS):
                grad = np.zeros_like(mobile)

                # Rebuild mobile KDTree periodically
                if _CMKDTree and (cm_step % CM_REBUILD_INTERVAL == 0):
                    mob_tree = _CMKDTree(mobile)

                # ── T6 Numba force kernel (opt-in) ────────────────
                if _use_numba_cm and mob_tree is not None:
                    # T7: cell-list neighbour search (O(N), ~1.35x faster than a
                    # rebuilt cKDTree) when UDB_CELL_LIST is set; identical pair
                    # set, so forces match within tolerance.
                    if _use_cell_list:
                        _pmm = _accel.cell_list_pairs(mobile, CM_CUTOFF)
                    else:
                        _pmm = np.asarray(list(mob_tree.query_pairs(CM_CUTOFF)),
                                          dtype=np.int64).reshape(-1, 2)
                    if frz_tree is not None:
                        _mf = mob_tree.query_ball_tree(frz_tree, CM_CUTOFF)
                        _pmf = np.array([(ii, kk) for ii, fl in enumerate(_mf)
                                         for kk in fl],
                                        dtype=np.int64).reshape(-1, 2)
                    else:
                        _pmf = np.zeros((0, 2), dtype=np.int64)
                    grad = _accel.cm_gradient(
                        mobile, mobile_r, mobile_h, _pmm,
                        frozen, frozen_r, frozen_h, _pmf,
                        _bond_mi, _bond_fk, _bond_d0,
                        cone_engine.vdw_scale, K_BOND)
                    g_norm = float(np.linalg.norm(grad))
                    if g_norm < 1e-8:
                        log(f"  [Alt-1] Collective min converged at step {cm_step}",'ok')
                        break
                    g_max = float(np.abs(grad).max())
                    mobile += lr * (grad / max(g_max, 1.0))
                    lr    *= CM_DAMPING
                    continue

                # ── Mobile vs mobile (Bug-5: KDTree pairs) ────────
                if mob_tree is not None:
                    pairs_mm = mob_tree.query_pairs(CM_CUTOFF)
                    for i, j in pairs_mm:
                        if mobile_h[i] and mobile_h[j]: continue
                        diff = mobile[i] - mobile[j]
                        d    = float(np.linalg.norm(diff))
                        if d < 1e-6: continue
                        r_min = cone_engine.vdw_scale * (mobile_r[i]+mobile_r[j])
                        if d < r_min:
                            ov  = r_min - d
                            uv  = diff / d
                            gm  = 2.0 * ov / d
                            grad[i] += gm * uv
                            grad[j] -= gm * uv
                else:
                    # Fallback: vectorised NumPy (only safe for small N)
                    for i in range(n_gen_atoms):
                        if mobile_h[i]: continue
                        for j in range(i+1, n_gen_atoms):
                            if mobile_h[j]: continue
                            diff = mobile[i] - mobile[j]
                            d    = float(np.linalg.norm(diff))
                            if d < 1e-6 or d > CM_CUTOFF: continue
                            r_min = cone_engine.vdw_scale*(mobile_r[i]+mobile_r[j])
                            if d < r_min:
                                ov = r_min - d
                                uv = diff / d
                                gm = 2.0 * ov / d
                                grad[i] += gm * uv
                                grad[j] -= gm * uv

                # ── Mobile vs frozen (Bug-5: KDTree ball-tree) ────
                if frz_tree is not None and mob_tree is not None:
                    pairs_mf = mob_tree.query_ball_tree(frz_tree, CM_CUTOFF)
                    for i, flist in enumerate(pairs_mf):
                        if mobile_h[i]: continue
                        for k in flist:
                            if frozen_h[k]: continue
                            diff = mobile[i] - frozen[k]
                            d    = float(np.linalg.norm(diff))
                            if d < 1e-6: continue
                            r_min = cone_engine.vdw_scale*(mobile_r[i]+frozen_r[k])
                            if d < r_min:
                                ov  = r_min - d
                                uv  = diff / d
                                grad[i] += 2.0 * ov / d * uv
                elif len(frozen) > 0:
                    for i in range(n_gen_atoms):
                        if mobile_h[i]: continue
                        for k in range(len(frozen)):
                            if frozen_h[k]: continue
                            diff = mobile[i] - frozen[k]
                            d    = float(np.linalg.norm(diff))
                            if d < 1e-6 or d > CM_CUTOFF: continue
                            r_min = cone_engine.vdw_scale*(mobile_r[i]+frozen_r[k])
                            if d < r_min:
                                ov  = r_min - d
                                uv  = diff / d
                                grad[i] += 2.0 * ov / d * uv

                for mi, fk, d0 in bond_constraints:
                    diff = mobile[mi] - frozen[fk]
                    d    = float(np.linalg.norm(diff))
                    if d < 1e-6: continue
                    stretch = d - d0       # +ve: too far, -ve: too close
                    uv  = diff / d
                    # Spring: restoring force toward d0
                    grad[mi] -= K_BOND * stretch * uv

                # ── Update coords ──────────────────────────────────
                g_norm = float(np.linalg.norm(grad))
                if g_norm < 1e-8:
                    log(f"  [Alt-1] Collective min converged at step {cm_step}",'ok')
                    break
                g_max = float(np.abs(grad).max())
                mobile += lr * (grad / max(g_max, 1.0))
                lr    *= CM_DAMPING

            # Write minimised coords back to all_atoms
            for i in range(n_gen_atoms):
                all_atoms[gen_start_idx+i]['x'] = float(mobile[i][0])
                all_atoms[gen_start_idx+i]['y'] = float(mobile[i][1])
                all_atoms[gen_start_idx+i]['z'] = float(mobile[i][2])

            log(f"  [Alt-1] Collective minimisation: {CM_STEPS} steps, "
                f"{n_gen_atoms} mobile atoms, "
                f"{len(bond_constraints)} bond constraints, "
                f"lr={CM_LR}→{lr:.5f}",'ok')

        missing=total_monomers-monomer_count
        log(f"\n  G{gen} complete:",'ok')
        log(f"    Sites     : {len(sites)}",'ok')
        log(f"    Monomers  : {monomer_count}  (expected {total_monomers})",
            'ok' if missing==0 else 'warn')
        if missing>0: log(f"    Missing   : {missing}",'warn')
        if skipped>0:
            log(f"    Skipped   : {skipped}",'warn')
            for r in skipped_reasons[:5]: log(f"      -> {r}",'warn')
        log(f"    Total atoms : {len(all_atoms)}",'ok')
        log(f"    Total bonds : {len(all_bonds)}",'ok')

        # Validate
        val_results,val_passed,val_total,mw,rg,n_term,n_branch=\
            validate_structure(all_atoms,all_bonds,mon_terminal_names)
        struct_rep=StructuralAnalysis.full_report(all_atoms,gen)
        log(f"\n  VALIDATION G{gen}  ({val_passed}/{val_total} passed):",'head')
        for label,ok,got,expected in val_results:
            icon="  ok" if ok else "  !!"
            tag='ok' if ok else ('warn' if 'clash' in label.lower() else 'err')
            log(f"{icon}  {label:<42} got={got}  exp={expected}",tag)
        log(f"  MW={mw:,.0f} Da   Rg={rg:.2f} A   "
            f"Rg_solv={struct_rep['Rg_solvent']:.2f} A   "
            f"delta={struct_rep['asphericity']:.4f}   "
            f"Term={n_term}   Branch={n_branch}",
            'ok' if val_passed==val_total else 'warn')

        # ─── Extended structural characterization (AS-BUILT) ────────────────
        try:
            ext_rep_built = StructuralAnalysis.extended_report(
                all_atoms, gen, mass_amu=mw, n_terminal=n_term)
            log(StructuralAnalysis.format_extended_table(
                ext_rep_built, label='AS-BUILT'), 'info')
        except Exception as _e:
            log(f"  [characterization] AS-BUILT failed: {_e}", 'warn')
            ext_rep_built = None

        # fire GUI callbacks — SAME SIGNATURE as v9
        val_str=f"{val_passed}/{val_total}"
        if gen_progress:
            gen_progress(gen, val_results, val_passed, val_total,
                         mw, rg, monomer_count, n_term)

        prefix=f"dendrimer_{den_type.lower()}_gen{gen}"
        if gen in gens_to_build:
            steps=pipeline_steps or {}
            if steps.get('save_mol2',True) or 'mol2' in out_formats:
                f_=os.path.join(output_dir,f"{prefix}_typed.mol2")
                write_mol2(f_,all_atoms,all_bonds,gen,den_type)
                log(f"\n    Saved: {os.path.basename(f_)}",'ok')
            if steps.get('save_pdb',True) or 'pdb' in out_formats:
                f_=os.path.join(output_dir,f"{prefix}.pdb")
                write_pdb(f_,all_atoms,all_bonds,gen,den_type)
                log(f"    Saved: {os.path.basename(f_)}",'ok')
            if 'xyz' in out_formats:
                f_=os.path.join(output_dir,f"{prefix}.xyz")
                write_xyz(f_,all_atoms,gen,den_type)
                log(f"    Saved: {os.path.basename(f_)}",'ok')
            if 'gro' in out_formats:
                f_=os.path.join(output_dir,f"{prefix}.gro")
                write_gro(f_,all_atoms,all_bonds,gen,den_type)
                log(f"    Saved: {os.path.basename(f_)}",'ok')
            if 'psf' in out_formats:
                f_=os.path.join(output_dir,f"{prefix}.psf")
                write_psf(f_,all_atoms,all_bonds,gen,den_type)
                log(f"    Saved: {os.path.basename(f_)}",'ok')
            if steps.get('save_script',True):
                min_out=os.path.join(output_dir,f"{prefix}_minimization.sh")
                write_min_script(min_out,gen,len(all_atoms),den_type,
                                 ff=steps.get('ff','gaff'),
                                 charge_method=steps.get('charge_method','bcc'),
                                 net_charge=net_charge,
                                 param_method=steps.get('param_method','full_antechamber'),
                                 core_file=steps.get('core_file',''),
                                 monomer_file=steps.get('monomer_file',''),
                                 terminal_file=steps.get('terminal_file',''),
                                 ncpus=steps.get('ncpus',1),
                                 conda_path=steps.get('conda_path',''),
                                 conda_env=steps.get('conda_env','AmberTools25'))
                log(f"    Saved: {os.path.basename(min_out)}",'ok')

            # the antechamber checkbox is unticked — it runs independently.
            # box unticked silently skipped parmchk2/tleap/sander entirely.
            any_amber=(
                steps.get('param_method')=='fragment_bcc' or
                any(steps.get(k) for k in ['antechamber','parmchk2','tleap',
                                            'minimize','ambpdb','solvate','add_ions','md']))
            if run_min and any_amber and steps.get('save_mol2',True):
                mol2_out=os.path.join(output_dir,f"{prefix}_typed.mol2")
                run_full_pipeline(mol2_out,output_dir,den_type,gen,
                                  net_charge,log,steps,cancel_flag)

            # OpenBabel geometry optimisation — runs after mol2 is saved
            if steps.get('obabel_minimize') and steps.get('save_mol2',True):
                mol2_out=os.path.join(output_dir,f"{prefix}_typed.mol2")
                mol2_opt=os.path.join(output_dir,f"{prefix}_obopt.mol2")
                ob_ff   =steps.get('obabel_ff','MMFF94')
                ob_steps=steps.get('obabel_steps',2500)
                ob_cmd  =(f"obabel {os.path.basename(mol2_out)} "
                          f"-O {os.path.basename(mol2_opt)} "
                          f"--minimize --ff {ob_ff} --steps {ob_steps}")
                log(f"\n  [OpenBabel]  $ {ob_cmd}",'head')
                try:
                    _cmd = shlex.split(ob_cmd) if isinstance(ob_cmd, str) else ob_cmd
                    r=subprocess.run(_cmd,shell=False,capture_output=True,
                                     text=True,cwd=output_dir,timeout=1800)
                    if r.returncode==0:
                        log(f"  [OpenBabel] done → {os.path.basename(mol2_opt)}",'ok')
                    else:
                        log(f"  [OpenBabel] FAILED (rc={r.returncode})",'err')
                        if r.stderr: log(r.stderr[:300],'err')
                except Exception as e:
                    log(f"  [OpenBabel] ERROR: {e}",'err')

            # ─── Extended structural characterization (MINIMIZED) ───────────
            # Re-runs the table on minimized coordinates if available.
            # Priority: AMBER minimized PDB (from ambpdb), then OpenBabel mol2.
            min_pdb_path = os.path.join(output_dir, f"{prefix}_min.pdb")
            ob_mol2_path = os.path.join(output_dir, f"{prefix}_obopt.mol2")
            min_src = None
            if os.path.exists(min_pdb_path):
                min_src = ('AMBER-MINIMIZED', min_pdb_path, 'pdb')
            elif os.path.exists(ob_mol2_path):
                min_src = ('OBABEL-MINIMIZED', ob_mol2_path, 'mol2')
            if min_src is not None:
                lbl, src_path, fmt = min_src
                try:
                    if fmt == 'pdb':
                        min_atoms = read_pdb_coords(src_path, all_atoms)
                    else:
                        # parse the obabel-optimized mol2 — preserve types
                        m_atoms, _ = parse_mol2(src_path)
                        if len(m_atoms) != len(all_atoms):
                            raise ValueError(
                                f"obopt mol2 atom count ({len(m_atoms)}) "
                                f"!= reference ({len(all_atoms)})")
                        min_atoms = []
                        for ref, m in zip(all_atoms, m_atoms):
                            a2 = dict(ref)
                            a2['x']=m['x']; a2['y']=m['y']; a2['z']=m['z']
                            min_atoms.append(a2)
                    ext_rep_min = StructuralAnalysis.extended_report(
                        min_atoms, gen, mass_amu=mw, n_terminal=n_term)
                    log(StructuralAnalysis.format_extended_table(
                        ext_rep_min, label=lbl), 'info')
                    # Full characterisation on minimised geometry
                    if _CHAR_MODULE_AVAILABLE:
                        try:
                            _steric2 = StericValidator.check(den_type, gen)
                            _char_min = compute_full_characterization(
                                atoms         = min_atoms,
                                generation    = gen,
                                mass_amu      = mw,
                                n_term        = n_term,
                                branching_factor = branching_factor,
                                den_type      = den_type,
                                P             = _steric2.get('P', 17),
                                g_max         = _steric2.get('g_max', 9.0),
                                strain_risk   = _steric2.get('strain_risk', 'NONE'),
                                area_per_term = _steric2.get('area_per_terminal', None),
                                terminal_names= mon_terminal_names,
                                core_centre   = cone_engine.core_centre,
                            )
                            log(format_characterization_log(_char_min, label=lbl), 'info')
                        except Exception as _cme:
                            log(f'  [Characterisation-min] {_cme}', 'warn')
                except Exception as _e:
                    log(f"  [characterization] {lbl} failed: {_e}", 'warn')

            # NAMD pipeline — generate input files then execute stages
            any_namd=(steps.get('namd_solvate') or steps.get('namd_minimize') or
                      steps.get('namd_heat') or steps.get('namd_equil') or
                      steps.get('namd_prod'))
            if any_namd:
                run_namd_pipeline(output_dir,den_type,gen,log,steps,cancel_flag)

            if steps.get('cgenff_use') and steps.get('save_pdb',True):
                write_cgenff_psfgen_script(
                    output_dir, den_type, gen,
                    core_str_path    = steps.get('core_str_file',''),
                    monomer_str_path = steps.get('monomer_str_file',''),
                    terminal_str_path= steps.get('terminal_str_file',''),
                    box_pad          = steps.get('namd_box', 15.0),
                    log_cb           = log)

            summary.append({
                'gen':gen,'atoms':len(all_atoms),'bonds':len(all_bonds),
                'mw':mw,'rg':rg,'n_term':n_term,'n_branch':n_branch,
                'val':f"{val_passed}/{val_total}",
                'mol2':os.path.join(output_dir,f"{prefix}_typed.mol2"),
                'sites':len(sites),'monomers':monomer_count,
                'rg_solv':struct_rep['Rg_solvent'],
                'asphericity':struct_rep['asphericity'],
            })

            # ─── Intergenerational minimization (v1.0) ──────────────────────
            # After each generation, minimize and feed minimized coords into
            # the next generation's starting geometry. Runs only between gens.
            if steps.get('intergen_enabled') and gen < max_gen:
                log(f"\n{'='*52}", 'head')
                log(f"  INTERGENERATIONAL MIN  G{gen} → G{gen+1}  (v1.0)", 'head')
                log(f"{'='*52}", 'head')
                _amber_atoms = None
                _namd_atoms  = None

                if steps.get('intergen_use_amber'):
                    _amber_atoms = run_intergen_amber_min(
                        prefix, output_dir, all_atoms, log)

                if steps.get('intergen_use_namd'):
                    _namd_atoms = run_intergen_namd_min(
                        prefix, output_dir, all_atoms, log,
                        steps, gen, cancel_flag)

                # Independent mode: NAMD takes priority; AMBER is fallback.
                _final_atoms = None
                if _namd_atoms is not None:
                    _final_atoms = _namd_atoms
                    _src = 'NAMD'
                elif _amber_atoms is not None:
                    _final_atoms = _amber_atoms
                    _src = 'AMBER'

                if _final_atoms is not None:
                    rg_min = StructuralAnalysis.radius_of_gyration(_final_atoms)
                    log(f"  [IntergenMin] {_src} coords accepted: "
                        f"Rg {rg:.2f} → {rg_min:.2f} Å  (Δ={rg_min-rg:+.2f} Å)", 'ok')
                    log(f"  [IntergenMin] G{gen+1} will build on {_src}-minimized G{gen}", 'ok')
                    all_atoms = _final_atoms
                else:
                    log(f"  [IntergenMin] No minimized coords obtained — "
                        f"G{gen+1} will use as-built G{gen} geometry.", 'warn')

        else:
            log(f"    G{gen} not in selection — kept in memory")

        prev_gen_boundary=len(curr_atoms)
        curr_atoms=all_atoms; curr_bonds=all_bonds
        if progress:
            progress(int(gen/max_gen*100), gen, 100, len(all_atoms))

    log(f"\n{'='*62}",'head')
    log(f"  BUILD COMPLETE — {den_type} Dendrimer  [v1.0]",'head')
    log(f"{'='*62}",'head')
    log(f"  {'Gen':<5} {'Atoms':<8} {'MW(Da)':<12} {'Rg(A)':<8} "
        f"{'Rg_solv(A)':<12} {'delta':<8} Validation",'head')
    log(f"  {'-'*64}",'head')
    for s in summary:
        log(f"  G{s['gen']:<4} {s['atoms']:<8} {s['mw']:<12,.0f} {s['rg']:<8.2f} "
            f"{s.get('rg_solv',0):<12.2f} {s.get('asphericity',0):<8.4f} {s['val']}",'ok')
    log(f"\n  Output: {output_dir}",'head')
    log(f"{'='*62}\n",'head')
    return summary


# =============================================================================
# SECTION 14.  3D MOLECULAR VIEWER & FRAGMENT MANAGER WIDGETS
# =============================================================================
# 3D molecular viewer widget (matplotlib embedded).
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class MolViewer3D(tk.Frame):
    """
    3D rotatable molecular viewer embedded in Tkinter via matplotlib.
    - Element-coloured spheres with atom name labels
    - Role highlight colours (5 roles, updated live)
    - Click atom -> callback(atom_idx, atom_name)
    - Atom name search -> highlight matching atoms
    - Equal aspect ratio: molecule always fills the box properly
    - Smart scroll: inside viewer = 3D zoom, outside = page scroll
    """

    def __init__(self, parent, T, title="Molecule",
                 on_click=None, width=430, height=320,
                 page_scroll_cmd=None):
        super().__init__(parent, bg=T['panel'])
        self.T               = T
        self.on_click        = on_click
        self._page_scroll    = page_scroll_cmd  # page scroll callback
        self._atoms          = []
        self._bonds          = []
        self._roles          = {}
        self._highlight      = set()
        self._pick_map       = {}
        self._mouse_inside   = False            # track hover

        # title
        tk.Label(self, text=f"  {title}", font=BOLD,
                 fg=T['cyan'], bg=T['panel']).pack(anchor='w', padx=6, pady=(4,0))

        body = tk.Frame(self, bg=T['panel'])
        body.pack(fill='both', expand=True)

        # left: matplotlib 3D canvas
        fig_bg = '#0d1117'
        self.fig = plt.Figure(figsize=(width/100, height/100), dpi=100)
        self.fig.patch.set_facecolor(fig_bg)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self._style_axes(fig_bg)

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg=fig_bg, highlightthickness=0)
        self.canvas_widget.pack(side='left', fill='both', expand=True,
                                padx=(6,2), pady=4)

        # smart scroll — track mouse enter/leave
        self.canvas_widget.bind('<Enter>', lambda e: setattr(self, '_mouse_inside', True))
        self.canvas_widget.bind('<Leave>', lambda e: setattr(self, '_mouse_inside', False))

        # connect matplotlib pick event for atom click
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)

        # right: legend + stats + search
        right = tk.Frame(body, bg=T['panel'], width=165)
        right.pack(side='right', fill='y', padx=(2,6), pady=4)
        right.pack_propagate(False)

        # search
        tk.Label(right, text="Search atom:", font=("Courier New",9),
                 fg=T['muted'], bg=T['panel']).pack(anchor='w')
        sf = tk.Frame(right, bg=T['panel']); sf.pack(fill='x', pady=(0,6))
        self._search_var = tk.StringVar()
        tk.Entry(sf, textvariable=self._search_var, font=("Courier New",10),
                 bg=T['entry'], fg=T['cyan'], insertbackground=T['cyan'],
                 relief='flat', bd=3, width=11).pack(side='left')
        tk.Button(sf, text="Go", font=("Courier New",10,"bold"),
                  bg=T['sbtn'], fg=T['green'], relief='flat', padx=4,
                  cursor='hand2', command=self._do_search).pack(side='left', padx=2)

        # stats
        self._stats_lbl = tk.Label(right, text="No file loaded",
                                   font=("Courier New",9), fg=T['muted'],
                                   bg=T['panel'], justify='left')
        self._stats_lbl.pack(anchor='w', pady=(0,6))

        # role legend
        tk.Label(right, text="Roles", font=("Courier New",9,"bold"),
                 fg=T['muted'], bg=T['panel']).pack(anchor='w')
        for role, color in ROLE_COLORS.items():
            if color is None: continue
            rf = tk.Frame(right, bg=T['panel']); rf.pack(anchor='w')
            c_ = tk.Canvas(rf, width=12, height=12, bg=T['panel'],
                           highlightthickness=0)
            c_.pack(side='left')
            c_.create_oval(1,1,11,11, fill=color, outline='')
            tk.Label(rf, text=ROLE_LABELS[role], font=("Courier New",8),
                     fg=T['muted'], bg=T['panel']).pack(side='left')

        tk.Label(right, text="\nDrag: rotate\nScroll: zoom\nClick atom: select",
                 font=("Courier New",8), fg=T['muted'],
                 bg=T['panel'], justify='left').pack(anchor='w', pady=(6,0))

        self._redraw()

    def _style_axes(self, fig_bg):
        ax = self.ax
        ax.set_facecolor(fig_bg)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#30363d')
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def load(self, atoms, bonds):
        self._atoms = atoms
        self._bonds = bonds
        self._roles = {}
        self._highlight = set()
        self._update_legend()
        self._redraw()

    def set_roles(self, roles_dict):
        self._roles = dict(roles_dict)
        self._redraw()

    def _atom_color(self, atom):
        role = self._roles.get(atom['idx'])
        if role and ROLE_COLORS.get(role):
            return ROLE_COLORS[role]
        if atom['idx'] in self._highlight:
            return '#ffffff'
        e = atom_element(atom)
        return ELEMENT_COLORS.get(e, ELEMENT_COLORS.get(e[:1], '#8b949e'))

    def _redraw(self):
        self.ax.cla()
        self._style_axes(self.fig.get_facecolor())
        self._pick_map = {}

        if not self._atoms:
            self.canvas.draw_idle()
            return

        amap = {a['idx']: a for a in self._atoms}

        # draw bonds
        segs = []
        for b in self._bonds:
            a1 = amap.get(b[0]); a2 = amap.get(b[1])
            if a1 and a2:
                segs.append([(a1['x'],a1['y'],a1['z']),
                              (a2['x'],a2['y'],a2['z'])])
        if segs:
            lc = Line3DCollection(segs, colors='#555e6a', linewidths=1.5, zorder=1)
            self.ax.add_collection3d(lc)

        # draw atoms grouped by colour for pick support
        from collections import defaultdict
        colour_groups = defaultdict(list)
        for a in self._atoms:
            colour_groups[self._atom_color(a)].append(a)

        for color, atoms_in_group in colour_groups.items():
            xs = [a['x'] for a in atoms_in_group]
            ys = [a['y'] for a in atoms_in_group]
            zs = [a['z'] for a in atoms_in_group]
            sizes = [55 if is_hydrogen(a) else 150 for a in atoms_in_group]
            sc = self.ax.scatter(xs, ys, zs, c=color, s=sizes,
                                 depthshade=True, picker=6, zorder=3,
                                 edgecolors='#ffffff' if color=='#ffffff' else '#00000033',
                                 linewidths=0.4)
            self._pick_map[id(sc)] = atoms_in_group

        # atom labels — skip H for large molecules
        for a in self._atoms:
            if not is_hydrogen(a) or len(self._atoms) < 35:
                self.ax.text(a['x'], a['y'], a['z'], f" {a['name']}",
                             fontsize=6, color='#c9d1d9', zorder=4,
                             ha='left', va='bottom')

        # Equal aspect ratio — molecule fills the box
        xs_all = [a['x'] for a in self._atoms]
        ys_all = [a['y'] for a in self._atoms]
        zs_all = [a['z'] for a in self._atoms]
        cx = (max(xs_all)+min(xs_all))/2
        cy = (max(ys_all)+min(ys_all))/2
        cz = (max(zs_all)+min(zs_all))/2
        # use the same half-range for all axes so molecule is not distorted
        half = max(
            (max(xs_all)-min(xs_all))/2,
            (max(ys_all)-min(ys_all))/2,
            (max(zs_all)-min(zs_all))/2,
            2.0   # minimum half-range so single atoms aren't huge
        ) * 1.15  # small padding

        self.ax.set_xlim(cx-half, cx+half)
        self.ax.set_ylim(cy-half, cy+half)
        self.ax.set_zlim(cz-half, cz+half)

        self.canvas.draw_idle()

    def _on_pick(self, event):
        art_id = id(event.artist)
        if art_id not in self._pick_map: return
        atoms_in_group = self._pick_map[art_id]
        ind = event.ind[0] if len(event.ind) > 0 else 0
        if ind >= len(atoms_in_group): return
        atom = atoms_in_group[ind]
        if self.on_click:
            self.on_click(atom['idx'], atom['name'])

    def _do_search(self):
        q = self._search_var.get().strip().upper()
        if not q:
            self._highlight = set()
        else:
            self._highlight = {a['idx'] for a in self._atoms
                               if a['name'].upper().startswith(q) or
                                  a['type'].upper().startswith(q)}
        self._redraw()

    def _update_legend(self):
        # Element legend removed in w11 — stats label still updated
        if not self._atoms: return
        T = self.T
        heavy = sum(1 for a in self._atoms if not is_hydrogen(a))
        self._stats_lbl.config(
            text=(f"Atoms : {len(self._atoms)}\n"
                  f"Heavy : {heavy}\n"
                  f"Bonds : {len(self._bonds)}\n"
                  f"MW    : {atom_mw(self._atoms):,.0f} Da"))

    def is_hovered(self):
        return self._mouse_inside

MolViewer = MolViewer3D

class FragmentManager(tk.Frame):
    """
    Dynamic add/remove listbox for atom names (terminal sites or core sites).
    Items: each entry is (atom_name, atom_idx_str).
    on_change() called whenever list changes.
    """
    def __init__(self, parent, T, title="Atoms", on_change=None,
                 atom_source=None):
        super().__init__(parent, bg=T['panel'])
        self.T = T
        self.on_change = on_change
        self.atom_source = atom_source  # callable returning list of atom dicts

        tk.Label(self, text=f"  {title}", font=("Courier New",10,"bold"),
                 fg=T['yellow'], bg=T['panel']).pack(anchor='w', pady=(4,2))

        # listbox
        lb_frame = tk.Frame(self, bg=T['panel'])
        lb_frame.pack(fill='x', padx=8)
        self.lb = tk.Listbox(lb_frame, font=("Courier New",10),
                             bg=T['entry'], fg=T['cyan'],
                             selectbackground=T['sel'],
                             selectforeground=T['white'],
                             activestyle='none',
                             relief='flat', bd=2, height=5)
        self.lb.pack(side='left', fill='x', expand=True)
        sb = ttk.Scrollbar(lb_frame, orient='vertical', command=self.lb.yview)
        sb.pack(side='right', fill='y')
        self.lb.config(yscrollcommand=sb.set)

        # add row
        add_row = tk.Frame(self, bg=T['panel'])
        add_row.pack(fill='x', padx=8, pady=(4,2))

        tk.Label(add_row, text="Name:", font=("Courier New",10),
                 fg=T['muted'], bg=T['panel']).pack(side='left')
        self._name_var = tk.StringVar()
        self._name_entry = tk.Entry(add_row, textvariable=self._name_var,
                                    font=("Courier New",10),
                                    bg=T['entry'], fg=T['cyan'],
                                    insertbackground=T['cyan'],
                                    relief='flat', bd=3, width=8)
        self._name_entry.pack(side='left', padx=4)

        tk.Label(add_row, text="Idx:", font=("Courier New",10),
                 fg=T['muted'], bg=T['panel']).pack(side='left')
        self._idx_var = tk.StringVar()
        tk.Entry(add_row, textvariable=self._idx_var,
                 font=("Courier New",10),
                 bg=T['entry'], fg=T['cyan'],
                 insertbackground=T['cyan'],
                 relief='flat', bd=3, width=5).pack(side='left', padx=4)

        if atom_source:
            tk.Button(add_row, text="↓", font=("Courier New",10,"bold"),
                      bg=T['sbtn'], fg=T['yellow'], relief='flat',
                      padx=4, cursor='hand2',
                      command=self._pick_from_source).pack(side='left', padx=2)

        btn_row = tk.Frame(self, bg=T['panel'])
        btn_row.pack(fill='x', padx=8, pady=(0,4))
        tk.Button(btn_row, text="+ Add", font=("Courier New",10,"bold"),
                  bg=T['sbtn'], fg=T['green'], relief='flat',
                  padx=6, cursor='hand2',
                  command=self._add).pack(side='left', padx=(0,6))
        tk.Button(btn_row, text="− Remove", font=("Courier New",10,"bold"),
                  bg=T['sbtn'], fg=T['red'], relief='flat',
                  padx=6, cursor='hand2',
                  command=self._remove).pack(side='left')

    def _add(self):
        name=self._name_var.get().strip()
        if not name: return
        idx=self._idx_var.get().strip()
        entry=f"{name}  (idx {idx})" if idx else name
        self.lb.insert('end', entry)
        self._name_var.set(''); self._idx_var.set('')
        if self.on_change: self.on_change()

    def _remove(self):
        sel=self.lb.curselection()
        if not sel: return
        self.lb.delete(sel[0])
        if self.on_change: self.on_change()

    def _pick_from_source(self):
        atoms=self.atom_source() if self.atom_source else []
        if not atoms:
            messagebox.showinfo("No atoms","Load a mol2 file first."); return
        win=tk.Toplevel(self)
        win.title("Pick atom"); win.configure(bg=self.T['bg'])
        win.geometry("260x320")
        tk.Label(win,text="  Select atom:",font=BOLD,
                 fg=self.T['cyan'],bg=self.T['bg']).pack(anchor='w',padx=8,pady=4)
        lb2=tk.Listbox(win,font=("Courier New",10),
                       bg=self.T['entry'],fg=self.T['cyan'],
                       selectbackground=self.T['sel'],
                       relief='flat',bd=2,height=15)
        lb2.pack(fill='both',expand=True,padx=8,pady=4)
        for a in atoms:
            lb2.insert('end',f"{a['idx']:3d}  {a['name']:<8}  {a['type']}")
        def pick():
            sel=lb2.curselection()
            if not sel: return
            a=atoms[sel[0]]
            self._name_var.set(a['name']); self._idx_var.set(str(a['idx']))
            win.destroy()
        tk.Button(win,text="Select",font=BOLD,
                  bg=self.T['cyan'],fg=self.T['bg'],
                  relief='flat',cursor='hand2',command=pick).pack(pady=4)

    def get_names(self):
        """Return list of atom names from the listbox."""
        items=[self.lb.get(i) for i in range(self.lb.size())]
        names=[]
        for item in items:
            name=item.split()[0]
            names.append(name)
        return names

    def get_idxs(self):
        """Return list of atom indices (int) from the listbox."""
        items=[self.lb.get(i) for i in range(self.lb.size())]
        idxs=[]
        for item in items:
            parts=item.split()
            for p in parts:
                if p.startswith('(idx'):
                    try: idxs.append(int(parts[parts.index(p)+1].rstrip(')')))
                    except (ValueError, IndexError): pass
                    break
        return idxs

    def set_name_entry(self, name, idx=''):
        """Called from viewer click → auto-fill."""
        self._name_var.set(name)
        self._idx_var.set(str(idx))


# =============================================================================
# SECTION 15.  CGENFF PARAMETER SUPPORT
# =============================================================================

def parse_cgenff_str(filepath):
    """
    Parse a CGenFF .str stream file from ParamChem.

    Returns a dict:
      {
        'atoms':   [{name, type, charge}, ...],
        'bonds':   [(a1, a2), ...],
        'rtf_lines':  [str, ...],   # full RTF block
        'prm_lines':  [str, ...],   # full PRM block
        'penalties':  [(section, key, score), ...]
        'max_penalty': float,
        'warn_count':  int,   # penalties > 50
        'ok':  bool,
        'error': str or None,
      }
    """
    result = {
        'atoms': [], 'bonds': [],
        'rtf_lines': [], 'prm_lines': [],
        'penalties': [], 'max_penalty': 0.0,
        'warn_count': 0, 'ok': False, 'error': None,
    }
    if not filepath or not os.path.exists(filepath):
        result['error'] = f"File not found: {filepath}"
        return result

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception as e:
        result['error'] = str(e)
        return result

    in_rtf = False
    in_prm = False
    current_section = None   # BONDS / ANGLES / DIHEDRALS / IMPROPERS / NONBONDED

    import re
    penalty_re = re.compile(r'penalty\s*=\s*([\d.]+)', re.IGNORECASE)

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        upper = stripped.upper()

        # detect block starts
        if upper.startswith('READ RTF') or upper == 'READ RTF':
            in_rtf = True; in_prm = False; continue
        if upper.startswith('READ PARAM') or upper.startswith('READ PAR'):
            in_prm = True; in_rtf = False; current_section = None; continue
        if upper == 'END':
            in_rtf = False; in_prm = False; current_section = None; continue

        if in_rtf:
            result['rtf_lines'].append(line)
            if upper.startswith('ATOM '):
                parts = stripped.split()
                if len(parts) >= 4:
                    result['atoms'].append({
                        'name':   parts[1],
                        'type':   parts[2],
                        'charge': parts[3],
                    })
            elif upper.startswith('BOND '):
                parts = stripped.split()[1:]
                for i in range(0, len(parts)-1, 2):
                    result['bonds'].append((parts[i], parts[i+1]))

        elif in_prm:
            result['prm_lines'].append(line)
            # track section headers
            for sec in ('BONDS','ANGLES','DIHEDRALS','IMPROPERS',
                        'NONBONDED','NBFIX','CMAP'):
                if upper.startswith(sec):
                    current_section = sec; break

            # extract penalty scores
            m = penalty_re.search(line)
            if m:
                score = float(m.group(1))
                # key = first two words of line (atom types)
                key_parts = stripped.split()[:3]
                key = ' '.join(key_parts)
                result['penalties'].append(
                    (current_section or 'PRM', key, score))
                if score > result['max_penalty']:
                    result['max_penalty'] = score
                if score > 50.0:
                    result['warn_count'] += 1

    result['ok'] = True
    return result

def write_cgenff_psfgen_script(output_dir, den_type, gen_num,
                                core_str_path, monomer_str_path,
                                terminal_str_path='',
                                box_pad=15.0, log_cb=None):
    """
    [w11.3] Write a psfgen .tcl script that uses the CGenFF .str files
    (molecule-specific RTF) instead of the generic CHARMM36 cgenff RTF.

    The script:
      1. Extracts RTF sections from each .str file into temp .rtf files
      2. Reads those topology files in psfgen
      3. Builds segments for core, monomer repeats, and terminal groups
      4. Solvates and ionizes
    """
    pre = f"{den_type.lower()}_cgenff_g{gen_num}"
    pdb_in = f"dendrimer_{den_type.lower()}_gen{gen_num}.pdb"
    psf_out = f"{pre}.psf"
    pdb_out = f"{pre}.pdb"

    def _str_to_rtf(str_path, out_name):
        """Extract RTF block from .str and write a valid standalone .rtf file.
        Fix: the .str RTF block already contains '36 1' as its version line.
        We must NOT prepend '20 1' — psfgen would see two version declarations
        and fail with 'FAILED TO RECOGNIZE 36'. Write '36 1' once as the
        single header, then strip any version-line duplicates from rtf_lines."""
        lines = []
        if str_path and os.path.exists(str_path):
            data = parse_cgenff_str(str_path)
            raw = data.get('rtf_lines', [])
            # strip any leading version line (e.g. '36 1') already in the block
            lines = [ln for ln in raw
                     if not (ln.strip() and ln.strip().split()[0].isdigit()
                             and len(ln.strip().split()) <= 2)]
        rtf_path = os.path.join(output_dir, out_name)
        with open(rtf_path, 'w') as f:
            f.write('* CGenFF RTF extracted by UDB v1.0\n*\n')
            f.write('36 1\n\n')   # correct CGenFF version header
            for ln in lines:
                f.write(ln + '\n')
            f.write('\nEND\n')
        return out_name   # relative filename for psfgen

    core_rtf = _str_to_rtf(core_str_path,     'cgenff_core.rtf')
    mon_rtf  = _str_to_rtf(monomer_str_path,   'cgenff_monomer.rtf')
    term_rtf = (_str_to_rtf(terminal_str_path, 'cgenff_terminal.rtf')
                if terminal_str_path and os.path.exists(terminal_str_path)
                else None)

    # Combine PRM sections into one file
    prm_lines = []
    for sp in [core_str_path, monomer_str_path, terminal_str_path]:
        if sp and os.path.exists(sp):
            data = parse_cgenff_str(sp)
            prm_lines += data.get('prm_lines', [])
    prm_path = os.path.join(output_dir, 'cgenff_combined.prm')
    with open(prm_path, 'w') as f:
        f.write('* CGenFF combined parameters — UDB v1.0\n*\n\n')
        f.write('BONDS\n')
        for ln in prm_lines:
            f.write(ln + '\n')
        f.write('\nEND\n')

    # Parse the residue name declared in the core and monomer .str files
    # so psfgen can match atoms in the built PDB (which are all labelled UNL)
    def _resi_name(str_path, default):
        if not str_path or not os.path.exists(str_path): return default
        data = parse_cgenff_str(str_path)
        for ln in data.get('rtf_lines', []):
            p = ln.strip().split()
            if p and p[0].upper() == 'RESI' and len(p) >= 2:
                return p[1].upper()
        return default

    core_resi = _resi_name(core_str_path, 'CORE')
    mon_resi  = _resi_name(monomer_str_path, 'MON')

    tcl_path = os.path.join(output_dir, f'{pre}_build_psf.tcl')
    with open(tcl_path, 'w') as f:
        f.write(
            f"# psfgen script — {den_type} G{gen_num} — CGenFF parameterization\n"
            f"# Generated by Universal Dendrimer Builder v1.0\n"
            f"# Requires VMD 1.9.3+ with psfgen plugin\n\n"
            f"package require psfgen\n\n"
            f"# Load molecule-specific CGenFF RTF topology files\n"
            f"topology {core_rtf}\n"
            f"topology {mon_rtf}\n"
        )
        if term_rtf:
            f.write(f"topology {term_rtf}\n")
        f.write(
            f"\n# Alias: map the generic residue name UNL used by UDB\n"
            f"# to the RESI names declared in the CGenFF .str files\n"
            f"pdbalias residue UNL {mon_resi}\n\n"
            f"# Build the full dendrimer as a single segment\n"
            f"# psfgen will use the RTF RESI {mon_resi} for all atoms\n"
            f"segment DEND {{\n"
            f"    first none\n"
            f"    last  none\n"
            f"    pdb {pdb_in}\n"
            f"}}\n"
            f"coordpdb {pdb_in} DEND\n\n"
            f"# Guess missing coordinates (none expected for a built structure)\n"
            f"guesscoord\n\n"
            f"# Write PSF + PDB\n"
            f"writepsf {psf_out}\n"
            f"writepdb {pdb_out}\n\n"
            f"# Solvate in TIP3P water box\n"
            f"package require solvate\n"
            f"solvate {psf_out} {pdb_out} -t {box_pad:.0f} -o {pre}_solv\n\n"
            f"# Neutralize with Na+/Cl-\n"
            f"package require autoionize\n"
            f"autoionize -psf {pre}_solv.psf -pdb {pre}_solv.pdb "
            f"-neutralize -o {pre}_ions\n\n"
            f"puts \"CGenFF psfgen complete — {pre}_ions.psf written\"\n"
            f"exit\n"
        )

    # Write NAMD conf files that use CGenFF parameters
    common = (
        f"# NAMD — {den_type} G{gen_num} — CGenFF\n"
        f"structure      {pre}_ions.psf\n"
        f"coordinates    {pre}_ions.pdb\n"
        f"paraTypeCharmm on\n"
        f"parameters     cgenff_combined.prm\n"
        f"PME yes\nPMEGridSpacing 1.0\n"
        f"cutoff 12.0\nswitchdist 10.0\npairlistdist 14.0\n"
        f"timestep 2.0\nrigidBonds all\n"
        f"nonbondedFreq 1\nfullElectFrequency 2\n"
    )
    stages = [
        ('01_min',  f"minimize 10000\nreinitvels 300\nrun 0\n"),
        ('02_heat', f"langevin on\nlangevinDamping 1.0\nlangevinTemp 300\n"
                    f"langevinHydrogen off\nreassignFreq 500\nreassignTemp 0\n"
                    f"reassignIncr 5\nreassignHold 300\nrun 25000\n"),
        ('03_nvt',  f"langevin on\nlangevinDamping 1.0\nlangevinTemp 300\n"
                    f"langevinHydrogen off\nrun 50000\n"),
        ('04_npt',  f"langevin on\nlangevinDamping 1.0\nlangevinTemp 300\n"
                    f"langevinHydrogen off\nuseGroupPressure yes\n"
                    f"useFlexibleCell no\nuseConstantArea no\n"
                    f"langevinPiston on\nlangevinPistonTarget 1.01325\n"
                    f"langevinPistonPeriod 500\nlangevinPistonDecay 250\n"
                    f"langevinPistonTemp 300\nmargin 5.0\nrun 500000\n"),
        ('05_prod', f"langevin on\nlangevinDamping 1.0\nlangevinTemp 300\n"
                    f"langevinHydrogen off\nuseGroupPressure yes\n"
                    f"useFlexibleCell no\nuseConstantArea no\n"
                    f"langevinPiston on\nlangevinPistonTarget 1.01325\n"
                    f"langevinPistonPeriod 500\nlangevinPistonDecay 250\n"
                    f"langevinPistonTemp 300\nmargin 5.0\n"
                    f"dcdFile {pre}_production.dcd\n"
                    f"dcdFreq 10000\noutputEnergies 10000\nrun 2500000\n"),
    ]
    conf_files = []
    for i, (tag, body) in enumerate(stages):
        conf = os.path.join(output_dir, f'{pre}_{tag}.conf')
        with open(conf, 'w') as f:
            f.write(f"# NAMD stage {tag}\noutputName {pre}_{tag}\n{common}")
            if i > 0:
                prev = stages[i-1][0]
                f.write(f"binCoordinates {pre}_{prev}.restart.coor\n"
                        f"extendedSystem {pre}_{prev}.restart.xsc\n")
            else:
                f.write(f"binCoordinates {pre}_ions.coor\n"
                        f"extendedSystem {pre}_ions.xsc\n")
            f.write(body)
        conf_files.append(os.path.basename(conf))

    # Run-all shell script
    sh_path = os.path.join(output_dir, f'{pre}_run_all.sh')
    with open(sh_path, 'w') as f:
        f.write(
            f"#!/bin/bash\n"
            f"# CGenFF NAMD pipeline — {den_type} G{gen_num}\n"
            f"# Generated by Universal Dendrimer Builder v1.0\n"
            f"set -e\n\n"
            f"# ── Check required executables ──────────────────────────\n"
            f"if ! command -v vmd &>/dev/null; then\n"
            f"    echo 'ERROR: vmd not found on PATH'\n"
            f"    echo 'Install VMD from https://www.ks.uiuc.edu/Research/vmd/'\n"
            f"    echo 'or add VMD to PATH: export PATH=/usr/local/lib/vmd:$PATH'\n"
            f"    exit 1\n"
            f"fi\n"
            f"if ! command -v namd2 &>/dev/null; then\n"
            f"    echo 'ERROR: namd2 not found on PATH'\n"
            f"    echo 'Install NAMD from https://www.ks.uiuc.edu/Research/namd/'\n"
            f"    echo 'or add NAMD to PATH: export PATH=/path/to/namd2:$PATH'\n"
            f"    echo ''\n"
            f"    echo 'Alternatively run each stage manually:'\n"
            f"    for conf in {pre}_01_min.conf {pre}_02_heat.conf "
            f"{pre}_03_nvt.conf {pre}_04_npt.conf {pre}_05_prod.conf; do\n"
            f"        echo \"  namd2 +p8 $conf > ${{conf%.conf}}.log\"\n"
            f"    done\n"
            f"    exit 1\n"
            f"fi\n\n"
            f"# ── Step 1: build PSF via VMD psfgen ────────────────────\n"
            f"echo 'Running psfgen via VMD...'\n"
            f"vmd -dispdev none -e {pre}_build_psf.tcl\n"
            f"echo 'psfgen done'\n\n"
            f"# ── Steps 2-6: NAMD stages ──────────────────────────────\n"
        )
        for tag, _ in stages:
            f.write(
                f"echo 'Running NAMD stage {tag}...'\n"
                f"namd2 +p8 {pre}_{tag}.conf > {pre}_{tag}.log\n"
                f"echo 'Stage {tag} done'\n\n"
            )
        f.write("echo 'CGenFF pipeline complete'\n")
    import stat as _st
    os.chmod(sh_path,
             os.stat(sh_path).st_mode | _st.S_IEXEC | _st.S_IXGRP | _st.S_IXOTH)

    if log_cb:
        log_cb(f"\n  [CGenFF] psfgen script   → {os.path.basename(tcl_path)}", 'ok')
        log_cb(f"  [CGenFF] combined params → cgenff_combined.prm", 'ok')
        log_cb(f"  [CGenFF] NAMD stages     → {', '.join(conf_files)}", 'ok')
        log_cb(f"  [CGenFF] run script      → {os.path.basename(sh_path)}", 'ok')

    return tcl_path


# =============================================================================
# SECTION 16.  MAIN APPLICATION GUI
# =============================================================================

class UniversalDendrimerBuilder:

    def __init__(self, root):
        self.root = root
        self.root.title("Universal Dendrimer Builder  v1.0")
        self.root.geometry("1140x1080")
        self.root.resizable(True, True)

        self._T     = DARK
        self._theme = 'dark'

        # molecule data
        self.core_atoms  = []; self.core_bonds  = []
        self.mon_atoms   = []; self.mon_bonds   = []
        self.term_atoms  = []; self.term_bonds  = []

        # cancel flag shared between GUI and build thread
        self._cancel_flag = [False]
        self._last_built_rg = None

        # ── StringVars ──────────────────────────────
        self.core_file     = tk.StringVar()
        self.monomer_file  = tk.StringVar()
        self.terminal_file = tk.StringVar()
        self.output_dir    = tk.StringVar(value=os.path.expanduser("~/dendrimer_output"))

        self.core_str_file     = tk.StringVar()
        self.monomer_str_file  = tk.StringVar()
        self.terminal_str_file = tk.StringVar()
        self.cgenff_use        = tk.BooleanVar(value=False)

        self.preset_var       = tk.StringVar(value="PAMAM")
        self.den_type_var     = tk.StringVar(value="PAMAM")
        self.bond_len_auto    = tk.BooleanVar(value=True)
        self.bond_len_var     = tk.DoubleVar(value=1.47)
        self.bond_ang_auto    = tk.BooleanVar(value=True)
        self.bond_ang_var     = tk.DoubleVar(value=109.5)
        self.branching_var    = tk.IntVar(value=2)
        self.num_gens_var     = tk.IntVar(value=6)

        self.mon_input_var    = tk.StringVar()
        self.mon_input_h_var  = tk.StringVar()   # kept for backward-compat; driven by labile widget
        self.term_input_var   = tk.StringVar()
        self.term_input_h_var = tk.StringVar()
        self.term_output_var  = tk.StringVar()

        # ── Atom-selection additions (w12.2) ─────────────────────────────
        # Every field defaults to auto / empty, which reproduces the pre-w12.2
        # behaviour exactly (auto-detect the leaving H). Unchecking an
        # "Auto-detect H" box enables manual selection of a non-H leaving atom.
        # #1 Monomer Terminal Leaving Atom(s) — leaving atom(s) on the monomer
        #    terminal growth site (previously the next-gen leaving H was always
        #    auto). A multi-atom manager (built in section 04b) holds the names.
        self.mon_term_leave_auto = tk.BooleanVar(value=True)
        # Auto-detect-H toggles for the existing monomer/core leaving selections
        self.mon_leave_auto      = tk.BooleanVar(value=True)
        self.core_leave_auto     = tk.BooleanVar(value=True)
        # #2/#3 Terminal Group binding + leaving atom (separate from Section 05
        #    Terminal Unit; when set they take precedence for the terminal-cap bond)
        self.tgroup_bind_var     = tk.StringVar()
        self.tgroup_leave_var    = tk.StringVar()
        self.tgroup_leave_auto   = tk.BooleanVar(value=True)

        # 'single' : one atom (H or any element) selected via combobox
        # 'group'  : multiple atoms removed together via FragmentManager
        self.labile_mode_var  = tk.StringVar(value='single')
        self._labile_input_mgr = None   # FragmentManager for group mode

        self.core_check_vars  = []
        self.gen_check_vars   = []

        self.net_charge_var   = tk.IntVar(value=0)

        # clash threshold
        self.clash_thresh_var = tk.DoubleVar(value=1.0)

        # output formats
        self.fmt_mol2 = tk.BooleanVar(value=True)
        self.fmt_pdb  = tk.BooleanVar(value=True)
        self.fmt_xyz  = tk.BooleanVar(value=False)
        self.fmt_gro  = tk.BooleanVar(value=False)
        self.fmt_psf  = tk.BooleanVar(value=False)  

        # Parameterization method: full_antechamber or fragment_bcc
        self.param_method_var = tk.StringVar(value="fragment_bcc")

        # AmberTools path settings
        self.amber_conda_env  = tk.StringVar(value="AmberTools25")
        self.amber_conda_path = tk.StringVar(value=os.path.expanduser("~/miniconda3"))

        # AMBER pipeline
        self.pipe_save_mol2    = tk.BooleanVar(value=True)
        self.pipe_save_pdb     = tk.BooleanVar(value=True)
        self.pipe_save_script  = tk.BooleanVar(value=True)
        self.pipe_antechamber  = tk.BooleanVar(value=False)
        self.pipe_parmchk2     = tk.BooleanVar(value=False)
        self.pipe_tleap        = tk.BooleanVar(value=False)
        self.pipe_minimize     = tk.BooleanVar(value=False)
        self.pipe_ambpdb       = tk.BooleanVar(value=False)
        self.pipe_solvate      = tk.BooleanVar(value=False)
        self.pipe_add_ions     = tk.BooleanVar(value=False)
        self.pipe_md           = tk.BooleanVar(value=False)
        self.pipe_box_size     = tk.DoubleVar(value=12.0)
        self.pipe_prod_ns      = tk.DoubleVar(value=5.0)
        self.ff_var            = tk.StringVar(value="gaff")
        self.charge_method_var = tk.StringVar(value="bcc")
        self.pipe_ncpus        = tk.IntVar(value=1)   

        # NAMD 3.x pipeline — AMBER/GAFF2 (tLeap → NAMD)
        self.namd_solvate    = tk.BooleanVar(value=True)
        self.namd_minimize   = tk.BooleanVar(value=True)
        self.namd_heat       = tk.BooleanVar(value=True)
        self.namd_equil      = tk.BooleanVar(value=True)
        self.namd_prod       = tk.BooleanVar(value=True)
        self.namd_ff_var     = tk.StringVar(value="GAFF2")
        self.namd_water_var  = tk.StringVar(value="TIP3P")
        self.namd_box_var    = tk.DoubleVar(value=15.0)
        self.namd_conc_var   = tk.DoubleVar(value=0.15)
        self.namd_prod_ns    = tk.DoubleVar(value=10.0)
        self.namd_equil_ns   = tk.DoubleVar(value=2.0)
        self.namd_heat_ps    = tk.DoubleVar(value=200.0)
        self.namd_timestep   = tk.DoubleVar(value=2.0)
        self.namd_cores_var  = tk.IntVar(value=8)
        self.namd_binary_var = tk.StringVar(value=NAMD_DEFAULT_BIN)
        self.namd_dcd_freq   = tk.IntVar(value=2500)
        self.namd_temp_var   = tk.DoubleVar(value=300.0)
        self.namd_psf_gen    = tk.BooleanVar(value=False)
        self.namd_equil_nvt  = tk.BooleanVar(value=False)
        self.namd_equil_npt  = tk.BooleanVar(value=False)
        self.namd_gpu_var    = tk.BooleanVar(value=False)

        # OpenBabel
        self.obabel_minimize  = tk.BooleanVar(value=False)
        self.obabel_ff_var    = tk.StringVar(value="MMFF94")
        self.obabel_steps_var = tk.IntVar(value=2500)

        # Intergenerational minimization (v1.0)
        self.intergen_enabled   = tk.BooleanVar(value=False)
        self.intergen_use_amber = tk.BooleanVar(value=True)
        self.intergen_use_namd  = tk.BooleanVar(value=False)
        self.intergen_namd_mode = tk.StringVar(value='vacuum')  # 'vacuum' | 'solvated'
        self.intergen_save_gaff = tk.BooleanVar(value=True)

        # live build data — preview and validator accumulate here
        self._live_preview_rows  = []   # list of dicts per gen
        self._live_val_rows      = []   # list of (gen, label, ok, got, expected)

        self._build_ui()
        self._apply_preset("PAMAM")

    def _toggle_theme(self):
        self._theme='light' if self._theme=='dark' else 'dark'
        self._T=LIGHT if self._theme=='light' else DARK
        self._retheme_all()

    def _retheme_all(self):
        T=self._T
        self.root.configure(bg=T['bg'])
        self._retheme_widget(self.root,T)
        self.log_box.config(bg=T['log_bg'],fg=T['log_fg'],
                            insertbackground=T['log_fg'])
        for tag,col in [('warn',T['yellow']),('err',T['red']),
                        ('ok',T['green']),('head',T['cyan']),('info',T['yellow'])]:
            self.log_box.tag_config(tag,foreground=col)
        s=ttk.Style()
        s.configure('TProgressbar',
                    background=T['prog'],
                    troughcolor=T['prog_tr'],
                    bordercolor=T['border'],
                    lightcolor=T['prog'],
                    darkcolor=T['prog'],
                    thickness=14)
        s.configure('TCombobox',fieldbackground=T['entry'],background=T['sbtn'],
                    foreground=T['cyan'],selectbackground=T['sel'])
        s.configure('Vertical.TScrollbar',background=T['sbtn'],
                    troughcolor=T['bg'],arrowcolor=T['cyan'])
        self._theme_btn.config(text=T['toggle_icon'],bg=T['sbtn'],fg=T['cyan'])

    def _retheme_widget(self,w,T):
        cls=w.winfo_class()
        try:
            if cls in ('Frame','Canvas'): w.config(bg=T['panel'] if w.master and w.master.winfo_class()!='Tk' else T['bg'])
            elif cls=='Label':
                cur_bg=w.cget('bg')
                rev={v:k for k,v in (LIGHT if self._theme=='dark' else DARK).items()
                     if isinstance(v,str) and v.startswith('#')}
                w.config(bg=T.get(rev.get(cur_bg,''),cur_bg))
            elif cls=='Button':
                for attr in ('bg','activebackground'):
                    cur=w.cget(attr)
                    rev={v:k for k,v in (LIGHT if self._theme=='dark' else DARK).items()
                         if isinstance(v,str) and v.startswith('#')}
                    w.config(**{attr:T.get(rev.get(cur,''),cur)})
            elif cls=='Entry':
                w.config(bg=T['entry'],fg=T['cyan'],insertbackground=T['cyan'])
            elif cls=='Checkbutton':
                w.config(selectcolor=T['sel'],activebackground=T['panel'])
            elif cls=='Radiobutton':
                w.config(selectcolor=T['panel'],activebackground=T['panel'])
            elif cls=='Spinbox':
                w.config(bg=T['entry'],fg=T['cyan'])
        except Exception: pass
        for child in w.winfo_children():
            self._retheme_widget(child,T)

    def _section(self,parent,title,color=None):
        T=self._T; c=color or T['cyan']
        outer=tk.Frame(parent,bg=T['border'])
        outer.pack(fill='x',padx=12,pady=5)
        inner=tk.Frame(outer,bg=T['panel'])
        inner.pack(fill='x',padx=1,pady=1)
        tk.Label(inner,text=f"  {title}",font=BOLD,fg=c,
                 bg=T['panel'],anchor='w').pack(fill='x',padx=8,pady=(8,4))
        tk.Frame(inner,bg=T['border'],height=1).pack(fill='x',padx=8,pady=(0,6))
        return inner

    def _file_row(self,parent,label,var,cmd,optional=False):
        T=self._T
        r=tk.Frame(parent,bg=T['panel']); r.pack(fill='x',padx=12,pady=3)
        tk.Label(r,text=label,width=26,anchor='w',font=FONT,
                 fg=T['muted'] if optional else T['white'],
                 bg=T['panel']).pack(side='left')
        _mk_entry(r,var,T,width=40).pack(side='left',fill='x',expand=True,padx=(0,8))
        _mk_btn(r,"Browse",cmd,T).pack(side='left')

    def _build_ui(self):
        T=self._T; self.root.configure(bg=T['bg'])

        hdr=tk.Frame(self.root,bg=T['bg']); hdr.pack(fill='x')
        title_row=tk.Frame(hdr,bg=T['bg']); title_row.pack(fill='x')
        tk.Label(title_row,text="  UNIVERSAL DENDRIMER BUILDER",
                 font=("Courier New",18,"bold"),
                 fg=T['cyan'],bg=T['bg']).pack(side='left',padx=(20,8),pady=(16,2))
        # designer credit
        credit_frame=tk.Frame(title_row,bg=T['bg'])
        credit_frame.pack(side='right',padx=(0,20),pady=(18,2))
        tk.Label(credit_frame,text="Designed by  Hirusha Walgampaya",
                 font=("Courier New",10),fg=T['muted'],bg=T['bg']).pack(side='top',anchor='e')
        btn_row=tk.Frame(credit_frame,bg=T['bg']); btn_row.pack(side='top',anchor='e',pady=(4,0))
        _mk_btn(btn_row,"Clear All",self._clear_all_fields,T,
                bg_key='red',fg_key='bg').pack(side='right',padx=(6,0))
        self._theme_btn=tk.Button(btn_row,text=T['toggle_icon'],
                                   command=self._toggle_theme,
                                   font=("Courier New",13,"bold"),
                                   bg=T['sbtn'],fg=T['cyan'],
                                   relief='flat',padx=8,pady=4,cursor='hand2',
                                   activebackground=T['border'],bd=0)
        self._theme_btn.pack(side='right')
        tk.Label(hdr,text="PAMAM  PPI  Triazine  Carbosilane  PLL  Phosphorus  Polyester  Polyglycerol  Custom      v1.0",
                 font=("Courier New",11),fg=T['muted'],bg=T['bg']).pack(pady=(0,4))
        tk.Frame(hdr,bg=T['border'],height=1).pack(fill='x',padx=20,pady=4)

        # PRESET BAR  — two rows for 9 presets
        pre=tk.Frame(self.root,bg=T['bg']); pre.pack(fill='x',padx=20,pady=6)
        tk.Label(pre,text="Preset:",font=BOLD,fg=T['yellow'],bg=T['bg']).pack(side='left',padx=(0,8))
        row1=tk.Frame(pre,bg=T['bg']); row1.pack(side='left')
        for name in PRESETS:
            tk.Radiobutton(row1,text=name,variable=self.preset_var,value=name,
                           command=lambda n=name: self._apply_preset(n),
                           font=FONT,fg=T['yellow'] if name!="Custom" else T['muted'],
                           bg=T['bg'],selectcolor=T['bg'],activebackground=T['bg'],
                           indicatoron=0,relief='flat',padx=8,pady=4,cursor='hand2'
                           ).pack(side='left',padx=2)
        self.preset_desc=tk.Label(pre,text=PRESETS["PAMAM"]["description"],
                                   font=("Courier New",10),fg=T['muted'],bg=T['bg'])
        self.preset_desc.pack(side='left',padx=12)

        # SCROLLABLE BODY  with smart scroll
        canvas=tk.Canvas(self.root,bg=T['bg'],highlightthickness=0)
        vsb=ttk.Scrollbar(self.root,orient='vertical',command=canvas.yview)
        self.sf=tk.Frame(canvas,bg=T['bg'])
        self.sf.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0),window=self.sf,anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left',fill='both',expand=True,padx=(16,0),pady=4)
        vsb.pack(side='right',fill='y')
        self._main_canvas = canvas

        # Smart scroll — fixed in w11:
        # scroll inside viewer = matplotlib zoom, outside = page scroll
        def _smart_scroll(event):
            core_hov = hasattr(self,'_core_viewer') and self._core_viewer.is_hovered()
            mon_hov  = hasattr(self,'_mon_viewer')  and self._mon_viewer.is_hovered()
            if core_hov or mon_hov:
                # Forward to the hovered viewer's matplotlib canvas as zoom
                viewer = self._core_viewer if core_hov else self._mon_viewer
                ax = viewer.ax
                # Get current limits
                xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
                xmid = sum(xlim)/2; ymid = sum(ylim)/2; zmid = sum(zlim)/2
                xrng = (xlim[1]-xlim[0])/2; yrng = (ylim[1]-ylim[0])/2; zrng = (zlim[1]-zlim[0])/2
                # Zoom in = shrink range, zoom out = expand range
                if event.num == 4 or (hasattr(event,'delta') and event.delta > 0):
                    factor = 0.85   # zoom in
                else:
                    factor = 1.15   # zoom out
                ax.set_xlim3d([xmid - xrng*factor, xmid + xrng*factor])
                ax.set_ylim3d([ymid - yrng*factor, ymid + yrng*factor])
                ax.set_zlim3d([zmid - zrng*factor, zmid + zrng*factor])
                viewer.canvas.draw_idle()
                return
            # Outside viewers — scroll the page
            if event.num == 4 or (hasattr(event,'delta') and event.delta > 0):
                canvas.yview_scroll(-1, "units")
            else:
                canvas.yview_scroll(1, "units")

        self.root.bind_all("<MouseWheel>", _smart_scroll)
        self.root.bind_all("<Button-4>",   _smart_scroll)
        self.root.bind_all("<Button-5>",   _smart_scroll)

        F=self.sf
        self._sec1_files(F)
        self._sec_viewer(F)        # [C] Section 02 — viewer right after files
        self._sec2_core_atoms(F)
        self._sec3_monomer_atoms(F)
        self._sec3a_mon_term_leave(F)
        self._sec3b_terminal_atoms(F)
        self._sec3c_terminal_unit_bond(F)
        self._sec4_parameters(F)
        self._sec5_gen_selector(F)
        self._sec6_pipeline(F)
        self._sec7_output(F)
        self._sec8_build(F)
        self._sec_gen_preview(F)
        self._sec_validator(F)
        self._sec9_log(F)

    def _sec1_files(self,F):
        T=self._T; sec=self._section(F,"01  MOLECULE FILES")
        tk.Label(sec,text="  Load the core and monomer mol2 files. Terminal unit is optional and replaces the monomer at the outermost generation.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))
        self._file_row(sec,"Core mol2  *required",self.core_file,
                       lambda: self._load_mol2('core'))
        self._file_row(sec,"Monomer mol2  *required",self.monomer_file,
                       lambda: self._load_mol2('monomer'))
        self._file_row(sec,"Terminal mol2  (optional)",self.terminal_file,
                       lambda: self._load_mol2('terminal'),optional=True)

        # ── CGenFF topology + parameter files ──────────────
        tk.Frame(sec,bg=T['border'],height=1).pack(fill='x',padx=12,pady=(10,0))
        cg_hdr=tk.Frame(sec,bg=T['panel']); cg_hdr.pack(fill='x',padx=12,pady=(6,2))
        tk.Label(cg_hdr,text="  CGenFF Topology + Parameter Files",
                 font=BOLD,fg=T['yellow'],bg=T['panel']).pack(side='left')
        tk.Checkbutton(cg_hdr,text="  Use CGenFF for NAMD pipeline",
                       variable=self.cgenff_use,font=FONT,
                       fg=T['cyan'],bg=T['panel'],selectcolor=T['sel'],
                       activebackground=T['panel'],cursor='hand2').pack(side='left',padx=16)
        tk.Label(sec,
                 text="  Optional. Load the .str files from ParamChem (cgenff.umaryland.edu) for each fragment.\n"
                      "  When loaded, the NAMD pipeline uses molecule-specific CGenFF RTF + parameters\n"
                      "  instead of the generic CHARMM36 topology. The building algorithm is unaffected.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],
                 wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,4))

        self._file_row(sec,"Core .str  (CGenFF)",self.core_str_file,
                       lambda: self._load_cgenff_str('core'),optional=True)
        self._file_row(sec,"Monomer .str  (CGenFF)",self.monomer_str_file,
                       lambda: self._load_cgenff_str('monomer'),optional=True)
        self._file_row(sec,"Terminal .str  (CGenFF, optional)",self.terminal_str_file,
                       lambda: self._load_cgenff_str('terminal'),optional=True)

        # penalty score display + validate button
        cg_row=tk.Frame(sec,bg=T['panel']); cg_row.pack(fill='x',padx=12,pady=(4,8))
        _mk_btn(cg_row,"Validate CGenFF Files",self._validate_cgenff,T,
                fg_key='yellow').pack(side='left',padx=(0,12))
        self._cgenff_status_lbl=tk.Label(cg_row,
            text="Load .str files then click Validate to check penalty scores.",
            font=("Courier New",10),fg=T['muted'],bg=T['panel'])
        self._cgenff_status_lbl.pack(side='left')

    def _load_cgenff_str(self, which):
        """Browse for a CGenFF .str file and store the path."""
        path = filedialog.askopenfilename(
            title=f"Select {which} CGenFF .str file",
            filetypes=[("CGenFF stream","*.str"),("All files","*.*")])
        if not path: return
        if which == 'core':
            self.core_str_file.set(path)
        elif which == 'monomer':
            self.monomer_str_file.set(path)
        elif which == 'terminal':
            self.terminal_str_file.set(path)
        self._log(f"  [CGenFF] Loaded {which} .str: {os.path.basename(path)}",'ok')
        # auto-validate immediately
        self._validate_cgenff()

    def _validate_cgenff(self):
        """
        [w11.3] Parse all loaded .str files, collect penalty scores,
        and display a summary in the status label and build log.
        Warns on any penalty > 50 (poorly fitted parameter).
        """
        T = self._T
        files = {
            'core':     self.core_str_file.get(),
            'monomer':  self.monomer_str_file.get(),
            'terminal': self.terminal_str_file.get(),
        }
        loaded = {k: v for k, v in files.items() if v and os.path.exists(v)}
        if not loaded:
            self._cgenff_status_lbl.config(
                text="No .str files loaded yet.",fg=T['muted'])
            return

        self._log(f"\n  [CGenFF] Validating {len(loaded)} .str file(s)...",'head')
        total_warn = 0; max_penalty = 0.0; summary_parts = []

        for frag, path in loaded.items():
            data = parse_cgenff_str(path)
            if not data['ok']:
                self._log(f"  [CGenFF] {frag}: ERROR — {data['error']}",'err')
                continue
            n_atoms   = len(data['atoms'])
            n_params  = len(data['penalties'])
            max_p     = data['max_penalty']
            warn_c    = data['warn_count']
            total_warn += warn_c
            if max_p > max_penalty: max_penalty = max_p

            status = 'ok' if warn_c == 0 else 'warn'
            self._log(
                f"  [CGenFF] {frag:8s}  atoms={n_atoms}  "
                f"params={n_params}  max_penalty={max_p:.1f}  "
                f"high_penalty(>50)={warn_c}", status)

            # log any high-penalty parameters individually
            for sec, key, score in data['penalties']:
                if score > 50.0:
                    self._log(
                        f"    !! {sec:<12} {key:<30} penalty={score:.1f}"
                        f"  → QM refinement recommended", 'warn')

            summary_parts.append(
                f"{frag}: max={max_p:.0f}"
                + (f" [{warn_c} high]" if warn_c else " [OK]"))

        summary = "  |  ".join(summary_parts)
        if total_warn == 0:
            color = T['green']
            label = f"All penalties OK  —  {summary}"
        else:
            color = T['yellow']
            label = f"{total_warn} high-penalty params (>50) — {summary}"

        self._cgenff_status_lbl.config(text=label, fg=color)
        self._log(f"  [CGenFF] Validation complete. {label}",
                  'ok' if total_warn == 0 else 'warn')

    def _sec2_core_atoms(self,F):
        T=self._T; sec=self._section(F,"03  CORE BINDING ATOMS",T['yellow'])

        cr=tk.Frame(sec,bg=T['panel']); cr.pack(fill='x',padx=12,pady=(4,2))
        self.core_counter_lbl=tk.Label(cr,text="Selected: 0 of 0 atoms",
                                        font=BOLD,fg=T['yellow'],bg=T['panel'])
        self.core_counter_lbl.pack(side='left')
        tk.Label(cr,text="  — updates as you tick checkboxes",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left',padx=8)

        tk.Label(sec,text="  Select atoms that will bond to monomers.\n"
                          "  H atoms on selected atoms are auto-removed before bonding.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,4))

        clr=tk.Frame(sec,bg=T['panel']); clr.pack(fill='x',padx=12,pady=(0,2))
        tk.Checkbutton(clr,text="Auto-detect H (core leaving atom)",variable=self.core_leave_auto,
                       font=("Courier New",10),fg=T['green'],bg=T['panel'],
                       selectcolor=T['panel'],activebackground=T['panel']).pack(side='left')
        tk.Label(clr,text="  untick to remove a non-H leaving atom on binding sites (by name, via Labile).",
                 font=("Courier New",9),fg=T['muted'],bg=T['panel']).pack(side='left')

        self.core_atom_frame=tk.Frame(sec,bg=T['panel'])
        self.core_atom_frame.pack(fill='x',padx=12,pady=6)
        tk.Label(self.core_atom_frame,text="  <- Load core mol2 to see atoms",
                 font=("Courier New",10),fg=T['border'],bg=T['panel']).pack(anchor='w')

        filt=tk.Frame(sec,bg=T['panel']); filt.pack(fill='x',padx=12,pady=(0,4))
        tk.Label(filt,text="Filter by element:",font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
        self.core_filter_var=tk.StringVar()
        _mk_entry(filt,self.core_filter_var,T,width=10).pack(side='left',padx=6)
        _mk_btn(filt,"Apply",self._filter_core_atoms,T).pack(side='left',padx=4)
        _mk_btn(filt,"Select All",self._select_all_core,T,fg_key='green').pack(side='left',padx=4)
        _mk_btn(filt,"Clear All",self._clear_all_core,T,fg_key='red').pack(side='left',padx=4)
        tk.Label(sec,text="  Tip: click any atom in the 3D viewer above to tick its checkbox instantly.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,8))

    def _on_core_mgr_changed(self):
        self._update_viewer_roles()

    def _sec3_monomer_atoms(self,F):
        T=self._T; sec=self._section(F,"04  MONOMER BONDING ATOMS",T['green'])
        tk.Label(sec,text="  Load the monomer mol2 file. Set the input bonding atom (attaches to core) and the terminal bonding atoms (grow next generation).",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))

        def combo_row(label,var,attr,on_change=None):
            r=tk.Frame(sec,bg=T['panel']); r.pack(fill='x',padx=12,pady=4)
            tk.Label(r,text=label,width=46,anchor='w',font=FONT,
                     fg=T['white'],bg=T['panel']).pack(side='left')
            cb=ttk.Combobox(r,textvariable=var,state='readonly',width=22,font=FONT)
            cb.pack(side='left')
            if on_change: cb.bind("<<ComboboxSelected>>",on_change)
            setattr(self,attr,cb)

        combo_row("Focal Point  (bonds to previous generation layer)",
                  self.mon_input_var,'mon_input_cb',on_change=self._on_mon_input_changed)

        labile_outer = tk.Frame(sec, bg=T['panel'])
        labile_outer.pack(fill='x', padx=12, pady=(2,0))
        self._labile_outer = labile_outer

        # header + radio row
        hdr_row = tk.Frame(labile_outer, bg=T['panel'])
        hdr_row.pack(fill='x', pady=(0,2))
        tk.Label(hdr_row,
                 text="Labile Atom(s)  (removed before bonding):",
                 width=46, anchor='w', font=FONT,
                 fg=T['white'], bg=T['panel']).pack(side='left')
        for val, txt in [('single','Single Atom'), ('group','Group (OH, NH2 …)')]:
            tk.Radiobutton(hdr_row, text=txt, variable=self.labile_mode_var,
                           value=val, font=("Courier New",10),
                           fg=T['yellow'], bg=T['panel'],
                           selectcolor=T['panel'], activebackground=T['panel'],
                           command=self._on_labile_mode_changed
                           ).pack(side='left', padx=(0,14))

        # ── SINGLE mode frame ────────────────────────────────────────────
        self._labile_single_frame = tk.Frame(labile_outer, bg=T['panel'])
        sr = tk.Frame(self._labile_single_frame, bg=T['panel'])
        sr.pack(fill='x')
        tk.Label(sr, text="  Atom to remove:", font=("Courier New",10),
                 fg=T['muted'], bg=T['panel']).pack(side='left')
        self.mon_input_h_cb = ttk.Combobox(sr, textvariable=self.mon_input_h_var,
                                            state='disabled', width=26, font=FONT)
        self.mon_input_h_cb.pack(side='left', padx=6)
        self.mon_input_h_cb.bind("<<ComboboxSelected>>",
                                 lambda e: self._update_viewer_roles())
        tk.Checkbutton(sr, text="Auto-detect H", variable=self.mon_leave_auto,
                       font=("Courier New",9), fg=T['green'], bg=T['panel'],
                       selectcolor=T['panel'], activebackground=T['panel'],
                       command=self._on_mon_leave_auto_changed).pack(side='left', padx=(6,0))
        tk.Label(sr, text="(H auto, or untick to pick Cl, Br, O …)",
                 font=("Courier New",9), fg=T['muted'], bg=T['panel']).pack(side='left')
        self._labile_single_frame.pack(fill='x', pady=(0,4))

        # ── GROUP mode frame ─────────────────────────────────────────────
        self._labile_group_frame = tk.Frame(labile_outer, bg=T['panel'])
        self._labile_input_mgr = FragmentManager(
            self._labile_group_frame, T,
            "Labile atoms  (add in order: first atom is anchor-bonded)",
            on_change=self._update_viewer_roles,
            atom_source=lambda: self.mon_atoms)
        self._labile_input_mgr.pack(fill='x')
        # Group frame hidden initially; shown when radio = 'group'
        self._labile_group_frame.pack_forget()

        # multiple terminal sites manager
        tk.Frame(sec,bg=T['border'],height=1).pack(fill='x',padx=12,pady=(6,0))
        tk.Label(sec,text="  Terminal Atom Manager  (add all terminal atom names for this monomer):",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(4,0))
        self._term_mgr=FragmentManager(sec,T,"Terminal atoms",
                                        on_change=self._on_term_mgr_changed,
                                        atom_source=lambda: self.mon_atoms)
        self._term_mgr.pack(fill='x',padx=12,pady=(0,4))
        tk.Label(sec,text="  Leaving atom for these terminal sites is set in the next section.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,8))

    def _sec3a_mon_term_leave(self,F):
        """04b — leaving atom(s) removed from each monomer terminal growth site when
        the next generation attaches. Default: auto-detect H. Add one or more atoms
        (matched by element, like the Section 04 terminal manager) to shed non-H
        leaving atoms instead."""
        T=self._T; sec=self._section(F,"04b  MONOMER TERMINAL LEAVING ATOM",T['green'])
        tk.Label(sec,text="  The atom(s) removed from each terminal atom (Section 04) when the next\n"
                          "  generation's monomer bonds onto it. Auto-detect H handles the usual\n"
                          "  case; untick and add one or more non-H leaving atoms (Cl, Br, O …).\n"
                          "  Atoms are matched by element, so every terminal site sheds the same species.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],
                 wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))
        chk = tk.Frame(sec, bg=T['panel']); chk.pack(fill='x', padx=12, pady=(2,2))
        tk.Checkbutton(chk, text="Auto-detect H", variable=self.mon_term_leave_auto,
                       font=("Courier New",9), fg=T['green'], bg=T['panel'],
                       selectcolor=T['panel'], activebackground=T['panel'],
                       command=self._on_mon_term_leave_auto_changed).pack(side='left')
        tk.Label(chk, text="(H auto, or untick to add Cl, Br, O …)",
                 font=("Courier New",9), fg=T['muted'], bg=T['panel']).pack(side='left', padx=(6,0))
        # multi-atom manager (mirrors the Section 04 terminal manager); shown only
        # when auto-detect-H is off (handled by _on_mon_term_leave_auto_changed).
        self._mon_term_leave_frame = tk.Frame(sec, bg=T['panel'])
        self._mon_term_leave_mgr = FragmentManager(
            self._mon_term_leave_frame, T,
            "Terminal leaving atoms  (add one or more; matched by element)",
            on_change=self._update_viewer_roles,
            atom_source=lambda: self.mon_atoms)
        self._mon_term_leave_mgr.pack(fill='x', padx=12, pady=(0,4))

    def _on_mon_leave_auto_changed(self):
        """Gate manual monomer-leaving selection. GUI-only: the build already
        uses the labile mechanism (auto-H fallback), so this only toggles editing
        — never clobbers a loaded/selected value."""
        self.mon_input_h_cb.config(
            state='disabled' if self.mon_leave_auto.get() else 'readonly')
        self._update_viewer_roles()

    def _on_mon_term_leave_auto_changed(self):
        """Show the manual multi-atom manager only when auto-detect-H is off."""
        if hasattr(self, '_mon_term_leave_frame'):
            if self.mon_term_leave_auto.get():
                self._mon_term_leave_frame.pack_forget()
            else:
                self._mon_term_leave_frame.pack(fill='x', padx=12, pady=(0,4))
        self._update_viewer_roles()

    def _on_term_mgr_changed(self):
        self._update_viewer_roles()

    def _on_labile_mode_changed(self):
        """[w11.5] Toggle single-atom combobox vs group FragmentManager."""
        if self.labile_mode_var.get() == 'single':
            self._labile_group_frame.pack_forget()
            self._labile_single_frame.pack(fill='x', pady=(0,4))
        else:
            self._labile_single_frame.pack_forget()
            self._labile_group_frame.pack(fill='x', pady=(0,4))
        self._update_viewer_roles()

    def _on_mon_input_changed(self,event=None):
        if not self.mon_atoms: return
        sel=self.mon_input_var.get()
        if not sel: return
        try: idx=int(sel.split(':')[0])
        except (ValueError,IndexError): return
        if self.labile_mode_var.get() == 'single':
            h_nbs=get_h_neighbors(self.mon_atoms,self.mon_bonds,idx)
            opts=self.mon_input_h_cb['values']
            if h_nbs:
                match=next((o for o in opts if o.startswith(f"{h_nbs[0]['idx']}:")),None)
                if match: self.mon_input_h_var.set(match)
        self._update_viewer_roles()

    def _sec3b_terminal_atoms(self,F):
        T=self._T
        self._term_outer=tk.Frame(F,bg=T['border'])
        inner=tk.Frame(self._term_outer,bg=T['panel'])
        inner.pack(fill='x',padx=1,pady=1)
        tk.Label(inner,text="  05  TERMINAL UNIT BONDING ATOMS",
                 font=BOLD,fg=T['green'],bg=T['panel'],anchor='w').pack(fill='x',padx=8,pady=(8,4))
        tk.Frame(inner,bg=T['border'],height=1).pack(fill='x',padx=8,pady=(0,6))
        tk.Label(inner,text="  Defines how the terminal mol2 bonds onto the outermost monomer layer.\n"
                            "  'Input atom' bonds to the monomer terminal atom; one H removed each side.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,4))

        def combo_row(label,var,attr):
            r=tk.Frame(inner,bg=T['panel']); r.pack(fill='x',padx=12,pady=4)
            tk.Label(r,text=label,width=52,anchor='w',font=FONT,
                     fg=T['white'],bg=T['panel']).pack(side='left')
            cb=ttk.Combobox(r,textvariable=var,state='readonly',width=22,font=FONT)
            cb.pack(side='left'); setattr(self,attr,cb)

        combo_row("Focal Point of Terminal Unit  (bonds to peripheral group)",self.term_input_var,'term_input_cb')
        combo_row("Labile Hydrogen (Terminal)  (stripped for coupling)",self.term_input_h_var,'term_input_h_cb')
        combo_row("Peripheral Functional Group  (surface, informational)",self.term_output_var,'term_output_cb')
        tk.Label(inner,text="  Falls back to monomer input/H settings if not set.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,8))

    def _sec3c_terminal_unit_bond(self,F):
        """05b — terminal-unit bond & leaving atom, standalone and ALWAYS visible
        (does not depend on _term_outer, which only appears after a terminal mol2
        loads). Combos populate from the terminal file once it is loaded."""
        T=self._T; sec=self._section(F,"05b  TERMINAL UNIT — BOND & LEAVING ATOM",T['cyan'])
        tk.Label(sec,text="  Atoms on the terminal unit used to form the bond with the monomer\n"
                          "  terminal atom. Bond atom = the terminal-unit atom that connects to the\n"
                          "  monomer terminal atom; leaving atom = the atom removed to make that bond.\n"
                          "  When set, these take precedence over the Terminal Unit fields (Section 05).",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],
                 wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,4))

        rb=tk.Frame(sec,bg=T['panel']); rb.pack(fill='x',padx=12,pady=4)
        tk.Label(rb,text="Bond Atom  (bonds to monomer terminal atom)",width=52,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        self.tgroup_bind_cb=ttk.Combobox(rb,textvariable=self.tgroup_bind_var,
                                          state='readonly',width=22,font=FONT)
        self.tgroup_bind_cb.pack(side='left')

        rl=tk.Frame(sec,bg=T['panel']); rl.pack(fill='x',padx=12,pady=4)
        tk.Label(rl,text="Leaving Atom  (removed to form the bond)",width=52,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        self.tgroup_leave_cb=ttk.Combobox(rl,textvariable=self.tgroup_leave_var,
                                          state='disabled',width=22,font=FONT)
        self.tgroup_leave_cb.pack(side='left')
        tk.Checkbutton(rl,text="Auto-detect H",variable=self.tgroup_leave_auto,
                       font=("Courier New",9),fg=T['green'],bg=T['panel'],
                       selectcolor=T['panel'],activebackground=T['panel'],
                       command=self._on_tgroup_leave_auto_changed).pack(side='left',padx=(6,0))
        tk.Label(sec,text="  Options populate once a terminal mol2 is loaded. Leave both empty to use\n"
                          "  the Terminal Unit settings (Section 05) — the default.",
                 font=("Courier New",9),fg=T['muted'],bg=T['panel'],
                 wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,8))

    def _on_tgroup_leave_auto_changed(self):
        """Enable/disable manual terminal-group-leaving selection."""
        self.tgroup_leave_cb.config(
            state='disabled' if self.tgroup_leave_auto.get() else 'readonly')

    def _populate_term_combos(self):
        amap_term = {a['idx']:a for a in self.term_atoms}
        opts=[f"{a['idx']}:{a['name']}  [{a['type']}]" for a in self.term_atoms]
        h_opts=[o for o in opts if
                amap_term.get(int(o.split(':')[0])) and (
                amap_term[int(o.split(':')[0])]['name'].upper().startswith('H') or
                amap_term[int(o.split(':')[0])]['type'].lower().startswith('h'))]
        for cb,choices in [(self.term_input_cb,opts),
                           (self.term_input_h_cb,h_opts),
                           (self.term_output_cb,opts),
                           (self.tgroup_bind_cb,opts),
                           (self.tgroup_leave_cb,opts)]:
            cb['values']=choices
        c_atoms=[o for o in opts if amap_term.get(int(o.split(':')[0])) and
                 amap_term[int(o.split(':')[0])]['name'].upper().startswith('C')]
        n_atoms=[o for o in opts if amap_term.get(int(o.split(':')[0])) and
                 amap_term[int(o.split(':')[0])]['name'].upper().startswith('N')]
        if c_atoms: self.term_input_var.set(c_atoms[0])
        if h_opts:  self.term_input_h_var.set(h_opts[0])
        if n_atoms: self.term_output_var.set(n_atoms[0])
        self._term_outer.pack(fill='x',padx=12,pady=5)

    def _sec4_parameters(self,F):
        T=self._T; sec=self._section(F,"06  BUILD PARAMETERS  (Dendrimer Architecture)")
        tk.Label(sec,text="  Optional. Set which atoms on the terminal unit will bond outward. Leave empty to reuse monomer terminal atoms.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))
        tk.Label(sec,text="  Set branching factor, bond geometry, number of generations, charge method, and clash detection threshold.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))

        def param_row(label,auto_var,val_var,unit=""):
            r=tk.Frame(sec,bg=T['panel']); r.pack(fill='x',padx=12,pady=5)
            tk.Label(r,text=label,width=30,anchor='w',font=FONT,
                     fg=T['white'],bg=T['panel']).pack(side='left')
            entry=tk.Entry(r,textvariable=val_var,font=FONT,width=10,
                           bg=T['entry'],fg=T['cyan'],insertbackground=T['cyan'],
                           relief='flat',bd=4,state='disabled',
                           disabledbackground=T['sbtn'],disabledforeground=T['border'])
            tk.Radiobutton(r,text="Auto",variable=auto_var,value=True,
                           font=FONT,fg=T['green'],bg=T['panel'],selectcolor=T['panel'],
                           activebackground=T['panel'],
                           command=lambda: entry.config(state='disabled')).pack(side='left')
            tk.Radiobutton(r,text="Manual",variable=auto_var,value=False,
                           font=FONT,fg=T['yellow'],bg=T['panel'],selectcolor=T['panel'],
                           activebackground=T['panel'],
                           command=lambda: entry.config(state='normal')).pack(side='left',padx=(0,8))
            entry.pack(side='left')
            tk.Label(r,text=f"  {unit}",font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left')
            return entry

        self.bl_entry=param_row("Bond Length (l_b)",self.bond_len_auto,self.bond_len_var,"Å")
        self.ba_entry=param_row("Valence Angle (θ)",self.bond_ang_auto,self.bond_ang_var,"°")

        r3=tk.Frame(sec,bg=T['panel']); r3.pack(fill='x',padx=12,pady=5)
        tk.Label(r3,text="Branching Multiplicity (f)",width=30,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        tk.Spinbox(r3,from_=1,to=8,textvariable=self.branching_var,
                   width=6,font=FONT,bg=T['entry'],fg=T['cyan'],
                   relief='flat',bd=4).pack(side='left')
        tk.Label(r3,text="  branches per junction (Branching Multiplicity)",font=("Courier New",10),
                 fg=T['muted'],bg=T['panel']).pack(side='left',padx=8)

        r4=tk.Frame(sec,bg=T['panel']); r4.pack(fill='x',padx=12,pady=5)
        tk.Label(r4,text="Maximum Generation (G_max)",width=30,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        sp=tk.Spinbox(r4,from_=1,to=10,textvariable=self.num_gens_var,
                      width=6,font=FONT,bg=T['entry'],fg=T['cyan'],relief='flat',bd=4)
        sp.pack(side='left')
        sp.bind("<FocusOut>",lambda e: self._rebuild_gen_selector())
        sp.bind("<ButtonRelease>",lambda e: self.root.after(100,self._rebuild_gen_selector))
        tk.Label(r4,text="  upper limit G_max, tick generations to save",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left',padx=8)

        r5=tk.Frame(sec,bg=T['panel']); r5.pack(fill='x',padx=12,pady=5)
        tk.Label(r5,text="Systematic Dendrimer Name",width=30,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        _mk_entry(r5,self.den_type_var,T,width=20).pack(side='left')
        tk.Label(r5,text="  used in output filenames (Systematic Dendrimer Name)",font=("Courier New",10),
                 fg=T['muted'],bg=T['panel']).pack(side='left',padx=8)

        # Clash threshold
        r6=tk.Frame(sec,bg=T['panel']); r6.pack(fill='x',padx=12,pady=5)
        tk.Label(r6,text="Steric Hindrance Threshold (d_min)",width=30,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        self._clash_lbl=tk.Label(r6,text="1.0 Å",font=FONT,fg=T['yellow'],bg=T['panel'],width=6)
        self._clash_lbl.pack(side='left')
        s_=ttk.Scale(r6,from_=0.3,to=2.0,orient='horizontal',
                     variable=self.clash_thresh_var,length=180,
                     command=lambda v: self._clash_lbl.config(text=f"{float(v):.1f} Å"))
        s_.pack(side='left',padx=8)
        tk.Label(r6,text="  flag if interatomic distance < d_min",font=("Courier New",10),
                 fg=T['muted'],bg=T['panel']).pack(side='left')

    def _sec5_gen_selector(self,F):
        T=self._T; sec=self._section(F,"07  GENERATION SELECTOR",T['yellow'])
        tk.Label(sec,text="  Tick the generations to save to disk.\n"
                          "  All are built internally — unticked ones are scaffolding only.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12)
        self.gen_selector_frame=tk.Frame(sec,bg=T['panel'])
        self.gen_selector_frame.pack(fill='x',padx=12,pady=(4,4))
        btn_row=tk.Frame(sec,bg=T['panel']); btn_row.pack(fill='x',padx=12,pady=(0,8))
        _mk_btn(btn_row,"All",self._gen_select_all,T,fg_key='green').pack(side='left',padx=(0,6))
        _mk_btn(btn_row,"None",self._gen_select_none,T,fg_key='red').pack(side='left',padx=(0,6))
        _mk_btn(btn_row,"Last only",self._gen_select_last,T,fg_key='yellow').pack(side='left',padx=(0,6))
        _mk_btn(btn_row,"Refresh",self._rebuild_gen_selector,T,fg_key='muted').pack(side='left')
        self._rebuild_gen_selector()

    def _rebuild_gen_selector(self):
        T=self._T
        for w in self.gen_selector_frame.winfo_children(): w.destroy()
        self.gen_check_vars=[]
        n=self.num_gens_var.get()
        row=tk.Frame(self.gen_selector_frame,bg=T['panel']); row.pack(anchor='w')
        for g in range(1,n+1):
            var=tk.BooleanVar(value=True)
            tk.Checkbutton(row,text=f"  G{g}  ",variable=var,
                           font=BOLD,fg=T['cyan'],bg=T['panel'],
                           selectcolor=T['sel'],activebackground=T['panel'],
                           activeforeground=T['green'],cursor='hand2'
                           ).grid(row=0,column=g-1,padx=6,pady=4)
            self.gen_check_vars.append((g,var))

    def _gen_select_all(self):
        for _,v in self.gen_check_vars: v.set(True)
    def _gen_select_none(self):
        for _,v in self.gen_check_vars: v.set(False)
    def _gen_select_last(self):
        for i,(_,v) in enumerate(self.gen_check_vars):
            v.set(i==len(self.gen_check_vars)-1)

    def _sec6_pipeline(self,F):
        T=self._T; sec=self._section(F,"08  SIMULATION PIPELINE  (AMBER / NAMD / OpenBabel)",T['red'])
        tk.Label(sec,text="  Configure parameterization, minimization, and MD. All steps are optional — untick to skip.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,4))

        # AmberTools path locator
        amber_frame=tk.Frame(sec,bg=T['border']); amber_frame.pack(fill='x',padx=12,pady=(0,6))
        amber_inner=tk.Frame(amber_frame,bg=T['panel']); amber_inner.pack(fill='x',padx=1,pady=1)
        tk.Label(amber_inner,text="  AmberTools Location",font=BOLD,fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=8,pady=(6,2))
        tk.Label(amber_inner,text="  Set the conda path and environment name so generated shell scripts use the correct AmberTools installation.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,4))

        r_env=tk.Frame(amber_inner,bg=T['panel']); r_env.pack(fill='x',padx=12,pady=3)
        tk.Label(r_env,text="Conda env name:",width=20,anchor='w',font=FONT,fg=T['white'],bg=T['panel']).pack(side='left')
        _mk_entry(r_env,self.amber_conda_env,T,width=22).pack(side='left',padx=(0,12))
        tk.Label(r_env,text="e.g.  AmberTools25",font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left')

        r_path=tk.Frame(amber_inner,bg=T['panel']); r_path.pack(fill='x',padx=12,pady=3)
        tk.Label(r_path,text="Conda base path:",width=20,anchor='w',font=FONT,fg=T['white'],bg=T['panel']).pack(side='left')
        _mk_entry(r_path,self.amber_conda_path,T,width=36).pack(side='left',padx=(0,8))
        _mk_btn(r_path,"Browse",lambda: self._browse_conda_path(),T).pack(side='left',padx=(0,8))
        _mk_btn(r_path,"  Auto-detect  ",self._auto_detect_ambertools,T,fg_key='cyan').pack(side='left',padx=(0,8))
        _mk_btn(r_path,"Test",self._test_ambertools,T,fg_key='green').pack(side='left')
        self._amber_test_lbl=tk.Label(amber_inner,text="",font=FONT,fg=T['muted'],bg=T['panel'])
        self._amber_test_lbl.pack(anchor='w',padx=12,pady=(0,6))

        # Parameterization method
        param_frame=tk.Frame(sec,bg=T['border']); param_frame.pack(fill='x',padx=12,pady=(0,6))
        param_inner=tk.Frame(param_frame,bg=T['panel']); param_inner.pack(fill='x',padx=1,pady=1)
        tk.Label(param_inner,text="  Parameterization Method",font=BOLD,fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=8,pady=(6,2))
        tk.Label(param_inner,text="  Fragment BCC is fast — runs antechamber only on the small core and monomer files, then applies those parameters to the full dendrimer. Full antechamber runs on the entire built dendrimer and is accurate but very slow for G4+.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))

        for val,label,desc in [
            ("fragment_bcc","Fragment BCC",
             "Antechamber on core + monomer only -- fast, works for any generation"),
            ("full_antechamber","Full Antechamber Parameterization",
             "Antechamber on entire built dendrimer -- accurate, very slow for G4+"),
        ]:
            r=tk.Frame(param_inner,bg=T['panel']); r.pack(fill='x',padx=12,pady=3)
            tk.Radiobutton(r,text=f"  {label}",variable=self.param_method_var,value=val,
                           font=BOLD,fg=T['cyan'],bg=T['panel'],selectcolor=T['panel'],
                           activebackground=T['panel'],cursor='hand2').pack(side='left')
            tk.Label(r,text=f"  {desc}",font=("Courier New",10),
                     fg=T['muted'],bg=T['panel']).pack(side='left')
        tk.Frame(param_inner,bg=T['bg'],height=6).pack()

        style=ttk.Style()
        style.configure('Pipeline.TNotebook',background=T['panel'],borderwidth=0)
        style.configure('Pipeline.TNotebook.Tab',background=T['sbtn'],foreground=T['muted'],
                        font=BOLD,padding=[12,4])
        style.map('Pipeline.TNotebook.Tab',background=[('selected',T['bg'])],
                  foreground=[('selected',T['cyan'])])
        nb=ttk.Notebook(sec,style='Pipeline.TNotebook')
        nb.pack(fill='x',padx=12,pady=(0,8))
        amber_tab=tk.Frame(nb,bg=T['panel']); nb.add(amber_tab,text="  AMBER / sander  ")
        self._amber_tab(amber_tab,T)
        namd_tab=tk.Frame(nb,bg=T['panel']);  nb.add(namd_tab,text="  NAMD  ")
        self._namd_tab(namd_tab,T)
        obabel_tab=tk.Frame(nb,bg=T['panel']); nb.add(obabel_tab,text="  OpenBabel  ")
        self._obabel_tab(obabel_tab,T)
        intergen_tab=tk.Frame(nb,bg=T['panel'])
        nb.add(intergen_tab,text="  Intergenerational Min  ")
        self._intergen_tab(intergen_tab,T)

    def _amber_tab(self,F,T):
        def sec_lbl(text):
            tk.Label(F,text=f"  {text}",font=BOLD,fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=12,pady=(8,2))
        def chk(text,var,color,indent=12):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Checkbutton(r,text=f"  {text}",variable=var,font=FONT,fg=T[color],
                           bg=T['panel'],selectcolor=T['sel'],activebackground=T['panel'],cursor='hand2').pack(side='left')
        def spinrow(label,var,frm,to,unit="",w=8,indent=36):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Label(r,text=label,width=24,anchor='w',font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
            tk.Spinbox(r,from_=frm,to=to,textvariable=var,width=w,font=FONT,
                       bg=T['entry'],fg=T['cyan'],relief='flat',bd=4,format="%.1f").pack(side='left')
            tk.Label(r,text=f"  {unit}",font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left')

        sec_lbl("FILE OUTPUTS")
        chk("Save mol2",self.pipe_save_mol2,'green')
        chk("Save PDB",self.pipe_save_pdb,'green')
        chk("Save minimization shell script",self.pipe_save_script,'green')

        sec_lbl("AMBER PARAMETERIZATION  (AmberTools25)")
        chk("antechamber — GAFF atom types + charges",self.pipe_antechamber,'cyan')

        # charge method row
        r_nc=tk.Frame(F,bg=T['panel']); r_nc.pack(fill='x',padx=36,pady=2)
        tk.Label(r_nc,text="Net Molecular Charge:",width=14,anchor='w',font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
        tk.Spinbox(r_nc,from_=-50,to=50,textvariable=self.net_charge_var,
                   width=6,font=FONT,bg=T['entry'],fg=T['cyan'],relief='flat',bd=4).pack(side='left')
        tk.Label(r_nc,text="  Molecular Mechanics Force Field:",font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left',padx=(12,4))
        ttk.Combobox(r_nc,textvariable=self.ff_var,state='readonly',
                     values=['gaff','gaff2'],width=8,font=FONT).pack(side='left')
        tk.Label(r_nc,text="  Partial Charge Model:",font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left',padx=(12,4))
        # Only valid antechamber -c flags: bcc (AM1-BCC), gas (Gasteiger), none (skip).
        # 'resp' would require a precomputed Gaussian/GAMESS ESP file — not supported here.
        ttk.Combobox(r_nc,textvariable=self.charge_method_var,state='readonly',
                     values=['bcc','gas','none'],
                     width=10,font=FONT).pack(side='left')

        chk("parmchk2  — missing parameters → .frcmod",self.pipe_parmchk2,'cyan')
        chk("tLeap     — topology + coords → .prmtop + .inpcrd",self.pipe_tleap,'cyan')

        sec_lbl("ENERGY MINIMIZATION  (vacuum)")
        chk("sander  — vacuum energy minimization",self.pipe_minimize,'yellow')
        chk("ambpdb  — minimized restart → PDB",self.pipe_ambpdb,'yellow')
        r_mpi=tk.Frame(F,bg=T['panel']); r_mpi.pack(fill='x',padx=36,pady=2)
        tk.Label(r_mpi,text="CPU cores (sander.MPI):",width=22,anchor='w',
                 font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
        tk.Spinbox(r_mpi,from_=1,to=64,textvariable=self.pipe_ncpus,
                   width=5,font=FONT,bg=T['entry'],fg=T['cyan'],
                   relief='flat',bd=4).pack(side='left')
        tk.Label(r_mpi,text="  1 = serial sander,  2+ = mpirun -np N sander.MPI",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left',padx=8)

        tk.Label(F,text="  Requires: conda activate AmberTools25",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(4,2))
        tk.Label(F,text="  Solvation and MD are in the NAMD tab  →  Generate NAMD Package",
                 font=("Courier New",10),fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,8))

    def _namd_tab(self,F,T):
        """NAMD 3.x / AMBER-GAFF2 pipeline tab (v1.0)."""
        def sec_lbl(text):
            tk.Label(F,text=f"  {text}",font=BOLD,fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=12,pady=(8,2))
        def chk(text,var,color='cyan',indent=12):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Checkbutton(r,text=f"  {text}",variable=var,font=FONT,fg=T[color],
                           bg=T['panel'],selectcolor=T['sel'],
                           activebackground=T['panel'],cursor='hand2').pack(side='left')
        def spinrow(label,var,frm,to,unit='',w=8,indent=36,fmt='%.1f'):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Label(r,text=label,width=24,anchor='w',font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
            tk.Spinbox(r,from_=frm,to=to,textvariable=var,width=w,font=FONT,
                       bg=T['entry'],fg=T['cyan'],relief='flat',bd=4,format=fmt).pack(side='left')
            tk.Label(r,text=f"  {unit}",font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left')
        def spinrow_int(label,var,frm,to,unit='',w=8,indent=36):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Label(r,text=label,width=24,anchor='w',font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
            tk.Spinbox(r,from_=frm,to=to,textvariable=var,width=w,font=FONT,
                       bg=T['entry'],fg=T['cyan'],relief='flat',bd=4).pack(side='left')
            tk.Label(r,text=f"  {unit}",font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left')
        def combo_row(label,var,choices,indent=36):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Label(r,text=label,width=18,anchor='w',font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
            ttk.Combobox(r,textvariable=var,values=choices,state='readonly',width=14,font=FONT).pack(side='left')
        def path_row(label,var,indent=36):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Label(r,text=label,width=18,anchor='w',font=FONT,fg=T['muted'],bg=T['panel']).pack(side='left')
            tk.Entry(r,textvariable=var,width=46,font=("Courier New",10),
                     bg=T['entry'],fg=T['cyan'],relief='flat',bd=4).pack(side='left')
            def _browse_binary(v=var):
                p = filedialog.askopenfilename(
                        title='Select NAMD binary',
                        filetypes=[('Executables','*'),('All','*.*')])
                if p: v.set(p)
            tk.Button(r,text='Browse',font=FONT,bg=T['sel'],fg=T['white'],relief='flat',cursor='hand2',
                      command=_browse_binary).pack(side='left',padx=4)

        # ── NAMD Binary ───────────────────────────────────────
        sec_lbl('NAMD BINARY')
        path_row('NAMD binary:',self.namd_binary_var)
        r_detect=tk.Frame(F,bg=T['panel']); r_detect.pack(fill='x',padx=36,pady=(0,4))
        def _detect_namd():
            import subprocess as _sp
            candidates = [
                NAMD_DEFAULT_BIN,
                '/opt/namd/NAMD_3.0.2_Linux-x86_64-multicore/namd3',
                '/opt/namd/NAMD_3.0.1_Linux-x86_64-multicore/namd3',
                os.path.expanduser('~/Documents/NAMD_3.0.2/namd3'),
                os.path.expanduser('~/Documents/NAMD_3.0.1/namd3'),
                os.path.expanduser('~/NAMD_3.0.1_Linux-x86_64-multicore/namd3'),
                'namd3', 'namd2',
            ]
            for candidate in candidates:
                try:
                    if '/' in candidate:
                        # Absolute path: check it exists AND is executable
                        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                            self.namd_binary_var.set(candidate)
                            self._log(f"  [NAMD] Detected: {candidate}",'ok'); return
                    else:
                        # Name only: use `which`
                        r = _sp.run(['which', candidate], capture_output=True, text=True)
                        if r.returncode == 0:
                            path = r.stdout.strip()
                            self.namd_binary_var.set(path)
                            self._log(f"  [NAMD] Detected: {path}",'ok'); return
                except Exception: pass
            self._log('  [NAMD] Could not auto-detect NAMD. Enter path manually.','warn')
        tk.Button(r_detect,text='  Auto-detect NAMD binary  ',font=FONT,
                  bg=T['sel'],fg=T['white'],relief='flat',cursor='hand2',
                  command=_detect_namd).pack(side='left')

        # ── Force field & Solvent ─────────────────────────────
        sec_lbl('FORCE FIELD  &  SOLVENT')
        combo_row('Force field:',self.namd_ff_var,['GAFF2','CHARMM36'])
        combo_row('Water model:',self.namd_water_var,['TIP3P','TIP4P','SPC/E'])
        spinrow('Box padding:',self.namd_box_var,8.0,30.0,'Å',w=6)
        spinrow('NaCl conc.:',self.namd_conc_var,0.0,2.0,'M  (0 = neutralise only)',w=6,fmt='%.2f')
        spinrow('Temperature:',self.namd_temp_var,273.0,400.0,'K',w=7)

        # ── Pipeline steps ────────────────────────────────────
        sec_lbl('PIPELINE STEPS')
        chk('Step 0 — tLeap solvation  (AMBER/GAFF2)',self.namd_solvate,'green')
        chk('Step 1 — Energy minimisation  (5 000 steps)',self.namd_minimize,'cyan')
        chk('Step 2 — Heating  0 K → T  (NVT)',self.namd_heat,'cyan')
        spinrow('  Heating duration:',self.namd_heat_ps,100.0,2000.0,'ps',w=8,indent=48)
        chk('Step 3 — NPT equilibration  (Langevin piston)',self.namd_equil,'cyan')
        spinrow('  Equilibration:',self.namd_equil_ns,1.0,100.0,'ns',w=8,indent=48)
        chk('Step 4 — NPT production MD',self.namd_prod,'cyan')
        spinrow('  Production:',self.namd_prod_ns,1.0,500.0,'ns',w=8,indent=48)

        # ── MD parameters ─────────────────────────────────────
        sec_lbl('MD PARAMETERS')
        spinrow('Timestep:',self.namd_timestep,1.0,4.0,'fs',w=6)
        spinrow_int('DCD frequency:',self.namd_dcd_freq,500,50000,'steps  (×2fs = ps)',w=8)
        spinrow_int('CPU cores:',self.namd_cores_var,1,256,'',w=6)

        # ── Time estimate ─────────────────────────────────────
        self._namd_time_lbl=tk.Label(F,text='  Load mol2 files to see time estimates.',
                                      font=('Courier New',10),fg=T['muted'],bg=T['panel'],justify='left')
        self._namd_time_lbl.pack(anchor='w',padx=12,pady=(8,4))
        for _v in (self.namd_prod_ns, self.namd_equil_ns, self.namd_heat_ps,
                   self.namd_timestep, self.namd_cores_var, self.namd_dcd_freq):
            _v.trace_add('write', self._update_namd_times)

        # ── Buttons ───────────────────────────────────────────
        btn_row=tk.Frame(F,bg=T['panel']); btn_row.pack(fill='x',padx=12,pady=(4,2))
        _mk_btn(btn_row,'  Generate NAMD Package  ',self._gen_namd_inputs,T,
                bg_key='cyan',fg_key='bg').pack(side='left',padx=(0,8))
        _mk_btn(btn_row,'  Open Package Folder  ',
                lambda:os.system(f"xdg-open {self.output_dir.get()}"),
                T,bg_key='sel',fg_key='muted').pack(side='left')
        tk.Label(F,text='  Generates: 00_solvate.tleap · 01_minimize.conf · 02_heat.conf · 03_equil.conf · 04_production.conf · run_namd.sh · analysis.tcl',
                 font=('Courier New',9),fg=T['muted'],bg=T['panel'],wraplength=680,justify='left').pack(anchor='w',padx=12,pady=(0,2))
        tk.Label(F,text='  Always use run_namd.sh (not conf files directly) — it creates the logs/ folder required by conf files.',
                 font=('Courier New',9),fg=T['yellow'],bg=T['panel'],wraplength=680,justify='left').pack(anchor='w',padx=12,pady=(0,8))

    def _obabel_tab(self,F,T):
        """[8] OpenBabel minimization tab."""
        def sec_lbl(text):
            tk.Label(F,text=f"  {text}",font=BOLD,fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=12,pady=(8,2))

        sec_lbl("OPENBABEL MINIMIZATION")
        r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=12,pady=2)
        tk.Checkbutton(r,text="  Run obabel geometry optimization after build",
                       variable=self.obabel_minimize,font=FONT,fg=T['cyan'],
                       bg=T['panel'],selectcolor=T['sel'],
                       activebackground=T['panel'],cursor='hand2').pack(side='left')

        r2=tk.Frame(F,bg=T['panel']); r2.pack(fill='x',padx=36,pady=2)
        tk.Label(r2,text="Force field:",width=14,anchor='w',font=FONT,
                 fg=T['muted'],bg=T['panel']).pack(side='left')
        ttk.Combobox(r2,textvariable=self.obabel_ff_var,state='readonly',
                     values=['MMFF94','UFF','GAFF','GHEMICAL'],
                     width=12,font=FONT).pack(side='left')

        r3=tk.Frame(F,bg=T['panel']); r3.pack(fill='x',padx=36,pady=2)
        tk.Label(r3,text="Steps:",width=14,anchor='w',font=FONT,
                 fg=T['muted'],bg=T['panel']).pack(side='left')
        tk.Spinbox(r3,from_=100,to=50000,textvariable=self.obabel_steps_var,
                   width=8,font=FONT,bg=T['entry'],fg=T['cyan'],
                   relief='flat',bd=4).pack(side='left')

        tk.Label(F,text="\n  Command used:\n"
                        "  obabel input.mol2 -O output.mol2 --minimize\n"
                        "         --ff MMFF94 --steps 2500\n\n"
                        "  Requires: openbabel (conda install -c conda-forge openbabel)",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],justify='left').pack(anchor='w',padx=12,pady=(4,8))

    def _intergen_tab(self,F,T):
        """Intergenerational minimization tab (v1.0).
        Minimize after each generation and feed minimized coords into the next."""
        FONT9=("Courier New",9)
        def sec_lbl(text):
            tk.Label(F,text=f"  {text}",font=BOLD,fg=T['yellow'],
                     bg=T['panel']).pack(anchor='w',padx=12,pady=(8,2))
        def chk(text,var,color='cyan',indent=12):
            r=tk.Frame(F,bg=T['panel']); r.pack(fill='x',padx=indent,pady=1)
            tk.Checkbutton(r,text=f"  {text}",variable=var,font=FONT,fg=T[color],
                           bg=T['panel'],selectcolor=T['sel'],
                           activebackground=T['panel'],cursor='hand2').pack(side='left')
        def note(text,indent=48):
            tk.Label(F,text=text,font=FONT9,fg=T['muted'],
                     bg=T['panel'],justify='left').pack(anchor='w',padx=indent,pady=(0,3))

        # ── Master enable ─────────────────────────────────────────────────────
        sec_lbl("INTERGENERATIONAL MINIMIZATION  (v1.0)")
        tk.Label(F,text=(
            "  After each generation is built, run energy minimization and seed the\n"
            "  next generation from the minimized geometry. Also writes a minimized\n"
            "  GAFF mol2 file with atom types and charges preserved from the prmtop."),
            font=FONT9,fg=T['muted'],bg=T['panel'],justify='left'
        ).pack(anchor='w',padx=12,pady=(0,4))
        en_row=tk.Frame(F,bg=T['panel']); en_row.pack(fill='x',padx=12,pady=4)
        tk.Checkbutton(en_row,text="  Enable intergenerational minimization",
                       variable=self.intergen_enabled,font=BOLD,fg=T['green'],
                       bg=T['panel'],selectcolor=T['sel'],
                       activebackground=T['panel'],cursor='hand2').pack(side='left')

        # ── Minimization engine ───────────────────────────────────────────────
        sec_lbl("MINIMIZATION ENGINE")
        tk.Label(F,text="  Both engines run independently on the as-built structure.\n"
                        "  NAMD result takes priority when both are enabled.",
                 font=FONT9,fg=T['muted'],bg=T['panel'],justify='left'
        ).pack(anchor='w',padx=12,pady=(0,4))

        chk("AMBER  (sander vacuum minimization)",self.intergen_use_amber,'cyan',indent=24)
        note("Reads {prefix}_min.pdb produced by sander + ambpdb.\n"
             "Requires: antechamber  parmchk2  tLeap  sander  ambpdb  (AMBER tab).")

        chk("NAMD  (energy minimization)",self.intergen_use_namd,'cyan',indent=24)
        note("Writes a per-generation NAMD conf, runs minimize, extracts coords via cpptraj.\n"
             "NAMD result takes priority over AMBER when both are enabled.")

        # NAMD sub-mode frame
        nf_outer=tk.Frame(F,bg=T['border']); nf_outer.pack(fill='x',padx=36,pady=(2,4))
        nf=tk.Frame(nf_outer,bg=T['panel']); nf.pack(fill='x',padx=1,pady=1)
        tk.Label(nf,text="  NAMD Minimization Mode:",
                 font=BOLD,fg=T['yellow'],bg=T['panel']).pack(anchor='w',padx=8,pady=(4,2))
        for val,label,desc in [
            ('vacuum',
             'Vacuum  (no PBC)',
             'Uses {prefix}.prmtop + inpcrd.  Requires tLeap in AMBER tab.'),
            ('solvated',
             'Solvated + cpptraj strip',
             'Uses {prefix}_solvated.prmtop.  Requires namd_solvate in NAMD tab.\n'
             '                Solvent is stripped by cpptraj after minimize.'),
        ]:
            rr=tk.Frame(nf,bg=T['panel']); rr.pack(fill='x',padx=12,pady=2)
            tk.Radiobutton(rr,text=f"  {label}",variable=self.intergen_namd_mode,
                           value=val,font=FONT,fg=T['cyan'],bg=T['panel'],
                           selectcolor=T['panel'],activebackground=T['panel'],
                           cursor='hand2').pack(side='left')
            tk.Label(rr,text=f"  — {desc}",font=FONT9,fg=T['muted'],
                     bg=T['panel']).pack(side='left')
        tk.Frame(nf,bg=T['bg'],height=4).pack()

        # ── Output ────────────────────────────────────────────────────────────
        sec_lbl("OUTPUT")
        chk("Save minimized GAFF mol2  ({prefix}_min_gaff.mol2)",
            self.intergen_save_gaff,'green',indent=24)
        note("GAFF atom types + partial charges from prmtop; minimized XYZ from rst.\n"
             "Requires: sander step enabled in AMBER tab  +  cpptraj available.")

        # ── Prerequisites ─────────────────────────────────────────────────────
        sec_lbl("PREREQUISITES  (per engine)")
        tk.Label(F,text=(
            "  AMBER path    :  AMBER tab — antechamber  parmchk2  tLeap  sander  ambpdb\n"
            "  NAMD vacuum   :  AMBER tab — tLeap  (prmtop + inpcrd)  +  NAMD binary\n"
            "  NAMD solvated :  NAMD tab  — namd_solvate  +  NAMD binary  +  cpptraj\n"
            "  GAFF mol2     :  AMBER tab — sander  +  cpptraj in AmberTools25 conda env\n\n"
            "  cpptraj ships with AmberTools25: conda activate AmberTools25 && cpptraj --version"),
            font=FONT9,fg=T['muted'],bg=T['panel'],justify='left'
        ).pack(anchor='w',padx=12,pady=(0,8))

    def _sec7_output(self,F):
        T=self._T; sec=self._section(F,"09  OUTPUT  &  CONFIG")
        tk.Label(sec,text="  Set the output folder, file format, and config save/load. Config stores all settings as JSON for reuse.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))

        r1=tk.Frame(sec,bg=T['panel']); r1.pack(fill='x',padx=12,pady=4)
        tk.Label(r1,text="Output folder:",width=18,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        _mk_entry(r1,self.output_dir,T,width=44).pack(side='left',fill='x',expand=True,padx=(0,8))
        _mk_btn(r1,"Choose",self._choose_out_dir,T).pack(side='left')

        # output format checkboxes
        fmt_row=tk.Frame(sec,bg=T['panel']); fmt_row.pack(fill='x',padx=12,pady=(4,4))
        tk.Label(fmt_row,text="Output formats:",width=18,anchor='w',font=FONT,
                 fg=T['white'],bg=T['panel']).pack(side='left')
        for text,var in [("mol2",self.fmt_mol2),("pdb",self.fmt_pdb),
                         ("xyz",self.fmt_xyz),("gro",self.fmt_gro),
                         ("psf",self.fmt_psf)]:
            tk.Checkbutton(fmt_row,text=f"  {text}",variable=var,font=FONT,
                           fg=T['cyan'],bg=T['panel'],selectcolor=T['sel'],
                           activebackground=T['panel'],cursor='hand2').pack(side='left',padx=4)

        r2=tk.Frame(sec,bg=T['panel']); r2.pack(fill='x',padx=12,pady=(4,10))
        _mk_btn(r2,"Save Config",self._save_config,T,fg_key='green').pack(side='left',padx=(0,8))
        _mk_btn(r2,"Load Config",self._load_config,T,fg_key='yellow').pack(side='left')
        tk.Label(r2,text="  Config saves file paths + atom selections + gen choices",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left',padx=12)

    def _sec8_build(self,F):
        T=self._T
        sec=self._section(F,"10  BUILD  &  RUN",T['cyan'])
        tk.Label(sec,text="  Click Build Dendrimer to start. Stop cancels at the next generation boundary. Progress updates live in sections 11 and 12 below.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))
        bf=tk.Frame(sec,bg=T['panel']); bf.pack(pady=14)

        btn_pair=tk.Frame(bf,bg=T['bg']); btn_pair.pack()
        self.build_btn=tk.Button(btn_pair,text="  BUILD DENDRIMER  ",
                                  font=("Courier New",15,"bold"),
                                  bg=T['cyan'],fg=T['bg'],
                                  relief='flat',padx=28,pady=12,cursor='hand2',
                                  activebackground=T['border'],
                                  command=self._start_build)
        self.build_btn.pack(side='left',padx=(0,8))

        # Stop button
        self.stop_btn=tk.Button(btn_pair,text="  STOP  ",
                                 font=("Courier New",15,"bold"),
                                 bg=T['red'],fg=T['bg'],
                                 relief='flat',padx=16,pady=12,cursor='hand2',
                                 activebackground=T['border'],
                                 state='disabled',
                                 command=self._stop_build)
        self.stop_btn.pack(side='left')

        pf=tk.Frame(F,bg=T['bg']); pf.pack(fill='x',padx=24,pady=(0,2))
        tk.Label(pf,text="Overall progress:",font=FONT,fg=T['muted'],bg=T['bg']).pack(anchor='w')
        self.progress_overall=ttk.Progressbar(pf,orient='horizontal',mode='determinate',length=980)
        self.progress_overall.pack(fill='x')
        self.prog_label=tk.Label(pf,text="Ready — configure above and click Build",
                                  font=FONT,fg=T['muted'],bg=T['bg'])
        self.prog_label.pack(anchor='w',pady=(2,8))

        self._gen_bar_header=tk.Label(pf,text="Current generation:",font=FONT,fg=T['muted'],bg=T['bg'])
        self._gen_bar_header.pack(anchor='w')
        self.progress_gen=ttk.Progressbar(pf,orient='horizontal',mode='determinate',length=980)
        self.progress_gen.pack(fill='x')
        self.prog_gen_label=tk.Label(pf,text="",font=FONT,fg=T['muted'],bg=T['bg'])
        self.prog_gen_label.pack(anchor='w',pady=(2,2))
        self._atom_lbl=tk.Label(pf,text="",font=BOLD,fg=T['cyan'],bg=T['bg'])
        self._atom_lbl.pack(anchor='w',pady=(0,4))

    def _stop_build(self):
        """[1] Signal the build thread to stop."""
        self._cancel_flag[0]=True
        self.stop_btn.config(state='disabled',text="  Stopping...  ")
        self._log("  Stop requested — will cancel at next generation boundary.",'warn')

    def _sec_viewer(self,F):
        T=self._T; sec=self._section(F,"02  MOLECULAR VIEWER  (3D)",T['green'])
        tk.Label(sec,text="  Click Build Dendrimer to start. The Stop button cancels at the next generation boundary.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))
        tk.Label(sec,text="  Drag to rotate, scroll to zoom, click any atom to auto-select it in the sections below.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,4))

        viewer_row=tk.Frame(sec,bg=T['panel'])
        viewer_row.pack(fill='x',padx=12,pady=(0,8))

        self._core_viewer=MolViewer3D(viewer_row,T,"Core fragment",
                                     on_click=self._viewer_click_core,
                                     width=440,height=300)
        self._core_viewer.pack(side='left',fill='both',expand=True,padx=(0,6))

        self._mon_viewer=MolViewer3D(viewer_row,T,"Monomer fragment",
                                    on_click=self._viewer_click_mon,
                                    width=440,height=300)
        self._mon_viewer.pack(side='left',fill='both',expand=True)

    def _viewer_click_core(self,idx,name):
        """[3] Click on core viewer → tick matching checkbox."""
        amap={a['idx']:a for a in self.core_atoms}
        a=amap.get(idx)
        if not a: return
        if is_hydrogen(a):
            self._log(f"  3D Viewer: clicked H atom {name} (idx {idx}) — select the heavy atom instead.",'warn')
            return
        # tick the checkbox for this atom
        for cidx,cname,var in self.core_check_vars:
            if cidx==idx:
                var.set(not var.get())
                state="selected" if var.get() else "deselected"
                self._log(f"  3D Viewer: core atom {name} (idx {idx}) → {state}",'ok')
                break
        self._update_core_counter()

    def _viewer_click_mon(self,idx,name):
        if not self.mon_atoms: return
        """[3] Click on monomer viewer → fill input/terminal/labile combos."""
        T=self._T
        amap={a['idx']:a for a in self.mon_atoms}
        a=amap.get(idx)
        if not a: return
        opts=[f"{at['idx']}:{at['name']}  [{at['type']}]" for at in self.mon_atoms]
        match=next((o for o in opts if o.startswith(f"{idx}:")),None)
        mode=self.labile_mode_var.get()
        if mode=='group' and self._labile_input_mgr is not None:
            # In group mode any clicked atom goes into the labile manager
            self._labile_input_mgr.set_name_entry(name,idx)
            self._log(f"  Viewer: {name} (idx {idx}) → filled into Labile Group manager",'ok')
        elif is_hydrogen(a):
            # Single mode: H click → set as labile atom
            if match and match in self.mon_input_h_cb['values']:
                self.mon_input_h_var.set(match)
                self._log(f"  Viewer: {name} (idx {idx}) → set as labile atom",'ok')
        else:
            # Heavy atom click → terminal manager or input combo
            self._term_mgr.set_name_entry(name,idx)
            self._log(f"  Viewer: {name} (idx {idx}) → filled into Terminal Manager",'ok')
            if match and match in self.mon_input_cb['values']:
                pass  # don't overwrite input unless nothing selected
        self._update_viewer_roles()

    def _update_viewer_roles(self):
        """[5] Recompute role highlights and push to both viewers."""
        core_roles={}
        mon_roles={}

        # Core: binding sites from checkboxes
        for idx,name,var in self.core_check_vars:
            if var.get():
                core_roles[idx]='core_binding'
                for h in get_h_neighbors(self.core_atoms,self.core_bonds,idx):
                    core_roles[h['idx']]='core_h'

        # Monomer: input atom
        sel=self.mon_input_var.get()
        if sel:
            try:
                idx=int(sel.split(':')[0])
                mon_roles[idx]='mon_input'
            except (ValueError, IndexError): pass
        # Monomer: input H / labile atom
        sel_h=self.mon_input_h_var.get()
        if sel_h and self.labile_mode_var.get()=='single':
            try:
                idx=int(sel_h.split(':')[0])
                mon_roles[idx]='mon_h'
            except (ValueError, IndexError): pass
        if (self.labile_mode_var.get()=='group'
                and self._labile_input_mgr is not None):
            for name in self._labile_input_mgr.get_names():
                for a in (self.mon_atoms or []):
                    if a['name']==name:
                        mon_roles[a['idx']]='mon_h'
        # Monomer: terminal atoms from manager
        for name in self._term_mgr.get_names():
            for a in (self.mon_atoms or []):
                if a['name']==name:
                    mon_roles[a['idx']]='mon_terminal'

        self._core_viewer.set_roles(core_roles)
        self._mon_viewer.set_roles(mon_roles)

    def _sec_gen_preview(self,F):
        T=self._T; sec=self._section(F,"11  GENERATION PREVIEW",T['yellow'])
        tk.Label(sec,text="  Shows estimated values before build, then updates with real values as each generation completes.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,4))
        btn_row=tk.Frame(sec,bg=T['panel']); btn_row.pack(fill='x',padx=12,pady=(0,4))
        _mk_btn(btn_row,"Refresh Preview",self._refresh_preview,T,fg_key='yellow').pack(side='left')
        tk.Label(btn_row,text="  updates automatically during build",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(side='left',padx=8)

        cols=('gen','monomers','term','atoms','mw','rg','val','status')
        self._preview_tree=ttk.Treeview(sec,columns=cols,show='headings',height=8)
        for col,txt,w in [('gen','Gen',45),('monomers','Monomers',80),
                          ('term','Terminals',80),('atoms','Atoms',90),
                          ('mw','MW (kDa)',90),('rg','Rg (A)',70),
                          ('val','Validation',90),('status','Status',90)]:
            self._preview_tree.heading(col,text=txt)
            self._preview_tree.column(col,width=w,anchor='center')
        self._preview_tree.tag_configure('done',foreground='#3fb950')
        self._preview_tree.tag_configure('pending',foreground='#8b949e')
        self._preview_tree.tag_configure('est',foreground='#d29922')
        style=ttk.Style()
        style.configure('Treeview',background=T['panel'],foreground=T['white'],
                        fieldbackground=T['panel'],font=FONT)
        style.configure('Treeview.Heading',background=T['sbtn'],foreground=T['cyan'],font=BOLD)
        self._preview_tree.pack(fill='x',padx=12,pady=(0,8))

    def _refresh_preview(self):
        """Show estimated values for all generations before build starts."""
        for item in self._preview_tree.get_children():
            self._preview_tree.delete(item)
        self._live_preview_rows=[]
        try:
            n_core_atoms=len(self.core_atoms) if self.core_atoms else 0
            n_mon_atoms =len(self.mon_atoms)  if self.mon_atoms  else 0
            n_core_sites=len([idx for idx,_,v in self.core_check_vars if v.get()]) or 4
            bf=self.branching_var.get(); n_gen=self.num_gens_var.get()
            if not n_core_atoms: return
            mw_mon =atom_mw(self.mon_atoms)  if self.mon_atoms  else 50.0
            mw_core=atom_mw(self.core_atoms) if self.core_atoms else 0.0
            current_atoms=n_core_atoms; current_mw=mw_core
            current_monomers=n_core_sites
            for g in range(1,n_gen+1):
                n_new   =current_monomers*bf if g>1 else n_core_sites*bf
                current_atoms+=n_new*n_mon_atoms
                current_mw   +=n_new*mw_mon
                n_term  =n_new*bf
                iid=self._preview_tree.insert('','end',
                    values=(f"G{g}",n_new,n_term,
                            f"~{current_atoms:,}",f"~{current_mw/1000:.1f}",
                            "--","--","Pending"),
                    tags=('pending',))
                self._live_preview_rows.append({'gen':g,'iid':iid,'done':False})
                current_monomers=n_term
        except Exception as e:
            self._log(f"  Preview error: {e}",'warn')

    def _update_preview_gen(self,gen,atoms,mw,rg,monomers,n_term,val_str):
        """[7] Called after each generation completes -- update that row with real values."""
        for row in self._live_preview_rows:
            if row['gen']==gen and not row['done']:
                self._preview_tree.item(row['iid'],
                    values=(f"G{gen}",monomers,n_term,
                            f"{atoms:,}",f"{mw/1000:.2f}",
                            f"{rg:.2f} A",val_str,"Done"),
                    tags=('done',))
                row['done']=True
                return
        # gen not in list yet (was not estimated) -- append
        self._preview_tree.insert('','end',
            values=(f"G{gen}",monomers,n_term,
                    f"{atoms:,}",f"{mw/1000:.2f}",
                    f"{rg:.2f} A",val_str,"Done"),
            tags=('done',))

    def _sec_validator(self,F):
        T=self._T; sec=self._section(F,"12  STRUCTURE VALIDATOR",T['cyan'])
        tk.Label(sec,text="  Validation results accumulate here after each generation completes. Green = pass, red = fail.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel']).pack(anchor='w',padx=12,pady=(0,4))

        cols=('gen','check','status','got','expected')
        self._val_tree=ttk.Treeview(sec,columns=cols,show='headings',height=12)
        for col,txt,w in [('gen','Gen',45),('check','Check',260),
                          ('status','Status',55),('got','Got',120),('expected','Expected',100)]:
            self._val_tree.heading(col,text=txt)
            self._val_tree.column(col,width=w,anchor='w' if col=='check' else 'center')
        self._val_tree.tag_configure('pass',foreground='#3fb950')
        self._val_tree.tag_configure('fail',foreground='#f85149')
        self._val_tree.tag_configure('sep', foreground='#30363d', background='#161b22')
        self._val_tree.pack(fill='x',padx=12,pady=(0,4))

        self._val_summary_lbl=tk.Label(sec,text="",font=BOLD,fg=T['muted'],bg=T['panel'])
        self._val_summary_lbl.pack(anchor='w',padx=12,pady=(0,8))

    def _populate_validator(self,val_results,val_passed,val_total,mw,rg,gen):
        """[8] Append this generation's results -- accumulates across all generations."""
        self._val_tree.insert('','end',
            values=(f"G{gen}",f"--- Generation {gen} ---","","",""),tags=('sep',))
        for label,ok,got,expected in val_results:
            tag='pass' if ok else 'fail'
            icon="pass" if ok else "FAIL"
            self._val_tree.insert('','end',
                values=(f"G{gen}",label,icon,got,expected),tags=(tag,))
        T=self._T
        total_fail=sum(1 for c in self._val_tree.get_children()
                       if 'fail' in (self._val_tree.item(c).get('tags') or []))
        color=T['green'] if val_passed==val_total else T['yellow']
        self._val_summary_lbl.config(
            text=(f"  G{gen}: {val_passed}/{val_total} passed   "
                  f"MW={mw:,.0f} Da   Rg={rg:.2f} A"
                  f"   (total fails so far: {total_fail})"),fg=color)
        children=self._val_tree.get_children()
        if children: self._val_tree.see(children[-1])

    def _sec9_log(self,F):
        T=self._T; sec=self._section(F,"13  BUILD LOG")
        tk.Label(sec,text="  Full build log. Export as a text file for records or troubleshooting.",
                 font=("Courier New",10),fg=T['muted'],bg=T['panel'],wraplength=900,justify='left').pack(anchor='w',padx=12,pady=(0,6))
        tb=tk.Frame(sec,bg=T['panel']); tb.pack(fill='x',padx=8,pady=(0,4))
        _mk_btn(tb,"Clear log",lambda: self.log_box.delete('1.0','end'),T).pack(side='right',padx=(4,0))
        _mk_btn(tb,"Export log (.txt)",self._save_log,T,fg_key='green').pack(side='right',padx=4)
        self.log_box=scrolledtext.ScrolledText(sec,height=22,font=("Courier New",10),
                                                bg=T['log_bg'],fg=T['log_fg'],
                                                insertbackground=T['log_fg'],
                                                relief='flat',bd=6)
        self.log_box.pack(fill='both',padx=8,pady=(0,8))
        self.log_box.tag_config('warn',foreground=T['yellow'])
        self.log_box.tag_config('err',foreground=T['red'])
        self.log_box.tag_config('ok',foreground=T['green'])
        self.log_box.tag_config('head',foreground=T['cyan'])
        self.log_box.tag_config('info',foreground=T['yellow'])
        self._log("Universal Dendrimer Builder  v1.0  ready.",'head')
        self._log("   1. Load core and monomer mol2 files")
        self._log("   2. Rotate molecules in 3D viewer (Section 02) — click atoms to select")
        self._log("   3. Tick core binding atoms in Section 03")
        self._log("   4. Add terminal atoms in Section 04 Terminal Manager")
        self._log("   5. Configure parameters and pipeline")
        self._log("   6. 🧬 Click BUILD DENDRIMER — use 🛑 STOP to cancel anytime\n")

    def _set_progress(self, overall_pct, gen=None, gen_pct=None, atoms=None):
        """Called from build thread. Schedules on main thread only when value changes."""
        # Only update when the integer percentage actually changes — avoids flooding
        # root.after with thousands of identical calls at G5 (2244 monomers).
        key = (int(overall_pct), gen, int(gen_pct) if gen_pct is not None else None)
        if getattr(self, '_last_progress_key', None) == key:
            return
        self._last_progress_key = key
        self.root.after(0, self._do_set_progress, overall_pct, gen, gen_pct, atoms)

    def _do_set_progress(self, overall_pct, gen=None, gen_pct=None, atoms=None):
        T = self._T
        self.progress_overall['value'] = overall_pct
        if gen is not None and gen_pct is not None:
            self.progress_gen['value'] = gen_pct
            max_gen = self.num_gens_var.get()
            self.prog_label.config(
                text=f"Building  G{gen} of G{max_gen}  —  {overall_pct:.0f}% overall",
                fg=T['cyan'])
            self._gen_bar_header.config(
                text=f"Generation {gen} of {max_gen}  —  placing monomers  ({gen_pct}%)",
                fg=T['cyan'])
            self.prog_gen_label.config(
                text=f"G{gen}  —  {gen_pct}%", fg=T['yellow'])
            if atoms is not None:
                self._atom_lbl.config(
                    text=f"  {atoms:,} atoms placed so far", fg=T['cyan'])

    def _set_gen_progress(self, gen, pct, placed=None, total=None, atoms=None):
        T = self._T
        max_gen = self.num_gens_var.get()
        self.progress_gen['value'] = pct
        overall = gen / max_gen * 100
        self.progress_overall['value'] = overall
        if pct >= 100:
            self.prog_label.config(
                text=f"G{gen} of G{max_gen} complete  —  {overall:.0f}% overall",
                fg=T['green'])
            self._gen_bar_header.config(
                text=f"Generation {gen} of {max_gen}  —  complete",
                fg=T['green'])
            self.prog_gen_label.config(
                text=f"G{gen} done", fg=T['green'])
        if atoms is not None:
            self._atom_lbl.config(
                text=f"  {atoms:,} atoms in structure", fg=T['cyan'])
        self.root.update_idletasks()

    def _mark_gen_done(self,gen,atoms):
        T=self._T
        self.progress_gen['value']=100
        self._gen_bar_header.config(text=f"Generation {gen}  —  complete",fg=T['green'])
        self.prog_gen_label.config(text=f"G{gen} done — {atoms:,} atoms total",fg=T['green'])
        self._atom_lbl.config(text=f"  {atoms:,} atoms",fg=T['green'])
        self.root.update_idletasks()

    def _log(self,msg,tag=None):
        self.log_box.insert('end',msg+'\n',tag or '')
        self.log_box.see('end')
        self.root.update_idletasks()

    def _save_log(self):
        """[14] Export build log to .txt file."""
        p=filedialog.asksaveasfilename(defaultextension=".txt",
                                        filetypes=[("Text","*.txt"),("All","*.*")],
                                        title="Export build log")
        if not p: return
        with open(p,'w') as f: f.write(self.log_box.get('1.0','end'))
        self._log(f"  Log exported → {p}",'ok')

    def _apply_preset(self,name):
        p=PRESETS[name]
        self.preset_desc.config(text=p["description"])
        self.den_type_var.set(name)
        self.bond_len_auto.set(p["bond_length_auto"]); self.bond_len_var.set(p["bond_length"])
        self.bond_ang_auto.set(p["bond_angle_auto"]);  self.bond_ang_var.set(p["bond_angle"])
        self.branching_var.set(p["branching_factor"]); self.preset_var.set(name)
        if hasattr(self,'bl_entry'):
            self.bl_entry.config(state='disabled' if p["bond_length_auto"] else 'normal')
        if hasattr(self,'ba_entry'):
            self.ba_entry.config(state='disabled' if p["bond_angle_auto"] else 'normal')
        if self.mon_atoms: self._auto_select_by_hint(p)

    def _auto_select_by_hint(self,p):
        if not self.mon_atoms: return
        atom_strs=[f"{a['idx']}:{a['name']}" for a in self.mon_atoms]
        if p.get("mon_input_hint"):
            m=next((s for s in atom_strs if s.split(':')[1].startswith(p["mon_input_hint"])),None)
            if m:
                self.mon_input_var.set(m); self._on_mon_input_changed(None)
        if p.get("mon_terminal_hint"):
            ms=[s for s in atom_strs if s.split(':')[1].startswith(p["mon_terminal_hint"])]
            if ms and not self._term_mgr.get_names():
                last=ms[-1].split(':')
                self._term_mgr._name_var.set(last[1].split()[0])
                self._term_mgr._idx_var.set(last[0])
        if self.core_check_vars and p.get("core_atom_hint"):
            hint=p["core_atom_hint"]
            for idx,name,var in self.core_check_vars:
                var.set(name.startswith(hint))
            self._update_core_counter()

    def _load_mol2(self,which):
        path=filedialog.askopenfilename(title=f"Select {which} mol2",
                                         filetypes=[("MOL2 files","*.mol2"),("All files","*.*")])
        if not path: return
        try:
            atoms,bonds=parse_mol2(path); bonds=deduplicate_bonds(bonds)
        except Exception as e:
            messagebox.showerror("Parse Error",f"Could not read mol2:\n{e}"); return

        if which=='core':
            self.core_file.set(path); self.core_atoms,self.core_bonds=atoms,bonds
            self._log(f"Loaded core: {os.path.basename(path)}"
                      f"  —  {len(atoms)} atoms, {len(bonds)} bonds",'ok')
            self._populate_core_atoms()
            self._core_viewer.load(atoms,bonds)  #
        elif which=='monomer':
            self.monomer_file.set(path); self.mon_atoms,self.mon_bonds=atoms,bonds
            self._log(f"Loaded monomer: {os.path.basename(path)}"
                      f"  —  {len(atoms)} atoms, {len(bonds)} bonds",'ok')
            self._populate_monomer_combos()
            self._auto_select_by_hint(PRESETS.get(self.preset_var.get(),{}))
            self._mon_viewer.load(atoms,bonds)  #
        elif which=='terminal':
            self.terminal_file.set(path); self.term_atoms,self.term_bonds=atoms,bonds
            self._log(f"Loaded terminal: {os.path.basename(path)}"
                      f"  —  {len(atoms)} atoms, {len(bonds)} bonds",'ok')
            self._populate_term_combos()

    def _browse_conda_path(self):
        d=filedialog.askdirectory(title="Select conda base directory (miniconda3 or anaconda3)")
        if d: self.amber_conda_path.set(d)

    def _auto_detect_ambertools(self):
        import subprocess as _sp
        candidates = [
            os.path.expanduser('~/miniconda3'),
            os.path.expanduser('~/anaconda3'),
            os.path.expanduser('~/mambaforge'),
            '/opt/conda','/usr/local/conda',
        ]
        envs_to_try = ['AmberTools25','AmberTools24','AmberTools23','amber22','base']
        for base in candidates:
            if not os.path.isdir(base): continue
            for env in envs_to_try:
                for bin_path in [
                    os.path.join(base,'envs',env,'bin','antechamber'),
                    os.path.join(base,'bin','antechamber'),
                ]:
                    if os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
                        self.amber_conda_path.set(base)
                        self.amber_conda_env.set(env)
                        self._amber_test_lbl.config(
                            text=f"  Auto-detected: {base}  env={env}",
                            fg=self._T['green'])
                        self._log(f"  [AmberTools] Auto-detected: {base}  env={env}",'ok')
                        return
        self._amber_test_lbl.config(
            text="  Not found — enter path manually",fg=self._T['red'])
        self._log("  [AmberTools] Could not auto-detect. Enter path manually.",'warn')

    def _test_ambertools(self):
        """[10] Test antechamber is reachable with current path settings."""
        env=self.amber_conda_env.get().strip()
        base=self.amber_conda_path.get().strip()
        cmd=f"source {base}/etc/profile.d/conda.sh && conda activate {env} && antechamber -h"
        self._amber_test_lbl.config(text="  Testing...",fg=self._T['yellow'])
        self.root.update_idletasks()
        try:
            # shell=True intentional here: cmd may be 'which antechamber' or similar
            r=subprocess.run(cmd,shell=True,capture_output=True,text=True,
                             timeout=20,executable='/bin/bash')
            if r.returncode==0 or 'antechamber' in (r.stdout+r.stderr).lower():
                self._amber_test_lbl.config(text="  OK — antechamber found and working",
                                             fg=self._T['green'])
                self._log(f"  AmberTools test OK: env={env}  base={base}",'ok')
            else:
                self._amber_test_lbl.config(
                    text="  FAIL — antechamber not found, check path and env name",
                    fg=self._T['red'])
                self._log(f"  AmberTools test FAILED\n  stdout: {r.stdout[:200]}\n  stderr: {r.stderr[:200]}",'err')
        except subprocess.TimeoutExpired:
            self._amber_test_lbl.config(text="  TIMEOUT — conda slow to activate, try again",
                                         fg=self._T['yellow'])
        except Exception as e:
            self._amber_test_lbl.config(text=f"  ERROR — {e}",fg=self._T['red'])

    def _amber_source_lines(self):
        """Return shell lines to source conda and activate AmberTools env."""
        env=self.amber_conda_env.get().strip() or "AmberTools25"
        base=self.amber_conda_path.get().strip()
        if base:
            return (f"source {base}/etc/profile.d/conda.sh\n"
                    f"conda activate {env}\n")
        return f"conda activate {env}\n"

    def _choose_out_dir(self):
        d=filedialog.askdirectory(title="Select output folder")
        if d: self.output_dir.set(d)

    def _load_mol2_silent(self,which,path):
        try:
            atoms,bonds=parse_mol2(path); bonds=deduplicate_bonds(bonds)
            if which=='core':
                self.core_file.set(path)
                self.core_atoms,self.core_bonds=atoms,bonds
                self._populate_core_atoms()
                self._core_viewer.load(atoms,bonds)
            elif which=='monomer':
                self.monomer_file.set(path)
                self.mon_atoms,self.mon_bonds=atoms,bonds
                self._populate_monomer_combos()
                self._mon_viewer.load(atoms,bonds)
            elif which=='terminal':
                self.terminal_file.set(path)
                self.term_atoms,self.term_bonds=atoms,bonds
                self._populate_term_combos()
        except Exception: pass

    def _populate_core_atoms(self):
        T=self._T
        for w in self.core_atom_frame.winfo_children(): w.destroy()
        self.core_check_vars=[]
        hdr=tk.Frame(self.core_atom_frame,bg=T['panel']); hdr.pack(fill='x',pady=(0,4))
        for txt,w in [("Idx",6),("Name",8),("Type",10)]:
            tk.Label(hdr,text=txt,width=w,anchor='w',
                     font=("Courier New",10,"bold"),fg=T['muted'],bg=T['panel']).pack(side='left')
        tk.Frame(self.core_atom_frame,bg=T['border'],height=1).pack(fill='x',pady=2)
        lc=tk.Canvas(self.core_atom_frame,bg=T['panel'],height=160,highlightthickness=0)
        lsb=ttk.Scrollbar(self.core_atom_frame,orient='vertical',command=lc.yview)
        lf=tk.Frame(lc,bg=T['panel'])
        lf.bind("<Configure>",lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.create_window((0,0),window=lf,anchor='nw')
        lc.configure(yscrollcommand=lsb.set)
        lc.pack(side='left',fill='both',expand=True); lsb.pack(side='right',fill='y')
        heavy=[a for a in self.core_atoms if not is_hydrogen(a)]
        cols=4
        for i,a in enumerate(heavy):
            var=tk.BooleanVar(value=False)
            lbl=f"{a['idx']:3d}  {a['name']:<6}  {a['type']}"
            tk.Checkbutton(lf,text=lbl,variable=var,
                           font=("Courier New",10),fg=T['cyan'],bg=T['panel'],
                           selectcolor=T['sel'],activebackground=T['panel'],cursor='hand2',
                           command=self._update_core_counter
                           ).grid(row=i//cols,column=i%cols,padx=4,pady=1,sticky='w')
            self.core_check_vars.append((a['idx'],a['name'],var))
        self._update_core_counter()

    def _update_core_counter(self):
        T=self._T
        sel=sum(1 for _,_,v in self.core_check_vars if v.get())
        tot=len(self.core_check_vars)
        self.core_counter_lbl.config(text=f"Selected: {sel} of {tot} atoms",
                                      fg=T['green'] if sel>0 else T['yellow'])
        self._update_viewer_roles()

    def _filter_core_atoms(self):
        filt=self.core_filter_var.get().strip().upper()
        if not filt: return
        for idx,name,var in self.core_check_vars: var.set(name.upper().startswith(filt))
        self._update_core_counter()

    def _select_all_core(self):
        for _,_,v in self.core_check_vars: v.set(True)
        self._update_core_counter()

    def _clear_all_core(self):
        for _,_,v in self.core_check_vars: v.set(False)
        self._update_core_counter()

    def _populate_monomer_combos(self):
        if not self.mon_atoms: return
        amap_mon = {a['idx']:a for a in self.mon_atoms}
        opts=[f"{a['idx']}:{a['name']}  [{a['type']}]" for a in self.mon_atoms]
        h_opts=[o for o in opts if
                amap_mon.get(int(o.split(':')[0])) and (
                amap_mon[int(o.split(':')[0])]['name'].upper().startswith('H') or
                amap_mon[int(o.split(':')[0])]['type'].lower().startswith('h'))]
        self.mon_input_cb['values'] = opts
        self.mon_input_h_cb['values'] = opts   # all atoms — user picks H or Cl/Br/O etc.
        # #1 terminal leaving atoms come from the multi-atom manager (04b), which
        #    pulls from self.mon_atoms via its atom_source — nothing to populate here.
        c_atoms=[o for o in opts if amap_mon.get(int(o.split(':')[0])) and
                 amap_mon[int(o.split(':')[0])]['name'].upper().startswith('C')]
        if c_atoms: self.mon_input_var.set(c_atoms[0]); self._on_mon_input_changed(None)

    def _update_namd_times(self, *_):
        """[v1.0] Recompute time estimates from current GUI state."""
        if not hasattr(self, '_namd_time_lbl'): return
        try:
            n_mon  = len(self.mon_atoms)  if self.mon_atoms  else 20
            n_core = len(self.core_atoms) if self.core_atoms else 8
            bf     = self.branching_var.get()
            n_gen  = self.num_gens_var.get()
            ts     = self.namd_timestep.get()
            heat_ps  = self.namd_heat_ps.get()
            equil_ns = self.namd_equil_ns.get()
            prod_ns  = self.namd_prod_ns.get()
            ncores   = self.namd_cores_var.get()
            # Total atom estimate (dendrimer only; solvent adds ~10×)
            n_den = n_core
            for g in range(1, n_gen + 1):
                n_den += (4 * bf**g) * n_mon
            n_total = n_den * 10   # rough solvated estimate
            # Perf model: ~5 ns/day per core for TIP3P at 2 fs (CPU)
            rate = max(0.001, 5.0 * ncores / max(1, n_total / 1000))
            dcd_freq  = self.namd_dcd_freq.get()
            traj_gb   = prod_ns * 1e6 / dcd_freq * (n_total * 3 * 4) / 1e9
            txt = (f"  Dendrimer: ~{n_den:,} atoms   Solvated: ~{n_total:,} atoms\n"
                   f"  Minimisation    :  ~5–15 min   ({ncores} cores, GAFF2/TIP3P)\n"
                   f"  Heating {heat_ps:.0f} ps   :  ~{max(1,int(heat_ps/1000/rate*24))} hr\n"
                   f"  NPT equil {equil_ns:.0f} ns :  ~{max(1,int(equil_ns/rate))} day(s)\n"
                   f"  Production {prod_ns:.0f} ns :  ~{max(1,int(prod_ns/rate))} day(s)\n"
                   f"  DCD trajectory  :  ~{traj_gb:.1f} GB   ({dcd_freq} steps × {ts:.0f} fs)")
            self._namd_time_lbl.config(text=txt)
        except Exception:
            pass

    def _gen_namd_inputs(self):
        """Generate a self-contained NAMD 3.x package per ticked generation.
        Inputs (per gen folder): {pre}_typed.mol2 and {pre}.frcmod from the AMBER pipeline.
        Writes: 00_solvate.tleap, 01_minimize.conf, 02_heat.conf, 03_nvt_equil.conf,
        04_npt_equil.conf, 05_production.conf, analysis.tcl, run_namd.sh."""
        import stat as _s, math
        out   = self.output_dir.get()
        dtype = self.den_type_var.get() or 'dendrimer'
        try: os.makedirs(out, exist_ok=True)
        except Exception as e: messagebox.showerror('Output dir', str(e)); return

        # ── Parameters from GUI ───────────────────────────────────────────────
        max_gen   = self.num_gens_var.get()
        namd_bin  = self.namd_binary_var.get()
        ncores    = self.namd_cores_var.get()
        ff        = self.namd_ff_var.get()
        water     = self.namd_water_var.get()
        box_pad   = self.namd_box_var.get()
        conc      = self.namd_conc_var.get()
        temp      = self.namd_temp_var.get()
        ts        = self.namd_timestep.get()
        heat_ps   = self.namd_heat_ps.get()
        equil_ns  = self.namd_equil_ns.get()
        prod_ns   = self.namd_prod_ns.get()
        dcd_freq  = self.namd_dcd_freq.get()
        ener_freq = max(500, dcd_freq // 5)

        # ── Step counts ───────────────────────────────────────────────────────
        heat_steps    = int(heat_ps  * 1000 / ts)
        nvt_steps     = int(500      * 1000 / ts)   # 500 ps NVT between heat and NPT
        equil_half    = int(equil_ns * 1e6  / ts) // 2
        prod_steps    = int(prod_ns  * 1e6  / ts)
        _n_reassign   = max(1, heat_steps // 1000)
        heat_incr     = round(min(2.0, temp / _n_reassign), 3)

        # ── Loop over ticked generations (Section 07) ─────────────────────────
        selected_gens = [g for g, v in self.gen_check_vars if v.get()]
        if not selected_gens:
            selected_gens = [max_gen]
            self._log("  [NAMD] No generations ticked — using max gen only", 'warn')

        self._log(f"  [NAMD] Generating packages for generations: {selected_gens}", 'head')
        all_folders = []
        all_file_counts = []

        for gen in selected_gens:
            pre     = f"dendrimer_{dtype.lower()}_gen{gen}"
            gen_out = os.path.join(out, f"{dtype.lower()}_g{gen}")
            os.makedirs(gen_out, exist_ok=True)

            self._log(f"\n  {'='*46}\n  G{gen}  ({pre})\n  {'='*46}", 'head')

# ── Box estimate from last built Rg or growth formula ─────────────────
            CORE_ARMS = {'PAMAM':4,'PPI':4,'PLL':2,'Triazine':3,
                         'Carbosilane':4,'Phosphorus':3,'Polyester':3,
                         'Polyglycerol':3,'Custom':self.branching_var.get()}
            f_arms = CORE_ARMS.get(dtype, self.branching_var.get())
            bf_val = self.branching_var.get()

            # Use actual last-built Rg if available, else estimate
            rg_est = None
            if hasattr(self, '_last_built_rg') and self._last_built_rg:
                rg_est = self._last_built_rg
                self._log(f"  [NAMD] Using actual built Rg = {rg_est:.2f} Å for box estimate", 'ok')
            if not rg_est:
                n_mon_at  = len(self.mon_atoms)  if self.mon_atoms  else 17
                n_core_at = len(self.core_atoms) if self.core_atoms else 8
                n_at = n_core_at
                for _g in range(1, gen + 1):
                    n_at += f_arms * (bf_val ** (_g - 1)) * 2 * n_mon_at
                rg_ref = {'PAMAM':6.7,'PPI':5.2,'PLL':5.0,'Triazine':5.5,
                          'Carbosilane':7.5,'Phosphorus':6.0,
                          'Polyester':4.5,'Polyglycerol':4.2,'Custom':5.0}
                ref_rg = rg_ref.get(dtype, 5.0)
                ref_N  = {'PAMAM':84,'PPI':58,'PLL':40,'Triazine':70,
                          'Carbosilane':120,'Phosphorus':80,
                          'Polyester':55,'Polyglycerol':40,'Custom':60}
                ref_n  = ref_N.get(dtype, 60)
                rg_est = max(5.0, ref_rg * (max(1, n_at) / ref_n) ** (1/3))

            # Ensures box edge ≥ 2×Rg + 24 Å so molecule never sees its periodic image
            auto_pad = max(box_pad, rg_est + 12.0)
            box_est  = max(60.0, rg_est * 4 + 2 * auto_pad)

            # ── Type-specific MD parameters ───────────────────────────────────────
            TYPE_PARAMS = {
                'PAMAM':       {'rigid':'all',   'ts':2.0, 'piston':200, 'decay':100},
                'PPI':         {'rigid':'all',   'ts':2.0, 'piston':200, 'decay':100},
                'PLL':         {'rigid':'all',   'ts':2.0, 'piston':200, 'decay':100},
                'Polyester':   {'rigid':'all',   'ts':2.0, 'piston':200, 'decay':100},
                'Polyglycerol':{'rigid':'all',   'ts':2.0, 'piston':500, 'decay':250},
                'Triazine':    {'rigid':'water', 'ts':1.0, 'piston':250, 'decay':125},
                'Phosphorus':  {'rigid':'water', 'ts':1.0, 'piston':250, 'decay':125},
                'Carbosilane': {'rigid':'all',   'ts':2.0, 'piston':250, 'decay':125},
                'Custom':      {'rigid':'all',   'ts':2.0, 'piston':200, 'decay':100},
            }
            tp = TYPE_PARAMS.get(dtype, TYPE_PARAMS['Custom'])
            rigid_bonds   = tp['rigid']
            piston_period = tp['piston']
            piston_decay  = tp['decay']
            # GUI override: use GUI timestep if user explicitly changed it
            if ts != 2.0: pass  # user overrode — keep their value
            else: ts = tp['ts']
            heat_steps = int(heat_ps  * 1000 / ts)
            nvt_steps  = int(500      * 1000 / ts)
            equil_half = int(equil_ns * 1e6  / ts) // 2
            prod_steps = int(prod_ns  * 1e6  / ts)
            _n_reassign = max(1, heat_steps // 1000)
            heat_incr   = round(min(2.0, temp / _n_reassign), 3)

            # ── PME grid — must be products of 2,3,5 ─────────────────────────────
            def next_pme(n):
                n = math.ceil(n)
                while True:
                    x = n
                    for p in [2, 3, 5]:
                        while x % p == 0: x //= p
                    if x == 1: return n
                    n += 1

            pme_g = next_pme(box_est)

            # ── Ion count from box volume ─────────────────────────────────────────
            box_vol_L = (box_est ** 3) * 1e-27 * 1e3   # Å³ → litres
            n_ions    = max(0, int(conc * box_vol_L * 6.022e23))
            if conc > 0:
                ions_line = f"addionsrand mol Na+ {n_ions} Cl- {n_ions}"
            else:
                # partial charges from the typed mol2 file (not from log)
                ions_line = "addionsrand mol Na+ __NACL_COUNT__ Cl- 0"

# ── Warn if upstream AMBER files missing ──────────────────────────────
            for _f in [f"{pre}_typed.mol2", f"{pre}.frcmod"]:
                if not os.path.exists(os.path.join(gen_out, _f)):
                    self._log(f"  [NAMD] WARNING: {_f} not found in output folder — "
                              f"run AMBER Fragment BCC pipeline first", 'warn')

            # ── Write helper ──────────────────────────────────────────────────────
            files = []
            def w(name, content):
                fp = os.path.join(gen_out, name)
                with open(fp, 'w') as f: f.write(content)
                files.append(name)
                return fp

            # 00_solvate.tleap
            w('00_solvate.tleap', f"""# ============================================================
# UDB v1.0  {dtype} G{gen}  tLeap Solvation
source leaprc.gaff2
source leaprc.water.tip3p

loadamberparams {pre}.frcmod
mol = loadmol2 {pre}_typed.mol2

check mol
charge mol

center mol
solvatebox mol TIP3PBOX {box_pad:.1f}
{ions_line}
charge mol

saveamberparm mol {pre}_solvated.prmtop {pre}_solvated.inpcrd
savepdb mol {pre}_solvated.pdb
quit
""")

            # ── Shared NAMD config block ──────────────────────────────────────────
            common = f"""amber               on
parmfile            {pre}_solvated.prmtop
exclude             scaled1-4
oneFourScaling      1.0
cutoff              12.0
switching           on
switchdist          10.0
pairlistdist        14.0
stepspercycle       20
PME                 yes
PMEGridSizeX        {pme_g}
PMEGridSizeY        {pme_g}
PMEGridSizeZ        {pme_g}
wrapAll             on
wrapWater           on
rigidBonds          {rigid_bonds}
rigidTolerance      1e-8
timestep            {ts:.1f}
"""

            # ── T10: NAMD 3 GPU-resident split ───────────────────────────────
            # Minimisation is NOT supported on the GPU-resident fast path, so
            # 01_minimize runs with it OFF; every dynamics stage runs it ON.
            # GPU-resident (NAMD 3.0+) requires: TIP3P water (used here),
            # Langevin piston not Berendsen (used), and no fixed atoms / Colvars
            # / GBIS / TCL forces (none used) — all satisfied by this generator.
            # Launch with ~1 PE per GPU (run_namd.sh uses +p1 +devices 0); more
            # cores slow GPU-resident down. Keep outputEnergies infrequent
            # (>=400) — frequent energy output stalls the GPU.
            _namd_gpu_var = getattr(self, 'namd_gpu_var', None)
            namd_gpu = _namd_gpu_var.get() if _namd_gpu_var is not None else True
            gpu_dyn = ("GPUresident         on        ;# NAMD 3.0+ "
                       "(replaces deprecated CUDASOAintegrate)\n") if namd_gpu else ""

            # 01_minimize.conf
            w('01_minimize.conf', f"""# ============================================================
# {dtype} G{gen}  Stage 1: Energy Minimisation
# Cell vectors are auto-patched from inpcrd by run_namd.sh
# GPUresident is intentionally OFF here — minimisation is unsupported on the
# NAMD 3 GPU-resident fast path. Dynamics stages (02-05) enable it.
{common}
ambercoor           {pre}_solvated.inpcrd
outputName          01_minimize
binaryoutput        yes
binaryrestart       yes
outputEnergies      100
restartfreq         500
dcdfreq             500
xstFreq             500
cellBasisVector1    {box_est:.4f}  0.0          0.0
cellBasisVector2    0.0          {box_est:.4f}  0.0
cellBasisVector3    0.0          0.0          {box_est:.4f}
cellOrigin          0.0          0.0          0.0
temperature         0
minimize            2000
""")

            # 02_heat.conf  — NVT 0 → T K
            w('02_heat.conf', f"""# ============================================================
# {dtype} G{gen}  Stage 2: NVT Heating  0 → {temp:.0f} K  ({heat_ps:.0f} ps)
{common}
{gpu_dyn}ambercoor           {pre}_solvated.inpcrd
bincoordinates      01_minimize.restart.coor
binvelocities       01_minimize.restart.vel
extendedSystem      01_minimize.restart.xsc
outputName          02_heat
binaryoutput        yes
binaryrestart       yes
outputEnergies      {ener_freq}
restartfreq         1000
dcdfreq             1000
xstFreq             1000
langevin            on
langevinDamping     1.0
langevinTemp        {temp:.1f}
langevinHydrogen    off
reassignFreq        1000
reassignTemp        0
reassignIncr        {heat_incr}
reassignHold        {temp:.1f}
run                 {heat_steps}
""")

            # 03_nvt_equil.conf  — NVT at T K (intermediate stage, 500 ps)
            w('03_nvt_equil.conf', f"""# ============================================================
# {dtype} G{gen}  Stage 3: NVT Equilibration  {temp:.0f} K  (500 ps)
# Intermediate stage — stabilises before barostat is switched on
{common}
{gpu_dyn}ambercoor           {pre}_solvated.inpcrd
bincoordinates      02_heat.restart.coor
binvelocities       02_heat.restart.vel
extendedSystem      02_heat.restart.xsc
outputName          03_nvt_equil
binaryoutput        yes
binaryrestart       yes
outputEnergies      {ener_freq}
restartfreq         2000
dcdfreq             2000
xstFreq             2000
langevin            on
langevinDamping     1.0
langevinTemp        {temp:.1f}
langevinHydrogen    off
run                 {nvt_steps}
""")

            # 04_npt_equil.conf  — NPT at T K / 1 atm with restraints then released
            w('04_npt_equil.conf', f"""# ============================================================
# {dtype} G{gen}  Stage 4: NPT Equilibration  {temp:.0f} K  ({equil_ns:.0f} ns)
# Phase 1 ({equil_ns/2:.0f} ns): 1 kcal/mol/A^2 restraints on dendrimer heavy atoms
# Phase 2 ({equil_ns/2:.0f} ns): fully unrestrained
# Restraint ref: {pre}_restrained.pdb  (B-col=1 for heavy atoms, 0 for water)
{common}
{gpu_dyn}ambercoor           {pre}_solvated.inpcrd
bincoordinates      03_nvt_equil.restart.coor
binvelocities       03_nvt_equil.restart.vel
extendedSystem      03_nvt_equil.restart.xsc
outputName          04_npt_equil
binaryoutput        yes
binaryrestart       yes
outputEnergies      {ener_freq}
restartfreq         5000
dcdfreq             5000
xstFreq             5000
langevin            on
langevinDamping     1.0
langevinTemp        {temp:.1f}
langevinHydrogen    off
useGroupPressure    yes
useFlexibleCell     no
useConstantArea     no
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    {piston_period:.1f}
langevinPistonDecay     {piston_decay:.1f}
langevinPistonTemp      {temp:.1f}
margin              5.0
constraints         on
consref             {pre}_restrained.pdb
conskfile           {pre}_restrained.pdb
conskcol            B
constraintScaling   1.0
run                 {equil_half}
constraintScaling   0.0
run                 {equil_half}
""")

            # 05_production.conf  — NPT production
            w('05_production.conf', f"""# ============================================================
# {dtype} G{gen}  Stage 5: NPT Production  {temp:.0f} K  ({prod_ns:.0f} ns)
{common}
{gpu_dyn}ambercoor           {pre}_solvated.inpcrd
bincoordinates      04_npt_equil.restart.coor
binvelocities       04_npt_equil.restart.vel
extendedSystem      04_npt_equil.restart.xsc
outputName          05_production
binaryoutput        yes
binaryrestart       yes
outputEnergies      {ener_freq}
restartfreq         {dcd_freq * 4}
dcdfreq             {dcd_freq}
xstFreq             {dcd_freq}
langevin            on
langevinDamping     1.0
langevinTemp        {temp:.1f}
langevinHydrogen    off
useGroupPressure    yes
useFlexibleCell     no
useConstantArea     no
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    {piston_period:.1f}
langevinPistonDecay     {piston_decay:.1f}
langevinPistonTemp      {temp:.1f}
margin              5.0
run                 {prod_steps}
""")

            # 05_production_restart.conf  — resume after walltime kill
            w('05_production_restart.conf', f"""# ============================================================
# {dtype} G{gen}  Stage 5 RESTART — resume production after walltime kill
# Usage: bash run_namd.sh restart
{common}
{gpu_dyn}ambercoor           {pre}_solvated.inpcrd
bincoordinates      05_production.restart.coor
binvelocities       05_production.restart.vel
extendedSystem      05_production.restart.xsc
outputName          05_production
binaryoutput        yes
binaryrestart       yes
outputEnergies      {ener_freq}
restartfreq         {dcd_freq * 4}
dcdfreq             {dcd_freq}
xstFreq             {dcd_freq}
firsttimestep       FIRSTTIMESTEP
langevin            on
langevinDamping     1.0
langevinTemp        {temp:.1f}
langevinHydrogen    off
useGroupPressure    yes
useFlexibleCell     no
useConstantArea     no
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    {piston_period:.1f}
langevinPistonDecay     {piston_decay:.1f}
langevinPistonTemp      {temp:.1f}
margin              5.0
run                 {prod_steps}
""")

            # analysis.tcl  — Rg, asphericity, RDF, SASA, end-to-end, H-bonds
            w('analysis.tcl', f"""# ============================================================
# UDB v1.0  {dtype} G{gen}  — Full Analysis Suite
mol new {pre}_solvated.prmtop type amber waitfor all
mol addfile 04_npt_equil.dcd type dcd waitfor all
mol addfile 05_production.dcd type dcd waitfor all
set nframes [molinfo top get numframes]
puts "Loaded $nframes frames"

# Atom selections
set den_heavy  [atomselect top "not water and not name Na Cl and not name H*"]
set den_all    [atomselect top "not water and not name Na Cl"]
set water_O    [atomselect top "water and name O"]
set terminal_N [atomselect top "not water and name N* and not name Na and numbonds 1"]

# Output files
set o_rg     [open "{pre}_Rg.dat"       w]
set o_delta  [open "{pre}_delta.dat"    w]
set o_rmsd   [open "{pre}_RMSD.dat"     w]
set o_ee     [open "{pre}_end2end.dat"  w]
set o_rdens  [open "{pre}_rdensity.dat" w]
puts $o_rg    "# Frame  Time_ns  Rg_heavy_A  Rg_all_A"
puts $o_delta "# Frame  Time_ns  delta  lambda1  lambda2  lambda3"
puts $o_rmsd  "# Frame  Time_ns  RMSD_A"
puts $o_ee    "# Frame  Time_ns  max_ee_dist_A  mean_ee_dist_A"
puts $o_rdens "# Frame  Time_ns  shell_Rg  density_at_0.5Rg  density_at_Rg  density_at_1.5Rg"

# Reference for RMSD (first frame)
$den_heavy frame 0
set ref_pos [$den_heavy get {{x y z}}]

for {{set fr 0}} {{$fr < $nframes}} {{incr fr}} {{
    $den_heavy frame $fr
    $den_all   frame $fr

    # ── Centre of mass ──────────────────────────────────────────────────────
    set coords_h [$den_heavy get {{x y z}}]
    set coords_a [$den_all   get {{x y z}}]
    set n_h [llength $coords_h]
    set n_a [llength $coords_a]
    set cx 0; set cy 0; set cz 0
    foreach c $coords_h {{
            set cx [expr {{$cx+[lindex $c 0]}}]
            set cy [expr {{$cy+[lindex $c 1]}}]
            set cz [expr {{$cz+[lindex $c 2]}}]
    }}
    set cx [expr {{$cx/$n_h}}]; set cy [expr {{$cy/$n_h}}]; set cz [expr {{$cz/$n_h}}]

    # ── Rg heavy ────────────────────────────────────────────────────────────
    set rg2_h 0
    set Gxx 0; set Gyy 0; set Gzz 0
    set Gxy 0; set Gxz 0; set Gyz 0
    foreach c $coords_h {{
            set dx [expr {{[lindex $c 0]-$cx}}]
            set dy [expr {{[lindex $c 1]-$cy}}]
            set dz [expr {{[lindex $c 2]-$cz}}]
            set rg2_h [expr {{$rg2_h+$dx*$dx+$dy*$dy+$dz*$dz}}]
            set Gxx [expr {{$Gxx+$dx*$dx}}]; set Gyy [expr {{$Gyy+$dy*$dy}}]; set Gzz [expr {{$Gzz+$dz*$dz}}]
            set Gxy [expr {{$Gxy+$dx*$dy}}]; set Gxz [expr {{$Gxz+$dx*$dz}}]; set Gyz [expr {{$Gyz+$dy*$dz}}]
    }}
    set rg_h [expr {{sqrt($rg2_h/$n_h)}}]
    set Gxx [expr {{$Gxx/$n_h}}]; set Gyy [expr {{$Gyy/$n_h}}]; set Gzz [expr {{$Gzz/$n_h}}]
    set Gxy [expr {{$Gxy/$n_h}}]; set Gxz [expr {{$Gxz/$n_h}}]; set Gyz [expr {{$Gyz/$n_h}}]

    # ── Rg all atoms ────────────────────────────────────────────────────────
    set cx2 0; set cy2 0; set cz2 0
    foreach c $coords_a {{
            set cx2 [expr {{$cx2+[lindex $c 0]}}]
            set cy2 [expr {{$cy2+[lindex $c 1]}}]
            set cz2 [expr {{$cz2+[lindex $c 2]}}]
    }}
    set cx2 [expr {{$cx2/$n_a}}]; set cy2 [expr {{$cy2/$n_a}}]; set cz2 [expr {{$cz2/$n_a}}]
    set rg2_a 0
    foreach c $coords_a {{
            set dx [expr {{[lindex $c 0]-$cx2}}]
            set dy [expr {{[lindex $c 1]-$cy2}}]
            set dz [expr {{[lindex $c 2]-$cz2}}]
            set rg2_a [expr {{$rg2_a+$dx*$dx+$dy*$dy+$dz*$dz}}]
    }}
    set rg_a [expr {{sqrt($rg2_a/$n_a)}}]
    set t_ns [expr {{$fr*{dcd_freq}*{ts:.1f}/1e6}}]
    puts $o_rg [format "%6d  %8.3f  %8.3f  %8.3f" $fr $t_ns $rg_h $rg_a]

    # ── Asphericity delta ────────────────────────────────────────────────────
    set I1 [expr {{$Gxx+$Gyy+$Gzz}}]
    set I2 [expr {{$Gxx*$Gyy+$Gyy*$Gzz+$Gxx*$Gzz-$Gxy*$Gxy-$Gxz*$Gxz-$Gyz*$Gyz}}]
    set delta [expr {{abs($I1)>1e-10 ? 1.0-3.0*$I2/($I1*$I1) : 0.0}}]
    puts $o_delta [format "%6d  %8.3f  %8.5f  %8.3f  %8.3f  %8.3f" \
            $fr $t_ns $delta $Gxx $Gyy $Gzz]

    # ── RMSD ────────────────────────────────────────────────────────────────
    set curr_pos [$den_heavy get {{x y z}}]
    set rmsd2 0
    foreach c $curr_pos r $ref_pos {{
            set dx [expr {{[lindex $c 0]-[lindex $r 0]}}]
            set dy [expr {{[lindex $c 1]-[lindex $r 1]}}]
            set dz [expr {{[lindex $c 2]-[lindex $r 2]}}]
            set rmsd2 [expr {{$rmsd2+$dx*$dx+$dy*$dy+$dz*$dz}}]
    }}
    set rmsd [expr {{sqrt($rmsd2/$n_h)}}]
    puts $o_rmsd [format "%6d  %8.3f  %8.3f" $fr $t_ns $rmsd]

    # ── End-to-end distances (terminal heavy atoms → core centre) ────────────
    $terminal_N frame $fr
    set tn_coords [$terminal_N get {{x y z}}]
    if {{[llength $tn_coords] > 0}} {{
            set max_ee 0; set sum_ee 0; set n_ee 0
            foreach c $tn_coords {{
                set dx [expr {{[lindex $c 0]-$cx}}]
                set dy [expr {{[lindex $c 1]-$cy}}]
                set dz [expr {{[lindex $c 2]-$cz}}]
                set d [expr {{sqrt($dx*$dx+$dy*$dy+$dz*$dz)}}]
                if {{$d > $max_ee}} {{set max_ee $d}}
                set sum_ee [expr {{$sum_ee+$d}}]; incr n_ee
            }}
            set mean_ee [expr {{$n_ee>0 ? $sum_ee/$n_ee : 0}}]
            puts $o_ee [format "%6d  %8.3f  %8.3f  %8.3f" $fr $t_ns $max_ee $mean_ee]
    }}

    # ── Radial density (atoms per shell at 0.5*Rg, Rg, 1.5*Rg) ─────────────
    set r1 [expr {{$rg_h*0.5}}]; set r2 $rg_h; set r3 [expr {{$rg_h*1.5}}]
    set dr 1.0; set vol [expr {{4*3.14159*$r2*$r2*$dr}}]
    set n1 0; set n2 0; set n3 0
    foreach c $coords_h {{
            set dx [expr {{[lindex $c 0]-$cx}}]
            set dy [expr {{[lindex $c 1]-$cy}}]
            set dz [expr {{[lindex $c 2]-$cz}}]
            set d [expr {{sqrt($dx*$dx+$dy*$dy+$dz*$dz)}}]
            if {{$d>=$r1-$dr&&$d<$r1+$dr}} {{incr n1}}
            if {{$d>=$r2-$dr&&$d<$r2+$dr}} {{incr n2}}
            if {{$d>=$r3-$dr&&$d<$r3+$dr}} {{incr n3}}
    }}
    puts $o_rdens [format "%6d  %8.3f  %8.3f  %8.4f  %8.4f  %8.4f" \
            $fr $t_ns $rg_h \
            [expr {{$n1/(4*3.14159*$r1*$r1*$dr)}}] \
            [expr {{$vol>0 ? $n2/$vol : 0}}] \
            [expr {{$n3/(4*3.14159*$r3*$r3*$dr)}}]]

    if {{$fr%100==0}} {{
            puts "Frame $fr/$nframes  t=[format %.1f $t_ns]ns  Rg=[format %.2f $rg_h]A  delta=[format %.4f $delta]"
            flush stdout
    }}
}}
close $o_rg; close $o_delta; close $o_rmsd; close $o_ee; close $o_rdens
puts "\nAnalysis complete:"
puts "  {pre}_Rg.dat          — radius of gyration vs time"
puts "  {pre}_delta.dat       — asphericity vs time"
puts "  {pre}_RMSD.dat        — RMSD from equilibrated structure"
puts "  {pre}_end2end.dat     — arm end-to-end distances"
puts "  {pre}_rdensity.dat    — radial density profile"
puts "\nReference (Maiti 2004 PAMAM G3): Rg~14-19A, delta<0.05"
quit
""")

            # run_namd.sh  — THE MASTER SCRIPT  (bash run_namd.sh)
            traj_gb = prod_ns * 1e6 / dcd_freq * (50000 * 3 * 4) / 1e9
            sh_path = w('run_namd.sh', f"""#!/bin/bash
# UDB v1.0  {dtype} G{gen}  Complete NAMD Pipeline
# Usage:          bash run_namd.sh
# Restart prod:   bash run_namd.sh restart
# Custom cores:   bash run_namd.sh 32
# Required in this folder:
#   {pre}_typed.mol2    (from UDB AMBER pipeline)
#   {pre}.frcmod        (from UDB AMBER pipeline)
# All other files are generated automatically.

NAMD_BIN="{namd_bin}"
NCORES="${{1:-{ncores}}}"
NAMD="$NAMD_BIN +p$NCORES"
# T10: GPU-resident dynamics launch — NAMD 3 wants ~1 PE per GPU; extra cores
# slow it down. Minimisation (01) keeps the multi-core CPU launch ($NAMD).
NAMD_GPU="$NAMD_BIN +p1 +setcpuaffinity +devices 0"
mkdir -p logs

# ── Colour output helpers ─────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; WHT='\033[1;37m'; NC='\033[0m'
ok()   {{ echo -e "${{GRN}}  [OK]${{NC}} $1"; }}
err()  {{ echo -e "${{RED}}  [ERROR]${{NC}} $1"; }}
info() {{ echo -e "${{CYN}}  $1${{NC}}"; }}
warn() {{ echo -e "${{YLW}}  [WARN]${{NC}} $1"; }}

# ── Restart mode ─────────────────────────────────────────────────────────────
RESTART_MODE=0
if [ "$1" = "restart" ]; then
    RESTART_MODE=1
    NCORES="${{2:-{ncores}}}"
    NAMD="$NAMD_BIN +p$NCORES"
fi

echo ""
echo -e "${{WHT}}======================================================${{NC}}"
echo -e "${{WHT}}  UDB v1.0  {dtype} G{gen}  NAMD Pipeline${{NC}}"
echo -e "${{WHT}}  NAMD:    $NAMD_BIN${{NC}}"
echo -e "${{WHT}}  Cores:   $NCORES  |  Start: $(date)${{NC}}"
echo -e "${{WHT}}======================================================${{NC}}"

# ── Validate binary ───────────────────────────────────────────────────────────
if [ -f "$NAMD_BIN" ] && [ -x "$NAMD_BIN" ]; then
    :
elif command -v "$NAMD_BIN" >/dev/null 2>&1; then
    NAMD_BIN="$(command -v "$NAMD_BIN")"
    NAMD="$NAMD_BIN +p$NCORES"
else
    err "NAMD binary not found or not executable: $NAMD_BIN"
    err "Edit NAMD_BIN= at the top of this script or add it to PATH."
    exit 1
fi

# ── Validate input files ──────────────────────────────────────────────────────
for f in "{pre}_typed.mol2" "{pre}.frcmod"; do
    if [ ! -f "$f" ]; then
            err "Required file missing: $f"
            err "Run UDB AMBER Fragment BCC pipeline first."
            exit 1
    fi
done
ok "Input files verified"

# ── Handle restart mode ───────────────────────────────────────────────────────
if [ $RESTART_MODE -eq 1 ]; then
    if [ ! -f "05_production.restart.coor" ]; then
            err "No production restart file found (05_production.restart.coor)"
            err "Cannot restart — no previous production run detected."
            exit 1
    fi
    info "RESTART MODE: resuming production from last checkpoint..."
    # Read last timestep from xsc file
    LAST_TS=$(tail -1 05_production.restart.xsc | awk '{{print $1}}')
    sed -i "s/firsttimestep.*FIRSTTIMESTEP/firsttimestep       $LAST_TS/" \
            05_production_restart.conf
    info "Resuming from timestep $LAST_TS"
    $NAMD 05_production_restart.conf >> logs/05_production.log
    [ $? -ne 0 ] && {{ err "Restart failed — check logs/05_production.log"; exit 1; }}
    ok "Production restart complete"
    echo ""
    echo -e "${{WHT}}======================================================"
    echo -e "  Restart complete!  $(date)${{NC}}"
    exit 0
fi

# STEP 0 — tLeap solvation
echo ""
info "[0/5] tLeap solvation (TIP3P, {box_pad:.0f} A padding)..."
if [ ! -f "{pre}_solvated.prmtop" ]; then
    if [ -f "{pre}_typed.mol2" ]; then
        MOL2_CHARGE=$(python3 -c "
import sys
total=0.0
in_atom=False
for l in open(\"{pre}_typed.mol2\"):
    if \'@<TRIPOS>ATOM\' in l: in_atom=True; continue
    if l.startswith(\'@\'): in_atom=False
    if in_atom:
        p=l.split()
        if len(p)>=9:
            try: total+=float(p[8])
            except: pass
print(max(0, round(abs(total))))
" 2>/dev/null || echo "0")
        if [ "$MOL2_CHARGE" -gt 0 ] 2>/dev/null; then
            warn "[v1.0] Fractional charge detected — adding $MOL2_CHARGE Na+ ions"
            sed -i "s/addionsrand mol Na+ __NACL_COUNT__ Cl- 0/addionsrand mol Na+ $MOL2_CHARGE Cl- 0/" 00_solvate.tleap
        else
            sed -i "s/addionsrand mol Na+ __NACL_COUNT__ Cl- 0/addionsrand mol Na+ 0 Cl- 0/" 00_solvate.tleap
        fi
    else
        sed -i "s/addionsrand mol Na+ __NACL_COUNT__ Cl- 0/addionsrand mol Na+ 0 Cl- 0/" 00_solvate.tleap
    fi
    tleap -f 00_solvate.tleap 2>&1 | tee logs/00_solvate.log
    if [ ! -f "{pre}_solvated.prmtop" ]; then
            err "tLeap failed — check logs/00_solvate.log"
            exit 1
    fi

    # ── Auto-read actual box from inpcrd and patch all conf files ────────────
    BOX_LINE=$(tail -1 {pre}_solvated.inpcrd)
    BX=$(echo $BOX_LINE | awk '{{print $1}}')
    BY=$(echo $BOX_LINE | awk '{{print $2}}')
    BZ=$(echo $BOX_LINE | awk '{{print $3}}')
    CX=$(echo "$BX / 2" | bc -l | xargs printf "%.4f")
    CY=$(echo "$BY / 2" | bc -l | xargs printf "%.4f")
    CZ=$(echo "$BZ / 2" | bc -l | xargs printf "%.4f")

    # ── PME grid: re-compute from actual box ─────────────────────────────────
    PME_FUNC() {{
            python3 -c "
import math, sys
n = math.ceil(float(sys.argv[1]))
while True:
    x = n
    for p in [2,3,5]:
            while x%p==0: x//=p
    if x==1: print(n); break
    n+=1
" $1
    }}
    PX=$(PME_FUNC $BX); PY=$(PME_FUNC $BY); PZ=$(PME_FUNC $BZ)

    info "  Actual box: ${{BX}} x ${{BY}} x ${{BZ}} A"
    info "  PME grid:   ${{PX}} x ${{PY}} x ${{PZ}}"
    info "  Cell origin: ${{CX}}  ${{CY}}  ${{CZ}}"

    # Patch 01_minimize.conf with real values
    sed -i "s/cellBasisVector1.*/cellBasisVector1    $BX  0.0    0.0/"  01_minimize.conf
    sed -i "s/cellBasisVector2.*/cellBasisVector2    0.0    $BY  0.0/"  01_minimize.conf
    sed -i "s/cellBasisVector3.*/cellBasisVector3    0.0    0.0    $BZ/" 01_minimize.conf
    sed -i "s/cellOrigin.*/cellOrigin          $CX  $CY  $CZ/"         01_minimize.conf
    for conf in 01_minimize.conf 02_heat.conf 03_nvt_equil.conf \
                    04_npt_equil.conf 05_production.conf 05_production_restart.conf; do
            [ -f "$conf" ] && \
            sed -i "s/PMEGridSizeX.*/PMEGridSizeX        $PX/" $conf && \
            sed -i "s/PMEGridSizeY.*/PMEGridSizeY        $PY/" $conf && \
            sed -i "s/PMEGridSizeZ.*/PMEGridSizeZ        $PZ/" $conf
    done
    ok "cellBasisVector + PME grid patched from actual inpcrd"

else
    ok "Solvated files found — skipping tLeap"
    BX=$(tail -1 {pre}_solvated.inpcrd | awk '{{print $1}}')
    BY=$(tail -1 {pre}_solvated.inpcrd | awk '{{print $2}}')
    BZ=$(tail -1 {pre}_solvated.inpcrd | awk '{{print $3}}')
    info "  Existing box: ${{BX}} x ${{BY}} x ${{BZ}} A"
fi

python3 - << 'PYEND'
import sys
with open("{pre}_solvated.pdb") as f: lines = f.readlines()
out = []
for l in lines:
    if l[:4] in ("ATOM","HETA"):
            name = l[12:16].strip()
            resn = l[17:20].strip()
            is_water = resn in ("WAT","HOH","TIP","TP3")
            is_ion   = name in ("Na","Cl","NA","CL")
            is_H     = name.startswith("H") or name.startswith("h")
            bfac = "0.00" if (is_water or is_ion or is_H) else "1.00"
            out.append(l[:60] + f"{{bfac:>6}}" + l[66:])
    else:
            out.append(l)
with open("{pre}_restrained.pdb","w") as f: f.writelines(out)
heavy = sum(1 for l in out if l[:4] in ("ATOM","HETA") and l[12:16].strip()[:1] not in ("H","h"))
print(f"  Restraint PDB: {pre}_restrained.pdb  ({{heavy}} constrained heavy atoms)")
PYEND
ok "Restraint PDB regenerated from current solvated structure"

# STEP 1 — Energy minimisation
echo ""
info "[1/5] Energy minimisation (5000 steps)..."
if [ ! -f "01_minimize.restart.coor" ] || [ ! -f "01_minimize.restart.vel" ] || [ ! -f "01_minimize.restart.xsc" ]; then
    $NAMD 01_minimize.conf > logs/01_minimize.log 2>&1
    if [ $? -ne 0 ]; then
            err "Minimisation failed"
            tail -30 logs/01_minimize.log
            exit 1
    fi
    ok "Minimisation complete"
else
    ok "Skipping — restart file found"
fi

# STEP 2 — NVT heating 0 → {temp:.0f} K
echo ""
info "[2/5] NVT heating  0 → {temp:.0f} K  ({heat_ps:.0f} ps)..."
if [ ! -f "02_heat.restart.coor" ] || [ ! -f "02_heat.restart.vel" ] || [ ! -f "02_heat.restart.xsc" ]; then
    {("$NAMD_GPU" if namd_gpu else "$NAMD")} 02_heat.conf > logs/02_heat.log 2>&1
    if [ $? -ne 0 ]; then
            err "Heating failed"
            tail -30 logs/02_heat.log
            exit 1
    fi
    ok "Heating complete"
else
    ok "Skipping — restart file found"
fi

# STEP 3 — NVT equilibration 500 ps
echo ""
info "[3/5] NVT equilibration at {temp:.0f} K  (500 ps)..."
if [ ! -f "03_nvt_equil.restart.coor" ] || [ ! -f "03_nvt_equil.restart.vel" ] || [ ! -f "03_nvt_equil.restart.xsc" ]; then
    {("$NAMD_GPU" if namd_gpu else "$NAMD")} 03_nvt_equil.conf > logs/03_nvt_equil.log 2>&1
    if [ $? -ne 0 ]; then
            err "NVT equilibration failed"
            tail -30 logs/03_nvt_equil.log
            exit 1
    fi
    ok "NVT equilibration complete"
else
    ok "Skipping — restart file found"
fi

# STEP 4 — NPT equilibration {equil_ns:.0f} ns
echo ""
info "[4/5] NPT equilibration  {temp:.0f} K / 1 atm  ({equil_ns:.0f} ns)..."
if [ ! -f "04_npt_equil.restart.coor" ] || [ ! -f "04_npt_equil.restart.vel" ] || [ ! -f "04_npt_equil.restart.xsc" ]; then
    {("$NAMD_GPU" if namd_gpu else "$NAMD")} 04_npt_equil.conf > logs/04_npt_equil.log 2>&1
    if [ $? -ne 0 ]; then
            err "NPT equilibration failed"
            tail -30 logs/04_npt_equil.log
            exit 1
    fi
    ok "NPT equilibration complete"
else
    ok "Skipping — restart file found"
fi

# STEP 5 — NPT production {prod_ns:.0f} ns
echo ""
info "[5/5] NPT production  {temp:.0f} K  ({prod_ns:.0f} ns)..."
info "      Estimated trajectory: ~{traj_gb:.1f} GB"
{("$NAMD_GPU" if namd_gpu else "$NAMD")} 05_production.conf > logs/05_production.log 2>&1
if [ $? -ne 0 ]; then
    err "Production failed"
    tail -30 logs/05_production.log
    exit 1
fi
ok "Production complete"

echo ""
info "Running VMD analysis..."
if command -v vmd &>/dev/null; then
    vmd -dispdev none -e analysis.tcl > logs/analysis.log 2>&1
    ok "Analysis complete"
    info "  Output files:"
    for f in {pre}_Rg.dat {pre}_delta.dat {pre}_RMSD.dat \
                 {pre}_end2end.dat {pre}_rdensity.dat; do
            [ -f "$f" ] && echo "    $f"
    done
else
    warn "VMD not found — skipping analysis"
    warn "Run manually: vmd -e analysis.tcl"
fi

echo ""
echo -e "${{WHT}}======================================================"
echo    "  Pipeline complete!  $(date)"
echo    "  To restart if killed:  bash run_namd.sh restart"
echo -e "======================================================${{NC}}"
""")

            os.chmod(sh_path, os.stat(sh_path).st_mode | _s.S_IEXEC | _s.S_IXGRP | _s.S_IXOTH)

            # ── continue_npt.sh  — patch & resume after NPT patch-grid crash ──────
            cont_path = w('continue_npt.sh', f"""#!/bin/bash
# ============================================================
# UDB v1.0  {dtype} G{gen}  — NPT continuation script
# Use this when step 4 (NPT equilibration) fails with:
#   "Periodic cell has become too small for original patch grid"
#
# What it does:
#   1. Patches 04_npt_equil.conf / 05_production.conf /
#      05_production_restart.conf  (adds margin, slows piston)
#   2. Re-runs run_namd.sh — steps 1-3 are skipped automatically
#      because their restart files already exist.
#
# Usage:
#   bash continue_npt.sh           (uses same core count as run_namd.sh)
#   bash continue_npt.sh 32        (override core count)
# ============================================================

set -e
NCORES="${{1:-{ncores}}}"
NAMD_BIN="{namd_bin}"
NAMD="$NAMD_BIN +p$NCORES"

RED='\\033[0;31m'; GRN='\\033[0;32m'; CYN='\\033[0;36m'
YLW='\\033[1;33m'; WHT='\\033[1;37m'; NC='\\033[0m'
ok()   {{ echo -e "${{GRN}}  [OK]${{NC}} $1"; }}
err()  {{ echo -e "${{RED}}  [ERROR]${{NC}} $1"; }}
info() {{ echo -e "${{CYN}}  $1${{NC}}"; }}
warn() {{ echo -e "${{YLW}}  [WARN]${{NC}} $1"; }}

echo ""
echo -e "${{WHT}}============================================================${{NC}}"
echo -e "${{WHT}}  UDB v1.0  NPT continuation — {dtype} G{gen}${{NC}}"
echo -e "${{WHT}}  Cores: $NCORES  |  Start: $(date)${{NC}}"
echo -e "${{WHT}}============================================================${{NC}}"

# ── Guard: make sure steps 1-3 are actually done ─────────────────────────────
for f in 01_minimize.restart.coor 02_heat.restart.coor 03_nvt_equil.restart.coor \\
         01_minimize.restart.vel  02_heat.restart.vel  03_nvt_equil.restart.vel  \\
         01_minimize.restart.xsc  02_heat.restart.xsc  03_nvt_equil.restart.xsc; do
    if [ ! -f "$f" ]; then
        err "Missing checkpoint: $f"
        err "Steps 1-3 must be complete before using this script."
        exit 1
    fi
done
ok "Steps 1-3 checkpoint files verified"

# ── Patch conf files ──────────────────────────────────────────────────────────
patch_conf() {{
    local conf="$1"
    [ ! -f "$conf" ] && warn "$conf not found — skipping" && return
    cp "$conf" "${{conf}}.bak"

    # Add margin 5.0 after langevinPistonTemp (idempotent — skip if already present)
    if ! grep -q "^margin" "$conf"; then
        sed -i '/^langevinPistonTemp/a margin              5.0' "$conf"
        ok "  $conf — added margin 5.0"
    else
        ok "  $conf — margin already present"
    fi

    # Slow piston: period 200→500, decay 100→250
    sed -i 's/^langevinPistonPeriod[[:space:]]\\+[0-9.]\\+/langevinPistonPeriod    500.0/' "$conf"
    sed -i 's/^langevinPistonDecay[[:space:]]\\+[0-9.]\\+/langevinPistonDecay     250.0/'  "$conf"
    ok "  $conf — piston period/decay set to 500/250 fs"
}}

info "Patching NPT conf files..."
patch_conf 04_npt_equil.conf
patch_conf 05_production.conf
patch_conf 05_production_restart.conf

# ── Remove any partial step-4 output so run_namd.sh re-runs it ───────────────
if [ -f "04_npt_equil.restart.coor" ]; then
    warn "Partial step-4 restart files found — removing so step 4 reruns cleanly"
    rm -f 04_npt_equil.restart.coor \\
          04_npt_equil.restart.vel  \\
          04_npt_equil.restart.xsc  \\
          04_npt_equil.restart.coor.BAK \\
          04_npt_equil.restart.vel.BAK  \\
          04_npt_equil.restart.xsc.BAK
    ok "Partial step-4 files removed"
fi

# ── Validate NAMD binary ──────────────────────────────────────────────────────
if [ ! -x "$NAMD_BIN" ] && ! command -v "$NAMD_BIN" &>/dev/null; then
    err "NAMD binary not found: $NAMD_BIN"
    err "Edit NAMD_BIN= at the top of this script."
    exit 1
fi

mkdir -p logs

echo ""
info "Resuming from step 4 (NPT equilibration)..."
echo ""

# STEP 4 — NPT equilibration
info "[4/5] NPT equilibration  300 K / 1 atm..."
$NAMD 04_npt_equil.conf > logs/04_npt_equil.log 2>&1
if [ $? -ne 0 ]; then
    err "NPT equilibration failed — check logs/04_npt_equil.log"
    tail -30 logs/04_npt_equil.log
    exit 1
fi
ok "NPT equilibration complete"

# STEP 5 — NPT production
echo ""
info "[5/5] NPT production..."
{("$NAMD_GPU" if namd_gpu else "$NAMD")} 05_production.conf > logs/05_production.log 2>&1
if [ $? -ne 0 ]; then
    err "Production failed — check logs/05_production.log"
    tail -30 logs/05_production.log
    exit 1
fi
ok "Production complete"

echo ""
echo -e "${{WHT}}============================================================${{NC}}"
echo    "  NPT continuation complete!  $(date)"
echo    "  To restart production if killed:  bash run_namd.sh restart"
echo -e "${{WHT}}============================================================${{NC}}"
""")
            os.chmod(cont_path, os.stat(cont_path).st_mode | _s.S_IEXEC | _s.S_IXGRP | _s.S_IXOTH)

            # ── Log and notify ────────────────────────────────────────────────────
            self._log(f"\n  NAMD package written to: {out}", 'head')
            self._log(f"  Type: {dtype} G{gen}  |  FF: {ff}  |  Water: {water}", 'ok')
            self._log(f"  Box: ~{box_est:.0f} A  |  Temp: {temp:.0f} K  |  Prod: {prod_ns:.0f} ns", 'ok')
            self._log(f"  rigidBonds: {rigid_bonds}  |  ts: {ts:.1f} fs  |  Piston: {piston_period} ps", 'ok')
            for nm in files:
                self._log(f"    -> {nm}", 'ok')
            all_folders.append(gen_out)
            all_file_counts.append(len(files))
            self._log(f"  [NAMD] G{gen} → {gen_out}  ({len(files)} files)", 'ok')

        # ── Final summary (after all generations) ─────────────────────────────
        total_files = sum(all_file_counts)
        folder_list = "\n".join(f"  {f}" for f in all_folders)
        self._log(f"\n  NAMD packages complete: {len(selected_gens)} generation(s), "
                  f"{total_files} files total", 'ok')
        messagebox.showinfo("NAMD Packages Ready",
            f"{len(selected_gens)} generation(s) | {total_files} files total\n\n"
            f"Folders:\n{folder_list}\n\n"
            f"For each folder copy the matching _typed.mol2 + .frcmod\n"
            f"then run:  bash run_namd.sh")

    def _clear_all_fields(self):
        if not messagebox.askyesno("Clear All","Reset ALL fields to defaults?\nThis cannot be undone."): return
        self.core_file.set(''); self.monomer_file.set(''); self.terminal_file.set('')
        self.core_atoms=[]; self.core_bonds=[]; self.mon_atoms=[]; self.mon_bonds=[]
        self.term_atoms=[]; self.term_bonds=[]
        self.core_str_file.set(''); self.monomer_str_file.set('')
        self.terminal_str_file.set(''); self.cgenff_use.set(False)
        if hasattr(self,'_cgenff_status_lbl'):
            self._cgenff_status_lbl.config(
                text="Load .str files then click Validate to check penalty scores.",
                fg=self._T['muted'])
        for w in self.core_atom_frame.winfo_children(): w.destroy()
        self.core_check_vars=[]; self._update_core_counter()
        for cb in [self.mon_input_cb,self.mon_input_h_cb]:
            cb['values']=[]; cb.set('')
        for v in [self.mon_input_var,self.mon_input_h_var]: v.set('')
        self.labile_mode_var.set('single')
        if self._labile_input_mgr is not None:
            self._labile_input_mgr.lb.delete(0,'end')
        self._on_labile_mode_changed()
        if hasattr(self,'term_input_cb'):
            for cb in [self.term_input_cb,self.term_input_h_cb,self.term_output_cb]:
                cb['values']=[]; cb.set('')
        for v in [self.term_input_var,self.term_input_h_var,self.term_output_var]: v.set('')
        if hasattr(self,'_term_outer'): self._term_outer.pack_forget()
        self._apply_preset("PAMAM"); self.preset_var.set("PAMAM")
        self.num_gens_var.set(6); self._rebuild_gen_selector()
        for var,val in [(self.pipe_save_mol2,True),(self.pipe_save_pdb,True),
                        (self.pipe_save_script,True),(self.pipe_antechamber,False),
                        (self.pipe_parmchk2,False),(self.pipe_tleap,False),
                        (self.pipe_minimize,False),(self.pipe_ambpdb,False),
                        (self.pipe_solvate,False),(self.pipe_add_ions,False),(self.pipe_md,False),
                        (self.intergen_enabled,False),(self.intergen_use_amber,True),
                        (self.intergen_use_namd,False),(self.intergen_save_gaff,True)]:
            var.set(val)
        self.intergen_namd_mode.set('vacuum')
        self.net_charge_var.set(0); self.pipe_box_size.set(12.0); self.pipe_prod_ns.set(5.0)
        self.ff_var.set('gaff'); self.charge_method_var.set('bcc')
        self.output_dir.set(os.path.expanduser("~/dendrimer_output"))
        self.progress_overall['value']=0; self.progress_gen['value']=0
        T=self._T
        self.prog_label.config(text="Ready — configure above and click Build",fg=T['muted'])
        self.prog_gen_label.config(text=""); self._atom_lbl.config(text="")
        self.log_box.delete('1.0','end')
        # clear viewers
        self._core_viewer.load([],[])
        self._mon_viewer.load([],[])
        # clear validator
        for item in self._val_tree.get_children(): self._val_tree.delete(item)
        for item in self._preview_tree.get_children(): self._preview_tree.delete(item)
        self._last_built_rg = None
        self._log("  All fields cleared. Ready for a new build.",'ok')

    def _save_config(self):
        p=filedialog.asksaveasfilename(defaultextension=".json",
                                        filetypes=[("JSON","*.json"),("All","*.*")],
                                        title="Save config")
        if not p: return
        cfg={
            # molecule files + atom selections
            'core_file':self.core_file.get(),
            'monomer_file':self.monomer_file.get(),
            'terminal_file':self.terminal_file.get(),
            'output_dir':self.output_dir.get(),
            'preset':self.preset_var.get(),
            'den_type':self.den_type_var.get(),
            'bond_length_auto':self.bond_len_auto.get(),
            'bond_length':self.bond_len_var.get(),
            'bond_angle_auto':self.bond_ang_auto.get(),
            'bond_angle':self.bond_ang_var.get(),
            'branching_factor':self.branching_var.get(),
            'num_gens':self.num_gens_var.get(),
            'mon_input':self.mon_input_var.get(),
            'mon_input_h':self.mon_input_h_var.get(),
            'terminal_names':self._term_mgr.get_names(),
            'labile_mode':  self.labile_mode_var.get(),
            'labile_names': (self._labile_input_mgr.get_names()
                             if self.labile_mode_var.get()=='group'
                             and self._labile_input_mgr else []),
            'labile_idxs':  (self._labile_input_mgr.get_idxs()
                             if self.labile_mode_var.get()=='group'
                             and self._labile_input_mgr else []),
            'core_binding':[(idx,name) for idx,name,v in self.core_check_vars if v.get()],
            # w12.2 atom-selection additions
            'mon_term_leave_names': (self._mon_term_leave_mgr.get_names()
                                     if hasattr(self, '_mon_term_leave_mgr') else []),
            'mon_term_leave_auto': self.mon_term_leave_auto.get(),
            'mon_leave_auto':      self.mon_leave_auto.get(),
            'core_leave_auto':     self.core_leave_auto.get(),
            'term_input':          self.term_input_var.get(),
            'term_input_h':        self.term_input_h_var.get(),
            'tgroup_bind':         self.tgroup_bind_var.get(),
            'tgroup_leave':        self.tgroup_leave_var.get(),
            'tgroup_leave_auto':   self.tgroup_leave_auto.get(),
            'gens_to_save':[g for g,v in self.gen_check_vars if v.get()],
            'out_formats':[f for f,v in [('mol2',self.fmt_mol2),('pdb',self.fmt_pdb),
                                          ('xyz',self.fmt_xyz),('gro',self.fmt_gro),
                                          ('psf',self.fmt_psf)] if v.get()],
            'param_method':self.param_method_var.get(),
            'ff':self.ff_var.get(),
            'charge_method':self.charge_method_var.get(),
            'net_charge':self.net_charge_var.get(),
            'pipe_save_mol2':self.pipe_save_mol2.get(),
            'pipe_save_pdb':self.pipe_save_pdb.get(),
            'pipe_save_script':self.pipe_save_script.get(),
            'pipe_antechamber':self.pipe_antechamber.get(),
            'pipe_parmchk2':self.pipe_parmchk2.get(),
            'pipe_tleap':self.pipe_tleap.get(),
            'pipe_minimize':self.pipe_minimize.get(),
            'pipe_ambpdb':self.pipe_ambpdb.get(),
            'pipe_solvate':self.pipe_solvate.get(),
            'pipe_add_ions':self.pipe_add_ions.get(),
            'pipe_md':self.pipe_md.get(),
            'pipe_box_size':self.pipe_box_size.get(),
            'pipe_prod_ns':self.pipe_prod_ns.get(),
            'cgenff_use':        self.cgenff_use.get(),
            'core_str_file':     self.core_str_file.get(),
            'monomer_str_file':  self.monomer_str_file.get(),
            'terminal_str_file': self.terminal_str_file.get(),
            'namd_binary':       self.namd_binary_var.get(),
            'namd_cores':        self.namd_cores_var.get(),
            'namd_ff':           self.namd_ff_var.get(),
            'namd_water':        self.namd_water_var.get(),
            'namd_box_pad':      self.namd_box_var.get(),
            'namd_conc':         self.namd_conc_var.get(),
            'namd_temp':         self.namd_temp_var.get(),
            'namd_heat_ps':      self.namd_heat_ps.get(),
            'namd_equil_ns':     self.namd_equil_ns.get(),
            'namd_prod_ns':      self.namd_prod_ns.get(),
            'namd_timestep':     self.namd_timestep.get(),
            'namd_dcd_freq':     self.namd_dcd_freq.get(),
            'namd_solvate':      self.namd_solvate.get(),
            'namd_minimize':     self.namd_minimize.get(),
            'namd_heat':         self.namd_heat.get(),
            'namd_equil':        self.namd_equil.get(),
            'namd_prod':         self.namd_prod.get(),
            'namd_cores':        self.namd_cores_var.get(),
            'pipe_ncpus':        self.pipe_ncpus.get(),
            'obabel_minimize':   self.obabel_minimize.get(),
            'obabel_ff':         self.obabel_ff_var.get(),
            'obabel_steps':      self.obabel_steps_var.get(),
            'clash_thresh':      self.clash_thresh_var.get(),
            'amber_conda_env':   self.amber_conda_env.get(),
            'amber_conda_path':  self.amber_conda_path.get(),
            'intergen_enabled':  self.intergen_enabled.get(),
            'intergen_use_amber':self.intergen_use_amber.get(),
            'intergen_use_namd': self.intergen_use_namd.get(),
            'intergen_namd_mode':self.intergen_namd_mode.get(),
            'intergen_save_gaff':self.intergen_save_gaff.get(),
        }
        with open(p,'w') as f: json.dump(cfg,f,indent=2)
        self._log(f"  Config saved → {p}",'ok')

    def _load_config(self):
        p=filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All","*.*")],title="Load config")
        if not p: return
        with open(p) as f: cfg=json.load(f)
        self.core_file.set(cfg.get('core_file',''))
        self.monomer_file.set(cfg.get('monomer_file',''))
        self.terminal_file.set(cfg.get('terminal_file',''))
        self.output_dir.set(cfg.get('output_dir',''))
        self.den_type_var.set(cfg.get('den_type','PAMAM'))
        self.bond_len_auto.set(cfg.get('bond_length_auto',True))
        self.bond_len_var.set(cfg.get('bond_length',1.47))
        self.bond_ang_auto.set(cfg.get('bond_angle_auto',True))
        self.bond_ang_var.set(cfg.get('bond_angle',109.5))
        self.branching_var.set(cfg.get('branching_factor',2))
        self.num_gens_var.set(cfg.get('num_gens',6))
        self._rebuild_gen_selector()
        # reload mol2 files silently first so atom lists are populated
        for kind in ('core','monomer','terminal'):
            fp=cfg.get(f'{kind}_file','')
            if fp and os.path.exists(fp): self._load_mol2_silent(kind,fp)
        # restore atom selections after mol2 lists are populated
        if cfg.get('mon_input'): self.mon_input_var.set(cfg['mon_input'])
        if cfg.get('mon_input_h'): self.mon_input_h_var.set(cfg['mon_input_h'])
        lmode = cfg.get('labile_mode', 'single')
        self.labile_mode_var.set(lmode)
        if self._labile_input_mgr is not None:
            self._labile_input_mgr.lb.delete(0,'end')
        if lmode == 'group' and self._labile_input_mgr is not None:
            names = cfg.get('labile_names', [])
            idxs  = cfg.get('labile_idxs',  [])
            for i, n in enumerate(names):
                self._labile_input_mgr._name_var.set(n)
                self._labile_input_mgr._idx_var.set(str(idxs[i]) if i < len(idxs) else '')
                self._labile_input_mgr._add()
        self._on_labile_mode_changed()
        # w12.2 atom-selection additions (missing keys default to auto/empty = legacy)
        self.mon_leave_auto.set(cfg.get('mon_leave_auto', True))
        self.core_leave_auto.set(cfg.get('core_leave_auto', True))
        self.mon_term_leave_auto.set(cfg.get('mon_term_leave_auto', True))
        # restore terminal leaving atoms into the multi-atom manager (w12.3);
        # accept the legacy single 'mon_term_leave' string too.
        if hasattr(self, '_mon_term_leave_mgr'):
            self._mon_term_leave_mgr.lb.delete(0, 'end')
            _mtl_names = cfg.get('mon_term_leave_names')
            if not _mtl_names and cfg.get('mon_term_leave'):
                _mtl_names = [cfg['mon_term_leave']]
            for _nm in (_mtl_names or []):
                _plain = _nm.split(':')[1].split()[0] if ':' in _nm else _nm.split()[0]
                self._mon_term_leave_mgr._name_var.set(_plain)
                self._mon_term_leave_mgr._idx_var.set('')
                self._mon_term_leave_mgr._add()
        if cfg.get('term_input'):   self.term_input_var.set(cfg['term_input'])
        if cfg.get('term_input_h'): self.term_input_h_var.set(cfg['term_input_h'])
        self.tgroup_leave_auto.set(cfg.get('tgroup_leave_auto', True))
        if cfg.get('tgroup_bind'):  self.tgroup_bind_var.set(cfg['tgroup_bind'])
        if cfg.get('tgroup_leave'): self.tgroup_leave_var.set(cfg['tgroup_leave'])
        # sync combobox enabled/disabled state to the restored auto flags
        self._on_mon_leave_auto_changed()
        self._on_mon_term_leave_auto_changed()
        self._on_tgroup_leave_auto_changed()
        # restore terminal names in manager
        for name in cfg.get('terminal_names',[]):
            self._term_mgr._name_var.set(name)
            self._term_mgr._add()
        # restore core binding checkboxes by atom index
        binding_set={idx for idx,_ in cfg.get('core_binding',[])}
        for idx,name,var in self.core_check_vars:
            var.set(idx in binding_set)
        self._update_core_counter()
        # restore generation selections
        saved_gens=set(cfg.get('gens_to_save',[]))
        for g,v in self.gen_check_vars:
            v.set(g in saved_gens if saved_gens else True)
        if 'param_method'    in cfg: self.param_method_var.set(cfg['param_method'])
        if 'ff'              in cfg: self.ff_var.set(cfg['ff'])
        if 'charge_method'   in cfg:
            _cm_clean, _cm_changed, _cm_reason = sanitise_charge_method(cfg['charge_method'])
            if _cm_changed:
                self._log(f"  [Config] {_cm_reason}", 'warn')
            self.charge_method_var.set(_cm_clean)
        if 'net_charge'      in cfg: self.net_charge_var.set(cfg['net_charge'])
        if 'pipe_save_mol2'  in cfg: self.pipe_save_mol2.set(cfg['pipe_save_mol2'])
        if 'pipe_save_pdb'   in cfg: self.pipe_save_pdb.set(cfg['pipe_save_pdb'])
        if 'pipe_save_script'in cfg: self.pipe_save_script.set(cfg['pipe_save_script'])
        if 'pipe_antechamber'in cfg: self.pipe_antechamber.set(cfg['pipe_antechamber'])
        if 'pipe_parmchk2'   in cfg: self.pipe_parmchk2.set(cfg['pipe_parmchk2'])
        if 'pipe_tleap'      in cfg: self.pipe_tleap.set(cfg['pipe_tleap'])
        if 'pipe_minimize'   in cfg: self.pipe_minimize.set(cfg['pipe_minimize'])
        if 'pipe_ambpdb'     in cfg: self.pipe_ambpdb.set(cfg['pipe_ambpdb'])
        if 'pipe_solvate'    in cfg: self.pipe_solvate.set(cfg['pipe_solvate'])
        if 'pipe_add_ions'   in cfg: self.pipe_add_ions.set(cfg['pipe_add_ions'])
        if 'pipe_md'         in cfg: self.pipe_md.set(cfg['pipe_md'])
        if 'pipe_box_size'   in cfg: self.pipe_box_size.set(cfg['pipe_box_size'])
        if 'pipe_prod_ns'    in cfg: self.pipe_prod_ns.set(cfg['pipe_prod_ns'])
        if 'cgenff_use'        in cfg: self.cgenff_use.set(cfg['cgenff_use'])
        if 'core_str_file'     in cfg: self.core_str_file.set(cfg['core_str_file'])
        if 'monomer_str_file'  in cfg: self.monomer_str_file.set(cfg['monomer_str_file'])
        if 'terminal_str_file' in cfg: self.terminal_str_file.set(cfg['terminal_str_file'])
        if 'namd_binary'   in cfg: self.namd_binary_var.set(cfg['namd_binary'])
        if 'namd_cores'    in cfg: self.namd_cores_var.set(cfg['namd_cores'])
        if 'namd_ff'       in cfg: self.namd_ff_var.set(cfg['namd_ff'])
        if 'namd_water'    in cfg: self.namd_water_var.set(cfg['namd_water'])
        if 'namd_box_pad'  in cfg: self.namd_box_var.set(cfg['namd_box_pad'])
        if 'namd_conc'     in cfg: self.namd_conc_var.set(cfg['namd_conc'])
        if 'namd_temp'     in cfg: self.namd_temp_var.set(cfg['namd_temp'])
        if 'namd_heat_ps'  in cfg: self.namd_heat_ps.set(cfg['namd_heat_ps'])
        if 'namd_equil_ns' in cfg: self.namd_equil_ns.set(cfg['namd_equil_ns'])
        if 'namd_prod_ns'  in cfg: self.namd_prod_ns.set(cfg['namd_prod_ns'])
        if 'namd_timestep' in cfg: self.namd_timestep.set(cfg['namd_timestep'])
        if 'namd_dcd_freq' in cfg: self.namd_dcd_freq.set(cfg['namd_dcd_freq'])
        if 'namd_solvate'  in cfg: self.namd_solvate.set(cfg['namd_solvate'])
        if 'namd_minimize' in cfg: self.namd_minimize.set(cfg['namd_minimize'])
        if 'namd_heat'     in cfg: self.namd_heat.set(cfg['namd_heat'])
        if 'namd_equil'    in cfg: self.namd_equil.set(cfg['namd_equil'])
        if 'namd_prod'     in cfg: self.namd_prod.set(cfg['namd_prod'])
        if 'namd_cores'       in cfg: self.namd_cores_var.set(cfg['namd_cores'])
        if 'pipe_ncpus'       in cfg: self.pipe_ncpus.set(cfg['pipe_ncpus'])
        if 'obabel_minimize'  in cfg: self.obabel_minimize.set(cfg['obabel_minimize'])
        if 'obabel_ff'        in cfg: self.obabel_ff_var.set(cfg['obabel_ff'])
        if 'obabel_steps'     in cfg: self.obabel_steps_var.set(cfg['obabel_steps'])
        if 'clash_thresh'     in cfg: self.clash_thresh_var.set(cfg['clash_thresh'])
        if 'amber_conda_env'   in cfg: self.amber_conda_env.set(cfg['amber_conda_env'])
        if 'amber_conda_path'  in cfg: self.amber_conda_path.set(cfg['amber_conda_path'])
        if 'intergen_enabled'  in cfg: self.intergen_enabled.set(cfg['intergen_enabled'])
        if 'intergen_use_amber'in cfg: self.intergen_use_amber.set(cfg['intergen_use_amber'])
        if 'intergen_use_namd' in cfg: self.intergen_use_namd.set(cfg['intergen_use_namd'])
        if 'intergen_namd_mode'in cfg: self.intergen_namd_mode.set(cfg['intergen_namd_mode'])
        if 'intergen_save_gaff'in cfg: self.intergen_save_gaff.set(cfg['intergen_save_gaff'])
        self._update_viewer_roles()
        self._log(f"  Config loaded: {p}",'ok')

    def _start_build(self):
        T=self._T
        if not self.core_atoms:
            messagebox.showerror("Missing","Load a core mol2 file!"); return
        if not self.mon_atoms:
            messagebox.showerror("Missing","Load a monomer mol2 file!"); return

        # Collect core binding idxs from checkboxes
        core_binding_idxs=[idx for idx,_,v in self.core_check_vars if v.get()]
        if not core_binding_idxs:
            messagebox.showerror("No Selection","Select at least one core binding atom!"); return

        if not self.mon_input_var.get():
            messagebox.showerror("Missing","Select monomer input atom!"); return

        # terminal names + indices from manager
        mon_terminal_names=self._term_mgr.get_names()
        mon_terminal_idxs =self._term_mgr.get_idxs()   
        if not mon_terminal_names:
            messagebox.showerror("Missing","Add at least one terminal atom in Terminal Manager!"); return

        # H atoms on terminal O/N are auto-detected by PATH A; adding them explicitly
        # causes duplicate removal and double-bonding at G2+.
        amap_mon={a['idx']:a for a in self.mon_atoms}
        h_in_terminal=[n for n in mon_terminal_names
                       if any(a['name']==n and is_hydrogen(a) for a in self.mon_atoms)]
        if h_in_terminal:
            if not messagebox.askyesno(
                "Warning — H atoms in Terminal Manager",
                f"Terminal Manager contains hydrogen atom(s): {h_in_terminal}\n\n"
                f"H atoms on terminal O/N sites are stripped automatically by the\n"
                f"PATH A mechanism — you do NOT need to add them here.\n\n"
                f"Adding H atoms as terminals causes duplicate removal and\n"
                f"double-bonding at G2+, corrupting the structure.\n\n"
                f"Continue anyway? (recommended: remove H atoms from Terminal Manager)"):
                return

        gens_to_build=[g for g,v in self.gen_check_vars if v.get()]
        if not gens_to_build:
            messagebox.showerror("No Generations","Tick at least one generation to build!"); return

        def strip_name(s):
            return s.split(':')[1].split()[0] if ':' in s else s

        mon_input_name    = strip_name(self.mon_input_var.get())

        labile_mode = self.labile_mode_var.get()
        if labile_mode == 'single':
            raw = self.mon_input_h_var.get()
            if not raw:
                messagebox.showerror("Missing",
                    "Select a labile atom (Labile Atom(s) → Single Atom)."); return
            labile_single_name = strip_name(raw)
            # parse idx from "idx:name  [type]"
            try:
                labile_single_idx = int(raw.split(':')[0])
            except (ValueError, IndexError):
                self._log("  [WARN] Cannot parse labile atom index — defaulting to 0",'warn')
                labile_single_idx = 0
            mon_input_h_name  = labile_single_name   # backward-compat key
            labile_names      = [labile_single_name]
            labile_idxs       = [labile_single_idx]
        else:
            # group: multiple atoms from FragmentManager
            if self._labile_input_mgr is None or not self._labile_input_mgr.get_names():
                messagebox.showerror("Missing",
                    "Add at least one atom to the Labile Group manager."); return
            labile_names = self._labile_input_mgr.get_names()
            labile_idxs  = self._labile_input_mgr.get_idxs()
            mon_input_h_name = labile_names[0]   # backward-compat: first atom name

        term_input_name   = strip_name(self.term_input_var.get()) if self.term_input_var.get() else mon_input_name
        term_input_h_name = strip_name(self.term_input_h_var.get()) if self.term_input_h_var.get() else mon_input_h_name

        # ── Atom-selection additions (w12.2) ─────────────────────────────
        # #1 Monomer Terminal Leaving Atom(s): manual names only when auto-detect-H off.
        mon_term_leave_auto = self.mon_term_leave_auto.get()
        mon_terminal_leave_names = (
            [strip_name(n) for n in self._mon_term_leave_mgr.get_names()]
            if (not mon_term_leave_auto
                and hasattr(self, '_mon_term_leave_mgr')
                and self._mon_term_leave_mgr.get_names())
            else None)
        # #2/#3 Terminal Group binding + leaving (override terminal-cap bond when set).
        tgroup_bind_name = (strip_name(self.tgroup_bind_var.get())
                            if self.tgroup_bind_var.get() else None)
        tgroup_leave_auto = self.tgroup_leave_auto.get()
        tgroup_leave_name = (strip_name(self.tgroup_leave_var.get())
                             if (not tgroup_leave_auto and self.tgroup_leave_var.get())
                             else None)

        if self.bond_len_auto.get():
            ce=next((a['name'] for a in self.core_atoms if a['idx'] in core_binding_idxs),'N')
            bond_length=auto_bond_length(ce,mon_input_name)
            self._log(f"Auto bond length: {bond_length} Å  ({ce}–{mon_input_name})")
        else:
            bond_length=self.bond_len_var.get()

        if self.bond_ang_auto.get():
            ta=next((a for a in self.mon_atoms if a['name']==mon_terminal_names[0]),None)
            atype=ta['type'] if ta else 'n3'
            bond_angle=auto_bond_angle(atype)
            self._log(f"Auto bond angle: {bond_angle}°  (type: {atype})")
        else:
            bond_angle=self.bond_ang_var.get()

        out_formats=[]
        if self.fmt_mol2.get(): out_formats.append('mol2')
        if self.fmt_pdb.get():  out_formats.append('pdb')
        if self.fmt_xyz.get():  out_formats.append('xyz')
        if self.fmt_gro.get():  out_formats.append('gro')
        if self.fmt_psf.get():  out_formats.append('psf')
        if not out_formats: out_formats=['mol2','pdb']

        pipeline_steps={
            'save_mol2':   self.pipe_save_mol2.get(),
            'save_pdb':    self.pipe_save_pdb.get(),
            'save_script': self.pipe_save_script.get(),
            'antechamber': self.pipe_antechamber.get(),
            'parmchk2':    self.pipe_parmchk2.get(),
            'tleap':       self.pipe_tleap.get(),
            'minimize':    self.pipe_minimize.get(),
            'ambpdb':      self.pipe_ambpdb.get(),
            'solvate':     self.pipe_solvate.get(),
            'add_ions':    self.pipe_add_ions.get(),
            'md':          self.pipe_md.get(),
            'box_size':    self.pipe_box_size.get(),
            'prod_ns':     self.pipe_prod_ns.get(),
            'ff':          self.ff_var.get(),
            'charge_method':self.charge_method_var.get(),
            'param_method':self.param_method_var.get(),  # fragment_bcc or full_antechamber
            'core_file':   self.core_file.get(),          # needed by fragment BCC
            'monomer_file':self.monomer_file.get(),       # needed by fragment BCC
            'terminal_file':self.terminal_file.get(),     # needed by fragment BCC
            'ncpus':        self.pipe_ncpus.get(),         
            'namd_run':     any([self.namd_solvate.get(), self.namd_minimize.get(),
                                 self.namd_heat.get(), self.namd_equil.get(),
                                 self.namd_prod.get()]),
            'namd_binary':  self.namd_binary_var.get(),
            'namd_cores':   self.namd_cores_var.get(),
            'namd_ff':      self.namd_ff_var.get(),
            'namd_water':   self.namd_water_var.get(),
            'namd_box':     self.namd_box_var.get(),
            'namd_conc':    self.namd_conc_var.get(),
            'namd_temp':    self.namd_temp_var.get(),
            'namd_prod_ns': self.namd_prod_ns.get(),
            'namd_equil_ns':self.namd_equil_ns.get(),
            'namd_heat_ps': self.namd_heat_ps.get(),
            'namd_timestep':self.namd_timestep.get(),
            'namd_dcd_freq':self.namd_dcd_freq.get(),
            'namd_solvate': self.namd_solvate.get(),
            'namd_minimize':self.namd_minimize.get(),
            'namd_heat':    self.namd_heat.get(),
            'namd_equil':   self.namd_equil.get(),
            'namd_prod':    self.namd_prod.get(),
            'namd_psf_gen': False,
            'namd_equil_nvt':False,
            'namd_equil_npt':False,
            'namd_gpu':     False,
            'obabel_minimize': self.obabel_minimize.get(),
            'obabel_ff':       self.obabel_ff_var.get(),
            'obabel_steps':    self.obabel_steps_var.get(),
            'cgenff_use':          self.cgenff_use.get(),
            'core_str_file':       self.core_str_file.get(),
            'monomer_str_file':    self.monomer_str_file.get(),
            'terminal_str_file':   self.terminal_str_file.get(),
            'conda_path':      self.amber_conda_path.get().strip(),
            'conda_env':       self.amber_conda_env.get().strip() or 'AmberTools25',
            # Intergenerational minimization (v1.0)
            'intergen_enabled':   self.intergen_enabled.get(),
            'intergen_use_amber': self.intergen_use_amber.get(),
            'intergen_use_namd':  self.intergen_use_namd.get(),
            'intergen_namd_mode': self.intergen_namd_mode.get(),
            'intergen_save_gaff': self.intergen_save_gaff.get(),
        }
        # Single source of truth: sanitise charge_method exactly once here.
        # Every downstream consumer (write_min_script, run_full_pipeline,
        # run_fragment_bcc) trusts the value in this dict.
        _cm_clean, _cm_changed, _cm_reason = sanitise_charge_method(pipeline_steps['charge_method'])
        if _cm_changed:
            self._log(f"  [WARN] {_cm_reason}", 'warn')
            self.charge_method_var.set(_cm_clean)
        pipeline_steps['charge_method'] = _cm_clean
        # the pipeline regardless of the antechamber checkbox state.
        any_amber=(
            pipeline_steps.get('param_method')=='fragment_bcc' or
            any(pipeline_steps[k] for k in
                ['antechamber','parmchk2','tleap','minimize','ambpdb','solvate','add_ions','md']))

        # ── Intergenerational min prerequisite warnings ────────────────────────
        if pipeline_steps.get('intergen_enabled'):
            _ig_gens = [g for g,v in self.gen_check_vars if v.get()]
            if len(_ig_gens) < 2:
                self._log("  [IntergenMin] WARN: Select ≥ 2 generations for "
                          "intergenerational minimization to have any effect.", 'warn')
            if pipeline_steps.get('intergen_use_amber'):
                _amber_ok = all(pipeline_steps.get(k) for k in
                                ['antechamber','parmchk2','tleap','minimize','ambpdb'])
                if not _amber_ok:
                    self._log("  [IntergenMin] WARN: AMBER intergen requires antechamber + "
                              "parmchk2 + tLeap + sander + ambpdb ticked in the AMBER tab. "
                              "Unticked steps will prevent AMBER coord feedback.", 'warn')
            if pipeline_steps.get('intergen_use_namd'):
                _nm = pipeline_steps.get('intergen_namd_mode', 'vacuum')
                if _nm == 'vacuum' and not pipeline_steps.get('tleap'):
                    self._log("  [IntergenMin] WARN: NAMD vacuum intergen requires tLeap "
                              "in the AMBER tab (for prmtop + inpcrd).", 'warn')
                if _nm == 'solvated' and not pipeline_steps.get('namd_solvate'):
                    self._log("  [IntergenMin] WARN: NAMD solvated intergen requires "
                              "namd_solvate ticked in the NAMD tab.", 'warn')

        config={
            'core_file':           self.core_file.get(),
            'monomer_file':        self.monomer_file.get(),
            'terminal_file':       self.terminal_file.get(),
            'core_binding_idxs':   core_binding_idxs,
            'mon_input_name':      mon_input_name,
            'mon_input_h_name':    mon_input_h_name,
            'mon_terminal_names':  mon_terminal_names,
            'mon_terminal_idxs':   mon_terminal_idxs,   
            'labile_mode':         labile_mode,
            'labile_names':        labile_names,
            'labile_idxs':         labile_idxs,
            'term_input_name':     term_input_name,
            'term_input_h_name':   term_input_h_name,
            # w12.2 atom-selection additions (None/auto = pre-w12.2 behaviour)
            'mon_terminal_leave_names': mon_terminal_leave_names,
            'mon_term_leave_auto':     mon_term_leave_auto,
            'mon_leave_auto':          self.mon_leave_auto.get(),
            'core_leave_auto':         self.core_leave_auto.get(),
            'tgroup_bind_name':        tgroup_bind_name,
            'tgroup_leave_name':       tgroup_leave_name,
            'tgroup_leave_auto':       tgroup_leave_auto,
            'bond_length':         bond_length,
            'bond_angle':          bond_angle,
            'branching_factor':    self.branching_var.get(),
            'generations_to_build':gens_to_build,
            'output_dir':          self.output_dir.get(),
            'den_type':            self.den_type_var.get(),
            'out_formats':         out_formats,
        }
        os.makedirs(config['output_dir'],exist_ok=True)

        # reset cancel flag, activate stop button
        self._cancel_flag[0]=False
        self.stop_btn.config(state='normal',text="  STOP  ")
        self.build_btn.config(state='disabled',text="  Building...  ",bg=T['border'])
        self.progress_overall['value']=0; self.progress_gen['value']=0

        # clear preview and validator for fresh build — then re-populate with estimates
        for item in self._val_tree.get_children(): self._val_tree.delete(item)
        self._val_summary_lbl.config(text="")
        self._refresh_preview()   # populate with estimates; live updates will overwrite

        self._log(f"\n{'='*54}",'head')
        self._log(f"  Starting build: {config['den_type']}  v1.0",'head')
        self._log(f"  Core binding atoms  : {core_binding_idxs}")
        self._log(f"  Monomer input       : {mon_input_name}  (labile: {labile_names}  mode: {labile_mode})")
        self._log(f"  Terminal atoms      : {mon_terminal_names}")
        self._log(f"  Bond length / angle : {bond_length} A  /  {bond_angle} deg")
        self._log(f"  Branching factor    : {config['branching_factor']}")
        self._log(f"  Generations to save : {gens_to_build}")
        self._log(f"  Output formats      : {out_formats}")
        self._log(f"  Output              : {config['output_dir']}")
        param_label = ("Bond Charge Correction (Fragment BCC)"
                       if pipeline_steps.get('param_method')=='fragment_bcc'
                       else "Full Antechamber Parameterization")
        self._log(f"  Parameterization    : {param_label}\n",'head')

        net_charge=self.net_charge_var.get()
        _atom_ref=[0]

        def log_wrapper(msg,tag=None):
            self._log(msg,tag)
            if 'Total atoms' in msg:
                try: _atom_ref[0]=int(msg.split(':')[1].strip().split()[0])
                except (ValueError, IndexError): pass

        def gen_progress(gen, val_results, val_passed, val_total, mw, rg, monomers, n_term):
            """Called from build thread after each generation validates."""
            atoms = _atom_ref[0]
            val_str = f"{val_passed}/{val_total}"
            max_gen = self.num_gens_var.get()
            self._last_built_rg = float(rg) if rg else None
            overall_pct = gen / max_gen * 100
            self.root.after(0, lambda g=gen,p=overall_pct,a=atoms:
                self._set_gen_progress(g, 100, atoms=a))
            self.root.after(0, lambda g=gen,a=atoms,m=mw,r=rg,mo=monomers,nt=n_term,vs=val_str:
                self._update_preview_gen(g, a, m, r, mo, nt, vs))
            self.root.after(0, lambda vr=val_results,vp=val_passed,vt=val_total,m=mw,r=rg,g=gen:
                self._populate_validator(vr, vp, vt, m, r, g))

        def run():
            try:
                summary=build_dendrimer(
                    config,log_wrapper,self._set_progress,gen_progress,
                    run_min=any_amber,net_charge=net_charge,
                    pipeline_steps=pipeline_steps,
                    cancel_flag=self._cancel_flag)
                self.root.after(0,self._on_build_done,True,summary,config)
            except Exception as e:
                self._log(f"\nERROR: {e}",'err')
                self._log(traceback.format_exc(),'err')
                # correctly shows "Build failed" not "Build cancelled"
                self.root.after(0,self._on_build_done,False,None,str(e))

        threading.Thread(target=run,daemon=True).start()

    def _on_build_done(self,success,summary,info):
        T=self._T
        was_cancelled = self._cancel_flag[0]
        # restore buttons
        self.build_btn.config(state='normal',text="  BUILD DENDRIMER  ",bg=T['cyan'])
        self.stop_btn.config(state='disabled',text="  STOP  ")
        self._cancel_flag[0]=False

        if success and summary:
            self.progress_overall['value']=100; self.progress_gen['value']=100
            self.prog_label.config(text="Build complete  ✓",fg=T['green'])
            lines=["DENDRIMER BUILD SUMMARY\n"]
            hdr=f"{'Gen':<5} {'Atoms':<8} {'MW(Da)':<12} {'Rg(Å)':<8} Validation"
            lines+=[hdr,"─"*52]
            for s in summary:
                lines.append(f"G{s['gen']:<4} {s['atoms']:<8} {s['mw']:<12,.0f} {s['rg']:<8.2f} {s['val']}")
            lines.append(f"\nOutput:\n{info['output_dir']}")
            messagebox.showinfo("Build Complete!","\n".join(lines))
        elif was_cancelled:
            # user pressed STOP
            self.prog_label.config(text="Build cancelled by user",fg=T['yellow'])
        else:
            # exception or pipeline failure
            self.prog_label.config(text="Build failed — see log for details",fg=T['red'])
            if isinstance(info, str):
                messagebox.showerror("Build Failed",
                    f"An error occurred during the build.\n\n"
                    f"Error:\n{info}\n\n"
                    f"Check the build log (Section 13) for the full traceback.")

# =============================================================================
# SECTION 17.  APPLICATION ENTRY POINT
# =============================================================================

if __name__=="__main__":
    root=tk.Tk()
    style=ttk.Style()
    style.theme_use('default')
    style.configure('TCombobox',fieldbackground='#0d1117',background='#21262d',
                    foreground='#58a6ff',selectbackground='#0f3460')
    style.configure('Vertical.TScrollbar',background='#21262d',
                    troughcolor='#0d1117',arrowcolor='#58a6ff')
    style.configure('TProgressbar',background='#58a6ff',troughcolor='#0d1117')
    style.configure('Treeview',background='#161b22',foreground='#c9d1d9',
                    fieldbackground='#161b22')
    style.configure('Treeview.Heading',background='#21262d',foreground='#58a6ff')
    style.map('Treeview',background=[('selected','#1c2733')])
    app=UniversalDendrimerBuilder(root)
    root.mainloop()
