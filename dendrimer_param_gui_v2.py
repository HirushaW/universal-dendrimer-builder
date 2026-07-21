#!/usr/bin/env python3
"""
Universal Dendrimer Parameterization GUI  v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Supports : PAMAM, PPI, Triazine, Carbosilane, Phosphorus,
           Polyester, Polyglycerol, PLL
Output   : AMBER (prmtop/inpcrd), NAMD/CHARMM (psf/pdb/prm),
           GROMACS (top/gro)
Validates: atom types, charges, bond lengths, penalty scores
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess, threading, os, math, shutil
from collections import Counter, defaultdict

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
PANEL2  = "#1c2128"
BORDER  = "#30363d"
ACCENT  = "#58a6ff"
ACCENT2 = "#79c0ff"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
RED     = "#f85149"
PURPLE  = "#bc8cff"
TEAL    = "#39d353"
TEXT    = "#c9d1d9"
SUBTEXT = "#8b949e"
MUTED   = "#484f58"
FMN     = ("Courier New", 10)
FUI     = ("TkDefaultFont", 10)
FH      = ("TkDefaultFont", 11, "bold")
FS      = ("TkDefaultFont", 8)
FSB     = ("TkDefaultFont", 8, "bold")

DENDRIMER_TYPES = [
    "PAMAM", "PPI", "Triazine", "Carbosilane",
    "Phosphorus", "Polyester", "Polyglycerol", "PLL"
]

SPECIAL = {
    "SI": {"label": "Silicon",    "color": YELLOW, "gaff2": "Si",
           "note": "Bypasses antechamber → manual GAFF2 typing"},
    "P":  {"label": "Phosphorus", "color": PURPLE, "gaff2": "p5",
           "note": "GAFF2 p5 — check frcmod penalty scores"},
    "CL": {"label": "Chlorine",   "color": TEAL,   "gaff2": "cl",
           "note": "Standard GAFF2 cl"},
    "BR": {"label": "Bromine",    "color": TEAL,   "gaff2": "br",
           "note": "Standard GAFF2 br"},
    "S":  {"label": "Sulfur",     "color": YELLOW, "gaff2": "ss",
           "note": "GAFF2 ss — check frcmod penalty scores"},
    "F":  {"label": "Fluorine",   "color": TEAL,   "gaff2": "f",
           "note": "Standard GAFF2 f"},
}

# Reference bond length ranges (Angstroms) for validation
BOND_RANGES = {
    frozenset(["C","C"]):   (1.40, 1.60),
    frozenset(["C","H"]):   (0.95, 1.15),
    frozenset(["C","N"]):   (1.30, 1.55),
    frozenset(["C","O"]):   (1.18, 1.50),
    frozenset(["SI","C"]):  (1.75, 1.98),
    frozenset(["SI","H"]):  (1.40, 1.55),
    frozenset(["C","CL"]):  (1.65, 1.85),
    frozenset(["P","O"]):   (1.45, 1.70),
    frozenset(["N","H"]):   (0.95, 1.10),
    frozenset(["O","H"]):   (0.90, 1.05),
}

# GAFF2 → Sybyl type conversion map
GAFF2_TO_SYBYL = {
    "c3":"C.3", "c2":"C.2", "ca":"C.ar", "c":"C.2", "c1":"C.1",
    "n3":"N.3", "n2":"N.2", "na":"N.ar", "nh":"N.am", "n4":"N.4",
    "nt":"N.3", "n":"N.am",
    "os":"O.3", "o":"O.2",  "oh":"O.3", "ow":"O.3",
    "Si":"Si",  "si":"Si",
    "p5":"P.3", "p4":"P.3",
    "ss":"S.3", "s2":"S.2", "sh":"S.3",
    "cl":"Cl",  "br":"Br",  "f":"F",    "i":"I",
    "hc":"H",   "ha":"H",   "hn":"H",   "ho":"H",
    "hs":"H",   "hp":"H",   "hw":"H",   "h1":"H",
    "h2":"H",   "h3":"H",   "h4":"H",   "h5":"H",
}

ATOM_COLORS = {
    "C":"#aaaaaa","H":"#ffffff","N":"#4466ff","O":"#ff4444",
    "SI":"#f0c040","P":"#cc88ff","CL":"#44ee88","BR":"#bb4422",
    "S":"#eeee44","F":"#44ffcc",
}
ATOM_RADII = {
    "C":7,"H":4,"N":7,"O":7,"SI":11,"P":9,"CL":8,"BR":9,"S":8,"F":6,
}

# ── Mol2 helpers ──────────────────────────────────────────────────────────────
def parse_mol2(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    # Normalise line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    secs = content.split("@<TRIPOS>")
    mol_sec = atom_sec = bond_sec = ""
    for s in secs:
        sl = s.strip()
        if sl.startswith("MOLECULE"):  mol_sec  = s
        elif sl.startswith("ATOM"):    atom_sec = s
        elif sl.startswith("BOND"):    bond_sec = s
    if not atom_sec:
        raise ValueError("No @<TRIPOS>ATOM section found in file")
    atoms = []
    for l in atom_sec.split("\n")[1:]:
        l = l.strip()
        if l and not l.startswith("#"):
            parts = l.split()
            if len(parts) >= 6:
                atoms.append(parts)
    if not atoms:
        raise ValueError("ATOM section is empty — check mol2 format")
    bonds = []
    for l in bond_sec.split("\n")[1:]:
        parts = l.split()
        if len(parts) >= 3:
            try:
                bonds.append((int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    adj = defaultdict(list)
    for a, b in bonds:
        adj[a].append(b); adj[b].append(a)
    return atoms, bonds, adj, mol_sec, bond_sec

def get_elem(name):
    n = name.upper()
    for e in ["SI","CL","BR"]:
        if n.startswith(e): return e
    return n[0] if n else "X"

def detect_elements(atoms):
    c = Counter()
    for a in atoms:
        c[get_elem(a[1])] += 1
    return c

def gaff2_type(idx, atoms, adj):
    elem  = get_elem(atoms[idx-1][1])
    sybyl = atoms[idx-1][5].lower()
    if elem == "SI": return "Si"
    if elem == "CL": return "cl"
    if elem == "BR": return "br"
    if elem == "N":  return "n3"
    if elem == "O":  return "o" if ("2" in sybyl or "co" in sybyl) else "os"
    if elem == "P":  return "p5"
    if elem == "S":  return "ss"
    if elem == "F":  return "f"
    if elem == "C":  return "ca" if ("ar" in sybyl or ".2" in sybyl) else "c3"
    if elem == "H":
        for nb in adj[idx]:
            ne = get_elem(atoms[nb-1][1])
            if ne == "SI": return "hs"
            if ne == "N":  return "hn"
            if ne == "O":  return "ho"
            if ne == "P":  return "hp"
        return "hc"
    return "DU"

def write_typed_mol2(atoms, bonds, adj, mol_sec, bond_sec, mol_name, outpath):
    counts = Counter()
    new_names = []
    for a in atoms:
        e = get_elem(a[1]); counts[e] += 1
        new_names.append(f"{e}{counts[e]}")
    out = f"@<TRIPOS>MOLECULE\n{mol_name}\n"
    out += "\n".join(mol_sec.split("\n")[2:]) + "\n"
    out += "@<TRIPOS>ATOM\n"
    for i, a in enumerate(atoms):
        idx = int(a[0]); gt = gaff2_type(idx, atoms, adj)
        si  = a[6] if len(a)>6 else "1"
        sn  = a[7] if len(a)>7 else "LIG"
        ch  = a[8] if len(a)>8 else "0.0000"
        out += (f"{idx:>7} {new_names[i]:<8} {float(a[2]):>10.4f}"
                f" {float(a[3]):>10.4f} {float(a[4]):>10.4f}"
                f" {gt:<8} {si:>2} {sn:<8} {float(ch):>8.4f}\n")
    out += "@<TRIPOS>BOND\n"
    for i,(ba,bb) in enumerate(bonds,1):
        out += f"{i:>6}  {ba:>4}  {bb:>4}  1\n"
    with open(outpath,"w") as f:
        f.write(out)

def needs_bypass(elem_counts):
    return any(e in elem_counts for e in ["SI","P","S"])

# ── Validation logic ──────────────────────────────────────────────────────────
def validate_outputs(orig_atoms, orig_bonds, typed_mol2, frcmod_path, nc):
    results = []

    # 1. typed mol2 exists
    if not os.path.exists(typed_mol2):
        results.append(("❌", "Typed mol2 not found", RED))
        return results
    results.append(("✅", "Typed mol2 generated", GREEN))

    # 2. parse typed mol2
    try:
        t_atoms, t_bonds, t_adj, _, _ = parse_mol2(typed_mol2)
    except Exception as e:
        results.append(("❌", f"Cannot parse typed mol2: {e}", RED))
        return results

    # 3. atom count match
    if len(t_atoms) == len(orig_atoms):
        results.append(("✅", f"Atom count matches: {len(t_atoms)}", GREEN))
    else:
        results.append(("❌", f"Atom count mismatch: {len(t_atoms)} vs {len(orig_atoms)}", RED))

    # 4. bond count match
    if len(t_bonds) == len(orig_bonds):
        results.append(("✅", f"Bond count matches: {len(t_bonds)}", GREEN))
    else:
        results.append(("❌", f"Bond count mismatch: {len(t_bonds)} vs {len(orig_bonds)}", RED))

    # 5. no DU types
    du_atoms = [a for a in t_atoms if a[5].upper() == "DU"]
    if not du_atoms:
        results.append(("✅", "No unassigned (DU) atom types", GREEN))
    else:
        results.append(("❌", f"{len(du_atoms)} atoms have DU type: "
                       + ", ".join(a[1] for a in du_atoms[:5]), RED))

    # 6. charge sum
    try:
        charges = [float(a[8]) for a in t_atoms if len(a) > 8]
        total_q = sum(charges)
        if abs(total_q - nc) < 0.5:
            results.append(("✅", f"Charge sum ≈ {nc}: {total_q:+.4f}", GREEN))
        else:
            results.append(("⚠️", f"Charge sum {total_q:+.4f} ≠ {nc} (charges may be 0.0)", YELLOW))
    except:
        results.append(("⚠️", "Could not check charge sum", YELLOW))

    # 7. bond lengths
    bad_bonds = []
    warn_bonds = []
    for ba, bb in t_bonds[:200]:  # sample first 200
        try:
            a1 = t_atoms[ba-1]; a2 = t_atoms[bb-1]
            e1 = get_elem(a1[1]); e2 = get_elem(a2[1])
            x1,y1,z1 = float(a1[2]),float(a1[3]),float(a1[4])
            x2,y2,z2 = float(a2[2]),float(a2[3]),float(a2[4])
            d = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            key = frozenset([e1,e2])
            if key in BOND_RANGES:
                lo, hi = BOND_RANGES[key]
                if d < lo or d > hi:
                    bad_bonds.append(f"{a1[1]}-{a2[1]} {d:.2f}Å (exp {lo}-{hi})")
        except:
            pass
    if not bad_bonds:
        results.append(("✅", "Bond lengths within reference ranges", GREEN))
    else:
        results.append(("⚠️", f"{len(bad_bonds)} bond(s) outside range: "
                       + "; ".join(bad_bonds[:3]), YELLOW))

    # 8. frcmod penalty scores
    if os.path.exists(frcmod_path):
        results.append(("✅", "frcmod generated", GREEN))
        high_penalty = []
        try:
            with open(frcmod_path) as f:
                for line in f:
                    if "penalty score" in line:
                        try:
                            score = float(line.split("=")[-1].strip())
                            if score > 10:
                                param = line.split()[0]
                                high_penalty.append(f"{param} (score={score:.0f})")
                        except:
                            pass
        except:
            pass
        if not high_penalty:
            results.append(("✅", "All frcmod penalty scores ≤ 10", GREEN))
        else:
            results.append(("⚠️", f"High penalty parameters: "
                           + ", ".join(high_penalty[:3]), YELLOW))
    else:
        results.append(("❌", "frcmod not found", RED))

    return results

# ── Molecule 2D Viewer ────────────────────────────────────────────────────────
class MolViewer(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#080c10", highlightthickness=0, **kw)
        self.atoms = []; self.bonds = []
        self._scale = 1.0; self._offset = [0.0, 0.0]; self._drag = None
        self._coords2d = []
        self.bind("<Configure>",      lambda e: self._draw())
        self.bind("<MouseWheel>",     self._zoom)
        self.bind("<Button-4>",       self._zoom)
        self.bind("<Button-5>",       self._zoom)
        self.bind("<ButtonPress-1>",  lambda e: setattr(self,'_drag',(e.x,e.y)))
        self.bind("<B1-Motion>",      self._pan)

    def load(self, atoms, bonds):
        self.atoms = atoms; self.bonds = bonds
        self._scale = 1.0; self._offset = [0.0, 0.0]
        self._project()
        # Draw immediately then again after widget fully renders
        self._draw()
        self.after(80, self._draw)

    def _project(self):
        if not self.atoms: self._coords2d = []; return
        try:
            xs=[float(a[2]) for a in self.atoms]
            ys=[float(a[3]) for a in self.atoms]
            zs=[float(a[4]) for a in self.atoms]
        except (ValueError, IndexError):
            self._coords2d = []; return
        px=[x+0.2*z for x,z in zip(xs,zs)]
        py=[y+0.1*z for y,z in zip(ys,zs)]
        mn_x,mx_x=min(px),max(px); mn_y,mx_y=min(py),max(py)
        span=max(mx_x-mn_x,mx_y-mn_y,1)
        self._coords2d=[((p-mn_x)/span,(q-mn_y)/span) for p,q in zip(px,py)]

    def _draw(self):
        self.delete("all")
        # Use max() not 'or' — winfo_width() returns 1 before render,
        # making scale negative and drawing everything off-screen
        W=max(self.winfo_width(), 280); H=max(self.winfo_height(), 240)
        if not self._coords2d:
            self.create_text(W//2,H//2,text="Upload mol2 to preview",
                             fill=SUBTEXT,font=("TkDefaultFont",9),justify="center")
            return
        pad=28; scale=max(min(W,H)-2*pad, 10)
        def tx(nx): return pad+(nx*scale*self._scale+self._offset[0])
        def ty(ny): return pad+(ny*scale*self._scale+self._offset[1])
        # Build atom index map — mol2 IDs may not be 0-based sequential
        idx_map = {int(a[0]): i for i, a in enumerate(self.atoms)}
        for ba,bb in self.bonds:
            if ba not in idx_map or bb not in idx_map: continue
            x1,y1=self._coords2d[idx_map[ba]]; x2,y2=self._coords2d[idx_map[bb]]
            self.create_line(tx(x1),ty(y1),tx(x2),ty(y2),fill=MUTED,width=1)
        for i,a in enumerate(self.atoms):
            elem=get_elem(a[1])
            col=ATOM_COLORS.get(elem,"#888")
            r=max(2,int(ATOM_RADII.get(elem,6)*max(0.4,self._scale)))
            cx,cy=tx(self._coords2d[i][0]),ty(self._coords2d[i][1])
            self.create_oval(cx-r,cy-r,cx+r,cy+r,fill=col,outline="")
            if elem in SPECIAL:
                self.create_text(cx,cy-r-5,text=elem.capitalize(),
                                 fill=SPECIAL[elem]["color"],
                                 font=("TkDefaultFont",7,"bold"))
    def _zoom(self, e):
        d=getattr(e,"delta",0)
        if e.num==4 or d>0: self._scale*=1.12
        elif e.num==5 or d<0: self._scale*=0.88
        self._scale=max(0.15,min(self._scale,6.0)); self._draw()

    def _pan(self, e):
        if self._drag:
            self._offset[0]+=e.x-self._drag[0]
            self._offset[1]+=e.y-self._drag[1]
            self._drag=(e.x,e.y); self._draw()

# ── Main Application ──────────────────────────────────────────────────────────
class ParamApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Universal Dendrimer Parameterization  v2.0")
        self.configure(bg=BG)
        self.geometry("1400x960")
        self.minsize(1100,720)

        # Variables
        self.mol2_path   = tk.StringVar()
        self.dend_type   = tk.StringVar(value="PAMAM")
        self.frag_type   = tk.StringVar(value="core")
        self.mol_name    = tk.StringVar(value="PAMAM_core")
        self.net_charge  = tk.IntVar(value=0)
        self.out_dir     = tk.StringVar(value="")
        self.fmt_amber   = tk.BooleanVar(value=True)
        self.fmt_namd    = tk.BooleanVar(value=False)
        self.fmt_gromacs = tk.BooleanVar(value=False)

        self._last_dir = os.path.expanduser('~')
        self._ext_stop_flag = False
        # Extract tab variables
        self.ext_prmtop   = tk.StringVar()
        self.ext_inpcrd   = tk.StringVar()
        self.ext_typed    = tk.StringVar()
        self.ext_name     = tk.StringVar(value="recovered")
        self.ext_dir      = tk.StringVar()
        self.ext_gaff2    = tk.BooleanVar(value=True)
        self.ext_sybyl    = tk.BooleanVar(value=True)
        self.ext_pdb      = tk.BooleanVar(value=True)
        self.atoms=[];self.bonds=[];self.adj={}
        self.mol_sec="";self.bond_sec=""
        self.elem_counts=Counter()
        self.special_vars={}

        self._build_ui()
        self._update_name()
        self.after(200, self._check_env_silent)

    # ── TOP-LEVEL LAYOUT ──────────────────────────────────────────────────
    def _build_ui(self):
        tk.Frame(self,bg=ACCENT,height=3).pack(fill="x")

        # Header
        hdr=tk.Frame(self,bg=PANEL,pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr,text="⬡  Universal Dendrimer Parameterization",
                 bg=PANEL,fg=ACCENT,font=("TkDefaultFont",15,"bold"),padx=18).pack(side="left")
        tk.Label(hdr,text="GAFF2 · AmberTools25 · AMBER / NAMD / GROMACS",
                 bg=PANEL,fg=SUBTEXT,font=FS,padx=6).pack(side="left")
        self.env_badge=tk.Label(hdr,text="checking env…",
                                bg=PANEL,fg=MUTED,font=FSB,padx=14)
        self.env_badge.pack(side="right")

        # Notebook tabs
        nb_frame=tk.Frame(self,bg=BG)
        nb_frame.pack(fill="both",expand=True,padx=0,pady=0)

        style=ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",background=BG,borderwidth=0)
        style.configure("TNotebook.Tab",background=PANEL,foreground=SUBTEXT,
                        padding=[14,6],font=FUI)
        style.map("TNotebook.Tab",
                  background=[("selected",PANEL2)],
                  foreground=[("selected",ACCENT)])

        self.nb=ttk.Notebook(nb_frame)
        self.nb.pack(fill="both",expand=True)

        self._tab_param   = tk.Frame(self.nb,bg=BG)
        self._tab_env     = tk.Frame(self.nb,bg=BG)
        self._tab_valid   = tk.Frame(self.nb,bg=BG)
        self._tab_extract = tk.Frame(self.nb,bg=BG)
        self._tab_refs    = tk.Frame(self.nb,bg=BG)

        self.nb.add(self._tab_param,   text="  Parameterization  ")
        self.nb.add(self._tab_env,     text="  Environment Check  ")
        self.nb.add(self._tab_valid,   text="  Validation  ")
        self.nb.add(self._tab_extract, text="  Extract mol2  ")
        self.nb.add(self._tab_refs,    text="  References  ")

        self._build_param_tab()
        self._build_env_tab()
        self._build_valid_tab()
        self._build_extract_tab()
        self._build_refs_tab()

        # Status bar
        tk.Frame(self,bg=BORDER,height=1).pack(fill="x",side="bottom")
        self.status_var=tk.StringVar(value="Ready")
        tk.Label(self,textvariable=self.status_var,bg=PANEL,fg=SUBTEXT,
                 font=FS,anchor="w",padx=14,pady=4).pack(fill="x",side="bottom")

    # ── TAB 1: PARAMETERIZATION ───────────────────────────────────────────
    def _build_param_tab(self):
        p=self._tab_param
        p.columnconfigure(0,weight=0,minsize=250)
        p.columnconfigure(1,weight=0,minsize=270)
        p.columnconfigure(2,weight=1)
        p.rowconfigure(0,weight=1)

        # Scrollable left column
        scroll_frame = tk.Frame(p, bg=BG)
        scroll_frame.grid(row=0,column=0,sticky="nsew",padx=(12,8),pady=12)
        scroll_frame.rowconfigure(0,weight=1)
        scroll_frame.columnconfigure(0,weight=1)

        left_canvas = tk.Canvas(scroll_frame, bg=BG, highlightthickness=0, width=240)
        left_scroll  = ttk.Scrollbar(scroll_frame, orient="vertical",
                                     command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.grid(row=0,column=1,sticky="ns")
        left_canvas.grid(row=0,column=0,sticky="nsew")

        col_l = tk.Frame(left_canvas, bg=BG)
        left_canvas_win = left_canvas.create_window((0,0), window=col_l, anchor="nw")

        def _on_frame_configure(e):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        def _on_canvas_configure(e):
            left_canvas.itemconfig(left_canvas_win, width=e.width)
        col_l.bind("<Configure>", _on_frame_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel scroll
        def _on_mousewheel(e):
            left_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        def _on_mousewheel_linux(e):
            if e.num == 4: left_canvas.yview_scroll(-1, "units")
            elif e.num == 5: left_canvas.yview_scroll(1, "units")
        left_canvas.bind("<Enter>", lambda e: (
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel),
            left_canvas.bind_all("<Button-4>",   _on_mousewheel_linux),
            left_canvas.bind_all("<Button-5>",   _on_mousewheel_linux)))
        left_canvas.bind("<Leave>", lambda e: (
            left_canvas.unbind_all("<MouseWheel>"),
            left_canvas.unbind_all("<Button-4>"),
            left_canvas.unbind_all("<Button-5>")))

        col_m=tk.Frame(p,bg=BG); col_r=tk.Frame(p,bg=BG)
        col_m.grid(row=0,column=1,sticky="nsew",padx=(0,8),pady=12)
        col_r.grid(row=0,column=2,sticky="nsew",padx=(0,12),pady=12)

        self._build_settings(col_l)
        self._build_viewer_col(col_m)
        self._build_log_col(col_r)

    def _build_settings(self, parent):
        # Dendrimer type
        self._lbl(parent,"Dendrimer Type")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        cb=ttk.Combobox(f,textvariable=self.dend_type,
                        values=DENDRIMER_TYPES,state="readonly",width=22)
        cb.pack(padx=12,pady=10)
        cb.bind("<<ComboboxSelected>>",lambda e:self._update_name())

        # Fragment
        self._lbl(parent,"Fragment")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        for txt,val in [("Core","core"),("Monomer","monomer")]:
            tk.Radiobutton(f,text=txt,variable=self.frag_type,value=val,
                           bg=PANEL2,fg=TEXT,selectcolor=BORDER,
                           activebackground=PANEL2,font=FUI,cursor="hand2",
                           command=self._update_name).pack(anchor="w",padx=14,pady=4)

        # Output name
        self._lbl(parent,"Output Name")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        tk.Label(f,text="Prefix for all output files",
                 bg=PANEL2,fg=SUBTEXT,font=FS).pack(anchor="w",padx=12,pady=(6,2))
        ttk.Entry(f,textvariable=self.mol_name,width=24).pack(padx=12,pady=(0,10),anchor="w")

        # Output directory
        self._lbl(parent,"Output Directory")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        self.outdir_lbl=tk.Label(f,text="Same as input file",
                                 bg=PANEL2,fg=SUBTEXT,font=FS,wraplength=195,justify="left")
        self.outdir_lbl.pack(anchor="w",padx=12,pady=(8,4))
        tk.Button(f,text="Browse…",bg=BORDER,fg=TEXT,activebackground=ACCENT,
                  activeforeground=BG,relief="flat",padx=10,pady=3,
                  font=FUI,cursor="hand2",
                  command=self._browse_outdir).pack(padx=12,pady=(0,8),anchor="w")

        # Net charge
        self._lbl(parent,"Net Charge  (-nc)")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        row=tk.Frame(f,bg=PANEL2); row.pack(padx=12,pady=10,anchor="w")
        ttk.Entry(row,textvariable=self.net_charge,width=5).pack(side="left")
        tk.Label(row,text=" (0 = neutral)",bg=PANEL2,fg=SUBTEXT,font=FS).pack(side="left")

        # Output formats
        self._lbl(parent,"Output Formats")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        for var,txt,note in [
            (self.fmt_amber,   "AMBER",        "prmtop + inpcrd"),
            (self.fmt_namd,    "NAMD/CHARMM",  "psf + pdb + prm  (ParmEd)"),
            (self.fmt_gromacs, "GROMACS",      "top + gro  (acpype)"),
        ]:
            row=tk.Frame(f,bg=PANEL2); row.pack(fill="x",padx=10,pady=2)
            cb2=tk.Checkbutton(row,text=txt,variable=var,
                               bg=PANEL2,fg=TEXT,selectcolor=BORDER,
                               activebackground=PANEL2,font=FUI,cursor="hand2")
            cb2.pack(side="left")
            tk.Label(row,text=f"  {note}",bg=PANEL2,fg=SUBTEXT,font=FS).pack(side="left")
        tk.Frame(f,bg=BG,height=4).pack()

        # Upload
        self._lbl(parent,"mol2 File")
        f=self._card(parent); f.pack(fill="x",pady=(2,8))
        self.fname_lbl=tk.Label(f,text="No file selected",bg=PANEL2,
                                fg=SUBTEXT,font=FS,wraplength=195,justify="left")
        self.fname_lbl.pack(anchor="w",padx=12,pady=(8,4))
        tk.Button(f,text="Browse…",bg=BORDER,fg=TEXT,activebackground=ACCENT,
                  activeforeground=BG,relief="flat",padx=10,pady=3,
                  font=FUI,cursor="hand2",command=self._browse_mol2).pack(
                  padx=12,pady=(0,8),anchor="w")

        tk.Frame(parent,bg=BG,height=8).pack()
        self.run_btn=tk.Button(parent,text="▶  Run Parameterization",
                               bg=ACCENT,fg=BG,activebackground=ACCENT2,
                               activeforeground=BG,relief="flat",
                               font=FH,padx=10,pady=10,cursor="hand2",
                               command=self._run)
        self.run_btn.pack(fill="x")
        tk.Frame(parent,bg=BG,height=4).pack()
        tk.Button(parent,text="💾  Save Script Only",
                  bg=PANEL,fg=TEXT,activebackground=PANEL2,
                  relief="flat",font=FUI,padx=10,pady=6,cursor="hand2",
                  command=self._save_script).pack(fill="x")

    def _build_viewer_col(self, parent):
        self._lbl(parent,"Molecule Preview")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        tk.Label(f,text="scroll=zoom · drag=pan",bg=PANEL2,
                 fg=MUTED,font=FS).pack(anchor="e",padx=10,pady=(4,0))
        self.viewer=MolViewer(f,width=260,height=240)
        self.viewer.pack(fill="x",padx=6,pady=(0,6))

        self._lbl(parent,"Detected Atoms")
        self.atom_card=self._card(parent)
        self.atom_card.pack(fill="both",expand=True,pady=(2,0))
        self.atom_inner=tk.Frame(self.atom_card,bg=PANEL2)
        self.atom_inner.pack(fill="both",expand=True,padx=4,pady=4)
        tk.Label(self.atom_inner,text="Upload a file to detect atoms",
                 bg=PANEL2,fg=SUBTEXT,font=FS).pack(padx=10,pady=12)

    def _build_log_col(self, parent):
        # Pipeline stages
        self._lbl(parent,"Pipeline Progress")
        f=self._card(parent); f.pack(fill="x",pady=(2,10))
        self.stage_dots={}
        stages=[("parse","Parse & fix mol2"),("type","GAFF2 atom typing"),
                ("parmchk","parmchk2 → frcmod"),("tleap","tleap → topology"),
                ("convert","Format conversion")]
        for key,lbl in stages:
            row=tk.Frame(f,bg=PANEL2); row.pack(fill="x",padx=12,pady=3)
            dot=tk.Label(row,text="○",bg=PANEL2,fg=MUTED,
                         font=("TkDefaultFont",10,"bold"))
            dot.pack(side="left")
            tk.Label(row,text=f"  {lbl}",bg=PANEL2,fg=SUBTEXT,font=FUI).pack(side="left")
            self.stage_dots[key]=dot
        tk.Frame(f,bg=BORDER,height=1).pack(fill="x",padx=10,pady=4)
        self.prog_bar=ttk.Progressbar(f,mode="determinate",maximum=5,value=0)
        self.prog_bar.pack(fill="x",padx=12,pady=(0,10))

        # Log
        self._lbl(parent,"Output Log")
        lf=self._card(parent); lf.pack(fill="both",expand=True,pady=(2,0))
        self.log=scrolledtext.ScrolledText(
            lf,bg="#070b0f",fg=TEXT,font=FMN,relief="flat",
            insertbackground=TEXT,wrap="word",state="disabled",
            padx=10,pady=8)
        self.log.pack(fill="both",expand=True,padx=2,pady=2)
        for tag,col in [("ok",GREEN),("warn",YELLOW),("err",RED),
                        ("info",ACCENT),("sec",ACCENT2),("cmd",SUBTEXT)]:
            self.log.tag_config(tag,foreground=col)
        self.log.tag_config("sec",font=("Courier New",10,"bold"))

    # ── TAB 2: ENVIRONMENT CHECK ──────────────────────────────────────────
    def _build_env_tab(self):
        p=self._tab_env
        p.columnconfigure(0,weight=1); p.columnconfigure(1,weight=1)
        p.rowconfigure(1,weight=1)

        tk.Label(p,text="Environment & Dependency Check",
                 bg=BG,fg=ACCENT,font=FH,anchor="w").grid(
                 row=0,column=0,columnspan=2,sticky="w",padx=16,pady=(14,6))

        # Tools panel
        tf=self._card(p); tf.grid(row=1,column=0,sticky="nsew",padx=(16,8),pady=(0,16))
        tk.Label(tf,text="REQUIRED TOOLS",bg=PANEL2,fg=SUBTEXT,
                 font=FSB,anchor="w").pack(fill="x",padx=12,pady=(10,4))
        self.env_rows={}
        tools=[
            ("antechamber", "AmberTools25",  "GAFF2 atom typing"),
            ("parmchk2",    "AmberTools25",  "Missing parameter generation"),
            ("tleap",       "AmberTools25",  "AMBER topology builder"),
            ("python3",     "System",        "Typing & validation scripts"),
            ("parmed",      "AmberTools25",  "NAMD/CHARMM conversion"),
            ("acpype",      "conda-forge",   "GROMACS conversion (optional)"),
        ]
        for name,pkg,desc in tools:
            row=tk.Frame(tf,bg=PANEL2); row.pack(fill="x",padx=10,pady=3)
            dot=tk.Label(row,text="○",bg=PANEL2,fg=MUTED,
                         font=("TkDefaultFont",12,"bold"),width=2)
            dot.pack(side="left")
            tk.Label(row,text=f" {name:<14}",bg=PANEL2,fg=TEXT,
                     font=FMN).pack(side="left")
            tk.Label(row,text=f"[{pkg}]",bg=PANEL2,fg=SUBTEXT,
                     font=FS).pack(side="left",padx=4)
            path_lbl=tk.Label(row,text="",bg=PANEL2,fg=MUTED,font=FS)
            path_lbl.pack(side="left",padx=8)
            self.env_rows[name]=(dot,path_lbl)

        tk.Button(tf,text="▶  Run Environment Check",
                  bg=ACCENT,fg=BG,activebackground=ACCENT2,activeforeground=BG,
                  relief="flat",font=FH,padx=10,pady=8,cursor="hand2",
                  command=self._check_env).pack(padx=12,pady=14,fill="x")

        # Versions / info panel
        vf=self._card(p); vf.grid(row=1,column=1,sticky="nsew",padx=(0,16),pady=(0,16))
        tk.Label(vf,text="VERSION INFO",bg=PANEL2,fg=SUBTEXT,
                 font=FSB,anchor="w").pack(fill="x",padx=12,pady=(10,4))
        self.env_log=scrolledtext.ScrolledText(
            vf,bg="#070b0f",fg=TEXT,font=FMN,relief="flat",
            state="disabled",padx=10,pady=8,height=20)
        self.env_log.pack(fill="both",expand=True,padx=6,pady=(0,8))
        for tag,col in [("ok",GREEN),("warn",YELLOW),("err",RED),("info",ACCENT)]:
            self.env_log.tag_config(tag,foreground=col)

        # Install hints
        ih=tk.Frame(p,bg=BG)
        ih.grid(row=2,column=0,columnspan=2,sticky="ew",padx=16,pady=(0,14))
        tk.Label(ih,text="Install missing tools:",bg=BG,fg=SUBTEXT,
                 font=FSB).pack(anchor="w")
        hints=[
            ("acpype",   "conda install -c conda-forge acpype -y"),
            ("parmed",   "Already in AmberTools25 — import parmed as pmd"),
        ]
        for name,cmd in hints:
            row=tk.Frame(ih,bg=BG); row.pack(fill="x",pady=1)
            tk.Label(row,text=f"  {name:<10}",bg=BG,fg=SUBTEXT,
                     font=FUI).pack(side="left")
            tk.Label(row,text=cmd,bg=BG,fg=MUTED,font=FMN).pack(side="left")

    # ── TAB 3: VALIDATION ─────────────────────────────────────────────────
    def _build_valid_tab(self):
        p=self._tab_valid
        p.columnconfigure(0,weight=1); p.rowconfigure(1,weight=1)

        hrow=tk.Frame(p,bg=BG); hrow.pack(fill="x",padx=16,pady=(14,6))
        tk.Label(hrow,text="Structure & Parameter Validation",
                 bg=BG,fg=ACCENT,font=FH).pack(side="left")
        tk.Button(hrow,text="▶  Run Validation",
                  bg=ACCENT,fg=BG,activebackground=ACCENT2,activeforeground=BG,
                  relief="flat",font=FUI,padx=12,pady=6,cursor="hand2",
                  command=self._run_validation).pack(side="right")

        self.valid_frame=tk.Frame(p,bg=BG)
        self.valid_frame.pack(fill="both",expand=True,padx=16,pady=(0,16))

        # Left: checks
        lf=self._card(self.valid_frame); lf.pack(side="left",fill="both",
                                                  expand=True,padx=(0,8))
        tk.Label(lf,text="CHECKS",bg=PANEL2,fg=SUBTEXT,
                 font=FSB,anchor="w").pack(fill="x",padx=12,pady=(10,4))
        self.valid_inner=tk.Frame(lf,bg=PANEL2)
        self.valid_inner.pack(fill="both",expand=True,padx=6,pady=(0,8))
        tk.Label(self.valid_inner,
                 text="Run parameterization first,\nthen click Run Validation.",
                 bg=PANEL2,fg=SUBTEXT,font=FS,justify="left").pack(padx=10,pady=14)

        # Right: frcmod viewer
        rf=self._card(self.valid_frame); rf.pack(side="right",fill="both",
                                                  expand=True,padx=(8,0))
        tk.Label(rf,text="FRCMOD CONTENT",bg=PANEL2,fg=SUBTEXT,
                 font=FSB,anchor="w").pack(fill="x",padx=12,pady=(10,4))
        self.frcmod_view=scrolledtext.ScrolledText(
            rf,bg="#070b0f",fg=TEXT,font=FMN,relief="flat",
            state="disabled",padx=8,pady=8)
        self.frcmod_view.pack(fill="both",expand=True,padx=6,pady=(0,8))
        self.frcmod_view.tag_config("hi",foreground=YELLOW)

    # ── TAB 4: REFERENCES ─────────────────────────────────────────────────
    def _build_refs_tab(self):
        p=self._tab_refs
        tk.Label(p,text="Key References & Validation Sources",
                 bg=BG,fg=ACCENT,font=FH,anchor="w").pack(
                 anchor="w",padx=16,pady=(14,8))

        refs=[
            ("GAFF / GAFF2 Force Field",
             "Wang et al. (2004) J. Comput. Chem. 25:1157–1174\n"
             "doi:10.1002/jcc.20035\n"
             "Validation: RMS displacement 0.26 Å for 74 crystal structures.\n"
             "Standard reference for GAFF2 bond/angle parameter quality.",
             ACCENT),

            ("AM1-BCC Charges for GAFF2",
             "He et al. (2020) J. Chem. Theory Comput. 16:7848–7862\n"
             "doi:10.1021/acs.jctc.0c00650\n"
             "Optimised AM1-BCC parameters for GAFF2; RESP charges give better\n"
             "solvation free energies but AM1-BCC sufficient for most structural work.",
             ACCENT),

            ("Carbosilane Dendrimer MD (AMBER)",
             "Mikhailov et al. (2004) Phys. Chem. Chem. Phys. 6:3426\n"
             "doi:10.1039/B311131H\n"
             "G5 carbosilane dendrimer MD with AMBER force field in CCl4.\n"
             "Reference Si–C bond: ~1.87 Å, Si–C–C angle: ~113°.",
             YELLOW),

            ("PAMAM Dendrimer MD",
             "Maiti et al. (2004) Macromolecules 37:6236–6254\n"
             "doi:10.1021/ma0491350\n"
             "G4-G6 PAMAM simulations; reference for dendrimer MD validation\n"
             "including radius of gyration, density profiles.",
             YELLOW),

            ("ACPYPE — Format Conversion",
             "Sousa da Silva & Vranken (2012) BMC Res. Notes 5:367\n"
             "doi:10.1186/1756-0500-5-367\n"
             "Tool used for AMBER → GROMACS / CHARMM topology conversion.\n"
             "Cite when publishing GROMACS/CHARMM outputs.",
             PURPLE),

            ("ParmEd — NAMD/CHARMM Conversion",
             "Case et al. AmberTools — ParmEd module\n"
             "https://parmed.github.io/ParmEd\n"
             "Python library for inter-format topology conversion.\n"
             "Supports AMBER → NAMD (psf/pdb), CHARMM (.str/.prm).",
             PURPLE),

            ("AMBER Parameter Database",
             "Bryce Group, University of Manchester\n"
             "http://amber.manchester.ac.uk/\n"
             "Repository of user-contributed AMBER parameters.\n"
             "Check here for existing Si-containing parameter sets\n"
             "before using parmchk2-estimated values.",
             TEAL),

            ("Si–C Bond Reference Values",
             "Pauling (1960) The Nature of the Chemical Bond\n"
             "Si–C:  1.87 Å  (range 1.75–1.98 Å acceptable)\n"
             "Si–H:  1.48 Å  (range 1.40–1.55 Å acceptable)\n"
             "C–C:   1.54 Å  (range 1.40–1.60 Å acceptable)\n"
             "Used for bond-length validation in this tool.",
             TEAL),
        ]

        canvas=tk.Canvas(p,bg=BG,highlightthickness=0)
        scrollbar=ttk.Scrollbar(p,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right",fill="y",padx=(0,8))
        canvas.pack(side="left",fill="both",expand=True,padx=(16,0))

        inner=tk.Frame(canvas,bg=BG)
        canvas.create_window((0,0),window=inner,anchor="nw")
        inner.bind("<Configure>",lambda e:canvas.configure(
            scrollregion=canvas.bbox("all")))

        for title,body,color in refs:
            f=tk.Frame(inner,bg=PANEL2,highlightthickness=1,
                       highlightbackground=BORDER)
            f.pack(fill="x",pady=5,ipady=6)
            tk.Label(f,text=title,bg=PANEL2,fg=color,
                     font=("TkDefaultFont",10,"bold"),
                     anchor="w").pack(fill="x",padx=14,pady=(8,2))
            tk.Label(f,text=body,bg=PANEL2,fg=TEXT,font=FS,
                     justify="left",anchor="w").pack(fill="x",padx=18,pady=(0,8))

    # ── Widget helpers ────────────────────────────────────────────────────
    def _card(self, parent):
        return tk.Frame(parent,bg=PANEL2,bd=0,
                        highlightthickness=1,highlightbackground=BORDER)

    def _lbl(self, parent, text):
        tk.Label(parent,text=text.upper(),bg=BG,fg=SUBTEXT,
                 font=FSB,anchor="w").pack(fill="x")

    # ── Name helpers ──────────────────────────────────────────────────────
    def _update_name(self):
        self.mol_name.set(f"{self.dend_type.get()}_{self.frag_type.get()}")

    # ── Directory helpers ─────────────────────────────────────────────────
    def _browse_outdir(self):
        start = self.out_dir.get() or self._last_dir
        d = filedialog.askdirectory(
            title="Select output directory",
            initialdir=start)
        if d:
            self._last_dir = d
            self.out_dir.set(d)
            short = d if len(d)<32 else "…"+d[-30:]
            self.outdir_lbl.config(text=short, fg=GREEN)

    def _get_outdir(self):
        if self.out_dir.get():
            return self.out_dir.get()
        if self.mol2_path.get():
            return os.path.dirname(self.mol2_path.get())
        return os.getcwd()

    # ── Browse mol2 ───────────────────────────────────────────────────────
    def _browse_mol2(self):
        # Start in last used dir, or output dir, or home
        start = self._last_dir
        if self.out_dir.get() and os.path.isdir(self.out_dir.get()):
            start = self.out_dir.get()
        path = filedialog.askopenfilename(
            title="Select mol2 file — navigate to your folder and select the .mol2 file",
            initialdir=start)
        if not path: return
        self._last_dir = os.path.dirname(path)
        self.mol2_path.set(path)
        self.fname_lbl.config(text=os.path.basename(path), fg=GREEN)
        if not self.out_dir.get():
            d = os.path.dirname(path)
            self.out_dir.set(d)
            short = d if len(d)<32 else "…"+d[-30:]
            self.outdir_lbl.config(text=short+" (same as input)", fg=SUBTEXT)
        self._analyze(path)

    # ── Analyze mol2 ──────────────────────────────────────────────────────
    def _analyze(self, path):
        self._log_clear()
        self._log(f"Loading: {os.path.basename(path)}\n", "info")
        self._status("Parsing mol2...")
        try:
            self.atoms, self.bonds, self.adj, self.mol_sec, self.bond_sec = \
                parse_mol2(path)
            self.elem_counts = detect_elements(self.atoms)
        except Exception as e:
            self._log(f"\n✗ Parse failed: {e}\n", "err")
            self._status(f"Parse error: {e}")
            messagebox.showerror("Parse Error",
                f"Could not load mol2 file:\n{e}\n\n"
                "Check that the file has @<TRIPOS>ATOM and @<TRIPOS>BOND sections.")
            return

        # Viewer — load and force redraw after layout
        self.viewer.load(self.atoms, self.bonds)
        self.after(150, self.viewer._draw)

        # Atom panel
        for w in self.atom_inner.winfo_children(): w.destroy()
        self.special_vars.clear()

        tk.Label(self.atom_inner,
                 text=f"Atoms: {len(self.atoms)}   Bonds: {len(self.bonds)}",
                 bg=PANEL2,fg=TEXT,font=("TkDefaultFont",9,"bold")).pack(
                 anchor="w",padx=10,pady=(8,2))
        summary="  ".join(f"{e}:{n}" for e,n in sorted(self.elem_counts.items()))
        tk.Label(self.atom_inner,text=summary,bg=PANEL2,fg=SUBTEXT,
                 font=FMN).pack(anchor="w",padx=10,pady=(0,8))
        tk.Frame(self.atom_inner,bg=BORDER,height=1).pack(fill="x",padx=8)

        found=False
        for elem,info in SPECIAL.items():
            if elem not in self.elem_counts: continue
            found=True
            n=self.elem_counts[elem]
            row=tk.Frame(self.atom_inner,bg=PANEL2); row.pack(fill="x",padx=8,pady=(5,0))
            tk.Label(row,text="●",bg=PANEL2,fg=info["color"],
                     font=("TkDefaultFont",9)).pack(side="left")
            var=tk.BooleanVar(value=True); self.special_vars[elem]=var
            tk.Checkbutton(row,text=f"{info['label']} ({n}) → {info['gaff2']}",
                           variable=var,bg=PANEL2,fg=TEXT,selectcolor=BORDER,
                           activebackground=PANEL2,font=FUI,
                           cursor="hand2").pack(side="left",padx=4)
            tk.Label(self.atom_inner,text=f"  ⚠ {info['note']}",
                     bg=PANEL2,fg=info["color"],font=FS,wraplength=230,
                     justify="left").pack(anchor="w",padx=18,pady=(0,4))

        if not found:
            tk.Label(self.atom_inner,
                     text="✓ No special atoms\n  Standard GAFF2 via antechamber",
                     bg=PANEL2,fg=GREEN,font=FS).pack(padx=10,pady=8,anchor="w")

        self._log(f"Atoms : {len(self.atoms)}   Bonds: {len(self.bonds)}\n","info")
        for e,n in sorted(self.elem_counts.items()):
            self._log(f"  {e:<4} ×{n}\n","warn" if e in SPECIAL else "ok")
        if needs_bypass(self.elem_counts):
            self._log("\n⚠ Special atoms → antechamber bypassed\n","warn")
        self._status(f"Loaded {len(self.atoms)} atoms, {len(self.bonds)} bonds — ready to run")

    def _log(self,text,tag=None):
        self.log.config(state="normal")
        self.log.insert("end",text,tag or ())
        self.log.see("end"); self.log.config(state="disabled")
        self.update_idletasks()

    def _log_clear(self):
        self.log.config(state="normal"); self.log.delete("1.0","end")
        self.log.config(state="disabled")

    def _status(self,msg):
        self.status_var.set(msg); self.update_idletasks()

    # ── Stage indicators ──────────────────────────────────────────────────
    def _stage(self,key,state):
        icons={"pending":"○","running":"◉","done":"●","error":"✗","skip":"–"}
        colors={"pending":MUTED,"running":ACCENT,"done":GREEN,"error":RED,"skip":MUTED}
        dot=self.stage_dots[key]
        dot.config(text=icons[state],fg=colors[state])
        self.update_idletasks()

    def _prog(self,val):
        self.prog_bar["value"]=val; self.update_idletasks()

    def _reset_stages(self):
        for k in self.stage_dots: self._stage(k,"pending")
        self._prog(0)

    # ── Script generation ─────────────────────────────────────────────────
    def _build_script(self):
        mn    = self.mol_name.get().strip() or "molecule"
        nc    = self.net_charge.get()
        src   = self.mol2_path.get()
        wdir  = self._get_outdir()
        typed = os.path.join(wdir,f"{mn}_typed.mol2")
        frc   = os.path.join(wdir,f"{mn}.frcmod")
        prm   = os.path.join(wdir,f"{mn}.prmtop")
        inc   = os.path.join(wdir,f"{mn}.inpcrd")

        py_type=f"""python3 << 'PYEOF'
from collections import Counter, defaultdict
with open("{src}") as f: content=f.read()
secs=content.split("@<TRIPOS>")
mol_sec=atom_sec=bond_sec=""
for s in secs:
    if s.startswith("MOLECULE"):  mol_sec=s
    elif s.startswith("ATOM"):    atom_sec=s
    elif s.startswith("BOND"):    bond_sec=s
atoms=[l.split() for l in atom_sec.split("\\n")[1:] if l.strip() and len(l.split())>=6]
bonds=[]
for l in bond_sec.split("\\n")[1:]:
    p=l.split()
    if len(p)>=3: bonds.append((int(p[1]),int(p[2])))
adj=defaultdict(list)
for a,b in bonds: adj[a].append(b); adj[b].append(a)
def get_elem(n):
    n=n.upper()
    for e in ["SI","CL","BR"]:
        if n.startswith(e): return e
    return n[0] if n else "X"
def gt(idx):
    elem=get_elem(atoms[idx-1][1]); sy=atoms[idx-1][5].lower()
    if elem=="SI": return "Si"
    if elem=="CL": return "cl"
    if elem=="BR": return "br"
    if elem=="N":  return "n3"
    if elem=="O":  return "o" if ("2" in sy or "co" in sy) else "os"
    if elem=="P":  return "p5"
    if elem=="S":  return "ss"
    if elem=="F":  return "f"
    if elem=="C":  return "ca" if ("ar" in sy or ".2" in sy) else "c3"
    if elem=="H":
        for nb in adj[idx]:
            ne=get_elem(atoms[nb-1][1])
            if ne=="SI": return "hs"
            if ne=="N":  return "hn"
            if ne=="O":  return "ho"
            if ne=="P":  return "hp"
        return "hc"
    return "DU"
counts=Counter(); new_names=[]
for a in atoms:
    e=get_elem(a[1]); counts[e]+=1; new_names.append(f"{{e}}{{counts[e]}}")
out=f"@<TRIPOS>MOLECULE\\n{mn}\\n"
out+="\\n".join(mol_sec.split("\\n")[2:])+"\\n"
out+="@<TRIPOS>ATOM\\n"
for i,a in enumerate(atoms):
    idx=int(a[0]); g=gt(idx)
    si=a[6] if len(a)>6 else "1"; sn=a[7] if len(a)>7 else "LIG"; ch=a[8] if len(a)>8 else "0.0000"
    out+=(f"{{idx:>7}} {{new_names[i]:<8}} {{float(a[2]):>10.4f}}"
          f" {{float(a[3]):>10.4f}} {{float(a[4]):>10.4f}}"
          f" {{g:<8}} {{si:>2}} {{sn:<8}} {{float(ch):>8.4f}}\\n")
out+="@<TRIPOS>BOND\\n"
for i,(ba,bb) in enumerate(bonds,1): out+=f"{{i:>6}}  {{ba:>4}}  {{bb:>4}}  1\\n"
with open("{typed}","w") as f: f.write(out)
tc=Counter(gt(int(a[0])) for a in atoms)
print("GAFF2 types:"); [print(f"  {{t}}: {{n}}") for t,n in sorted(tc.items())]
print("Written: {typed}")
PYEOF"""

        # Conversion blocks
        namd_block=""
        if self.fmt_namd.get():
            psf=os.path.join(wdir,f"{mn}.psf")
            pdb=os.path.join(wdir,f"{mn}.pdb")
            namd_block=f"""
echo ""
echo "=== [4/5] NAMD/CHARMM conversion (ParmEd) ==="
python3 << 'PARMEDEOF'
try:
    import parmed as pmd
    p=pmd.load_file("{prm}",xyz="{inc}")
    p.save("{psf}",overwrite=True)
    p.save("{pdb}",overwrite=True)
    print("Written: {psf}")
    print("Written: {pdb}")
except ImportError:
    print("WARNING: parmed not available — skipping NAMD/CHARMM output")
except Exception as e:
    print(f"ParmEd error: {{e}}")
PARMEDEOF"""

        gmx_block=""
        if self.fmt_gromacs.get():
            gmx_block=f"""
echo ""
echo "=== [5/5] GROMACS conversion (acpype) ==="
if command -v acpype &>/dev/null; then
    acpype -p {prm} -x {inc} -o gmx -b {mn} -d
    echo "GROMACS files written to {mn}.acpype/"
else
    echo "WARNING: acpype not found — skipping GROMACS output"
    echo "Install: conda install -c conda-forge acpype -y"
fi"""

        script=f"""#!/bin/bash
# Universal Dendrimer Parameterization GUI  v2.0
# Molecule : {mn}
# Source   : {src}
# Output   : {wdir}
set -e
mkdir -p "{wdir}"

echo "=== [1/5] Parse & fix mol2 ==="
{py_type}

echo ""
echo "=== [2/5] parmchk2 → frcmod ==="
parmchk2 -i "{typed}" -f mol2 -o "{frc}" -s gaff2
echo "Penalty scores:"; grep "penalty score" "{frc}" || echo "  None"

echo ""
echo "=== [3/5] tleap → AMBER topology ==="
tleap -f - << 'LEAPEOF'
source leaprc.gaff2
loadamberparams {frc}
mol = loadmol2 {typed}
check mol
saveamberparm mol {prm} {inc}
savemol2 mol {os.path.join(wdir,mn+"_final.mol2")} 1
quit
LEAPEOF
{namd_block}
{gmx_block}

echo ""
echo "=== Complete ==="
echo "  {typed}"
echo "  {frc}"
echo "  {prm}"
echo "  {inc}"
"""
        return script,wdir,typed,frc,prm,inc

    # ── Save script ───────────────────────────────────────────────────────
    def _save_script(self):
        if not self.mol2_path.get():
            messagebox.showwarning("No file","Upload a mol2 file first."); return
        script,wdir,*_=self._build_script()
        mn=self.mol_name.get().strip() or "molecule"
        p=filedialog.asksaveasfilename(
            initialdir=wdir,initialfile=f"parameterize_{mn}.sh",
            filetypes=[("Shell script","*.sh")])
        if p:
            with open(p,"w") as f: f.write(script)
            os.chmod(p,0o755)
            self._log(f"\nScript saved → {p}\n","ok")
            self._status(f"Script saved: {os.path.basename(p)}")

    # ── Run ───────────────────────────────────────────────────────────────
    def _run(self):
        if not self.mol2_path.get():
            messagebox.showwarning("No file","Upload a mol2 file first."); return
        if not self.atoms:
            messagebox.showwarning("Not loaded","File not parsed yet."); return
        if not self.fmt_amber.get() and not self.fmt_namd.get() and not self.fmt_gromacs.get():
            messagebox.showwarning("No format","Select at least one output format."); return

        self.run_btn.config(state="disabled",text="Running…")
        self._log_clear(); self._reset_stages(); self._status("Running…")

        script,wdir,typed,frc,prm,inc=self._build_script()
        tmp=os.path.join(wdir,"_param_tmp.sh")
        with open(tmp,"w") as f: f.write(script)
        os.chmod(tmp,0o755)

        self._last_typed=typed; self._last_frc=frc
        self._last_prm=prm; self._last_inc=inc

        stage_map={"[1/5]":"parse","GAFF2 types:":"type","[2/5]":"parmchk","[3/5]":"tleap",
                   "[4/5]":"convert","[5/5]":"convert"}
        cur_stage=[0]
        stage_keys=["parse","type","parmchk","tleap","convert"]

        def _worker():
            try:
                self.after(0,self._stage,"parse","running")
                proc=subprocess.Popen(["bash",tmp],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,
                                      text=True,cwd=wdir)
                for line in proc.stdout:
                    ls=line.rstrip()
                    for trigger,sk in stage_map.items():
                        if trigger in ls:
                            idx=stage_keys.index(sk)
                            for j in range(cur_stage[0],idx):
                                self.after(0,self._stage,stage_keys[j],"done")
                                self.after(0,self._prog,j+1)
                            cur_stage[0]=idx
                            self.after(0,self._stage,sk,"running")
                            break
                    tag=None
                    if any(x in ls for x in ["Fatal","Error","error","failed","FATAL"]):
                        tag="err"
                    elif any(x in ls for x in ["Warning","warning","penalty","ATTN"]):
                        tag="warn"
                    elif ls.startswith("==="): tag="sec"
                    elif any(x in ls for x in ["Written","Done","Complete","Errors = 0"]): tag="ok"
                    elif ":" in ls and ls.startswith("  "): tag="info"
                    self.after(0,self._log,line,tag)
                proc.wait()
                try: os.remove(tmp)
                except: pass
                if proc.returncode==0:
                    for k in stage_keys: self.after(0,self._stage,k,"done")
                    self.after(0,self._prog,5)
                    self.after(0,self._log,"\n✓ Complete — Errors = 0\n","ok")
                    self.after(0,self._status,"✓ Complete — click Validation tab to verify")
                    self.after(0,self._auto_validate)
                else:
                    for k in stage_keys:
                        if self.stage_dots[k].cget("text")=="◉":
                            self.after(0,self._stage,k,"error")
                    self.after(0,self._log,f"\n✗ Exit {proc.returncode}\n","err")
                    self.after(0,self._status,"✗ Failed — check log")
            except Exception as e:
                self.after(0,self._log,f"\nException: {e}\n","err")
            finally:
                self.after(0,lambda:self.run_btn.config(
                    state="normal",text="▶  Run Parameterization"))

        threading.Thread(target=_worker,daemon=True).start()

    # ── Validation ────────────────────────────────────────────────────────
    def _auto_validate(self):
        self.nb.select(self._tab_valid)
        self._run_validation()

    def _run_validation(self):
        if not hasattr(self,"_last_typed"):
            messagebox.showinfo("Not ready","Run parameterization first."); return

        for w in self.valid_inner.winfo_children(): w.destroy()

        results=validate_outputs(
            self.atoms, self.bonds,
            self._last_typed, self._last_frc,
            self.net_charge.get())

        for icon,msg,color in results:
            row=tk.Frame(self.valid_inner,bg=PANEL2)
            row.pack(fill="x",padx=8,pady=3)
            tk.Label(row,text=icon,bg=PANEL2,fg=color,
                     font=("TkDefaultFont",11)).pack(side="left",padx=4)
            tk.Label(row,text=msg,bg=PANEL2,fg=TEXT,
                     font=FUI,wraplength=280,justify="left").pack(side="left",padx=4)

        # Load frcmod
        self.frcmod_view.config(state="normal")
        self.frcmod_view.delete("1.0","end")
        if os.path.exists(self._last_frc):
            with open(self._last_frc) as f:
                content=f.read()
            self.frcmod_view.insert("end",content)
            # Highlight penalty lines
            start="1.0"
            while True:
                pos=self.frcmod_view.search("penalty score",start,"end")
                if not pos: break
                line_end=f"{pos} lineend"
                self.frcmod_view.tag_add("hi",pos,line_end)
                start=line_end
        self.frcmod_view.config(state="disabled")

    # ── Extract / Convert tab ────────────────────────────────────────────
    def _build_extract_tab(self):
        p = self._tab_extract
        p.columnconfigure(0, weight=0, minsize=290)
        p.columnconfigure(1, weight=1)
        p.rowconfigure(0, weight=1)

        # Scrollable left column
        scroll_frame2 = tk.Frame(p, bg=BG)
        scroll_frame2.grid(row=0, column=0, sticky="nsew", padx=(12,8), pady=12)
        scroll_frame2.rowconfigure(0, weight=1)
        scroll_frame2.columnconfigure(0, weight=1)

        left_canvas2 = tk.Canvas(scroll_frame2, bg=BG, highlightthickness=0, width=280)
        left_scroll2  = ttk.Scrollbar(scroll_frame2, orient="vertical",
                                      command=left_canvas2.yview)
        left_canvas2.configure(yscrollcommand=left_scroll2.set)
        left_scroll2.grid(row=0, column=1, sticky="ns")
        left_canvas2.grid(row=0, column=0, sticky="nsew")

        col_l = tk.Frame(left_canvas2, bg=BG)
        win2 = left_canvas2.create_window((0,0), window=col_l, anchor="nw")

        def _on_fc2(e): left_canvas2.configure(scrollregion=left_canvas2.bbox("all"))
        def _on_cc2(e): left_canvas2.itemconfig(win2, width=e.width)
        col_l.bind("<Configure>", _on_fc2)
        left_canvas2.bind("<Configure>", _on_cc2)

        col_r = tk.Frame(p, bg=BG)
        col_r.grid(row=0, column=1, sticky="nsew", padx=(0,12), pady=12)

        # ── Title ──────────────────────────────────────────────────────
        tk.Label(col_l, text="Convert Dendrimer mol2",
                 bg=BG, fg=ACCENT, font=FH, anchor="w").pack(fill="x", pady=(0,4))
        tk.Label(col_l,
                 text="UDB v11 output (GAFF2 typed) → raw Sybyl mol2\n"
                      "for inspection, editing, and re-use",
                 bg=BG, fg=SUBTEXT, font=FS,
                 justify="left", anchor="w").pack(fill="x", pady=(0,10))

        # ── Mode selector ──────────────────────────────────────────────
        self._lbl(col_l, "Conversion Mode")
        mode_f = self._card(col_l); mode_f.pack(fill="x", pady=(2,10))
        self.ext_mode = tk.StringVar(value="A")
        for val, txt, sub in [
            ("A", "Mode A — mol2 only  (primary)",
                  "Input: dendrimer mol2 from UDB v11\nNo AMBER files needed"),
            ("B", "Mode B — mol2 + prmtop + inpcrd",
                  "Input: all 3 files\nUses prmtop coords for accuracy"),
        ]:
            row = tk.Frame(mode_f, bg=PANEL2); row.pack(fill="x", padx=10, pady=4)
            tk.Radiobutton(row, text=txt, variable=self.ext_mode, value=val,
                           bg=PANEL2, fg=TEXT, selectcolor=BORDER,
                           activebackground=PANEL2, font=FUI, cursor="hand2",
                           command=self._ext_mode_changed).pack(anchor="w")
            tk.Label(row, text=sub, bg=PANEL2, fg=SUBTEXT,
                     font=FS, justify="left").pack(anchor="w", padx=22)

        # ── Input files ────────────────────────────────────────────────
        self._lbl(col_l, "Input Files")
        inp_f = self._card(col_l); inp_f.pack(fill="x", pady=(2,10))

        self._ext_file_rows = {}
        file_defs = [
            ("mol2",   "Dendrimer mol2  (UDB v11 output)", self.ext_typed,  True),
            ("prmtop", "prmtop  (AMBER topology)",         self.ext_prmtop, False),
            ("inpcrd", "inpcrd  (AMBER coordinates)",      self.ext_inpcrd, False),
        ]
        for key, label, var, required in file_defs:
            tk.Label(inp_f, text=label, bg=PANEL2,
                     fg=TEXT if required else SUBTEXT,
                     font=FS, anchor="w").pack(fill="x", padx=12, pady=(8,1))
            row = tk.Frame(inp_f, bg=PANEL2); row.pack(fill="x", padx=12, pady=(0,4))
            lbl = tk.Label(row, text="Not set", bg=PANEL2, fg=MUTED,
                           font=FS, width=24, anchor="w")
            lbl.pack(side="left")
            btn = tk.Button(row, text="Browse",
                            bg=BORDER, fg=TEXT, activebackground=ACCENT,
                            activeforeground=BG, relief="flat", padx=8, pady=2,
                            font=FS, cursor="hand2",
                            command=lambda v=var, k=key: self._ext_browse(v, k))
            btn.pack(side="left", padx=(6,0))
            self._ext_file_rows[key] = (lbl, btn)
        tk.Frame(inp_f, bg=BG, height=4).pack()

        # ── Output types ───────────────────────────────────────────────
        self._lbl(col_l, "Output Files")
        out_f = self._card(col_l); out_f.pack(fill="x", pady=(2,10))
        for var, txt, note in [
            (self.ext_sybyl, "Sybyl mol2  ✦ primary",
                             "_sybyl.mol2   types: C.3, Si, H  ← raw"),
            (self.ext_gaff2, "GAFF2 mol2",
                             "_gaff2.mol2   types: c3, Si, hc"),
            (self.ext_pdb,   "PDB",
                             "_recovered.pdb  for VMD visualization"),
        ]:
            row = tk.Frame(out_f, bg=PANEL2); row.pack(fill="x", padx=10, pady=3)
            tk.Checkbutton(row, text=txt, variable=var,
                           bg=PANEL2, fg=TEXT, selectcolor=BORDER,
                           activebackground=PANEL2, font=FUI,
                           cursor="hand2").pack(side="left")
            tk.Label(row, text=f"  {note}", bg=PANEL2, fg=SUBTEXT,
                     font=FS).pack(side="left")
        tk.Frame(out_f, bg=BG, height=4).pack()

        # ── Output name + dir ──────────────────────────────────────────
        self._lbl(col_l, "Output Name Prefix")
        f = self._card(col_l); f.pack(fill="x", pady=(2,10))
        ttk.Entry(f, textvariable=self.ext_name, width=26).pack(
                  padx=12, pady=10, anchor="w")

        self._lbl(col_l, "Output Directory")
        f = self._card(col_l); f.pack(fill="x", pady=(2,10))
        self.ext_dir_lbl = tk.Label(f, text="Same as input mol2",
                                    bg=PANEL2, fg=SUBTEXT, font=FS,
                                    wraplength=220, justify="left")
        self.ext_dir_lbl.pack(anchor="w", padx=12, pady=(8,4))
        tk.Button(f, text="Browse…", bg=BORDER, fg=TEXT,
                  activebackground=ACCENT, activeforeground=BG,
                  relief="flat", padx=10, pady=3, font=FUI, cursor="hand2",
                  command=self._ext_browse_dir).pack(padx=12, pady=(0,8), anchor="w")

        # ── Right: run + stop buttons, progress bar, log ──────────────
        btn_row = tk.Frame(col_r, bg=BG)
        btn_row.pack(fill="x", pady=(0,6))
        self.ext_run_btn = tk.Button(btn_row,
                                     text="▶  Convert mol2",
                                     bg=ACCENT, fg=BG,
                                     activebackground=ACCENT2,
                                     activeforeground=BG, relief="flat",
                                     font=FH, padx=10, pady=10,
                                     cursor="hand2", command=self._run_extract)
        self.ext_run_btn.pack(side="left", fill="x", expand=True)
        self.ext_stop_btn = tk.Button(btn_row,
                                      text="⏹ Stop",
                                      bg=RED, fg=TEXT,
                                      activebackground="#ff7070",
                                      activeforeground=BG, relief="flat",
                                      font=FUI, padx=12, pady=10,
                                      cursor="hand2", state="disabled",
                                      command=self._stop_extract)
        self.ext_stop_btn.pack(side="left", padx=(6,0))

        # Progress bar
        prog_f = self._card(col_r); prog_f.pack(fill="x", pady=(0,8))
        prog_inner = tk.Frame(prog_f, bg=PANEL2); prog_inner.pack(fill="x", padx=10, pady=8)
        self.ext_prog = ttk.Progressbar(prog_inner, mode="determinate",
                                         maximum=3, value=0)
        self.ext_prog.pack(fill="x")
        self.ext_stage_lbl = tk.Label(prog_inner, text="Idle",
                                       bg=PANEL2, fg=SUBTEXT, font=FS, anchor="w")
        self.ext_stage_lbl.pack(fill="x", pady=(4,0))

        self._lbl(col_r, "Conversion Log")
        lf = self._card(col_r); lf.pack(fill="both", expand=True, pady=(2,0))
        self.ext_log = scrolledtext.ScrolledText(
            lf, bg="#070b0f", fg=TEXT, font=FMN, relief="flat",
            insertbackground=TEXT, wrap="word", state="disabled",
            padx=10, pady=8)
        self.ext_log.pack(fill="both", expand=True, padx=2, pady=2)
        for tag, col2 in [("ok",GREEN),("warn",YELLOW),("err",RED),
                          ("info",ACCENT),("sec",ACCENT2),("code",PURPLE)]:
            self.ext_log.tag_config(tag, foreground=col2)
        self.ext_log.tag_config("sec",  font=("Courier New",10,"bold"))
        self.ext_log.tag_config("code", font=("Courier New", 9))

        # Init mode state
        self._ext_mode_changed()

    def _ext_mode_changed(self):
        """Show/hide prmtop+inpcrd rows based on mode"""
        mode = self.ext_mode.get()
        for key in ("prmtop","inpcrd"):
            lbl, btn = self._ext_file_rows[key]
            state = "normal" if mode == "B" else "disabled"
            fg    = SUBTEXT  if mode == "B" else MUTED
            lbl.config(fg=fg)
            btn.config(state=state,
                       bg=BORDER if mode=="B" else PANEL,
                       fg=TEXT   if mode=="B" else MUTED)
        self.ext_run_btn.config(
            text="▶  Convert mol2" if mode=="A" else "▶  Extract mol2")

    def _stop_extract(self):
        self._ext_stop_flag = True
        self.ext_stop_btn.config(state="disabled")
        self.after(0, self._ext_log, "\n⏹ Stop requested — aborting after current step\n","warn")

    def _ext_browse(self, var, key):
        start = self._last_dir
        path = filedialog.askopenfilename(
            title=f"Select {key} file",
            initialdir=start)
        if not path: return
        self._last_dir = os.path.dirname(path)
        var.set(path)
        short = os.path.basename(path)
        self._ext_file_rows[key][0].config(text=short, fg=GREEN)
        # Auto-populate name + outdir from mol2
        if key == "mol2":
            d = os.path.dirname(path)
            if not self.ext_dir.get():
                self.ext_dir.set(d)
                short_d = d if len(d)<32 else "…"+d[-30:]
                self.ext_dir_lbl.config(text=short_d, fg=SUBTEXT)
            base = os.path.basename(path).replace(".mol2","")
            self.ext_name.set(base + "_raw")

    def _ext_browse_dir(self):
        start = self.ext_dir.get() or self._last_dir
        d = filedialog.askdirectory(title="Select output directory",
                                    initialdir=start)
        if d:
            self._last_dir = d
            self.ext_dir.set(d)
            short = d if len(d)<32 else "…"+d[-30:]
            self.ext_dir_lbl.config(text=short, fg=GREEN)

    def _ext_log(self, text, tag=None):
        self.ext_log.config(state="normal")
        self.ext_log.insert("end", text, tag or ())
        self.ext_log.see("end")
        self.ext_log.config(state="disabled")
        self.update_idletasks()

    def _ext_log_clear(self):
        self.ext_log.config(state="normal")
        self.ext_log.delete("1.0","end")
        self.ext_log.config(state="disabled")

    def _run_extract(self):
        mode   = self.ext_mode.get()
        typed  = self.ext_typed.get()
        name   = self.ext_name.get().strip() or "recovered"
        outdir = self.ext_dir.get() or (
                 os.path.dirname(typed) if typed else os.getcwd())

        # Validate
        if not typed or not os.path.exists(typed):
            messagebox.showerror("Missing file",
                "Please select the dendrimer mol2 file (UDB v11 output).")
            return
        if mode == "B":
            prmtop = self.ext_prmtop.get()
            inpcrd = self.ext_inpcrd.get()
            missing = []
            for lbl2, p2 in [("prmtop",prmtop),("inpcrd",inpcrd)]:
                if not p2:           missing.append(f"{lbl2} not set")
                elif not os.path.exists(p2): missing.append(f"{lbl2} not found")
            if missing:
                messagebox.showerror("Missing files",
                    "Mode B requires prmtop and inpcrd:\n" + "\n".join(missing))
                return
        if not any([self.ext_gaff2.get(), self.ext_sybyl.get(), self.ext_pdb.get()]):
            messagebox.showwarning("No output","Select at least one output type.")
            return

        self._ext_stop_flag = False
        self.ext_run_btn.config(state="disabled", text="Running…")
        self.ext_stop_btn.config(state="normal")
        self.ext_prog["value"] = 0
        self.ext_stage_lbl.config(text="Starting…", fg=ACCENT)
        self._ext_log_clear()
        os.makedirs(outdir, exist_ok=True)

        gaff2_out = os.path.join(outdir, f"{name}_gaff2.mol2")
        sybyl_out = os.path.join(outdir, f"{name}_sybyl.mol2")
        pdb_out   = os.path.join(outdir, f"{name}_recovered.pdb")

        want_gaff2 = self.ext_gaff2.get()
        want_sybyl = self.ext_sybyl.get()
        want_pdb   = self.ext_pdb.get()

        def _code(line):
            self.after(0, self._ext_log, f"  ▶ {line}\n", "code")

        def _result(line):
            self.after(0, self._ext_log, f"  → {line}\n", "ok")

        def _stage(n, label):
            self.after(0, lambda: self.ext_prog.config(value=n))
            self.after(0, lambda: self.ext_stage_lbl.config(text=label, fg=ACCENT))

        def _worker():
            try:
                mode_label = "Mode A — mol2 only" if mode=="A" else "Mode B — mol2 + AMBER files"
                self.after(0, self._ext_log,
                           f"{'═'*52}\n {mode_label}\n{'═'*52}\n","sec")

                # ══ Step 1: Load mol2 ══════════════════════════════════
                if self._ext_stop_flag: return
                _stage(0, "Step 1/3 — Loading mol2…")
                self.after(0, self._ext_log, "\n══ Step 1: Load & Parse mol2 ══\n","sec")

                _code(f'with open("{typed}", encoding="utf-8") as f:')
                _code(f'    content = f.read()')
                _code(f'sections = content.split("@<TRIPOS>")')

                with open(typed, encoding="utf-8", errors="replace") as f:
                    tc = f.read()
                tc = tc.replace("\r\n","\n").replace("\r","\n")
                secs = tc.split("@<TRIPOS>")
                t_mol_sec = t_atom_sec = t_bond_sec = ""
                for s in secs:
                    sl = s.strip()
                    if sl.startswith("MOLECULE"):  t_mol_sec  = s
                    elif sl.startswith("ATOM"):    t_atom_sec = s
                    elif sl.startswith("BOND"):    t_bond_sec = s

                _code(f'atoms = [l.split() for l in atom_sec if len(l.split()) >= 6]')
                t_atoms = []
                for l in t_atom_sec.split("\n")[1:]:
                    l = l.strip()
                    if l and not l.startswith("#"):
                        parts = l.split()
                        if len(parts) >= 6:
                            t_atoms.append(parts)

                _code(f'bonds = [(int(p[1]),int(p[2])) for p in bond_sec lines]')
                t_bonds = []
                for l in t_bond_sec.split("\n")[1:]:
                    parts = l.split()
                    if len(parts) >= 3:
                        try: t_bonds.append((int(parts[0]),int(parts[1]),int(parts[2])))
                        except: pass

                _result(f'{len(t_atoms)} atoms, {len(t_bonds)} bonds parsed')

                types_found = set(a[5] for a in t_atoms)
                gaff_types  = {t for t in types_found if t.lower() in GAFF2_TO_SYBYL}
                _result(f'GAFF2 types detected: {sorted(gaff_types)}')
                _stage(1, "Step 1/3 — mol2 loaded ✓")

                # ══ Step 2: Coordinates ════════════════════════════════
                if self._ext_stop_flag: return
                _stage(1, "Step 2/3 — Getting coordinates…")
                self.after(0, self._ext_log, "\n══ Step 2: Extract Coordinates ══\n","sec")

                coords = [(float(a[2]),float(a[3]),float(a[4])) for a in t_atoms]

                if mode == "B":
                    prmtop = self.ext_prmtop.get()
                    inpcrd = self.ext_inpcrd.get()
                    _code(f'subprocess.run(["ambpdb", "-p", "{prmtop}", "-c", "{inpcrd}"])')
                    result = subprocess.run(
                        ["ambpdb", "-p", prmtop, "-c", inpcrd],
                        capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        pdb_coords = []
                        for line in result.stdout.split("\n"):
                            if line.startswith(("ATOM","HETATM")):
                                try:
                                    x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
                                    pdb_coords.append((x,y,z))
                                except: pass
                        if len(pdb_coords) == len(t_atoms):
                            coords = pdb_coords
                            _result(f'prmtop coords loaded: {len(coords)} atoms')
                            if want_pdb:
                                with open(pdb_out,"w") as f: f.write(result.stdout)
                                _result(f'PDB written → {pdb_out}')
                        else:
                            self.after(0, self._ext_log,
                                f"  ⚠ coord mismatch ({len(pdb_coords)} vs {len(t_atoms)}) — using mol2 coords\n","warn")
                    else:
                        self.after(0, self._ext_log,"  ⚠ ambpdb failed — using mol2 coords\n","warn")
                else:
                    _code(f'coords = [(float(a[2]),float(a[3]),float(a[4])) for a in t_atoms]')
                    _result(f'Using mol2 coordinates — {len(coords)} atoms')

                _stage(2, "Step 2/3 — Coordinates ready ✓")

                # ══ Step 3: Write outputs ══════════════════════════════
                if self._ext_stop_flag: return
                _stage(2, "Step 3/3 — Writing output files…")
                self.after(0, self._ext_log, "\n══ Step 3: Write mol2 Output Files ══\n","sec")

                mol_lines  = t_mol_sec.split("\n")
                mol_name_l = mol_lines[1].strip() if len(mol_lines)>1 else name
                counts_l   = mol_lines[2].strip() if len(mol_lines)>2 else f"{len(t_atoms)} {len(t_bonds)} 0 0 0"
                ff_l       = mol_lines[3].strip() if len(mol_lines)>3 else "SMALL"
                chg_l      = mol_lines[4].strip() if len(mol_lines)>4 else "GASTEIGER"

                def write_mol2(outpath, use_sybyl):
                    out  = f"@<TRIPOS>MOLECULE\n{mol_name_l}\n{counts_l}\n{ff_l}\n{chg_l}\n\n"
                    out += "@<TRIPOS>ATOM\n"
                    for i, a in enumerate(t_atoms):
                        idx=int(a[0]); nm=a[1]; gaff=a[5]
                        atype = GAFF2_TO_SYBYL.get(gaff.lower(), gaff) if use_sybyl else gaff
                        si=a[6] if len(a)>6 else "1"; sn=a[7] if len(a)>7 else "LIG"
                        ch=a[8] if len(a)>8 else "0.0000"
                        x,y,z = coords[i] if i < len(coords) else (float(a[2]),float(a[3]),float(a[4]))
                        out += (f"{idx:>7} {nm:<8} {x:>10.4f} {y:>10.4f} {z:>10.4f}"
                                f" {atype:<8} {si:>2} {sn:<8} {float(ch):>8.4f}\n")
                    out += "@<TRIPOS>BOND\n"
                    for bid,ba,bb in t_bonds:
                        out += f"{bid:>6}  {ba:>4}  {bb:>4}  1\n"
                    with open(outpath,"w") as f: f.write(out)

                if want_sybyl:
                    _code(f'write_mol2("{sybyl_out}", use_sybyl=True)')
                    _code(f'  GAFF2_TO_SYBYL map: c3→C.3, Si→Si, hc→H, n3→N.3, os→O.3 ...')
                    write_mol2(sybyl_out, use_sybyl=True)
                    _result(f'Sybyl mol2 written → {sybyl_out}')

                if want_gaff2:
                    _code(f'write_mol2("{gaff2_out}", use_sybyl=False)')
                    _code(f'  Keeping GAFF2 types as-is from typed mol2')
                    write_mol2(gaff2_out, use_sybyl=False)
                    _result(f'GAFF2 mol2 written → {gaff2_out}')

                if want_pdb and mode == "A":
                    _code(f'antechamber -i "{typed}" -fi mol2 -o "{pdb_out}" -fo pdb -dr no')
                    result2 = subprocess.run(
                        ["antechamber","-i",typed,"-fi","mol2",
                         "-o",pdb_out,"-fo","pdb","-dr","no"],
                        capture_output=True, text=True)
                    if os.path.exists(pdb_out):
                        _result(f'PDB written → {pdb_out}')
                    else:
                        self.after(0, self._ext_log,"  ⚠ PDB generation failed\n","warn")

                _stage(3, "Complete ✓")
                self.after(0, self._ext_log, "\n" + "═"*52 + "\n","sec")
                self.after(0, self._ext_log, " ✓ Conversion complete\n","ok")
                self.after(0, self._ext_log, "═"*52 + "\n","sec")
                self.after(0, self._status, "✓ mol2 conversion complete")

            except Exception as e:
                import traceback
                self.after(0, self._ext_log, f"\n✗ Error: {e}\n","err")
                self.after(0, self._ext_log, traceback.format_exc(),"err")
                self.after(0, self._status, "✗ Conversion failed")
                self.after(0, lambda: self.ext_stage_lbl.config(text="Failed ✗", fg=RED))
            finally:
                self.after(0, lambda: self.ext_run_btn.config(
                    state="normal",
                    text="▶  Convert mol2" if self.ext_mode.get()=="A" else "▶  Extract mol2"))
                self.after(0, lambda: self.ext_stop_btn.config(state="disabled"))

    def _check_env_silent(self):
        """Quick check on startup to update badge"""
        tools=["antechamber","parmchk2","tleap"]
        all_ok=all(shutil.which(t) is not None for t in tools)
        if all_ok:
            self.env_badge.config(text="✓ env OK",fg=GREEN)
        else:
            self.env_badge.config(text="⚠ env issues",fg=YELLOW)

    def _check_env(self):
        self.env_log.config(state="normal")
        self.env_log.delete("1.0","end")

        def _elog(text,tag=None):
            self.env_log.insert("end",text,tag or ())
            self.env_log.see("end"); self.update_idletasks()

        tools_check=[
            ("antechamber","antechamber -h"),
            ("parmchk2",   "parmchk2 --version"),
            ("tleap",      "tleap -h"),
            ("python3",    "python3 --version"),
            ("parmed",     "python3 -c import_parmed"),
            ("acpype",     "acpype --version"),
        ]

        all_ok=True
        for name,cmd in tools_check:
            dot,path_lbl=self.env_rows[name]
            path=shutil.which(name.split()[0]) if name!="parmed" else "python3"
            if path:
                try:
                    if "import_parmed" in cmd:
                        actual_cmd=["python3","-c","import parmed; print(parmed.__version__)"]
                    else:
                        actual_cmd=cmd.split()
                    result=subprocess.run(actual_cmd,capture_output=True,
                                          text=True,timeout=8)
                    version=(result.stdout+result.stderr).split("\n")[0][:50].strip()
                    dot.config(text="●",fg=GREEN)
                    path_lbl.config(text=path,fg=MUTED)
                    _elog(f"✓ {name:<14} {version}\n","ok")
                except Exception as e:
                    dot.config(text="⚠",fg=YELLOW)
                    _elog(f"⚠ {name:<14} found but check failed: {e}\n","warn")
            else:
                if name in ("acpype",):
                    dot.config(text="○",fg=MUTED)
                    _elog(f"○ {name:<14} not found (optional)\n","warn")
                else:
                    dot.config(text="✗",fg=RED)
                    _elog(f"✗ {name:<14} NOT FOUND\n","err")
                    all_ok=False

        _elog("\n")
        if all_ok:
            _elog("✓ All required tools available\n","ok")
            self.env_badge.config(text="✓ env OK",fg=GREEN)
        else:
            _elog("✗ Missing required tools — check installation\n","err")
            self.env_badge.config(text="✗ env issues",fg=RED)

        self.env_log.config(state="disabled")

if __name__=="__main__":
    app=ParamApp()
    app.mainloop()
