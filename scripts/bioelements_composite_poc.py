"""Probe current bioelements + prototype the composite-material idea."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
REPO = r"C:/Users/skite/Documents/Github/biotuner/.claude/worktrees/epic-morse-ded0d5"
sys.path.insert(0, REPO)

print("pandas version:", pd.__version__)
df = pd.read_csv(f"{REPO}/data/air_elements.csv")
print("columns:", list(df.columns))
print("n lines:", len(df), " n elements:", df['element'].nunique())
print("wavelength unit: Angstrom, range:", round(df.wavelength.min(),1), "-", round(df.wavelength.max(),1))

# element name check for common biosignal-relevant atoms
for e in ["Hydrogen","Oxygen","Carbon","Nitrogen","Sodium","Calcium","Iron","Potassium"]:
    n = (df.element==e).sum()
    print(f"  {e:10s}: {n:5d} lines" if n else f"  {e:10s}: ABSENT")

# --- test the current matching function ---
from biotuner import bioelements as be
peaks_hz = np.array([10.0, 20.0, 40.0])  # alpha + harmonics
try:
    m = be.find_matching_spectral_lines(df, peaks_hz, tolerance=1.0, max_divisions=10)
    print("\nfind_matching_spectral_lines: OK, matched rows =", len(m))
except Exception as ex:
    print("\nfind_matching_spectral_lines FAILED:", type(ex).__name__, ex)

# --- test compute_ratios_df (suspected pandas-2 .append breakage) ---
small = df[df.element=="Hydrogen"].head(6).copy()
try:
    r = be.compute_ratios_df(small, "all", "element")
    print("compute_ratios_df: OK, rows =", len(r))
except Exception as ex:
    print("compute_ratios_df FAILED:", type(ex).__name__, str(ex)[:80])

# ============================================================== #
print("\n" + "="*66)
print("PROOF OF CONCEPT: composite material spectra from element formulas")
print("="*66)
from biotuner.biotuner_utils import compute_peak_ratios

# atomic composition (atom counts) of a few complex materials
MATERIALS = {
    "Water (H2O)":        {"Hydrogen": 2, "Oxygen": 1},
    "Air (dry)":          {"Nitrogen": 78, "Oxygen": 21, "Argon": 1},   # ~mole fractions
    "Cellulose (wood)":   {"Carbon": 6, "Hydrogen": 10, "Oxygen": 5},
    "Soft tissue":        {"Hydrogen": 63, "Oxygen": 26, "Carbon": 9, "Nitrogen": 1.4},  # atomic %
    "Table salt (NaCl)":  {"Sodium": 1, "Chlorine": 1},
}

def element_lines(elem, top=30):
    """Top-`top` most intense emission lines of an element: (wavelength_A, intensity)."""
    sub = df[df.element == elem].nlargest(top, "intensity")
    return sub["wavelength"].values, sub["intensity"].values

def composite_spectrum(formula, top=30):
    """Weighted superposition: each element's lines scaled by (atom_count * intensity)."""
    wl, inten, lbl = [], [], []
    for elem, count in formula.items():
        w, it = element_lines(elem, top)
        if len(w) == 0:
            continue
        wl.append(w); inten.append(it * float(count)); lbl += [elem]*len(w)
    return np.concatenate(wl), np.concatenate(inten), np.array(lbl)

for name, formula in MATERIALS.items():
    wl, inten, lbl = composite_spectrum(formula, top=25)
    # material tuning: fold the composite line wavelengths to ratios
    order = np.argsort(inten)[::-1][:12]                 # 12 strongest composite lines
    hz = 2.998e8 / (wl[order] * 1e-10)                   # Angstrom -> Hz
    ratios = compute_peak_ratios(list(hz), rebound=True)
    ratios = sorted(set(round(r, 3) for r in ratios if 1.0 <= r <= 2.0))[:8]
    # which element dominates the composite by summed weighted intensity?
    dom = pd.Series(inten, index=lbl).groupby(level=0).sum().sort_values(ascending=False)
    domstr = ", ".join(f"{k} {100*v/dom.sum():.0f}%" for k,v in dom.items())
    print(f"\n  {name}")
    print(f"    composite lines: {len(wl)}   dominance: {domstr}")
    print(f"    material tuning (folded ratios): {ratios}")

# --- match a biosignal to MATERIALS (not just atoms) ---
print("\n" + "="*66)
print("PROOF OF CONCEPT: match a biosignal to composite materials")
print("="*66)
def fold_to_optical(hz, lo=3000, hi=7000):
    """Octave-transpose a Hz value into the optical wavelength band [lo,hi] Angstrom."""
    wl = 2.998e18 / hz  # Hz -> Angstrom (c in Angstrom/s = 2.998e18)
    while wl > hi: wl /= 2
    while wl < lo: wl *= 2
    return wl

def material_affinity(peaks_hz, formula, top=40, tol=15.0):
    """Score how strongly a signal's peaks align with a material's composite lines."""
    wl, inten, _ = composite_spectrum(formula, top)
    score = 0.0
    for pk in peaks_hz:
        w = fold_to_optical(pk)
        near = np.abs(wl - w) <= tol
        if near.any():
            score += inten[near].sum()
    return score / (inten.sum() + 1e-9)

# a synthetic 'signal' (peaks in Hz)
sig = np.array([7.83, 14.3, 20.8, 27.3, 33.8])   # Schumann-like
print("  signal peaks (Hz):", sig)
aff = {name: material_affinity(sig, f) for name, f in MATERIALS.items()}
for name, a in sorted(aff.items(), key=lambda x: -x[1]):
    print(f"    {name:20s} affinity {a:.4f}")
print("\n  -> a biosignal now has a graded affinity to composite MATERIALS,")
print("     computed from the weighted superposition of their element spectra.")
