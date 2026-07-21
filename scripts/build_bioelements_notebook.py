"""Build (and execute) the bioelements docs notebook.

Run:  python scripts/build_bioelements_notebook.py
Writes docs/examples/bioelements/bioelements.ipynb with executed outputs.
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "examples" / "bioelements" / "bioelements.ipynb"


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


cells = [
    md("# bioelements — from biosignals to the periodic table, and to materials\n"
       "\n"
       "Every atom emits light at a characteristic set of wavelengths — its emission\n"
       "spectrum. `bioelements` octave-folds a biosignal's peaks into the optical band and\n"
       "asks which elements' lines they land on: the same octave-transpose `biocolors` uses\n"
       "to send a frequency to a *colour*, `bioelements` uses to send it to the *periodic\n"
       "table*.\n"
       "\n"
       "The expansion lifts this from **atoms** to **materials**. A `Composition` is a\n"
       "weighted bag of elements — or of other compositions — so water, air, wood, a cloud,\n"
       "any composition of matter, has its own composite spectrum, and therefore its own\n"
       "chord, tuning, palette, geometry, and biosignal affinity."),

    code("import warnings; warnings.filterwarnings('ignore')\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from biotuner import bioelements as be\n"
         "df = be.load_elements('air')\n"
         "print(f'{len(df):,} NIST emission lines across {df.element.nunique()} elements')\n"
         "print('element categories:', ', '.join(sorted(df.type.unique())))"),

    md("## An element is a spectrum is a chord is a tuning\n"
       "The atom is already a first-class harmonic object. Take hydrogen's strongest lines,\n"
       "fold them to audio, reduce to ratios — a **material tuning**."),

    code("h = be.element_spectrum('Hydrogen', top=12)\n"
         "print('hydrogen — strongest folded ratios:', be.material_tuning(be.MATERIALS['Water'], n_steps=6))\n"
         "print('an element leaf spectrum:', h)"),

    md("## Matter is compositions of atoms\n"
       "Water is H₂O. The composite spectrum is the weighted superposition of its elements'\n"
       "spectra — but each element must be **budget-normalised** first, or line-rich elements\n"
       "(oxygen has 3× hydrogen's lines) would dominate regardless of stoichiometry. With\n"
       "budget-normalisation, water is 2/3 hydrogen, 1/3 oxygen *by design*."),

    code("water = be.MATERIALS['Water']\n"
         "print('water.elements()      ', {k: round(v,3) for k,v in water.elements().items()})\n"
         "print('spectral dominance    ', {k: round(v,3) for k,v in water.dominant(top=None).items()})\n"
         "print('air.elements()        ', {k: round(v,3) for k,v in list(be.MATERIALS[\"Air\"].elements().items())[:3]})"),

    md("## Composition is recursive: a cloud is water + air\n"
       "A **structure**'s parts are themselves materials, so composition bottoms out at\n"
       "element leaves through any depth."),

    code("cloud = be.MATERIALS['Cloud']\n"
         "print('cloud.elements()', {k: round(v,3) for k,v in list(cloud.elements().items())[:5]})\n"
         "print('cloud composite spectrum:', len(cloud.spectrum()), 'lines')"),

    md("## The dictionary spans matter — a measured claim\n"
       "Materials are tagged on four axes (compositional level · material class · natural\n"
       "domain · element category). `coverage_report()` is a build gate: every class and\n"
       "domain represented, and the falsifiable metric — ≥ 8 of the 10 NIST element\n"
       "categories exercised."),

    code("rep = be.coverage_report(be.MATERIALS)"),

    md("## A biosignal resonates with materials, not just atoms\n"
       "Fold a signal's peaks to the optical band and score them against every material's\n"
       "composite lines (relative, cents-based tolerance). A biosignal now has a graded\n"
       "affinity to *water*, *wood*, *iron* — the materials it most resonates with."),

    code("sig = np.array([7.83, 14.3, 20.8, 27.3, 33.8])   # a Schumann-like signal\n"
         "ranked = be.match_materials(sig, tol_cents=60)\n"
         "print(ranked.head(8).to_string(index=False))"),

    md("## Every material affords a palette — its flame colours\n"
       "Each element burns with a characteristic colour (how chemists identify them by flame:\n"
       "Na→amber, Cu→green, K→violet, Li→crimson…). A material's palette is the flame colours\n"
       "of its dominant elements, weighted by composition — so water (H/O) reads nothing like a\n"
       "copper alloy (green) or table salt (amber+green)."),

    code("mats = ['Water','Fire','Wood','Air','Blood','Halite','Bronze','Chlorophyll','Quartz']\n"
         "fig, ax = plt.subplots(figsize=(9, 5))\n"
         "for row, name in enumerate(mats):\n"
         "    pal = be.material_flame_palette(be.MATERIALS[name], n=7)\n"
         "    for col, hexc in enumerate(pal):\n"
         "        ax.add_patch(plt.Rectangle((col, row), 1, 0.9, color=hexc))\n"
         "    ax.text(-0.2, row + 0.45, name, ha='right', va='center', fontsize=11)\n"
         "ax.set_xlim(-2.5, 7); ax.set_ylim(0, len(mats)); ax.axis('off')\n"
         "ax.set_title('material flame-colour palettes — each element by how it burns')\n"
         "plt.tight_layout(); plt.show()"),

    md("## And a tuning, a chord, a form\n"
       "The same composite spectrum reduces to a material scale, sounds as a material chord,\n"
       "and hands off to `harmonic_geometry` as a `HarmonicInput` for a material's *shape*."),

    code("for name in ['Water','Fire','Iron']:\n"
         "    m = be.MATERIALS[name]\n"
         "    print(f'{name:6s} tuning={m.tuning(n_steps=6)}')\n"
         "gi = be.MATERIALS['Water'].geometry(top=6)\n"
         "print('\\nwater.geometry() ->', type(gi).__name__, 'with', len(gi.peaks), 'partials')"),

    md("## The compositional hierarchy\n"
       "```\n"
       "  Element     H, O, C, N, Fe …            leaf spectra (NIST lines)\n"
       "     |  stoichiometry\n"
       "  Compound    H2O, NaCl, CO2, cellulose   fixed atom ratios\n"
       "     |  proportion\n"
       "  Mixture     air, wood, tissue, blood    blends of compounds/elements\n"
       "     |  arrangement\n"
       "  Structure   cloud, fire, lightning      compositions of materials (-> geometry)\n"
       "```\n"
       "One recursive operation — *superpose the component spectra, weighted, each\n"
       "budget-normalised* — carries all four levels. Everything a material affords is a\n"
       "transform of that one composite spectrum."),

    md("## Applied: a night of sleep, read as changing matter\n"
       "The affinity is not a still image. Run it across a **real night of sleep** (Sleep-EDF\n"
       "SC4012, EEG Fpz-Cz, 12 real 30-second epochs per stage) and the matter a biosignal\n"
       "resonates with **shifts with sleep depth** — the same recording, read as a changing\n"
       "composition of matter."),

    code("from pathlib import Path\n"
         "from scipy.signal import welch, find_peaks\n"
         "p = Path('sleep_stages.npz')\n"
         "if not p.exists(): p = Path('docs/examples/bioelements/sleep_stages.npz')\n"
         "S = np.load(p); epochs, stages, sf = S['epochs'], S['stages'], float(S['sf'])\n"
         "ORDER = ['W','N1','N2','N3','REM']\n"
         "print('real sleep epochs:', epochs.shape, '|',\n"
         "      {s:int((stages==s).sum()) for s in ORDER})\n"
         "\n"
         "def epoch_peaks(x, sf, n=6, fmin=1.5, fmax=30):\n"
         "    f, pxx = welch(x, sf, nperseg=int(sf*4)); b=(f>=fmin)&(f<=fmax)\n"
         "    f, db = f[b], 10*np.log10(pxx[b]+1e-30)\n"
         "    idx,_ = find_peaks(db, distance=2)\n"
         "    if len(idx)==0: return np.array([f[np.argmax(db)]])\n"
         "    return f[np.sort(idx[np.argsort(db[idx])[::-1][:n]])]"),

    md("### Which materials does each sleep stage resonate with?\n"
       "Per-epoch affinity to a curated set of materials, averaged within each stage — shown\n"
       "**relative per material** (z-scored across stages) so a stage reads as resonating\n"
       "*above* or *below* that material's own average, rather than being swamped by\n"
       "line-dense species that match almost anything."),

    code("MATS = ['Water','Air','Fire','Wood','Blood','Seawater','Quartz','Calcite','Bone',\n"
         "        'Chlorophyll','Cloud','Lightning','Methane','Glucose','StellarPlasma']\n"
         "aff = {s: [] for s in ORDER}\n"
         "for x, st in zip(epochs, stages):\n"
         "    pk = epoch_peaks(x, sf)\n"
         "    aff[str(st)].append([be.material_affinity(pk, be.MATERIALS[m], tol_cents=60) for m in MATS])\n"
         "H = np.array([np.mean(aff[s], axis=0) for s in ORDER])\n"
         "Z = (H - H.mean(0)) / (H.std(0) + 1e-9)   # per-material: which stages sit above/below average\n"
         "fig, ax = plt.subplots(figsize=(11, 4))\n"
         "im = ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-1.6, vmax=1.6)\n"
         "ax.set_xticks(range(len(MATS))); ax.set_xticklabels(MATS, rotation=45, ha='right')\n"
         "ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)\n"
         "ax.set_ylabel('sleep stage')\n"
         "ax.set_title('relative material affinity per sleep stage (per-material z-score) — Sleep-EDF SC4012')\n"
         "fig.colorbar(im, ax=ax, label='below  →  above average', shrink=0.85)\n"
         "plt.tight_layout(); plt.show()"),

    md("### The elemental *type* signature of each stage\n"
       "Summing `match_elements` by periodic **category** gives each stage a fingerprint over\n"
       "the kinds of matter — a stacked profile that shifts as the night deepens."),

    code("from biotuner.bioelements.tables import ELEMENT_CATEGORIES\n"
         "import matplotlib.cm as cm\n"
         "cat_mix = {}\n"
         "for s in ORDER:\n"
         "    xs = epochs[stages==s]\n"
         "    ff, pp = welch(xs[0], sf, nperseg=int(sf*4)); acc = np.zeros_like(pp)\n"
         "    for x in xs: acc += welch(x, sf, nperseg=int(sf*4))[1]\n"
         "    b=(ff>=1.5)&(ff<=30); fb, db = ff[b], 10*np.log10(acc[b]/len(xs)+1e-30)\n"
         "    idx,_ = find_peaks(db, distance=2); top = np.sort(idx[np.argsort(db[idx])[::-1][:6]])\n"
         "    me = be.match_elements(fb[top], top=30, tol_cents=60)\n"
         "    bc = me.groupby('category')['score'].sum()\n"
         "    cat_mix[s] = (bc/bc.sum()).reindex(ELEMENT_CATEGORIES).fillna(0)\n"
         "fig, ax = plt.subplots(figsize=(10,5)); bottom = np.zeros(len(ORDER))\n"
         "for c, col in zip(ELEMENT_CATEGORIES, cm.tab10(np.linspace(0,1,len(ELEMENT_CATEGORIES)))):\n"
         "    vals = [cat_mix[s][c] for s in ORDER]\n"
         "    ax.bar(ORDER, vals, bottom=bottom, label=c, color=col); bottom = bottom + vals\n"
         "ax.set_ylabel('share of element-match'); ax.set_title('elemental category signature per sleep stage')\n"
         "ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=8)\n"
         "plt.tight_layout(); plt.show()"),

    md("### The periodic table itself, lit by resonance\n"
       "Full granularity: score every one of the 99 elements (`match_elements`) and paint the\n"
       "result **onto the periodic table**. Each stage lights a different region of the table —\n"
       "the resonance has structure across the whole of chemistry, not just a top-few list."),

    code("from biotuner.bioelements import periodic as P\n"
         "PT_STAGES = ['W', 'N2', 'N3', 'REM']\n"
         "pt_scores = {}\n"
         "for s in PT_STAGES:\n"
         "    xs = epochs[stages == s]; acc = None\n"
         "    for x in xs:\n"
         "        me = be.match_elements(epoch_peaks(x, sf), top=30, tol_cents=60).set_index('element')['score']\n"
         "        acc = me if acc is None else acc.add(me, fill_value=0)\n"
         "    pt_scores[s] = acc / len(xs)\n"
         "\n"
         "def draw_pt(ax, sc, title):\n"
         "    vmax = float(sc.max()) or 1e-6\n"
         "    for name in P.NAMES:\n"
         "        pos = P.element_position(name)\n"
         "        if pos is None: continue\n"
         "        r, c = pos; v = float(sc.get(name, 0.0)) / vmax\n"
         "        ax.add_patch(plt.Rectangle((c, -r), 0.92, 0.92, color=plt.cm.magma(v)))\n"
         "        ax.text(c + 0.46, -r + 0.46, P.symbol(name), ha='center', va='center',\n"
         "                fontsize=5.5, color='white' if v < 0.55 else 'black')\n"
         "    ax.set_xlim(-0.5, 18); ax.set_ylim(-9, 1); ax.set_aspect('equal'); ax.axis('off')\n"
         "    ax.set_title(title, fontsize=12)\n"
         "fig, axs = plt.subplots(2, 2, figsize=(13, 8))\n"
         "for ax, s in zip(axs.ravel(), PT_STAGES): draw_pt(ax, pt_scores[s], f'stage {s}')\n"
         "fig.suptitle('element resonance across the periodic table, per sleep stage — Sleep-EDF SC4012', fontsize=13)\n"
         "plt.tight_layout(); plt.show()"),

    md("### A whole night, continuously — the resonance river\n"
       "Not five averages but **139 contiguous epochs** across ~9 hours: each folded to the\n"
       "optical band and scored by element category, stacked over time under the hypnogram.\n"
       "The composition of matter the EEG resonates with **flows** as the night unfolds."),

    code("from biotuner.bioelements.tables import ELEMENT_CATEGORIES\n"
         "import matplotlib.cm as cm\n"
         "nx, nstg, nt = S['night_epochs'], S['night_stages'], S['night_times_h']\n"
         "riv = np.zeros((len(nx), len(ELEMENT_CATEGORIES)))\n"
         "for i, x in enumerate(nx):\n"
         "    me = be.match_elements(epoch_peaks(x, sf), top=30, tol_cents=60)\n"
         "    bc = me.groupby('category')['score'].sum().reindex(ELEMENT_CATEGORIES).fillna(0)\n"
         "    riv[i] = (bc / (bc.sum() + 1e-9)).values\n"
         "fig, (axh, axr) = plt.subplots(2, 1, figsize=(12, 6.2),\n"
         "                               gridspec_kw={'height_ratios': [1, 4]}, sharex=True)\n"
         "sy = {'W': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}\n"
         "axh.plot(nt, [sy[s] for s in nstg], drawstyle='steps-post', color='#444', lw=1.5)\n"
         "axh.set_yticks(list(sy.values())); axh.set_yticklabels(list(sy.keys())); axh.set_ylabel('stage')\n"
         "axh.set_title('a night of sleep, read as a continuous resonance with matter — Sleep-EDF SC4012')\n"
         "axr.stackplot(nt, riv.T, labels=list(ELEMENT_CATEGORIES),\n"
         "              colors=cm.tab10(np.linspace(0, 1, len(ELEMENT_CATEGORIES))))\n"
         "axr.set_xlabel('time (hours)'); axr.set_ylabel('element-category share')\n"
         "axr.set_ylim(0, 1); axr.set_xlim(float(nt.min()), float(nt.max()))\n"
         "axr.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)\n"
         "plt.tight_layout(); plt.show()"),

    md("As the sleeper descends wake → N1 → N2 → slow-wave N3 and back through REM, the\n"
       "elements and materials the EEG most resonates with move across the periodic table —\n"
       "a biosignal's affinity with matter is **stateful**, tracking the brain's own rhythms."),
]

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

print("executing notebook (cwd = repo so the worktree biotuner is used) ...")
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(REPO)}})
client.execute()

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT.relative_to(REPO))
