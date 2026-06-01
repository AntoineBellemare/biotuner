"""Validate that all code cells in the cookbook execute cleanly.

Extracts code cells from resonance_cookbook.ipynb and runs them in a shared
namespace, printing pass/fail per cell. Used because the dev environment
doesn't have jupyter/nbclient installed.
"""
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NB = Path(__file__).parent / "resonance_cookbook.ipynb"
nb = json.loads(NB.read_text())

ns = {"__name__": "__cookbook__"}
failures = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    # strip ipython magics like %matplotlib inline
    src = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("%"))
    label = f"cell #{i}"
    try:
        exec(compile(src, label, "exec"), ns)
        plt.close("all")
        print(f"[PASS] {label}  ({len(src)} chars)")
    except Exception as exc:
        plt.close("all")
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failures.append((i, exc))

print()
if failures:
    print(f"FAILED: {len(failures)} cell(s)")
    sys.exit(1)
else:
    print(f"All code cells executed cleanly.")
