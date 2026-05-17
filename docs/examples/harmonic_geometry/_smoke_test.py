"""Smoke-test the harmonic_geometry notebooks by executing every code cell
as a single Python script. Failures are reported per notebook with the
offending cell's first line so they can be fixed in _build_notebooks.py.
"""
from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path

# Force UTF-8 stdout on Windows so traceback rendering doesn't crash on the
# unicode characters used in notebook source / suptitles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # headless

HERE = Path(__file__).resolve().parent

# Make the worktree's biotuner importable first.
WORKTREE = HERE.parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

NOTEBOOKS = sorted(HERE.glob("*.ipynb"))


def run_notebook(path: Path) -> tuple[bool, str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            exec(compile(src, f"{path.name}#cell{i}", "exec"), ns)
        except Exception:  # noqa: BLE001
            return False, (
                f"FAILED in {path.name} cell {i}:\n"
                f"--- src ---\n{src}\n--- traceback ---\n{traceback.format_exc()}"
            )
    return True, f"OK  {path.name}  ({sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells)"


def main() -> int:
    rc = 0
    for nb in NOTEBOOKS:
        ok, msg = run_notebook(nb)
        print(msg)
        if not ok:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
