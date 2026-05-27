"""
build_harmonic_sequence_paper_pdf.py
====================================
Compile docs/papers/harmonic_sequence_use_cases.md to a single PDF using
the `markdown_pdf` package (GitHub-flavoured Markdown via mistune + WeasyPrint).

Run:
    python scripts/build_harmonic_sequence_paper_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

ROOT = Path(__file__).resolve().parents[1]
PAPER_MD = ROOT / "docs" / "papers" / "harmonic_sequence_use_cases.md"
PAPER_PDF = ROOT / "docs" / "papers" / "harmonic_sequence_use_cases.pdf"


# ---------------------------------------------------------------------------
# Pre-process the markdown so paths and links work in the PDF
# ---------------------------------------------------------------------------

def preprocess(md_text: str) -> str:
    """Rewrite relative image paths to be relative to the markdown file.

    The markdown lives at docs/papers/harmonic_sequence_use_cases.md, and
    images reference figures/NN_*.png — that's already relative to the
    markdown file, so it works directly with markdown_pdf when we pass
    the file's directory as the resource root via Section(root=...).

    We do strip the worktree-relative `../../` repo-source links because
    they don't resolve to anything browseable from a PDF (they were
    GitHub navigation links). Leave them as text references.
    """
    # Convert `[label](../../path/to/file.py:LINE)` markdown links to a
    # plain monospace `label` followed by the path in parentheses. This
    # keeps the readable label while removing the broken-in-PDF link.
    pattern = re.compile(r"\[([^\]]+)\]\((\.\./\.\./[^\)]+)\)")
    md_text = pattern.sub(r"`\1` _(\2)_", md_text)
    # Same for `[label](figures/...)` — those WORK so leave them.
    return md_text


# ---------------------------------------------------------------------------
# CSS for clean printable layout
# ---------------------------------------------------------------------------

CSS = """
@page {
    size: A4;
    margin: 1.8cm 1.6cm;
}
body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.42;
    color: #222;
}
h1 {
    font-size: 18pt;
    color: #1a4480;
    border-bottom: 2px solid #1a4480;
    padding-bottom: 4px;
    margin-top: 14pt;
    page-break-before: auto;
}
h2 {
    font-size: 13.5pt;
    color: #1a4480;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3px;
    margin-top: 16pt;
}
h3 {
    font-size: 11.5pt;
    color: #1f3a5f;
    margin-top: 12pt;
}
h4 {
    font-size: 10.5pt;
    color: #1f3a5f;
}
p {
    margin: 6pt 0;
    text-align: justify;
}
img {
    max-width: 100%;
    display: block;
    margin: 8pt auto;
    page-break-inside: avoid;
}
pre, code {
    font-family: "Cascadia Mono", "Consolas", "Menlo", monospace;
    font-size: 9pt;
}
pre {
    background: #f4f6fa;
    border: 1px solid #d3dae3;
    border-radius: 3px;
    padding: 8pt 10pt;
    line-height: 1.30;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
}
code {
    background: #f4f6fa;
    padding: 1px 3px;
    border-radius: 2px;
}
pre code {
    background: transparent;
    padding: 0;
}
table {
    border-collapse: collapse;
    margin: 10pt 0;
    font-size: 9.5pt;
    width: 100%;
}
th, td {
    border: 1px solid #c9d2dc;
    padding: 4pt 6pt;
    text-align: left;
}
th {
    background: #eef3f8;
    font-weight: 600;
}
blockquote {
    border-left: 3px solid #1a4480;
    background: #f4f6fa;
    margin: 8pt 0;
    padding: 4pt 10pt;
    color: #444;
}
hr {
    border: 0;
    border-top: 1px solid #ccc;
    margin: 14pt 0;
}
em { color: #1a4480; }
strong { color: #111; }
a { color: #1a4480; text-decoration: none; }
"""


def main() -> int:
    print(f"Reading {PAPER_MD.relative_to(ROOT)}...")
    md_text = PAPER_MD.read_text(encoding="utf-8")
    md_text = preprocess(md_text)

    pdf = MarkdownPdf(toc_level=2, mode="gfm-like")
    pdf.meta["title"] = "Modelling temporal harmonic structure in biosignals"
    pdf.meta["author"] = "biotuner.harmonic_sequence"
    pdf.meta["subject"] = (
        "Use-case survey of the biotuner.harmonic_sequence module"
    )

    # The Section's `root` parameter tells markdown_pdf where to resolve
    # relative image/asset paths from. We point it at docs/papers/ so
    # `figures/NN_*.png` references work out of the box.
    pdf.add_section(
        Section(md_text, root=str(PAPER_MD.parent)),
        user_css=CSS,
    )

    print(f"Writing {PAPER_PDF.relative_to(ROOT)}...")
    pdf.save(str(PAPER_PDF))

    size_mb = PAPER_PDF.stat().st_size / 1e6
    print(f"  done. PDF size: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
