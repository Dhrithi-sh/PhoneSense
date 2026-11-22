"""Assemble report/sections/*.md into one PDF: report/PhoneSense_report.pdf.

    python scripts/build_report.py            # = make report
    python scripts/build_report.py --html-only

Steps:
1. Concatenate the sections in report order (ORDER below). Each section's internal status note
   (the blockquote right under its title: "Status: draft", "Report section ... (owner X)") and
   the literature review's "References (to pass to C)" working list are dropped.
2. Expand ``<!-- include: results/<file>.md[#test|#dev] -->`` with the generated table (the
   ``#test`` / ``#dev`` suffix keeps only that split's part of a file), so no table number is
   typed by hand.
3. Convert Markdown to HTML (tables, LaTeX math via KaTeX), embed the figures, add a title page
   and a table of contents built from the numbered headings.
4. Print to A4 PDF with a headless Chromium browser (Google Chrome or Microsoft Edge, with a
   temporary profile so it works while the browser is open), page numbers in the footer.

Needs ``markdown`` and ``pymdown-extensions`` (requirements-dev.txt), a local Edge or Chrome,
and network access for the KaTeX script (cdn.jsdelivr.net).
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "report" / "sections"
FIGURES = ROOT / "report" / "figures"
OUT_PDF = ROOT / "report" / "PhoneSense_report.pdf"
OUT_HTML = ROOT / "report" / "build" / "PhoneSense_report.html"

# (file, heading-level shift). C's graph/attack file is subsection 8.5 of Experiments.
ORDER: list[tuple[str, int]] = [
    ("01_abstract_A.md", 0),
    ("02_introduction_B.md", 0),
    ("03_literature_review_B.md", 0),
    ("04_objective_B.md", 0),
    ("05_proposed_model_B.md", 0),
    ("06_dataset_A.md", 0),
    ("07_methodology_C.md", 0),
    ("08a_experiments_retrieval_A.md", 0),
    ("08b_experiments_graph_attack_C.md", 1),
    ("09_conclusions_limitations_C.md", 0),
    ("10_references_C.md", 0),
    ("11_appendix_A.md", 0),
]
TITLE = "PhoneSense: Smartphone Search with Classic, Neural and Trust-Aware Ranking"
SUBTITLE = "CSD358 Information Retrieval, final project report"
AUTHORS = [("A", "Dhrithi Shashibushan"), ("B", "Palleti Viraja"), ("C", "Rishit Kamboj")]
NOTE_WORDS = ("Status:", "Report section", "Report appendix", "(owner")
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page); font: 9pt 'Times New Roman', serif;
                         color: #444; } }
@page :first { @bottom-center { content: none; } }
body { font: 10.5pt/1.42 'Times New Roman', Georgia, serif; color: #111; }
h1 { font-size: 16pt; margin: 0 0 8pt; break-before: page; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; }
h1, h2, h3 { break-after: avoid; font-family: 'Times New Roman', Georgia, serif; }
p, li { text-align: justify; hyphens: auto; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 8.3pt; break-inside: avoid; }
th, td { border-top: 0.5pt solid #999; border-bottom: 0.5pt solid #999; padding: 2pt 5pt; }
thead th { border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
img { display: block; max-width: 88%; max-height: 105mm; margin: 6pt auto 2pt; }
p > em:only-child { display: block; font-size: 9pt; text-align: center; }
code { font-size: 8.8pt; font-family: Consolas, monospace; }
.katex { font-size: 1.02em; }
.title { break-after: page; text-align: center; padding-top: 70mm; }
.title h1 { break-before: auto; font-size: 22pt; border: 0; }
.title .sub { font-size: 13pt; margin: 10pt 0 26pt; }
.title .authors { font-size: 12pt; line-height: 1.8; }
.toc { break-after: page; }
.toc h1 { break-before: auto; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { margin: 2pt 0; text-align: left; }
.toc li.l2 { padding-left: 16pt; font-size: 9.8pt; }
"""


def strip_note(text: str) -> str:
    """Drop the status/owner blockquote right after the section title."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines) and not lines[i].startswith("#"):
        out.append(lines[i])
        i += 1
    if i < len(lines):
        out.append(lines[i])
        i += 1
    j = i
    while j < len(lines) and not lines[j].strip():
        j += 1
    k = j
    while k < len(lines) and lines[k].startswith(">"):
        k += 1
    if k > j and any(w in " ".join(lines[j:k]) for w in NOTE_WORDS):
        i = k
    return "\n".join(out + lines[i:])


def drop_working_lists(text: str) -> str:
    return re.sub(r"\n## References \(to pass to C\).*?(?=\n# |\n## |\Z)", "\n", text, flags=re.S)


def include(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        rel, part = m.group(1), m.group(2)
        body = (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        if part:
            mm = re.search(rf"^### {part.capitalize()} split\n(.*?)(?=^### |\Z)", body,
                           flags=re.S | re.M)
            if not mm:
                raise SystemExit(f"include {rel}#{part}: section not found")
            body = mm.group(1)
        return "\n" + body.strip() + "\n"

    return re.sub(r"<!--\s*include:\s*([^#\s]+)(?:#(\w+))?\s*-->", repl, text)


def shift_headings(text: str, shift: int) -> str:
    if not shift:
        return text
    return re.sub(r"^(#{1,5}) ", lambda m: "#" * (len(m.group(1)) + shift) + " ", text, flags=re.M)


def embed_images(html_text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        path = FIGURES / Path(m.group(1)).name
        if not path.exists():
            raise SystemExit(f"figure not found: {path}")
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="(?:\.\./)?figures/([^"]+)"', repl,
                  html_text.replace('src="../figures/', 'src="figures/'))


def toc(html_text: str) -> str:
    items = []
    for level, ident, title in re.findall(r'<h([12]) id="([^"]+)">(.*?)</h\1>', html_text):
        clean = re.sub(r"<[^>]+>", "", title)
        if level == "2" and not re.match(r"\d+\.\d+ ", clean):
            continue
        items.append(f'<li class="l{level}"><a href="#{ident}">{clean}</a></li>')
    return '<div class="toc"><h1>Contents</h1><ul>' + "".join(items) + "</ul></div>"


def build_html() -> str:
    parts = []
    for name, shift in ORDER:
        text = (SECTIONS / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        text = drop_working_lists(strip_note(text))
        parts.append(shift_headings(include(text), shift))
    md = markdown.Markdown(extensions=["tables", "attr_list", "sane_lists", "toc",
                                       "pymdownx.arithmatex"],
                           extension_configs={"pymdownx.arithmatex": {"generic": True}})
    body = embed_images(md.convert("\n\n".join(parts)))
    authors = "<br>".join(f"{html.escape(n)} (member {r})" for r, n in AUTHORS)
    title = (f'<div class="title"><h1>{html.escape(TITLE)}</h1><div class="sub">'
             f"{html.escape(SUBTITLE)}</div><div class=\"authors\">{authors}</div></div>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '\\\\[', right: '\\\\]', display: true}},
    {{left: '\\\\(', right: '\\\\)', display: false}}],
    throwOnError: false}}); document.body.dataset.math = 'done';"></script>
<style>{CSS}</style></head><body>{title}{toc(body)}{body}</body></html>"""


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        if shutil.which(name):
            return shutil.which(name)  # type: ignore[return-value]
    raise SystemExit("no Edge or Chrome found to print the PDF; open the HTML and print it")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_PDF)
    args = ap.parse_args(argv)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    if args.html_only:
        return 0
    # own throw-away profile, so printing works while the browser is open
    profile = tempfile.mkdtemp(prefix="phonesense_pdf_")
    cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--no-first-run", f"--user-data-dir={profile}",
           "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=20000", f"--print-to-pdf={args.out.resolve()}",
           OUT_HTML.resolve().as_uri()]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if not args.out.exists() or args.out.stat().st_size == 0:
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("PDF was not written")
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
