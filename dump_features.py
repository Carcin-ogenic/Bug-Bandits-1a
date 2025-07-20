#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextBoxHorizontal, LTTextLineHorizontal

NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")

def normalize(text: str) -> str:
    """Collapse whitespace, strip, lowercase for consistent matching."""
    txt = re.sub(r"\s+", " ", text)
    return txt.strip().lower()

def iter_lines(pdf: Path):
    for page_i, layout in enumerate(extract_pages(pdf), 1):
        for el in layout:
            if not isinstance(el, (LTTextBoxHorizontal, LTTextLineHorizontal)):
                continue
            lines = [el] if isinstance(el, LTTextLineHorizontal) else \
                    [l for l in el if isinstance(l, LTTextLineHorizontal)]
            for ln in lines:
                chars = [c for c in ln if isinstance(c, LTChar)]
                if not chars:
                    continue
                txt = ln.get_text().strip()
                if len(txt) < 3:
                    continue
                yield {
                    "text": txt,
                    "size": max(c.size for c in chars),
                    "bold": int(any("Bold" in c.fontname or "Black" in c.fontname
                                    for c in chars)),
                    "indent": ln.x0,
                    "caps": sum(map(str.isupper, txt)) / len(txt),
                    "num": int(bool(NUM_RE.match(txt)))
                }

def build_gold_map(label_path: Path):
    """Return {normalized_text: level} or empty dict if .json not found."""
    if not label_path.exists():
        return {}
    jd = json.loads(label_path.read_text(encoding="utf-8"))
    gold = {}
    title = jd.get("title", "").strip()
    if title:
        gold[normalize(title)] = "TITLE"
    for h in jd.get("outline", []):
        text = h.get("text", "").strip()
        if text:
            gold[normalize(text)] = h.get("level", "")
    return gold

def main(pdf_dir, label_dir, out_csv):
    pdf_dir, label_dir = Path(pdf_dir), Path(label_dir)
    out_csv = Path(out_csv)

    # Compute median font size per PDF
    med = {}
    for pdf in pdf_dir.glob("*.pdf"):
        sizes = [l["size"] for l in iter_lines(pdf)]
        if sizes:
            med[pdf.name] = sorted(sizes)[len(sizes)//2]

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text","rel_size","bold","indent","caps","numbered","label"])

        for pdf in pdf_dir.glob("*.pdf"):
            gold = build_gold_map(label_dir / f"{pdf.stem}.json")
            m = med.get(pdf.name, 1.0)
            for ln in iter_lines(pdf):
                key = normalize(ln["text"])
                label = gold.get(key, "BODY")
                w.writerow([
                    ln["text"],
                    ln["size"] / m,
                    ln["bold"],
                    ln["indent"] / 600,
                    ln["caps"],
                    ln["num"],
                    label
                ])

    print(f"✅  Wrote {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: dump_features.py <pdf_dir> <label_dir> <out_csv>")
    main(*sys.argv[1:])
