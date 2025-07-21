#!/usr/bin/env python3
"""
Dump the raw PyMuPDF text extraction to JSON so you can inspect
what the parser really returns.

usage:
    python debug_pymu_dump.py input.pdf output.json
"""
import json, sys
from pathlib import Path
import fitz                                # PyMuPDF

def dump(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []
    for pno, page in enumerate(doc, 1):
        page_dict = {"page": pno, "blocks": []}
        for blk in page.get_text("dict")["blocks"]:
            b_entry = {"bbox": blk["bbox"], "lines": []}
            for ln in blk.get("lines", []):
                l_entry = {"bbox": ln["bbox"], "spans": []}
                for sp in ln.get("spans", []):
                    span_info = {
                        "text"  : sp["text"],
                        "size"  : sp["size"],
                        "font"  : sp["font"],
                        "color" : sp["color"],
                        "flags" : sp["flags"],   # bit-field: 2=bold, 1=italic
                        "bbox"  : sp["bbox"]
                    }
                    l_entry["spans"].append(span_info)
                b_entry["lines"].append(l_entry)
            page_dict["blocks"].append(b_entry)
        pages.append(page_dict)
    doc.close()
    return pages

def main(pdf, out_json):
    pdf, out_json = map(Path, (pdf, out_json))
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(dump(pdf), f, indent=2, ensure_ascii=False)
    print(f"✅  wrote {out_json}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: debug_pymu_dump.py input.pdf output.json")
    main(*sys.argv[1:])
