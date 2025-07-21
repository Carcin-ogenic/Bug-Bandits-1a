#!/usr/bin/env python3
"""
Rebuild train.csv with canonical text and full gold-label coverage
usage: python make_train_csv.py <pdf_dir> <label_dir> <out_csv>
"""
import csv, json, re, sys, html, unicodedata
from pathlib import Path
from statistics import median
import fitz                         # PyMuPDF

HDR_RE          = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")
CAPS_TH, RATIO  = 0.60, 1.20        # heading heuristics

# ---------- text canonicalisation (identical for train & inference)
def norm(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", html.unescape(txt))
    return re.sub(r"\s+", " ", txt).strip()

def gold_map(label_json: Path) -> dict:
    if not label_json.exists():
        return {}
    jd = json.load(label_json.open(encoding="utf-8"))
    m  = {norm(jd["title"]): "TITLE"}
    m.update({norm(h["text"]): h["level"] for h in jd["outline"]})
    return m

# ---------- line extraction + simple line-merge
def pdf_rows(pdf: Path):
    doc   = fitz.open(pdf)
    sizes = [s["size"]
             for p in doc
             for b in p.get_text("dict")["blocks"]
             for l in b.get("lines", [])
             for s in l.get("spans", [])]
    med   = median(sizes) if sizes else 1.0
    raw   = []
    for pg, page in enumerate(doc):
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                txt = "".join(s["text"] for s in ln.get("spans", [])).strip()
                if len(txt) < 3:
                    continue
                sp   = ln["spans"][0]
                raw.append({
                    "page" : pg,
                    "y"    : sp["bbox"][1],          # y-top
                    "text" : norm(txt),
                    "rel"  : sp["size"]/med,
                    "bold" : int(sp["flags"] & 2 != 0),
                    "caps" : sum(map(str.isupper, txt))/len(txt),
                    "num"  : int(bool(HDR_RE.match(txt)))
                })
    doc.close()

    # merge consecutive fragments on the same page & y-coord (±1 pt)
    raw.sort(key=lambda r: (r["page"], r["y"]))
    merged = []
    for r in raw:
        if merged and r["page"] == merged[-1]["page"] and abs(r["y"]-merged[-1]["y"]) < 1:
            merged[-1]["text"] += " " + r["text"]
        else:
            merged.append(r)
    return merged, med

# ---------- main ------------------------------------------------------
def main(pdf_dir, lbl_dir, out_csv):
    pdf_dir, lbl_dir, out_csv = map(Path, (pdf_dir, lbl_dir, out_csv))
    unmatched = {}
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdf","page","text","rel_size","bold","caps","numbered","label"])

        for pdf in pdf_dir.glob("*.pdf"):
            gmap = gold_map(lbl_dir / f"{pdf.stem}.json")
            seen = set()

            rows, _ = pdf_rows(pdf)
            for row in rows:
                key = row["text"]
                if key in gmap:
                    row["label"] = gmap[key]
                    seen.add(key)
                else:
                    keep = (row["rel"] >= RATIO or row["bold"] or
                            row["caps"] > CAPS_TH or row["num"])
                    if not keep:
                        continue
                    row["label"] = "BODY"
                w.writerow([pdf.name, row["page"], row["text"],
                            f"{row['rel']:.3f}", row["bold"],
                            f"{row['caps']:.3f}", row["num"], row["label"]])

            for miss in gmap.keys() - seen:
                unmatched.setdefault(pdf.name, []).append(miss)

    if unmatched:
        print("⚠ Unmatched gold strings:")
        for pdf, misses in unmatched.items():
            for m in misses:
                print(f"  {pdf}: '{m}'")
    print("✅  train.csv rebuilt:", out_csv)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: make_train_csv.py <pdf_dir> <label_dir> <out_csv>")
    main(*sys.argv[1:])
