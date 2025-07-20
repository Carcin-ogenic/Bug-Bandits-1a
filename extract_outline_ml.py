#!/usr/bin/env python3
"""
PDF → JSON outline
• MiniLM (sentence-transformers, PyTorch CPU)
• 5 layout features
• logistic-regression head in models/clf.pkl
"""
import json, re, sys, time, joblib
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar

# ───────── constants ─────────
NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")

# ───────── load models ───────
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

root = Path(__file__).resolve()
while root != root.parent and not (root / "models" / "clf.pkl").exists():
    root = root.parent
clf_path = root / "models" / "clf.pkl"
ART = joblib.load(clf_path)
clf, LABELS = ART["clf"], ART["labels"]

# ───────── helpers ───────────
def parse_pdf(pdf: Path):
    rows, sizes = [], []
    for pg, layout in enumerate(extract_pages(pdf), 0):
        for box in layout:
            if not isinstance(box, (LTTextBoxHorizontal, LTTextLineHorizontal)):
                continue
            for ln in ([box] if isinstance(box, LTTextLineHorizontal)
                       else [l for l in box if isinstance(l, LTTextLineHorizontal)]):
                chars = [c for c in ln if isinstance(c, LTChar)]
                if not chars:
                    continue
                txt = ln.get_text().strip()
                if len(txt) < 3:
                    continue
                size = max(c.size for c in chars)
                rows.append({
                    "text": txt,
                    "page": pg,
                    "size": size,
                    "bold": int(any("Bold" in c.fontname or "Black" in c.fontname
                                    for c in chars)),
                    "indent": ln.x0,
                    "caps":  sum(map(str.isupper, txt)) / len(txt),
                    "num":   int(bool(NUM_RE.match(txt)))
                })
                sizes.append(size)
    return rows, (np.median(sizes) if sizes else 1.0)

# ───────── core ──────────────
def predict_outline(pdf: Path):
    rows, med = parse_pdf(pdf)
    if not rows:
        return {"title": "", "outline": []}

    num = np.array([[r["size"]/med, r["bold"], r["indent"]/600, r["caps"], r["num"]]
                    for r in rows], dtype=np.float32)

    emb = embedder.encode([r["text"] for r in rows],
                          batch_size=256, convert_to_numpy=True)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    X = np.hstack([emb, num])
    preds = clf.predict(X)

    title, outline = None, []
    for r, idx in zip(rows, preds):
        lbl = LABELS[idx]
        if lbl == "TITLE":
            if title is None:                      # keep first only
                title = r["text"]
            continue                               # don't duplicate in outline
        if lbl in {"H1", "H2", "H3"}:
            outline.append({"level": lbl,
                            "text":  r["text"],
                            "page":  r["page"]})

    # leave title empty if the model never predicted TITLE
    if title is None:
        title = ""

    return {"title": title, "outline": outline}

# ───────── CLI ───────────────
def main(inp_dir, out_dir):
    inp, out = map(Path, (inp_dir, out_dir))
    out.mkdir(parents=True, exist_ok=True)
    for pdf in inp.glob("*.pdf"):
        t0 = time.time()
        res = predict_outline(pdf)
        with open(out / f"{pdf.stem}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"{pdf.name}: {time.time()-t0:4.1f}s")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: extract_outline_ml.py <input_dir> <output_dir>")
    main(sys.argv[1], sys.argv[2])
