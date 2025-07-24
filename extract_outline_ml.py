#!/usr/bin/env python3
"""
Fast inference – PyMuPDF + ONNX MiniLM-L6   (canonical text, merged lines)
"""
import json, re, sys, time, html, unicodedata, numpy as np, joblib, onnxruntime as ort
from pathlib import Path
import fitz
from transformers import AutoTokenizer

HDR_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")
CAPS_TH, RATIO = 0.60, 1.20

root   = Path(__file__).resolve().parent
# Update model directory and ONNX file for L12 multilingual model
onnxdir = root / "models" / "paraphrase-multilingual-MiniLM-L12-v2-quantized"
sess = ort.InferenceSession(str(onnxdir / "model_quantized.onnx"), providers=["CPUExecutionProvider"])
tok = AutoTokenizer.from_pretrained(onnxdir, local_files_only=True)
clf, LABELS = joblib.load(root/"models"/"clf.pkl").values()

# ---------- helpers ---------------------------------------------------
def norm(t): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html.unescape(t))).strip()

def embed(txt):
    batch = tok(txt, padding=True, truncation=True,
                max_length=64, return_tensors="np")
    inputs = {k: v.astype("int64") for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    outputs = sess.run(None, inputs)
    hidden = outputs[0]
    mask = inputs["attention_mask"][:, :, None]
    return (hidden * mask).sum(1) / mask.sum(1)

# Update feature extraction to match script7.py (PyMuPDF version)
def extract_features_from_pdf(pdf):
    doc = fitz.open(pdf)
    sizes = [s["size"] for p in doc
             for b in p.get_text("dict")["blocks"]
             for l in b.get("lines", [])
             for s in l.get("spans", [])]
    med = np.median(sizes) if sizes else 1.0
    lines = []
    for pg, page in enumerate(doc, 1):
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                spans = ln.get("spans", [])
                if not spans:
                    continue
                txt = "".join(s["text"] for s in spans).strip()
                if len(txt) < 3:
                    continue
                fontnames = set(s["font"] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                x0 = min(s["bbox"][0] for s in spans)
                x1 = max(s["bbox"][2] for s in spans)
                size = max(s["size"] for s in spans)
                bold = int(any("Bold" in s["font"] or "Black" in s["font"] for s in spans))
                caps = sum(map(str.isupper, txt)) / len(txt)
                num = int(bool(HDR_RE.match(txt)))
                line_len = len(txt)
                font_hash = hash(tuple(sorted(fontnames))) & 0xFFFFFFFF
                lines.append({
                    "text": txt,
                    "rel_size": size / med,
                    "bold": bold,
                    "indent": x0 / 600,
                    "caps": caps,
                    "numbered": num,
                    "line_len": line_len,
                    "page": pg,
                    "y0_pos": y0 / 800,
                    "font_hash": font_hash
                })
    doc.close()
    return lines

# ---------- inference -------------------------------------------------
def outline(pdf):
    rows = extract_features_from_pdf(pdf)
    if not rows:
        return {"title": "", "outline": []}
    # Prepare features in the correct order
    feats = np.array([
        [r["rel_size"], r["bold"], r["indent"], r["caps"], r["numbered"],
         r["line_len"], r["page"], r["y0_pos"], r["font_hash"]]
        for r in rows
    ], np.float32)
    X = np.hstack([embed([r["text"] for r in rows]), feats])
    pred = clf.predict(X)
    title, out = "", []
    for r, lid in zip(rows, pred):
        lbl = LABELS[lid]
        if lbl == "TITLE" and not title:
            title = r["text"]
            continue
        if lbl in {"H1", "H2", "H3"}:
            out.append({"level": lbl, "text": r["text"], "page": r["page"]})
    return {"title": title, "outline": out}

def main(src,dst):
    src,dst=map(Path,(src,dst)); dst.mkdir(exist_ok=True,parents=True)
    for pdf in src.glob("*.pdf"):
        t=time.time(); res=outline(pdf)
        with open(dst/f"{pdf.stem}.json","w",encoding="utf-8") as f:
            json.dump(res,f,indent=2,ensure_ascii=False)
        print(f"{pdf.name}: {time.time()-t:4.2f}s")

if __name__=="__main__":
    if len(sys.argv)!=3:
        sys.exit("usage: extract_outline_onnx.py <input_dir> <output_dir>")
    main(sys.argv[1],sys.argv[2])
