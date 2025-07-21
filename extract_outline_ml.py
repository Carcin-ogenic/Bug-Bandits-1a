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
onnxdir = root / "models" / "all-MiniLM-L6-v2-onnx"
sess   = ort.InferenceSession(str(onnxdir/"model.onnx"),
                              providers=["CPUExecutionProvider"])
tok    = AutoTokenizer.from_pretrained(onnxdir, local_files_only=True)
clf, LABELS = joblib.load(root/"models"/"clf.pkl").values()

# ---------- helpers ---------------------------------------------------
def norm(t): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html.unescape(t))).strip()

def embed(txt):
    batch = tok(txt, padding=True, truncation=True,
                max_length=64, return_tensors="np",
                return_token_type_ids=True)
    inp = {k: v.astype("int64") for k, v in batch.items()}
    hid, = sess.run(None, inp)
    mask = inp["attention_mask"][:, :, None]
    return (hid*mask).sum(1) / mask.sum(1)

def rows_of(pdf):
    doc = fitz.open(pdf)
    sizes = [s["size"] for p in doc
             for b in p.get_text("dict")["blocks"]
             for l in b.get("lines", [])
             for s in l.get("spans", [])]
    med = np.median(sizes) if sizes else 1.0
    raw = []
    for pg, page in enumerate(doc):
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                txt = "".join(s["text"] for s in ln.get("spans", [])).strip()
                if len(txt) < 3: continue
                sp  = ln["spans"][0]
                raw.append({"page":pg, "y":sp["bbox"][1], "text":norm(txt),
                            "rel":sp["size"]/med,
                            "bold":int(sp["flags"]&2!=0),
                            "caps":sum(map(str.isupper, txt))/len(txt),
                            "num":int(bool(HDR_RE.match(txt)))})
    doc.close()

    # merge split lines
    raw.sort(key=lambda r:(r["page"], r["y"]))
    merged=[]
    for r in raw:
        if merged and r["page"]==merged[-1]["page"] and abs(r["y"]-merged[-1]["y"])<1:
            merged[-1]["text"] += " " + r["text"]
        else:
            merged.append(r)
    return [r for r in merged if r["rel"]>=RATIO or r["bold"] or r["caps"]>CAPS_TH or r["num"]]

# ---------- inference -------------------------------------------------
def outline(pdf):
    rows = rows_of(pdf)
    if not rows:
        return {"title":"", "outline":[]}

    feats = np.array([[r["rel"],r["bold"],r["caps"],r["num"]] for r in rows], np.float32)
    X = np.hstack([embed([r["text"] for r in rows]), feats])
    pred = clf.predict(X)

    title, out = "", []
    for r,lid in zip(rows,pred):
        lbl=LABELS[lid]
        if lbl=="TITLE" and not title:
            title=r["text"]; continue
        if lbl in {"H1","H2","H3"}:
            out.append({"level":lbl,"text":r["text"],"page":r["page"]})
    return {"title":title, "outline":out}

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
