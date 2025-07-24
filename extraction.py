#!/usr/bin/env python3
import os
import json
import pickle
import argparse
import numpy as np
import fitz  # PyMuPDF
import onnxruntime as ort
from transformers import AutoTokenizer
import re
import statistics
from collections import defaultdict

# Regex for numbering detection
NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")

# ——— 1) Heading grouping logic ———
def compute_gap_stats(lines):
    y_gaps, prev_page, prev_y0 = [], None, None
    for ln in sorted(lines, key=lambda x: (x["page"], -x["y0"])):
        if prev_page == ln["page"] and prev_y0 is not None:
            y_gaps.append(abs(prev_y0 - ln["y0"]))
        prev_page, prev_y0 = ln["page"], ln["y0"]
    return statistics.median(y_gaps) if y_gaps else 14

def heading_similarity(a, b, gap_med):
    if b["indent"] - a["indent"] > 30: return 0
    if re.match(r"^[a-z]", b["text"]) or re.search(r"[.:]$", b["text"]): return 0
    if abs(a["y0"] - b["y0"]) < 20 and abs(a["x0"] - b["x0"]) > 100: return 0

    vg, ig, sz = abs(a["y0"] - b["y0"]), abs(a["indent"] - b["indent"]), abs(a["size"] - b["size"])
    font_sim = a["font_hash"] == b["font_hash"]
    short = max(a["line_len"], b["line_len"]) < 60
    caps = min(a["caps"], b["caps"]) > 0.15

    score = (
        (0 < vg < gap_med * 1.25) +
        (ig < 150) +
        (sz < 1.7) +
        font_sim +
        short +
        caps
    )
    return score

def fuzzy_group_headings(lines):
    gap_med = compute_gap_stats(lines)
    lines = sorted(lines, key=lambda x: (x["page"], -x["y0"], x["x0"]))
    used, groups, i, n = set(), [], 0, len(lines)
    while i < n:
        if i in used:
            i += 1
            continue
        run, curr, j = [lines[i]], lines[i], i+1
        while j < n and lines[j]["page"] == curr["page"]:
            if heading_similarity(curr, lines[j], gap_med) >= 4:
                run.append(lines[j])
                used.add(j)
                curr = lines[j]
                j += 1
            else:
                break
        if len(run) > 1:
            merged_text = " ".join(r["text"] for r in run)
            groups.append({
                **run[0],
                "text": merged_text,
                "y0": min(r["y0"] for r in run),
                "line_len": len(merged_text),
                "caps": sum(c.isupper() for r in run for c in r["text"]) / (
                    sum(len(r["text"]) for r in run) or 1
                )
            })
        else:
            groups.append(run[0])
        i += len(run)
    return groups

# ——— 2) Extract & group lines via PyMuPDF ———
def iter_lines(pdf_path):
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc, start=1):
        spans = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if len(text) < 3:
                    continue
                sizes = [s["size"] for s in line["spans"]]
                fonts = [s["font"] for s in line["spans"]]
                x0s = [s["bbox"][0] for s in line["spans"]]
                y0s = [s["bbox"][1] for s in line["spans"]]
                spans.append({
                    "text": text,
                    "size": max(sizes),
                    "bold": int(any("Bold" in f or "Black" in f for f in fonts)),
                    "indent": min(x0s),
                    "caps": sum(c.isupper() for c in text) / len(text),
                    "num": int(bool(NUM_RE.match(text))),
                    "line_len": len(text),
                    "page": page_index,
                    "y0": min(y0s),
                    "x0": min(x0s),
                    "x1": max(s["bbox"][2] for s in line["spans"]),
                    "font_hash": hash(tuple(sorted(fonts))) & 0xFFFFFFFF,
                })
        # horizontal merge by y0
        clusters = defaultdict(list)
        for ln in spans:
            key = round(ln["y0"] / 2)
            clusters[key].append(ln)
        merged = []
        for grp in clusters.values():
            grp = sorted(grp, key=lambda x: x["x0"])
            full = " ".join(g["text"] for g in grp)
            base = grp[0].copy()
            base.update({
                "text": full,
                "line_len": len(full),
                "caps": sum(c.isupper() for c in full) / (len(full) or 1),
                "x1": grp[-1]["x1"],
            })
            merged.append(base)
        # apply fuzzy grouping
        for ln in fuzzy_group_headings(merged):
            yield ln

# ——— 3) Embedding helper ———
def load_text_embedder(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(
        f"{model_dir}/model_quantized.onnx",
        providers=["CPUExecutionProvider"]
    )
    input_names = {inp.name for inp in session.get_inputs()}

    def embed(text: str) -> np.ndarray:
        toks = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        ort_in = {}
        for k, v in toks.items():
            if k in input_names:
                if v.dtype == np.int32:
                    v = v.astype(np.int64)
                ort_in[k] = v
        out = session.run(None, ort_in)[0]
        if out.ndim == 3:
            out = out[0]
        return out.mean(axis=0)
    return embed

# ——— 4) Main: label PDFs and write JSON ———
def main(pdf_dir, output_dir, model_pickle):
    # load classifier bundle
    data = pickle.load(open(model_pickle, "rb"))
    clf, scaler, num_cols = data["clf"], data["scaler"], data["num_cols"]
    embed_text = load_text_embedder(data["model_dir"])

    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        # 1) extract & group
        lines = list(iter_lines(path))
        # 2) compute per‐PDF median size & rel_size
        sizes = [ln["size"] for ln in lines]
        m = statistics.median(sizes) if sizes else 1.0
        for ln in lines:
            ln["rel_size"] = ln["size"] / m

        # 3) build feature arrays
        # 3a) text embeddings
        embs = np.vstack([embed_text(ln["text"]) for ln in lines])
        # 3b) numeric feature normalizations (must match training!)
        for ln in lines:
            # rel_size was computed earlier
            # indent feature was scaled by /600
            ln["indent"] = ln["indent"] / 600
            # y0_pos feature was scaled by y0/800
            ln["y0_pos"] = ln["y0"] / 800
        # 3c) assemble numeric features
        X_num = np.vstack([
            [ln[col] if col != "numbered" else ln["num"] for col in num_cols]
            for ln in lines
        ])
        X_num_scaled = scaler.transform(X_num)
        X = np.hstack([embs, X_num_scaled])

        # 4) predict labels
        preds = clf.predict(X)

        # 5) assemble JSON
        titles = [ln["text"] for ln, p in zip(lines, preds) if p == "TITLE"]
        title = titles[0] if titles else ""
        outline = [
            {"level": p, "text": ln["text"], "page": ln["page"]}
            for ln, p in zip(lines, preds)
            if p in ("H1", "H2", "H3", "H4")
        ]

        # 6) write JSON
        base, _ = os.path.splitext(fname)
        with open(os.path.join(output_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump({"title": title, "outline": outline}, fh, ensure_ascii=False, indent=2)

        print(f"Generated {base}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", help="Input PDFs directory")
    parser.add_argument("output_dir", help="Where to write JSON outlines")
    parser.add_argument(
        "--model_pickle",
        default="./models/classifier.pkl",
        help="Path to trained classifier bundle"
    )
    args = parser.parse_args()
    main(args.pdf_dir, args.output_dir, args.model_pickle)
