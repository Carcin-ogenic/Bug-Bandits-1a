#!/usr/bin/env python3
import csv
import json
import re
import sys
import statistics
from pathlib import Path
from collections import defaultdict
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextBoxHorizontal, LTTextLineHorizontal

NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")

def normalize(text: str) -> str:
    txt = re.sub(r"\s+", " ", text)
    return txt.strip().lower()

def compute_gap_stats(lines):
    y_gaps = []
    prev_page = prev_y0 = None
    for ln in sorted(lines, key=lambda x: (x["page"], -x["y0"])):
        if prev_page == ln["page"] and prev_y0 is not None:
            y_gaps.append(abs(prev_y0 - ln["y0"]))
        prev_page = ln["page"]
        prev_y0 = ln["y0"]
    med_gap = statistics.median(y_gaps) if y_gaps else 14
    return med_gap

def heading_similarity(line1, line2, gap_med):
    # 🚫 Hard guards to avoid paragraph merging
    if line2['indent'] - line1['indent'] > 30:
        return 0
    if re.match(r"^[a-z]", line2["text"]) or re.search(r"[.:]$", line2["text"]):
        return 0

    # 🚫 New: Prevent merging across separate columns at similar y-levels
    if abs(line1['y0'] - line2['y0']) < 20 and abs(line1['x0'] - line2['x0']) > 100:
        return 0

    vert_gap = abs(line1['y0'] - line2['y0'])
    indent_sim = abs(line1['indent'] - line2['indent'])
    size_sim = abs(line1['size'] - line2['size'])
    font_sim = (line1['font_hash'] == line2['font_hash'])
    short_lines = max(line1['line_len'], line2['line_len']) < 60
    caps_sim = min(line1['caps'], line2['caps']) > 0.15

    score = 0
    if 0 < vert_gap < gap_med * 1.25:  # ⬅️ tightened vertical sensitivity
        score += 1
    if indent_sim < 150:
        score += 1
    if size_sim < 1.7:
        score += 1
    if font_sim:
        score += 1
    if short_lines:
        score += 1
    if caps_sim:
        score += 1
    return score

def fuzzy_group_headings(lines):
    gap_med = compute_gap_stats(lines)
    lines = sorted(lines, key=lambda x: (x['page'], -x['y0'], x['x0']))
    n = len(lines)
    used = set()
    groups = []
    i = 0
    while i < n:
        if i in used:
            i += 1
            continue
        run = [lines[i]]
        curr = lines[i]
        j = i + 1
        while j < n:
            nxt = lines[j]
            if nxt['page'] != curr['page']:
                break
            sim = heading_similarity(curr, nxt, gap_med)
            if sim >= 4:  # 🧠 conservative merge
                run.append(nxt)
                used.add(j)
                curr = nxt
                j += 1
            else:
                break
        if len(run) > 1:
            merged_text = " ".join(r["text"] for r in run)
            new_y0 = min(r["y0"] for r in run)
            merged = {**run[0], "text": merged_text, "y0": new_y0,
                      "line_len": len(merged_text),
                      "caps": sum(map(str.isupper, merged_text)) / (len(merged_text) or 1)}
            groups.append(merged)
        else:
            groups.append(run[0])
        i += len(run)
    return groups

def iter_lines(pdf: Path):
    all_lines = []
    for page_i, layout in enumerate(extract_pages(pdf), 0):
        line_objs = []
        for el in layout:
            if not isinstance(el, (LTTextBoxHorizontal, LTTextLineHorizontal)):
                continue
            lines = [el] if isinstance(el, LTTextLineHorizontal) else el._objs
            for ln in lines:
                if not isinstance(ln, LTTextLineHorizontal):
                    continue
                chars = [c for c in ln if isinstance(c, LTChar)]
                if not chars:
                    continue
                txt = ln.get_text().strip()
                if len(txt) < 3:
                    continue
                fontnames = set(c.fontname for c in chars)
                y0 = min(c.y0 for c in chars)
                x0 = min(c.x0 for c in chars)
                line_objs.append({
                    "text": txt,
                    "size": max(c.size for c in chars),
                    "bold": int(any("Bold" in c.fontname or "Black" in c.fontname for c in chars)),
                    "indent": x0,
                    "caps": sum(map(str.isupper, txt)) / len(txt),
                    "num": int(bool(NUM_RE.match(txt))),
                    "line_len": len(txt),
                    "page": page_i,
                    "y0": y0,
                    "x0": x0,
                    "x1": max(c.x1 for c in chars),
                    "font_hash": hash(tuple(sorted(fontnames))) & 0xFFFFFFFF,
                })
        # Horizontal merge
        clusters_by_y0 = defaultdict(list)
        for line in line_objs:
            key = round(line["y0"] / 2)
            clusters_by_y0[key].append(line)
        merged_lr_lines = []
        for group in clusters_by_y0.values():
            group = sorted(group, key=lambda x: x["x0"])
            full_text = " ".join(part["text"].strip() for part in group)
            base = group[0]
            base.update({
                "text": full_text,
                "line_len": len(full_text),
                "caps": sum(map(str.isupper, full_text)) / len(full_text) if len(full_text) else 0,
                "x1": group[-1]["x1"]
            })
            merged_lr_lines.append(base)
        all_lines.extend(merged_lr_lines)
    # Robust vertical merge
    merged_lines = fuzzy_group_headings(all_lines)
    for ln in merged_lines:
        yield ln

def build_gold_map(label_path: Path):
    if not label_path.exists():
        return {}
    jd = json.loads(label_path.read_text(encoding="utf-8"))
    gold = defaultdict(list)
    title = jd.get("title", "").strip()
    if title:
        gold[normalize(title)].append(("TITLE", 1, 0))
    for h in jd.get("outline", []):
        text = h.get("text", "").strip()
        page = h.get("page", 0)
        if text:
            gold[normalize(text)].append((h.get("level", ""), page, -1))
    return gold

def find_best_label(key, page, y0, gold, max_page_distance=1, debug=False):
    candidates = gold.get(key, [])
    if not candidates:
        return "BODY"
    close_matches = [
        (i, cand) for i, cand in enumerate(candidates)
        if abs(cand[1] - page) <= max_page_distance
    ]
    if not close_matches:
        if debug and candidates:
            closest_page = min(candidates, key=lambda c: abs(c[1] - page))[1]
            print(f"[find_best_label] No close match for key={key!r} on page={page}. Nearest in gold is page={closest_page}")
        return "BODY"
    best_idx = min(
        close_matches,
        key=lambda x: (
            abs(x[1][1] - page),
            abs(x[1][2] - y0) if x[1][2] >= 0 and y0 >= 0 else 0
        )
    )[0]
    best = candidates.pop(best_idx)
    if not candidates:
        del gold[key]
    return best[0]

def main(pdf_dir, label_dir, out_csv):
    pdf_dir, label_dir = Path(pdf_dir), Path(label_dir)
    out_csv = Path(out_csv)
    med = {}
    for pdf in pdf_dir.glob("*.pdf"):
        sizes = [l["size"] for l in iter_lines(pdf)]
        if sizes:
            med[pdf.name] = sorted(sizes)[len(sizes)//2]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "text", "rel_size", "bold", "indent", "caps", "numbered",
            "line_len", "page", "y0_pos", "font_hash", "label"
        ])
        for pdf in pdf_dir.glob("*.pdf"):
            gold = build_gold_map(label_dir / f"{pdf.stem}.json")
            m = med.get(pdf.name, 1.0)
            for ln in iter_lines(pdf):
                key = normalize(ln["text"])
                label = find_best_label(key, ln["page"], ln["y0"], gold)
                w.writerow([
                    ln["text"],
                    ln["size"] / m,
                    ln["bold"],
                    ln["indent"] / 600,
                    ln["caps"],
                    ln["num"],
                    ln["line_len"],
                    ln["page"],
                    ln["y0"] / 800,
                    ln["font_hash"],
                    label
                ])
    print(f"✅ Wrote {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: script.py <pdf_dir> <label_dir> <out_csv>")
    main(*sys.argv[1:])
