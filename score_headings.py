import csv
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import re

def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def load_gold_labels(label_dir):
    gold = defaultdict(dict)  # {(pdf_stem, norm_text): level}
    for json_file in Path(label_dir).glob("*.json"):
        pdf_stem = json_file.stem.lower()
        with open(json_file, encoding="utf-8") as f:
            jd = json.load(f)
        if jd.get("title"):
            gold[(pdf_stem, normalize(jd["title"]))] = "TITLE"
        for item in jd.get("outline", []):
            txt = item.get("text", "")
            if txt:
                gold[(pdf_stem, normalize(txt))] = item["level"]
    return gold

def load_predictions(csv_file):
    pred = defaultdict(dict)  # {(pdf_stem, norm_text): label}
    pdf_stem = Path(csv_file).stem.lower()
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().upper()
            if label == "BODY":
                continue
            text = normalize(row["text"])
            pred[(pdf_stem, text)] = label
    return pred

def evaluate(pred, gold):
    levels = ["TITLE", "H1", "H2", "H3", "H4"]
    tp = Counter()
    fp = Counter()
    fn = Counter()

    pred_keys = set(pred.keys())
    gold_keys = set(gold.keys())

    for key in pred_keys:
        pred_label = pred[key]
        gold_label = gold.get(key)
        if gold_label and pred_label == gold_label:
            tp[pred_label] += 1
        else:
            fp[pred_label] += 1

    for key in gold_keys:
        gold_label = gold[key]
        if key not in pred or pred[key] != gold_label:
            fn[gold_label] += 1

    print("\n📊 Heading Detection Metrics\n")
    print(f"{'Level':<8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
    for lvl in levels:
        t, f, n = tp[lvl], fp[lvl], fn[lvl]
        precision = t / (t + f) if (t + f) else 0.0
        recall = t / (t + n) if (t + n) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{lvl:<8} {precision:7.2%} {recall:7.2%} {f1:7.2%} {t:4} {f:4} {n:4}")

def main():
    csv_file = "trainFinal.csv"
    label_dir = "./labels"

    if not Path(csv_file).exists():
        print(f"❌ CSV not found: {csv_file}")
        return
    if not Path(label_dir).exists():
        print(f"❌ Label directory not found: {label_dir}")
        return

    gold = load_gold_labels(label_dir)
    pred = load_predictions(csv_file)
    evaluate(pred, gold)

if __name__ == "__main__":
    main()
