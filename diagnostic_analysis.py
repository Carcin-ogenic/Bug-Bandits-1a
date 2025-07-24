#!/usr/bin/env python3
import json
import os
from collections import defaultdict, Counter

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(text):
    """Normalize text for comparison"""
    return " ".join(text.strip().split())

def analyze_predictions():
    gt_dir = "labels"
    pred_dir = "out6"  # Use corrected predictions
    
    issues = defaultdict(list)
    total_stats = Counter()
    
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".json"):
            continue
            
        print(f"\n=== {fname} ===")
        
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        
        if not os.path.exists(pred_path):
            print(f"❌ Missing prediction file")
            continue
            
        gt = load_json(gt_path)
        pred = load_json(pred_path)
        
        # Check title
        gt_title = normalize_text(gt.get("title", ""))
        pred_title = normalize_text(pred.get("title", ""))
        
        if gt_title and not pred_title:
            print(f"🔴 MISSING TITLE: GT='{gt_title}' -> PRED=''")
            issues["missing_title"].append((fname, gt_title))
            total_stats["missing_title"] += 1
        elif gt_title != pred_title and gt_title:
            print(f"🟡 TITLE MISMATCH: GT='{gt_title}' -> PRED='{pred_title}'")
            issues["title_mismatch"].append((fname, gt_title, pred_title))
            total_stats["title_mismatch"] += 1
        elif pred_title and not gt_title:
            print(f"🟠 EXTRA TITLE: PRED='{pred_title}' (GT has no title)")
            issues["extra_title"].append((fname, pred_title))
            total_stats["extra_title"] += 1
        else:
            print(f"✅ TITLE OK: '{gt_title}'")
            total_stats["title_correct"] += 1
        
        # Create maps for outline comparison
        gt_outline = {}
        for item in gt.get("outline", []):
            key = (normalize_text(item["text"]), item["level"])
            gt_outline[key] = item
            
        pred_outline = {}
        for item in pred.get("outline", []):
            key = (normalize_text(item["text"]), item["level"])
            pred_outline[key] = item
        
        # Check for missing headings
        for key, gt_item in gt_outline.items():
            text, level = key
            if key not in pred_outline:
                print(f"🔴 MISSING {level}: '{text}' (page {gt_item['page']})")
                issues["missing_heading"].append((fname, level, text, gt_item['page']))
                total_stats[f"missing_{level}"] += 1
                
        # Check for extra headings  
        for key, pred_item in pred_outline.items():
            text, level = key
            if key not in gt_outline:
                print(f"🟠 EXTRA {level}: '{text}' (page {pred_item['page']})")
                issues["extra_heading"].append((fname, level, text, pred_item['page']))
                total_stats[f"extra_{level}"] += 1
                
        # Check for page mismatches in matching headings
        for key in gt_outline.keys() & pred_outline.keys():
            gt_page = gt_outline[key]['page']
            pred_page = pred_outline[key]['page']
            if gt_page != pred_page:
                text, level = key
                print(f"🟡 PAGE MISMATCH {level}: '{text}' GT_page={gt_page} -> PRED_page={pred_page}")
                issues["page_mismatch"].append((fname, level, text, gt_page, pred_page))
                total_stats["page_mismatch"] += 1
    
    print(f"\n\n📊 OVERALL STATISTICS:")
    print(f"Title Issues:")
    print(f"  - Missing titles: {total_stats['missing_title']}")
    print(f"  - Title mismatches: {total_stats['title_mismatch']}")
    print(f"  - Extra titles: {total_stats['extra_title']}")
    print(f"  - Correct titles: {total_stats['title_correct']}")
    
    print(f"\nHeading Issues:")
    for level in ["H1", "H2", "H3", "H4"]:
        missing = total_stats[f"missing_{level}"]
        extra = total_stats[f"extra_{level}"]
        if missing or extra:
            print(f"  - {level}: {missing} missing, {extra} extra")
    
    print(f"\nOther Issues:")
    print(f"  - Page mismatches: {total_stats['page_mismatch']}")
    
    return issues, total_stats

if __name__ == "__main__":
    analyze_predictions()
