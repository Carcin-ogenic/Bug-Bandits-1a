#!/usr/bin/env python3
"""
Train logistic regression on ONNX MiniLM-L6 embeddings + layout features.
"""
import pandas as pd, numpy as np, joblib, onnxruntime as ort
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

root    = Path(__file__).resolve().parent
onnxdir = root / "models" / "all-MiniLM-L6-v2-onnx"

# ONNX session + tokenizer (offline)
sess = ort.InferenceSession(str(onnxdir/"model.onnx"),
                            providers=["CPUExecutionProvider"])
tok  = AutoTokenizer.from_pretrained(onnxdir, local_files_only=True)

def embed(lines):
    batch = tok(lines, padding=True, truncation=True,
                max_length=64, return_tensors="np",
                return_token_type_ids=True)
    inputs = {k: v.astype("int64") for k, v in batch.items()}
    hidden, = sess.run(None, inputs)              # (B, L, 384)
    mask = inputs["attention_mask"][:, :, None]   # (B, L, 1)
    return (hidden * mask).sum(1) / mask.sum(1)   # masked mean-pool

df = pd.read_csv("train.csv")                     # BODY kept
Xtxt = embed(df.text.tolist())
Xnum = df[["rel_size","bold","caps","numbered"]].to_numpy("float32")
X    = np.hstack([Xtxt, Xnum])

le = LabelEncoder(); y = le.fit_transform(df.label)
clf = LogisticRegression(max_iter=400, multi_class="multinomial",
                         class_weight="balanced").fit(X, y)

out = root / "models"; out.mkdir(exist_ok=True)
joblib.dump({"clf": clf, "labels": le.classes_}, out/"clf.pkl")
print("✅  saved models/clf.pkl")
