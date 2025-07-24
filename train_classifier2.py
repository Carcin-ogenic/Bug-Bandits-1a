#!/usr/bin/env python3
"""
Train logistic regression on multilingual ONNX MiniLM embeddings + layout features.
"""
import pandas as pd, numpy as np, joblib, onnxruntime as ort
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

root    = Path(__file__).resolve().parent

# ✅ Updated model directory
onnxdir = root / "models" / "paraphrase-multilingual-MiniLM-L12-v2-quantized"

# Load ONNX session + tokenizer
sess = ort.InferenceSession(str(onnxdir / "model_quantized.onnx"),
                            providers=["CPUExecutionProvider"])
tok  = AutoTokenizer.from_pretrained(onnxdir, local_files_only=True)

def embed(lines):
    batch = tok(lines, padding=True, truncation=True,
                max_length=64, return_tensors="np")

    inputs = {k: v.astype("int64") for k, v in batch.items() if k in ["input_ids", "attention_mask"]}

    outputs = sess.run(None, inputs)
    hidden = outputs[0]  # typically token embeddings (B, L, D)

    mask = inputs["attention_mask"][:, :, None]
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    assert pooled.shape[1] in (384, 768), "Unexpected embedding size"
    return pooled

# ✅ Load training data
df = pd.read_csv("trainFinal.csv")
Xtxt = embed(df.text.tolist())
# Use all new features in the correct order
Xnum = df[[
    "rel_size", "bold", "indent", "caps", "numbered",
    "line_len", "page", "y0_pos", "font_hash"
]].to_numpy("float32")
X    = np.hstack([Xtxt, Xnum])

le = LabelEncoder(); y = le.fit_transform(df.label)

clf = LogisticRegression(max_iter=400, multi_class="multinomial",
                         class_weight="balanced").fit(X, y)

out = root / "models"; out.mkdir(exist_ok=True)
joblib.dump({"clf": clf, "labels": le.classes_}, out/"clf.pkl")
print("✅  saved models/clf.pkl")
