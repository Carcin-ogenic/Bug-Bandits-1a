#!/usr/bin/env python3
import pandas as pd
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# 1) Load training CSV
CSV_PATH = "trainFinal.csv"
df = pd.read_csv(CSV_PATH)

# 2) Load ONNX MiniLM model & tokenizer
MODEL_DIR = "./models/paraphrase-multilingual-MiniLM-L12-v2-quantized"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(
    f"{MODEL_DIR}/model_quantized.onnx",
    providers=["CPUExecutionProvider"]
)

# Cache the valid ONNX input names
onnx_input_names = {inp.name for inp in session.get_inputs()}

def embed_text(text: str) -> np.ndarray:
    """
    Tokenize, run ONNX to get token embeddings,
    then mean‑pool to obtain a single vector.
    """
    # 1) Tokenize
    toks = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True
    )
    # 2) Keep only inputs ONNX expects, and cast ints to int64
    ort_inputs = {}
    for name, arr in toks.items():
        if name in onnx_input_names:
            if arr.dtype == np.int32:
                arr = arr.astype(np.int64)
            ort_inputs[name] = arr
    # 3) Inference
    outputs = session.run(None, ort_inputs)
    # outputs[0] might be shape (1, seq_len, dim) or (seq_len, dim)
    emb_array = outputs[0]
    # Drop batch axis if present
    if emb_array.ndim == 3:
        emb_array = emb_array[0]
    # Mean‑pool across tokens → (dim,)
    return emb_array.mean(axis=0)

# 3) Compute embeddings for all lines
print("Embedding all text lines (this may take a while)…")
embs = np.vstack(df["text"].apply(embed_text).values)

# 4) Prepare numeric features & scale
NUM_COLS = [
    "rel_size", "bold", "indent",
    "caps", "numbered", "line_len",
    "page", "y0_pos", "font_hash"
]
X_num = df[NUM_COLS].to_numpy()
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# 5) Assemble feature matrix & labels
X = np.hstack([embs, X_num_scaled])
y = df["label"]

# 6) Train logistic‑regression (multinomial)
print("Training logistic‑regression classifier…")
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000
)
clf.fit(X, y)

# 7) Persist classifier + scaler + metadata
OUT_PATH = "./models/classifierColl2.pkl"
with open(OUT_PATH, "wb") as fout:
    pickle.dump({
        "scaler": scaler,
        "clf": clf,
        "num_cols": NUM_COLS,
        "model_dir": MODEL_DIR
    }, fout)

print(f"✅ Trained model saved to {OUT_PATH}")
