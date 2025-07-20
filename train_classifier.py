"""
Embed texts with MiniLM (PyTorch backend) + numeric layout features,
train logistic-regression, save clf.pkl
"""
from pathlib import Path
import joblib, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

df = pd.read_csv("train.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # auto-download
emb = embedder.encode(df.text.tolist(), batch_size=128, show_progress_bar=True)

nums = df[["rel_size","bold","indent","caps","numbered"]].astype("float32").values
X = np.hstack([emb, nums])

lab = LabelEncoder()
y = lab.fit_transform(df.label)

clf = LogisticRegression(
        max_iter=400,
        multi_class="multinomial",
        class_weight="balanced")      # <— evens out class frequencies
clf.fit(X, y)

model_dir = (Path(__file__).resolve().parent / "models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump({"clf": clf, "labels": lab.classes_}, model_dir / "clf.pkl")
print("✅  wrote", model_dir / "clf.pkl")
