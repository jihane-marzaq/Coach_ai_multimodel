"""
training_sbert.py

Train SBERT + regression model
"""

import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv("dataset_v2.csv")

texts = df["text"].tolist()
scores = df["score"].tolist()

# -----------------------------
print("🧠 Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
print("🔄 Encoding texts...")
X = sbert_model.encode(texts, show_progress_bar=True)
y = scores

# -----------------------------
print("✂️ Train / Test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
print("🚀 Training regressor...")
regressor = Ridge()
regressor.fit(X_train, y_train)

# -----------------------------
print("📊 Evaluation...")
y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {round(mae, 3)}")
print(f"R2:  {round(r2, 3)}")

# -----------------------------
print("💾 Saving model...")
joblib.dump(regressor, "model_sbert3.pkl")

print("✅ Training complete!")