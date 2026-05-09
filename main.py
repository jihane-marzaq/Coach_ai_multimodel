"""
main.py
Pipeline : SBERT → score  |  Gemini → feedback
"""

import sys
import joblib
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
MODEL_PATH  = "model_sbert3.pkl"
SBERT_NAME  = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────
def run_pipeline(text: str):
    print("\n" + "=" * 50)
    print("📝 TEXTE ANALYSÉ :")
    print(text)
    print("=" * 50)

    from preprocessing import preprocess_texts
    from features import extract_features
    from feedback import generate_feedback

    # --- Chargement SBERT (depuis HuggingFace, pas joblib) ---
    print("\n📦 Chargement de SBERT...")
    sbert_model = SentenceTransformer(SBERT_NAME)
    print("✅ SBERT chargé.")

    # --- Chargement regressor ---
    print("📦 Chargement du regressor...")
    regressor = joblib.load(MODEL_PATH)
    print("✅ Regressor chargé.")

    # --- Preprocessing ---
    print("🔄 Preprocessing...")
    processed = preprocess_texts([text])
    if not processed:
        print("❌ Texte trop court ou vide.")
        sys.exit(1)
    tokens, doc = processed[0]

    # --- Features ---
    print("📐 Extraction des features...")
    features = extract_features(tokens, doc)

    # --- Score ---
    print("🔢 Calcul du score...")
    embedding = sbert_model.encode([text])
    score = round(float(regressor.predict(embedding)[0]), 2)

    # --- Feedback Gemini ---
    #print("🤖 Génération du feedback Gemini...")
    #feedback = generate_feedback(features, score, text)

    # --- Résultat ---
    print("\n" + "=" * 50)
    print(f"🏆 SCORE FINAL : {score} / 10")
    print("=" * 50)
    print("\n💬 FEEDBACK & CONSEILS :\n")
    #print(feedback)
    print("=" * 50)


# ─────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline("I present something and I repeat many time")