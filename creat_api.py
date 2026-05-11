"""
creat_api.py
FastAPI app - Score + Feedback Gemini
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from feedback import generate_feedback
from features import extract_features
from preprocessing import preprocess
import spacy
import joblib

app = FastAPI()

# Chargement des modèles au démarrage
print("📦 Chargement SBERT...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ SBERT chargé.")

print("📦 Chargement du modèle...")
model = joblib.load("model_sbert3.pkl")
print("✅ Modèle chargé.")

print("📦 Chargement spaCy...")
nlp = spacy.load("en_core_web_sm")
print("✅ spaCy chargé.")


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI API is running"}


@app.post("/predict")
def predict(input: TextInput):
    try:
        # 1. Générer l'embedding + Score
        embedding = sbert.encode(input.text)
        prediction = model.predict([embedding])
        score = float(prediction[0])

        # 2. Preprocessing + Features
        tokens, doc = preprocess(input.text, nlp)
        features = extract_features(tokens, doc)

        # 3. Feedback Gemini
        feedback = generate_feedback(features, score, input.text)

        return {
            "text": input.text,
            "score": round(score, 2),
            "feedback": feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))