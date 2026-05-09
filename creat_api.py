from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib

app = FastAPI()

# Charger le modèle SBERT pour générer les embeddings
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Charger ton modèle classifieur
model = joblib.load("model_sbert3.pkl")

# Schéma d'entrée — texte brut
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "AI API is running"}

@app.post("/predict")
def predict(input: TextInput):
    # 1. Générer l'embedding (384 dimensions)
    embedding = sbert.encode(input.text)
    
    # 2. Prédire avec ton modèle
    prediction = model.predict([embedding])
    
    return {
        "text": input.text,
        "prediction": prediction.tolist()
    }