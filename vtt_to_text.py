import webvtt
import os

def convert_vtt_to_text(file_path):
    """
    Convert a .vtt subtitle file into clean text
    """
    captions = webvtt.read(file_path)

    text = []
    for caption in captions:
        line = caption.text.strip()
        if line:
            text.append(line)

    # fusionner toutes les lignes
    full_text = " ".join(text)

    return full_text

import re

def clean_text(text):
    # enlever (Music), (Applause), [Music], etc.
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # enlever "Text:" ou balises récurrentes
    text = text.replace("Text:", "")

    # enlever espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_all_vtt(folder_path):
    """
    Load all VTT files and convert to clean texts
    """
    texts = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".vtt"):
                path = os.path.join(root, file)
                if file.endswith(".vtt") and "en" in file:
                    try:
                        text = convert_vtt_to_text(path)
                        text = clean_text(text)
                        if len(text) > 200:
                            texts.append(text)
                    except Exception as e:
                        print(f"Erreur avec {file}: {e}")

    return texts




if __name__ == "__main__":
    folder = "tedDirector/subtitles"

    texts = load_all_vtt(folder)

    print("Nombre de textes:", len(texts))
    print("\nExemple:\n", texts[0][:500])
    
    
# pour chaque .vtt :

# ✔ enlève timestamps
# ✔ récupère uniquement le texte
# ✔ fusionne en un seul paragraphe