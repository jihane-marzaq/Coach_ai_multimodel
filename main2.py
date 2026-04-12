# main 1:


"""
main.py

Pipeline orchestration:
preprocessing → feature extraction → scoring
"""

from preprocessing import preprocess_texts
from features import extract_features
from scoring import compute_score


def run_pipeline(text: str):
    """
    Run full NLP evaluation pipeline.

    Args:
        text (str): input text
    """

    # Preprocessing
    processed = preprocess_texts([text])[0]
    tokens, doc = processed

    # Feature extraction
    features = extract_features(tokens, doc)

    # Scoring
    score = compute_score(features)

    # Output
    print("\n=== FEATURES ===")
    for k, v in features.items():
        print(f"{k}: {v}")

    print("\n=== FINAL SCORE ===")
    print(score)


if __name__ == "__main__":
    sample_text = """
    In tis project , i created a nlp module where i evaluate user text , 
    fistt we preparing the data we clean it with nlp.pip() spacy method then we Implement intelligent filtering: 
    remove stopwords, remove punctuation ,keep only alphabetic tokens, use lemmatization
    then we extract some features wich we'll use to score the text , and we creat a rules scoring system in scoring.py ,
    and finally we call all this in the main.py file 
    """

    run_pipeline(sample_text)