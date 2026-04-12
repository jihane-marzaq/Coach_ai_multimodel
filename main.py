"""
main.py

Pipeline orchestration:
preprocessing → features → scoring → feedback
"""

from preprocessing import preprocess_texts
from features import extract_features
from scoring import compute_score
from feedback import generate_feedback


def run_pipeline(text: str):
    """
    Run full NLP evaluation pipeline with AI feedback.

    Args:
        text (str): input text
    """

    # --- Preprocessing ---
    tokens, doc = preprocess_texts([text])[0]

    # --- Feature extraction ---
    features = extract_features(tokens, doc)

    # --- Scoring ---
    score = compute_score(features)

    # --- Feedback (Gemini) ---
    feedback = generate_feedback(features, score, text)

    # --- Output ---
    print("\n=== FEATURES ===")
    for k, v in features.items():
        print(f"{k}: {v}")

    print("\n=== FINAL SCORE ===")
    print(score)

    print("\n=== AI FEEDBACK ===")
    print(feedback)


if __name__ == "__main__":
    sample_text = """
    Today I want to present my idea. It is good. It is very useful.
    However, it still needs improvements because some parts are not clear.
    """

    run_pipeline(sample_text)