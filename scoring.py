"""
scoring.py

Computes interpretable presentation quality score using heuristic rules.
"""

from typing import Dict


def compute_score(features: Dict) -> float:
    """
    Compute final score based on extracted features.

    Args:
        features (Dict): Feature dictionary

    Returns:
        float: score between 0 and 10
    """

    score = 0.0

    # --- Clarity (sentence length) ---
    if 8 <= features["avg_sentence_length"] <= 20:
        score += 2
    elif 5 <= features["avg_sentence_length"] < 8:
        score += 1
    else:
        score += 0.5

    # --- Vocabulary richness ---
    if features["vocab_richness"] > 0.6:
        score += 2
    elif features["vocab_richness"] > 0.4:
        score += 1.5
    else:
        score += 1

    # --- Logical connectors ---
    if features["connector_count"] >= 3:
        score += 2
    elif features["connector_count"] >= 1:
        score += 1

    # --- Complexity ---
    if 0.2 <= features["complexity"] <= 0.4:
        score += 2
    else:
        score += 1

    # --- Fluency (repetition penalty) ---
    if features["repetition_rate"] < 0.1:
        score += 2
    elif features["repetition_rate"] < 0.2:
        score += 1
    else:
        score += 0.5

    # Normalize to 10
    return round(min(score, 10), 2)