"""
preprocessing.py

Handles text preprocessing using spaCy.
Optimized with nlp.pipe() for scalability.
"""

import spacy
from typing import List, Tuple

# Load spaCy model once (global for performance)
nlp = spacy.load("en_core_web_sm")


def preprocess_texts(texts: List[str]) -> List[Tuple[List[str], spacy.tokens.Doc]]:
    """
    Preprocess a list of texts using spaCy pipeline.

    Args:
        texts (List[str]): List of raw text inputs

    Returns:
        List of tuples:
            - cleaned_tokens (List[str])
            - original spaCy Doc object
    """
    results = []

    # Use nlp.pipe for efficient batch processing
    for doc in nlp.pipe(texts, batch_size=32):
        cleaned_tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]

        results.append((cleaned_tokens, doc))

    return results