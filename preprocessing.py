"""
preprocessing.py

Handles text preprocessing using spaCy.
Optimized with nlp.pipe() for scalability.
"""

import spacy
from typing import List, Tuple

# Load spaCy WITHOUT heavy components
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ✅ AJOUT CRUCIAL
if "parser" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")




def preprocess_texts(texts: List[str]) -> List[Tuple[List[str], spacy.tokens.Doc]]:
    """
    Preprocess a list of texts using spaCy pipeline.
    """

    results = []

    for doc in nlp.pipe(texts, batch_size=32, n_process=2):

        # ✅ skip empty docs (safety)
        if len(doc) == 0:
            continue

        cleaned_tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]

        # ✅ skip if no tokens
        if len(cleaned_tokens) == 0:
            continue

        results.append((cleaned_tokens, doc))

    return results