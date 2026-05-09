"""
features.py

Extracts linguistic and structural features from processed text.
"""

from collections import Counter
from typing import List, Dict
import spacy

# Logical connectors list
CONNECTORS = {
    "however", "therefore", "moreover", "furthermore",
    "in conclusion", "first", "second", "finally",
    "thus", "consequently", "for example", "because"}


def extract_features(tokens: List[str], doc: spacy.tokens.Doc) -> Dict:
    """
    Extract features from tokens and spaCy doc.

    Args:
        tokens (List[str]): Cleaned tokens
        doc (spacy.tokens.Doc): Original spaCy doc

    Returns:
        Dict of features
    """

    # Sentences
    sentences = list(doc.sents)
    num_sentences = len(sentences)

    # Average sentence length
    avg_sentence_length = (
        sum(len(sent) for sent in sentences) / num_sentences
        if num_sentences > 0 else 0
    )

    # Vocabulary richness
    total_lemmas = len(tokens)
    unique_lemmas = len(set(tokens))
    vocab_richness = unique_lemmas / total_lemmas if total_lemmas > 0 else 0

    # Logical connectors
    connector_count = sum(1 for token in tokens if token in CONNECTORS)

    # Complexity (long words > 6 letters)
    long_words = [word for word in tokens if len(word) > 6]
    complexity = len(long_words) / total_lemmas if total_lemmas > 0 else 0

    # Fluency (repetition rate)
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    repetition_rate = repeated / total_lemmas if total_lemmas > 0 else 0

    # Part-of-Speech densities
    total_doc_tokens = len(doc)
    if total_doc_tokens > 0:
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN") / total_doc_tokens
        verb_count = sum(1 for token in doc if token.pos_ == "VERB") / total_doc_tokens
        adj_count = sum(1 for token in doc if token.pos_ == "ADJ") / total_doc_tokens
    else:
        noun_count = verb_count = adj_count = 0.0

    return {
        "avg_sentence_length": avg_sentence_length,
        "vocab_richness": vocab_richness,
        "connector_count": connector_count,
        "complexity": complexity,
        "repetition_rate": repetition_rate,
        "num_sentences": num_sentences,
        "total_words": total_lemmas,
        "noun_ratio": noun_count,
        "verb_ratio": verb_count,
        "adj_ratio": adj_count
    }