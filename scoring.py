"""
scoring.py

Robust automatic scoring using:
- Readability
- Lexical diversity
- Grammar quality
"""

import textstat
import language_tool_python


# Init grammar checker (lazy loading conseillé en prod)
tool = language_tool_python.LanguageTool('en-US')


def compute_score_v2(text: str, tokens: list) -> float:
    """
    Compute a quality score (0 → 10)
    """

    # -------------------------
    # 1. Readability (Flesch)
    # -------------------------
    try:
        readability = textstat.flesch_reading_ease(text)
        readability_score = max(0, min(readability / 100, 1))
    except:
        readability_score = 0.5

    # -------------------------
    # 2. Lexical Diversity (TTR)
    # -------------------------
    if len(tokens) > 0:
        ttr = len(set(tokens)) / len(tokens)
    else:
        ttr = 0

    # -------------------------
    # 3. Grammar Errors
    # -------------------------
    try:
        matches = tool.check(text)
        error_rate = len(matches) / max(len(tokens), 1)
        grammar_score = max(0, 1 - error_rate)
    except:
        grammar_score = 0.5

    # -------------------------
    # 4. Length penalty
    # -------------------------
    length = len(tokens)
    if length < 5:
        length_penalty = 0.3
    elif length < 10:
        length_penalty = 0.6
    else:
        length_penalty = 1.0

    # -------------------------
    # FINAL SCORE
    # -------------------------
    score = (
        readability_score * 3 +
        ttr * 3 +
        grammar_score * 4
    ) * length_penalty

    return round(max(0, min(score, 10)), 2)