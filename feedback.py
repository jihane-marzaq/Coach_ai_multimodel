"""
feedback.py
Generates personalized coaching feedback using Google Gemini API.
Designed with a provider abstraction for easy switching (Gemini ↔ OpenAI).
"""
import os
from google import genai


def _build_prompt(features: dict, score: float, user_text: str) -> str:
    """
    Build the prompt dynamically using extracted features and user text.
    """
    return f"""
You are an expert public speaking coach and AI assistant.
Your task is to generate personalized, actionable, and structured feedback for a user based on the analysis of their presentation.
## Context:
* avg_sentence_length: {features['avg_sentence_length']}
* vocab_richness: {features['vocab_richness']}
* connector_count: {features['connector_count']}
* complexity: {features['complexity']}
* repetition_rate: {features['repetition_rate']}
* num_sentences: {features['num_sentences']}
* total_words: {features['total_words']}
* final_score: {score}
Text:
"{user_text}"
## Instructions:
* Be specific (NO generic advice)
* Explain strengths and weaknesses
* Give actionable improvements
* Rewrite improved version of the text
Structure your response clearly.
"""


def _generate_with_gemini(prompt: str) -> str:
    """
    Internal function to call Gemini API (new google-genai syntax).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    # ✅ Nouvelle syntaxe google-genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()


def generate_feedback(features: dict, score: float, user_text: str) -> str:
    """
    Main function used by the pipeline.
    Args:
        features (dict): NLP extracted features
        score (float): final score (0-10)
        user_text (str): original input text
    Returns:
        str: structured AI feedback
    """
    prompt = _build_prompt(features, score, user_text)

    # Provider abstraction (future-proof)
    provider = "gemini"

    if provider == "gemini":
        return _generate_with_gemini(prompt)
    # Future extension (example)
    elif provider == "openai":
        raise NotImplementedError("OpenAI provider not implemented yet.")
    else:
        raise ValueError("Unsupported AI provider.")