"""
feedback.py
Generates personalized coaching feedback using Groq API.
"""
import os
from groq import Groq


def _build_prompt(features: dict, score: float, user_text: str) -> str:
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


def _generate_with_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment variables.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generate_feedback(features: dict, score: float, user_text: str) -> str:
    prompt = _build_prompt(features, score, user_text)
    return _generate_with_groq(prompt)