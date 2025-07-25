REFLECTION_PROMPT = """
Original:
{original}
Summary:
{summary}
Respond with evaluation and score (1–10).
"""

HALLUCINATION_PROMPT = """
Original:
{original}
Summary:
{summary}
List hallucinated claims. Score: 0–10 (0 = fully hallucinated).
"""
