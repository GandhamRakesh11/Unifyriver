REFINEMENT_PROMPT = """
You are a professional {tgt_lang} translator. ONLY return the improved translation.
DO NOT add explanations or comments. ONLY refine the translation.
Improve this translation while:
1. Preserving original meaning and numbers
2. Maintaining technical terms and names
3. Ensuring contextual accuracy
Source text:
{source}
Current translation:
{translation}
Improved translation in {tgt_lang}:
"""
