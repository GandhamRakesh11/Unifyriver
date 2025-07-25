from models.translator_models import (
    TranslationManager
)

def translate_text(text: str, target_lang: str) -> str:
    """External function to perform full translation pipeline."""
    manager = TranslationManager()
    return manager.translate_from_text(text, target_lang)
