import os
import re
import requests
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

from dotenv import load_dotenv

load_dotenv()

from utilities.translator_utils import (
    DocumentProcessor,
    LANGUAGE_MAP,
    HALLUCINATION_THRESHOLD
)
from prompts.translator_prompt import REFINEMENT_PROMPT

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")


class TranslationEngine:
    def __init__(self):
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def translate_chunk(self, chunk, src_lang, tgt_lang):
        chunk = chunk.strip()
        if not chunk:
            return "[Skipped empty chunk]"
        if len(chunk) > 2000:
            chunk = chunk[:2000]

        self.tokenizer.src_lang = src_lang
        try:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=False)
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            return f"[Translation Error] {str(e)}"


class TranslationManager:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.engine = TranslationEngine()
        self.semantic_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def translate_from_text(self, raw_text, target_lang):
        try:
            if isinstance(raw_text, bytes):
                raw_text = raw_text.decode("utf-8", errors="ignore")
            else:
                raw_text = raw_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"Failed to decode input: {e}")

        cleaned = self.processor.clean_text(raw_text)

        try:
            detected = detect(cleaned[:500])
        except Exception:
            detected = "english"

        src_code = LANGUAGE_MAP.get(detected.lower(), "en_XX")
        tgt_code = LANGUAGE_MAP.get(target_lang.lower())
        if not tgt_code:
            raise ValueError(f"Unsupported target language: {target_lang}")

        chunks = self.processor.chunk_text(cleaned)
        translations = []

        for i, chunk in enumerate(chunks):
            print(f"\n[Translating chunk {i+1}/{len(chunks)}] Length: {len(chunk)}")
            translated = self.engine.translate_chunk(chunk, src_code, tgt_code)
            hallucinated, score = self.detect_hallucination(chunk, translated)
            print(f"[Similarity Score: {score:.2f}]")

            if hallucinated:
                print(f"[Refining chunk {i+1}] Score below threshold {HALLUCINATION_THRESHOLD}")
                translated = self.refine_translation(chunk, translated, target_lang)

            translations.append(f"{translated}\n\n[Similarity Score: {score:.2f}]")

        return "\n\n".join(translations)

    def refine_translation(self, source, translation, tgt_lang):
        prompt = REFINEMENT_PROMPT.format(source=source, translation=translation, tgt_lang=tgt_lang)
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=60
            )
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
            return self._clean_llm_output(raw_output)
        except Exception as e:
            print(f"[Refinement Error] {e}")
            return translation

    def _clean_llm_output(self, text):
        text = re.sub(r"^(improved|refined) translation:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[\"']|[\"']$", "", text)
        return text.strip()

    def compute_similarity(self, source, translation):
        try:
            src_embed = self.semantic_model.encode(source, convert_to_tensor=True)
            trans_embed = self.semantic_model.encode(translation, convert_to_tensor=True)
            return util.pytorch_cos_sim(src_embed, trans_embed).item()
        except Exception as e:
            print(f"[Similarity Error] {e}")
            return 0.0

    def detect_hallucination(self, source, translation):
        semantic_score = self.compute_similarity(source, translation)
        src_keywords = set(re.findall(r'\b\w{4,}\b', source.lower()))
        trans_keywords = set(re.findall(r'\b\w{4,}\b', translation.lower()))
        coverage = len(src_keywords & trans_keywords) / max(1, len(src_keywords))
        if semantic_score < HALLUCINATION_THRESHOLD or coverage < 0.5:
            return True, semantic_score
        return False, semantic_score
