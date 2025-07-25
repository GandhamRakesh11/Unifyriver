from models.summarizer_models import (
    HF_MODELS, OR_MODELS, HF_CHUNK_LIMITS, OR_CHUNK_LIMITS,
    call_hf_model, call_or_model, call_openrouter, OR_KEY
)
from utilities.summarizer_utils import clean_text, compute_summary_metrics, chunk_text
from prompts.summarizer_prompt import REFLECTION_PROMPT, HALLUCINATION_PROMPT


class Summarizer:
    def __init__(self, use_reflection=False, use_hallucination=False):
        self.use_reflection = use_reflection
        self.use_hallucination = use_hallucination

    def summarize(self, text):
        cleaned = clean_text(text)
        summaries = {}

        for key, model_id in HF_MODELS.items():
            summaries[key] = self._summarize_in_chunks(
                model_id, cleaned, call_hf_model, HF_CHUNK_LIMITS[model_id]
            )

        for key, model_id in OR_MODELS.items():
            summaries[key] = self._summarize_in_chunks(
                model_id, cleaned, call_or_model, OR_CHUNK_LIMITS[model_id]
            )

        fallback_summary = next(
            (s["text"] for s in summaries.values()
             if isinstance(s, dict) and isinstance(s.get("text"), str)
             and not s["text"].startswith("[HF Error") and not s["text"].startswith("[OpenRouter Error")),
            ""
        )

        reflection = None
        hallucination = None

        if self.use_reflection and fallback_summary:
            prompt = REFLECTION_PROMPT.format(original=cleaned[:1500], summary=fallback_summary)
            reflection = call_openrouter(prompt, "mistralai/mistral-7b-instruct", OR_KEY)

        if self.use_hallucination and fallback_summary:
            prompt = HALLUCINATION_PROMPT.format(original=cleaned[:1500], summary=fallback_summary)
            hallucination = call_openrouter(prompt, "meta-llama/llama-3-8b-instruct", OR_KEY)

        return {
            "summaries": {k: v["text"] if isinstance(v, dict) else v for k, v in summaries.items()},
            "reflection": reflection,
            "hallucination": hallucination
        }

    def _summarize_in_chunks(self, model_id, text, model_fn, chunk_size_words):
        chunks = chunk_text(text, chunk_size_words)
        results = []

        for chunk in chunks:
            summary = model_fn(model_id, chunk)
            metrics = compute_summary_metrics(summary)
            results.append({"chunk": summary, "metrics": metrics})

        full_summary = "\n\n".join([r["chunk"] for r in results])
        return {"text": full_summary, "chunks": results}
