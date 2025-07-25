import re
from collections import Counter

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()

def chunk_text(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def compute_summary_metrics(text):
    words = text.split()
    word_count = len(words)
    lower = text.lower()

    density_terms = [
        "key", "important", "notable", "significant", "highlight", "trend",
        "increase", "decrease", "finding", "objective", "method", "result",
        "conclusion", "impact", "outcome"
    ]

    density_score = sum(lower.count(term) for term in density_terms)
    redundancy_score = sum(c for _, c in Counter(words).items() if c > 1) / max(word_count, 1)
    noun_like = [w for w in words if w.istitle()]
    specificity_score = len(set(noun_like)) / max(word_count, 1)
    insight_score = sum(lower.count(k) for k in ["trend", "result", "finding", "impact", "conclusion"]) / max(word_count, 1)
    hallucination_risk = word_count > 120 and "may not be accurate" in lower

    return {
        "length": word_count,
        "density_score": density_score,
        "redundancy_score": redundancy_score,
        "specificity_score": specificity_score,
        "insight_score": insight_score,
        "hallucination_risk": hallucination_risk
    }
