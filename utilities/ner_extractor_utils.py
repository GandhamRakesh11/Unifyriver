from collections import Counter

def compute_caption_metrics(text):
    length_score = len(text.split())
    lower = text.lower()

    density_terms = [
        "axis", "legend", "trend", "outlier", "distribution", "variance", "correlat", "increase", "decrease",
        "sales", "growth", "profit", "income", "market", "expense", "cost", "price", "chart", "revenue"
    ]
    density_score = sum(lower.count(term) for term in density_terms)

    words = text.split()
    word_counts = Counter(words)
    redundancy_score = sum(c for w, c in word_counts.items() if c > 1) / max(len(words), 1)

    noun_like = [w for w in words if w[0].isupper()]
    specificity_score = len(set(noun_like)) / max(len(words), 1)

    insight_keywords = ["trend", "increase", "decrease", "anomaly", "correlat", "cluster", "outlier", "distribution"]
    insight_score = sum(lower.count(k) for k in insight_keywords) / max(len(words), 1)

    hallucination_risk = length_score > 100 and "no graph" in lower

    return {
        "length_score": length_score,
        "density_score": density_score,
        "redundancy_score": redundancy_score,
        "specificity_score": specificity_score,
        "insight_score": insight_score,
        "hallucination_risk": hallucination_risk
    }

def chunk_text(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]

def load_prompt(filename):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        return f.read()
