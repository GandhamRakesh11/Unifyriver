import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

from utilities.document_classifier_utils import clean_text, chunk_text, get_mean_embedding
from models.document_classifier_models import (
    ST_MODEL_NAMES,
    HF_MODEL_NAMES,
    load_sentence_transformer_models,
    load_huggingface_models
)

class DocumentClassifier:
    def __init__(self):
        self.labels = [
            "invoice", "contract", "resume", "medical report",
            "academic paper", "bank statement", "insurance policy",
            "purchase order", "letter", "manual", "presentation", "report"
        ]
        self.st_models = load_sentence_transformer_models()
        self.hf_tokenizers, self.hf_models = load_huggingface_models()

    @torch.no_grad()
    def classify(self, text):
        results = {}
        predictions = []
        chunks = chunk_text(text)

        for name, model in self.st_models.items():
            doc_embedding = get_mean_embedding(model, chunks)
            label_embeddings = model.encode(self.labels, convert_to_tensor=True)

            sims = cosine_similarity(
                doc_embedding.cpu().numpy().reshape(1, -1),
                label_embeddings.cpu().numpy()
            ).flatten()

            probs = softmax(sims).tolist()
            top1_idx = int(np.argmax(probs))
            sorted_probs = sorted(probs, reverse=True)
            predicted = self.labels[top1_idx]
            predictions.append(predicted)

            results[name] = {
                "type": "sentence-transformer",
                "predicted_label": predicted,
                "probabilities": dict(zip(self.labels, probs)),
                "tlm_score": float(sims[top1_idx]),
                "hallucination_score": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            }

        for name in self.hf_models:
            tokenizer = self.hf_tokenizers[name]
            model = self.hf_models[name]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            doc_embedding = model(**inputs).last_hidden_state[:, 0, :].detach().numpy()

            label_embeddings = []
            for label in self.labels:
                label_inputs = tokenizer(label, return_tensors="pt", truncation=True, padding=True, max_length=16)
                label_embedding = model(**label_inputs).last_hidden_state[:, 0, :].detach().numpy()
                label_embeddings.append(label_embedding)

            similarities = [cosine_similarity(doc_embedding, le).item() for le in label_embeddings]
            probs = softmax(similarities).tolist()
            top1_idx = int(np.argmax(probs))
            sorted_probs = sorted(probs, reverse=True)
            predicted = self.labels[top1_idx]
            predictions.append(predicted)

            results[name] = {
                "type": "transformer",
                "predicted_label": predicted,
                "probabilities": dict(zip(self.labels, probs)),
                "tlm_score": float(similarities[top1_idx]),
                "hallucination_score": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            }

        consistency_score = len(set(predictions)) / len(predictions) if predictions else 0.0
        for model in results:
            results[model]["consistency_score"] = 1.0 - consistency_score

        print("\n==== DOCUMENT CLASSIFICATION RESULTS ====")
        for model, info in results.items():
            print(f"\nModel: {model}")
            print(f"  ➤ Predicted: {info['predicted_label']}")
            print(f"  ➤ TLM Score: {info['tlm_score']:.3f}")
            print(f"  ➤ Hallucination Score: {info['hallucination_score']:.3f}")
            print(f"  ➤ Consistency Score: {info['consistency_score']:.3f}")

        return results
