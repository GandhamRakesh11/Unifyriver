from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Define model names (exposed for central management)
ST_MODEL_NAMES = [
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1"
]

HF_MODEL_NAMES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "prajjwal1/bert-tiny"
]

def load_sentence_transformer_models():
    """Load all sentence-transformer models."""
    return {name: SentenceTransformer(name) for name in ST_MODEL_NAMES}

def load_huggingface_models():
    """Load all HuggingFace transformers and their tokenizers."""
    tokenizers = {}
    models = {}
    for name in HF_MODEL_NAMES:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval()
        tokenizers[name] = tokenizer
        models[name] = model
    return tokenizers, models
