import re
import torch

def clean_text(text):
    """Remove extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=512):
    """Split long text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_mean_embedding(model, chunks):
    """Compute mean embedding from text chunks."""
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings.mean(dim=0)
