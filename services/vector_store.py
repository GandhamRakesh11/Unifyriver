# vector_store.py

import os
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "document_store")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# === Initialize Qdrant client ===
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional vector

# === Ensure collection exists in Qdrant ===
def init_collection():
    collections = client.get_collections().collections
    if COLLECTION_NAME not in [col.name for col in collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

# Call once on import to verify/initialize collection
init_collection()

# === Generate a consistent ID based on filename (MD5 hash → int) ===
def compute_id(filename):
    return int(hashlib.md5(filename.encode()).hexdigest()[:16], 16)

# === Retrieve entry by filename ===
def get_entry(filename):
    point_id = compute_id(filename)
    result = client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id])
    return result[0].payload if result else None

# === Add or update an entry (filename → vector + payload) ===
def upsert_entry(filename, **fields):
    init_collection()

    if "filename" in fields:
        fields.pop("filename")  # prevent multiple values for 'filename'

    point_id = compute_id(filename)
    existing = get_entry(filename) or {}

    # Merge old and new fields; prefer non-null new values
    payload = {**existing, **{k: v for k, v in fields.items() if v is not None}}

    # Extract text field and encode into a vector
    base_text = payload.get("text", "")
    if not isinstance(base_text, str):
        base_text = str(base_text)

    try:
        vector = model.encode(base_text, normalize_embeddings=True).tolist()
    except Exception as e:
        print(f"❌ Vector encoding failed for {filename}: {e}")
        vector = [0.0] * 384

    # Final payload with file reference
    payload = {"filename": filename, **payload}

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)]
    )

