import os
import json
import pickle

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "chunks",
    "knowledge_base_chunks.json"
)

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_embeddings.pkl"
)

FAISS_INDEX = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base.faiss"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

TOP_K = 5

# ============================================================
# Load resources once
# ============================================================

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX)

print("Loading knowledge base...")

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading embeddings...")

with open(EMBEDDING_FILE, "rb") as f:
    embedding_data = pickle.load(f)

print("Building BM25 index...")

tokenized_corpus = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)

print("Hybrid Retriever Ready.")

# ============================================================
# Semantic Retrieval (FAISS)
# ============================================================

def semantic_search(query, top_k=TOP_K):
    """
    Retrieve the most semantically similar chunks using FAISS.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx].copy()

        chunk["semantic_score"] = float(score)

        results.append(chunk)

    return results