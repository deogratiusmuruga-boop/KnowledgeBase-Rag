import os
import json
import pickle

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHUNK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "chunks",
    "knowledge_base_chunks.json"
)

FAISS_INDEX = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base.faiss"
)


# ============================================================
# Models
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

RERANK_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# Retrieval Parameters
# ============================================================

TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_RERANK = 20
FINAL_RESULTS = 5


# ============================================================
# Load Resources
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Loading CrossEncoder...")
reranker = CrossEncoder(
    RERANK_MODEL_NAME
)

print("Loading FAISS index...")
faiss_index = faiss.read_index(
    FAISS_INDEX
)

print("Loading chunks...")

with open(
    CHUNK_FILE,
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)


print("Building BM25 index...")

tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)


print("Hybrid Retriever Ready")


# ============================================================
# FAISS Semantic Search
# ============================================================

def semantic_search(query):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)


    scores, indices = faiss_index.search(
        query_embedding,
        TOP_K_VECTOR
    )


    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue


        chunk = chunks[idx].copy()

        chunk["dense_score"] = float(score)

        results.append(chunk)


    return results



# ============================================================
# BM25 Search
# ============================================================

def bm25_search(query):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(
        query_tokens
    )


    top_indices = np.argsort(
        scores
    )[::-1][:TOP_K_BM25]


    results = []


    for idx in top_indices:

        chunk = chunks[idx].copy()

        chunk["bm25_score"] = float(
            scores[idx]
        )

        results.append(chunk)


    return results



# ============================================================
# Hybrid Retrieval + Reranking
# ============================================================

def hybrid_search(query):

    dense_results = semantic_search(
        query
    )

    sparse_results = bm25_search(
        query
    )


    combined = {}


    # Dense
    for item in dense_results:

        idx = item["chunk_id"]

        combined[idx] = {
            "chunk": item,
            "dense_score": item.get(
                "dense_score",
                0
            ),
            "bm25_score": 0
        }



    # BM25
    for item in sparse_results:

        idx = item["chunk_id"]


        if idx in combined:

            combined[idx]["bm25_score"] = (
                item.get(
                    "bm25_score",
                    0
                )
            )

        else:

            combined[idx] = {
                "chunk": item,
                "dense_score": 0,
                "bm25_score": item.get(
                    "bm25_score",
                    0
                )
            }



    results = list(
        combined.values()
    )


    # Normalize

    max_dense = max(
        [
            x["dense_score"]
            for x in results
        ],
        default=1
    )


    max_bm25 = max(
        [
            x["bm25_score"]
            for x in results
        ],
        default=1
    )


    for item in results:

        dense = (
            item["dense_score"]
            /
            max_dense
            if max_dense
            else 0
        )


        sparse = (
            item["bm25_score"]
            /
            max_bm25
            if max_bm25
            else 0
        )


        item["hybrid_score"] = (
            0.6 * dense
            +
            0.4 * sparse
        )


    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    # CrossEncoder

    candidates = results[
        :TOP_K_RERANK
    ]


    pairs = [
        (
            query,
            item["chunk"]["text"]
        )
        for item in candidates
    ]


    scores = reranker.predict(
        pairs
    )


    for item, score in zip(
        candidates,
        scores
    ):

        item["rerank_score"] = float(
            score
        )


    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    return [
        item["chunk"]
        for item in candidates[
            :FINAL_RESULTS
        ]
    ]