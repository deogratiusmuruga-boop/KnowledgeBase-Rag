import os
import pickle

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAISS_INDEX = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base.faiss"
)

EMBEDDING_DATA = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_embeddings.pkl"
)

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_RERANK = 20
FINAL_RESULTS = 5

# ==========================================================
# Load Models
# ==========================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading CrossEncoder reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX)

print("Loading knowledge base...")

with open(EMBEDDING_DATA, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]

tokenized_corpus = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)

print("=" * 70)
print("Hybrid Retrieval Ready")
print("=" * 70)

# ==========================================================
# Retrieval
# ==========================================================

def retrieve(query):

    # --------------------------------------------------
    # Dense Retrieval
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    dense_scores, dense_indices = index.search(
        query_embedding,
        TOP_K_VECTOR
    )

    results = {}

    for score, idx in zip(
        dense_scores[0],
        dense_indices[0]
    ):

        if idx == -1:
            continue

        results[idx] = {
            "dense_score": float(score),
            "bm25_score": 0.0
        }

    # --------------------------------------------------
    # BM25 Retrieval
    # --------------------------------------------------

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25 = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]

    for idx in top_bm25:

        score = float(bm25_scores[idx])

        if idx in results:

            results[idx]["bm25_score"] = score

        else:

            results[idx] = {
                "dense_score": 0.0,
                "bm25_score": score
            }

    # --------------------------------------------------
    # Normalize Scores
    # --------------------------------------------------

    max_dense = max(
        [r["dense_score"] for r in results.values()],
        default=1
    )

    max_bm25 = max(
        [r["bm25_score"] for r in results.values()],
        default=1
    )

    hybrid_results = []

    for idx, scores in results.items():

        dense = (
            scores["dense_score"] / max_dense
            if max_dense else 0
        )

        sparse = (
            scores["bm25_score"] / max_bm25
            if max_bm25 else 0
        )

        hybrid = (0.6 * dense) + (0.4 * sparse)

        hybrid_results.append({
            "index": idx,
            "hybrid_score": hybrid,
            "dense_score": dense,
            "bm25_score": sparse,
            "chunk": chunks[idx]
        })

    hybrid_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    # --------------------------------------------------
    # CrossEncoder Reranking
    # --------------------------------------------------

    candidates = hybrid_results[:TOP_K_RERANK]

    sentence_pairs = [
        (query, item["chunk"]["text"])
        for item in candidates
    ]

    rerank_scores = reranker.predict(sentence_pairs)

    for item, score in zip(candidates, rerank_scores):
        item["rerank_score"] = float(score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print("FINAL RERANKED RESULTS")
    print("=" * 70)

    for rank, item in enumerate(
        candidates[:FINAL_RESULTS],
        start=1
    ):

        chunk = item["chunk"]

        print(f"\nRank             : {rank}")
        print(f"Cross Score      : {item['rerank_score']:.4f}")
        print(f"Hybrid Score     : {item['hybrid_score']:.4f}")
        print(f"Dense Score      : {item['dense_score']:.4f}")
        print(f"BM25 Score       : {item['bm25_score']:.4f}")

        print(f"Document ID      : {chunk['document_id']}")
        print(f"Title            : {chunk['title']}")
        print(f"Category         : {chunk['category']}")
        print(f"Organization     : {chunk['organization']}")
        print(f"Source           : {chunk['source_document']}")
        print(f"Chunk ID         : {chunk['chunk_id']}")

        print("-" * 70)

        preview = chunk["text"][:700]

        print(preview)

        if len(chunk["text"]) > 700:
            print("...")

        print("-" * 70)

# ==========================================================
# Main
# ==========================================================

def main():

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("Question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        retrieve(query)

if __name__ == "__main__":
    main()