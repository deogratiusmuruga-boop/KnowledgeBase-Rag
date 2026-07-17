import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base.faiss"
)

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_embeddings.pkl"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():
    if not os.path.isfile(INDEX_FILE):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
    if not os.path.isfile(EMBEDDING_FILE):
        raise FileNotFoundError(f"Embedding file not found: {EMBEDDING_FILE}")

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    print("Loading chunks...")

    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise ValueError("Embedding file must contain a 'chunks' list.")

    chunks = data["chunks"]

    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and embedding file contain different numbers of vectors. "
            "Rebuild the index with build_faiss_index.py."
        )
    if index.ntotal == 0:
        raise ValueError("The FAISS index is empty.")

    saved_model_name = data.get("model_name")
    if saved_model_name and saved_model_name != MODEL_NAME:
        raise ValueError(
            f"Embeddings were generated with {saved_model_name!r}, but retrieval is "
            f"configured for {MODEL_NAME!r}. Regenerate embeddings or use the same model."
        )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    while True:

        try:
            query = input("\nAsk a question (or type 'exit'): ").strip()
        except EOFError:
            break

        if query.lower() == "exit":
            break

        if not query:
            continue

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        if query_embedding.shape[1] != index.d:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[1]}) does not match "
                f"the FAISS index dimension ({index.d})."
            )

        distances, indices = index.search(query_embedding, min(5, index.ntotal))

        print("\n==============================")
        print("Top Retrieved Chunks")
        print("==============================")
        print(f"Query: {query}")

        for rank, idx in enumerate(indices[0], start=1):

            if idx == -1:
                continue

            chunk = chunks[idx]

            print(f"\nResult {rank}")
            print(f"Source Document : {chunk['source_document']}")
            print(f"Chunk ID        : {chunk['chunk_id']}")
            print(f"Similarity      : {distances[0][rank-1]:.4f}")
            print("-" * 60)
            preview = chunk["text"][:900]

            if len(chunk["text"]) > 900:
                preview += "..."

            print(preview)
            print("-" * 60)


if __name__ == "__main__":
    main()
