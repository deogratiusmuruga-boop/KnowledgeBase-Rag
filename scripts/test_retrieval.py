import os
import pickle

import faiss
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

TOP_K = 5


def retrieve(query):

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

    # Load chunk metadata
    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Encode query
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search
    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print(f"TOP {TOP_K} RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, (score, idx) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        chunk = chunks[idx]

        print(f"\nRank: {rank}")
        print(f"Document   : {chunk['source_document']}")
        print(f"Chunk ID   : {chunk['chunk_id']}")
        print(f"Similarity : {score:.4f}")

        print("-" * 70)

        preview = chunk["text"][:600]

        print(preview)

        if len(chunk["text"]) > 600:
            print("...")

        print("-" * 70)


def main():

    print("=" * 70)
    print("KNOWLEDGE BASE RETRIEVAL TEST")
    print("Type 'exit' to quit.")
    print("=" * 70)

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        retrieve(query)


if __name__ == "__main__":
    main()