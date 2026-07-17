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


def retrieve_context(query):

    index = faiss.read_index(INDEX_FILE)

    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)

    context = ""

    for rank, idx in enumerate(indices[0], start=1):

        if idx == -1:
            continue

        chunk = chunks[idx]

        context += (
            f"\n[Document: {chunk['source_document']}]\n"
            f"{chunk['text']}\n"
        )

    return context


def main():

    while True:

        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        context = retrieve_context(query)

        print("\n==============================")
        print("CONTEXT FOR LLM")
        print("==============================")
        print(context)


if __name__ == "__main__":
    main()