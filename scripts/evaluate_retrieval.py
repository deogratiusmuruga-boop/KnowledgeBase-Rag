import json
import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "evaluation_queries.json",
)
INDEX_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base.faiss",
)
EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_embeddings.pkl",
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_evaluation_queries():
    """Load and validate the standard retrieval evaluation queries."""
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list) or not queries:
        raise ValueError("Evaluation queries must be a non-empty JSON list.")

    for position, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {position} must be an object.")
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"Evaluation item {position} has an invalid query.")
        if not isinstance(item.get("expected_source"), str) or not item["expected_source"].strip():
            raise ValueError(f"Evaluation item {position} has an invalid expected_source.")

    return queries


def load_retrieval_data():
    """Load the FAISS index and the chunk metadata in matching order."""
    if not os.path.isfile(INDEX_FILE):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
    if not os.path.isfile(EMBEDDING_FILE):
        raise FileNotFoundError(f"Embedding file not found: {EMBEDDING_FILE}")

    index = faiss.read_index(INDEX_FILE)

    with open(EMBEDDING_FILE, "rb") as file:
        data = pickle.load(file)

    chunks = data.get("chunks") if isinstance(data, dict) else None
    if not isinstance(chunks, list):
        raise ValueError("Embedding file must contain a 'chunks' list.")
    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and embedding file contain different numbers of vectors. "
            "Run build_faiss_index.py."
        )
    if index.ntotal == 0:
        raise ValueError("The FAISS index is empty.")

    saved_model_name = data.get("model_name")
    if saved_model_name and saved_model_name != MODEL_NAME:
        raise ValueError(
            f"Embeddings use {saved_model_name!r}, but evaluation uses {MODEL_NAME!r}."
        )

    return index, chunks


def retrieve_top_source(query, index, chunks, model):
    """Return the source document for the highest-scoring retrieval result."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension ({query_embedding.shape[1]}) does not match "
            f"the FAISS index dimension ({index.d})."
        )

    _, indices = index.search(query_embedding, 1)
    top_index = int(indices[0][0])

    if top_index == -1:
        return None

    source_document = chunks[top_index].get("source_document")
    if not isinstance(source_document, str):
        raise ValueError(f"Retrieved chunk {top_index} has no valid source_document.")

    return source_document


def main():
    evaluation_queries = load_evaluation_queries()
    index, chunks = load_retrieval_data()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    passed = 0

    print("\n" + "=" * 40)
    print("Retrieval Evaluation")
    print("=" * 40)

    for item in evaluation_queries:
        query = item["query"]
        expected_source = item["expected_source"]
        retrieved_source = retrieve_top_source(query, index, chunks, model)
        is_match = retrieved_source == expected_source
        passed += is_match

        print(f"\nQuery: {query}")
        print(f"Expected : {expected_source}")
        print(f"Retrieved: {retrieved_source or 'No result'}")
        print(f"Result   : {'PASS' if is_match else 'FAIL'}")

    total = len(evaluation_queries)
    accuracy = passed / total * 100

    print("\n" + "=" * 40)
    print(f"Accuracy: {passed}/{total} ({accuracy:.0f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
