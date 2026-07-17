import os
import pickle

import faiss
import ollama
from sentence_transformers import SentenceTransformer

from evidence_aggregation import aggregate_evidence

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
LLM_MODEL = "llama3.2:latest"

TOP_K = 5

def retrieve_context(query, embedding_model):

    index = faiss.read_index(INDEX_FILE)

    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)
    evidence_items = aggregate_evidence(chunks, distances, indices)

    context = "".join(
        (
            f"\n[Source: {item['source_document']} | "
            f"Category: {item['document_category']} | "
            f"Authority: {item['authority_score']:.2f} | "
            f"Similarity: {item['similarity_score']:.4f}]\n"
            f"{item['text']}\n"
        )
        for item in evidence_items
    )

    return context

def generate_answer(query, embedding_model):

    context = retrieve_context(query, embedding_model)

    prompt = f"""
You are an AI assistant for elderly care.

Answer the user's question ONLY using the information provided in the context below.

If the answer is not contained in the context, reply:

"I couldn't find that information in the knowledge base."

Keep the answer:
- Clear
- Friendly
- Easy for older adults to understand

==========================
CONTEXT
==========================

{context}

==========================
QUESTION
==========================

{query}

==========================
ANSWER
==========================
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

def main():

    print("=" * 60)
    print("Elderly Care RAG Assistant")
    print("Type 'exit' to quit.")
    print("=" * 60)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        print("\nGenerating answer...\n")

        answer = generate_answer(query, embedding_model)

        print("=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()

