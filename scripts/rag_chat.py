import os
import pickle

import faiss
import ollama
from sentence_transformers import SentenceTransformer

from evidence_aggregation import aggregate_evidence
from reliability_evaluation import evaluate_reliability
from adaptive_decision_controller import make_reliability_decision
from build_grounded_prompt import build_grounded_prompt

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

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

    # Load stored chunks
    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]

    # Encode query
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search
    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    # Aggregate retrieved evidence
    evidence_items = aggregate_evidence(
        chunks,
        distances,
        indices
    )

    # Evaluate retrieval quality
    reliability = evaluate_reliability(
        query,
        evidence_items
    )

    # Adaptive decision
    decision = make_reliability_decision(
        reliability["overall_reliability"]
    )

    return evidence_items, reliability, decision


def generate_answer(query, embedding_model):

    evidence_items, reliability, decision = retrieve_context(
        query,
        embedding_model
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL REPORT")
    print("=" * 60)
    print(f"Authority             : {reliability['authority']:.2f}")
    print(f"Relevance             : {reliability['relevance']:.2f}")
    print(f"Support               : {reliability['support']:.2f}")
    print(f"Coverage              : {reliability['coverage']:.2f}")
    print(f"Consistency           : {reliability['consistency']:.2f}")
    print(f"Overall Reliability   : {reliability['overall_reliability']:.2f}")
    print(f"Decision              : {decision['decision']}")
    print(f"Reason                : {decision['reason']}")
    print("=" * 60)

    # --------------------------------------------------
    # Additional Safety Check
    # --------------------------------------------------

    if reliability["support"] < 0.50:

        return (
            "I couldn't find reliable information about that topic "
            "in the knowledge base."
        )

    # --------------------------------------------------
    # Adaptive Decision Controller
    # --------------------------------------------------

    if decision["decision"] == "REJECT":

        return (
            "The retrieved information is not reliable enough.\n\n"
            "I couldn't find reliable information in the knowledge base."
        )

    elif decision["decision"] == "RE-RETRIEVE":

        return (
            "The retrieved evidence is not reliable enough.\n\n"
            "Please try rephrasing your question or ask something more specific."
        )

    elif decision["decision"] == "REFINE":

        print("\nModerate reliability detected.")
        print("Generating answer from retrieved evidence...\n")

    # ACCEPT continues automatically

    # --------------------------------------------------
    # Build Evidence-Grounded Prompt
    # --------------------------------------------------

    prompt = build_grounded_prompt(
        query=query,
        evidence_items=evidence_items,
        reliability=reliability,
        decision=decision
    )

    # --------------------------------------------------
    # Call LLM
    # --------------------------------------------------

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a trustworthy elderly-care assistant.\n"
                    "Answer ONLY from the retrieved evidence.\n"
                    "Never use outside knowledge.\n"
                    "Never invent medical facts.\n"
                    "If the evidence does not answer the question, reply exactly:\n"
                    "\"I couldn't find that information in the knowledge base.\""
                )
            },
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

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        print("\nGenerating answer...\n")

        answer = generate_answer(
            query,
            embedding_model
        )

        print("=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()