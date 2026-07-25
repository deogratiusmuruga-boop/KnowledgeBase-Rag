import pickle
import os

from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_embeddings.pkl"
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5


def rerank(query, candidate_chunks):

    print("\nLoading CrossEncoder model...")

    model = CrossEncoder(MODEL_NAME)

    sentence_pairs = [
        (query, chunk["text"])
        for chunk in candidate_chunks
    ]

    scores = model.predict(sentence_pairs)

    ranked = sorted(
        zip(scores, candidate_chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:TOP_K]


def main():

    print("=" * 70)
    print("CrossEncoder Reranker Test")
    print("=" * 70)

    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        # Temporary test:
        # Instead of Hybrid Retrieval, just use the first 20 chunks
        candidate_chunks = chunks[:20]

        results = rerank(query, candidate_chunks)

        print("\n" + "=" * 70)
        print("TOP RERANKED RESULTS")
        print("=" * 70)

        for rank, (score, chunk) in enumerate(results, start=1):

            print(f"\nRank             : {rank}")
            print(f"Cross Score      : {score:.4f}")
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


if __name__ == "__main__":
    main()