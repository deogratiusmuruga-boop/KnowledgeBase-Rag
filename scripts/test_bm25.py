import os
import pickle

from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BM25_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_bm25.pkl"
)

TOP_K = 5


def tokenize(text):
    return text.lower().split()


def retrieve(query):

    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print(f"TOP {TOP_K} BM25 DOCUMENTS")
    print("=" * 80)

    for rank, (idx, score) in enumerate(ranked[:TOP_K], start=1):

        chunk = chunks[idx]

        print(f"\nRank             : {rank}")
        print(f"BM25 Score       : {score:.4f}")
        print(f"Document ID      : {chunk['document_id']}")
        print(f"Title            : {chunk['title']}")
        print(f"Category         : {chunk['category']}")
        print(f"Organization     : {chunk['organization']}")
        print(f"Source           : {chunk['source_document']}")
        print(f"Chunk ID         : {chunk['chunk_id']}")
        print(f"Last Updated     : {chunk['last_updated']}")

        print("-" * 80)

        preview = chunk["text"][:700]

        print(preview)

        if len(chunk["text"]) > 700:
            print("...")

        print("-" * 80)


def main():

    print("=" * 80)
    print("BM25 Retrieval Test")
    print("=" * 80)

    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"Indexed Chunks : {len(data['chunks'])}")
    print(f"Top-K Returned : {TOP_K}")
    print("\nType 'exit' to quit.")

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        retrieve(query)


if __name__ == "__main__":
    main()