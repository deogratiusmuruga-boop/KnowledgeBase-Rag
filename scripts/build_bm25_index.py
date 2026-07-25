import os
import json
import pickle

from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "chunks",
    "knowledge_base_chunks.json"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "vector_db"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "knowledge_base_bm25.pkl"
)


def tokenize(text):
    """
    Basic BM25 tokenizer.
    """
    return text.lower().split()


def main():

    print("Loading knowledge base...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    print("Tokenizing documents...")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks
            },
            f
        )

    print("\n" + "=" * 60)
    print("BM25 index created successfully.")
    print("=" * 60)
    print(f"Indexed documents : {len(chunks)}")
    print(f"Saved index       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()