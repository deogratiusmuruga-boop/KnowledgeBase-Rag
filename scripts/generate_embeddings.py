import os
import json
import pickle
from collections import Counter

from sentence_transformers import SentenceTransformer

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
    "knowledge_base_embeddings.pkl"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():

    print("=" * 60)
    print("Knowledge Base Embedding Generation")
    print("=" * 60)

    # ---------------------------------------------------------
    # Check knowledge base
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Knowledge base not found:\n{INPUT_FILE}"
        )

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Reading knowledge base chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(chunks) == 0:
        raise ValueError(
            "knowledge_base_chunks.json contains no chunks."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    category_counts = Counter(
        chunk["category"]
        for chunk in chunks
    )

    print(f"Total chunks found: {len(texts)}")

    print("\nKnowledge Base Summary")

    for category, count in sorted(category_counts.items()):
        print(f"  {category:<25} {count}")

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    data = {
        "chunks": chunks,
        "embeddings": embeddings
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print("\n" + "=" * 60)
    print("Embedding generation complete.")
    print("=" * 60)

    print(f"Embedding model      : {MODEL_NAME}")
    print(f"Total chunks         : {len(chunks)}")
    print(f"Embedding dimension  : {embeddings.shape[1]}")
    print(f"Embedding shape      : {embeddings.shape}")
    print(f"Output file          : {OUTPUT_FILE}")

    print("\nKnowledge Base Categories")

    for category, count in sorted(category_counts.items()):
        print(f"  {category:<25} {count}")

    print("\nVector embeddings successfully generated.")


if __name__ == "__main__":
    main()