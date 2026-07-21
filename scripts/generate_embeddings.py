import os
import json
import pickle

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

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Reading knowledge base chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

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
    print(f"Total chunks       : {len(chunks)}")
    print(f"Embedding shape    : {embeddings.shape}")
    print(f"Embedding model    : {MODEL_NAME}")
    print(f"Saved embeddings   : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()