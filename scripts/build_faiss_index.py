import os
import pickle
import faiss
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base_embeddings.pkl"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vector_db",
    "knowledge_base.faiss"
)


def main():

    print("Loading embeddings...")

    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    if "embeddings" not in data or "chunks" not in data:
        raise ValueError("Embedding file must contain 'chunks' and 'embeddings'.")

    embeddings = data["embeddings"]

    embeddings = np.ascontiguousarray(
        np.array(
        embeddings,
        dtype="float32"
        )
    )

    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    if len(data["chunks"]) != embeddings.shape[0]:
        raise ValueError("The number of chunks does not match the number of embeddings.")

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    print("Creating FAISS index...")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    faiss.write_index(index, OUTPUT_FILE)

    print("\nFAISS index created successfully.")
    print(f"Total vectors indexed: {index.ntotal}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
