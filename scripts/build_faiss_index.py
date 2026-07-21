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

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "vector_db"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "knowledge_base.faiss"
)


def main():

    print("Loading embeddings...")

    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"].astype(np.float32)

    print(f"Embedding matrix shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    print("Building FAISS index...")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(index, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("FAISS index created successfully.")
    print("=" * 60)
    print(f"Vectors indexed : {index.ntotal}")
    print(f"Embedding size  : {dimension}")
    print(f"Saved index     : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()