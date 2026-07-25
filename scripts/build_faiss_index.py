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

    print("=" * 60)
    print("Building FAISS Vector Database")
    print("=" * 60)

    # ---------------------------------------------------------
    # Check embedding file
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Embedding file not found:\n{INPUT_FILE}"
        )

    print("Loading embeddings...")

    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    if "embeddings" not in data or "chunks" not in data:
        raise ValueError(
            "Embedding file is missing required data."
        )

    embeddings = data["embeddings"].astype(np.float32)
    chunks = data["chunks"]

    if len(embeddings) == 0:
        raise ValueError(
            "No embeddings found."
        )

    print(f"Knowledge Base Chunks : {len(chunks)}")
    print(f"Embedding Matrix      : {embeddings.shape}")

    dimension = embeddings.shape[1]

    print("\nBuilding FAISS index...")

    # Inner Product (Cosine Similarity because embeddings are normalized)
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("Saving vector database...")

    faiss.write_index(index, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("FAISS Index Successfully Created")
    print("=" * 60)

    print(f"Embedding Model Dimension : {dimension}")
    print(f"Indexed Vectors           : {index.ntotal}")
    print(f"Vector Database           : {OUTPUT_FILE}")

    print("\nKnowledge Base is now ready for retrieval.")


if __name__ == "__main__":
    main()