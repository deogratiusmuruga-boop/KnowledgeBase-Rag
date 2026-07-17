import os
import json
import pickle

from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "chunks"
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
    all_chunks = []

    print("\nReading chunk files...\n")

    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"Chunk folder not found: {INPUT_FOLDER}")

    for file in sorted(os.listdir(INPUT_FOLDER)):

        if not file.endswith("_chunks.json"):
            continue

        file_path = os.path.join(INPUT_FOLDER, file)

        with open(file_path, "r", encoding="utf-8") as f:

            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError(f"Chunk file must contain a JSON list: {file_path}")

        invalid_chunks = [
            index for index, chunk in enumerate(chunks)
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("text"), str)
                or not chunk["text"].strip()
            )
        ]
        if invalid_chunks:
            raise ValueError(
                f"Invalid chunk records in {file_path}: {invalid_chunks[:5]}"
            )

        all_chunks.extend(chunks)

        print(f"{file} -> {len(chunks)} chunks")

    print("\n----------------------------------------")
    print(f"Total chunks loaded: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError(f"No '*_chunks.json' files with chunks found in: {INPUT_FOLDER}")

    texts = [chunk["text"] for chunk in all_chunks]

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    data = {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "model_name": MODEL_NAME,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print("\n========================================")
    print("Embedding generation complete.")
    print(f"Embedding shape : {embeddings.shape}")
    print(f"Knowledge chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
