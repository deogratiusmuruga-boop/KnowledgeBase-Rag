import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_text"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "chunks"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHUNK_SIZE = 500
OVERLAP = 100


def chunk_text(text, source_document, chunk_size=500, overlap=100):
    """
    Split text into overlapping word chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to zero and less than chunk_size.")

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 1

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = {
            "chunk_id": chunk_id,
            "source_document": source_document,
            "text": " ".join(chunk_words)
        }

        chunks.append(chunk)

        chunk_id += 1

        start += chunk_size - overlap

    return chunks


def main():

    print("Searching for cleaned text files...\n")

    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"Cleaned text folder not found: {INPUT_FOLDER}")

    total_documents = 0
    total_chunks = 0
    failed_count = 0

    for file in sorted(os.listdir(INPUT_FOLDER)):

        if not file.endswith("_cleaned.txt"):
            continue

        total_documents += 1

        input_path = os.path.join(INPUT_FOLDER, file)

        print(f"Chunking: {file}")

        try:

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            source_document = file.removesuffix("_cleaned.txt") + ".pdf"

            chunks = chunk_text(
                text,
                source_document,
                CHUNK_SIZE,
                OVERLAP
            )

            output_name = file.removesuffix("_cleaned.txt") + "_chunks.json"

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    chunks,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            total_chunks += len(chunks)

            print(f"   Chunks: {len(chunks)}")
            print(f"   Saved : {output_name}\n")

        except Exception as e:

            failed_count += 1
            print(f"   Failed: {file}")
            print(f"   Error : {e}\n")

    print("=" * 60)
    print(f"Documents processed : {total_documents}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Chunk files saved in: {OUTPUT_FOLDER}")

    if failed_count:
        raise RuntimeError(f"Failed to chunk {failed_count} document(s).")


if __name__ == "__main__":
    main()
