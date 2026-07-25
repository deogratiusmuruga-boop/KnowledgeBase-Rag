import argparse
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

CHUNK_SIZE = 180
OVERLAP = 40
DOCUMENT_METADATA = {
    "nia_caregivers_handbook.pdf": {
        "document_id": "DOC0001",
        "title": "NIA Caregivers Handbook",
        "organization": "National Institute on Aging",
        "last_updated": "Unknown"
    },
    "understanding-memory-loss.pdf": {
        "document_id": "DOC0002",
        "title": "Understanding Memory Loss",
        "organization": "National Institute on Aging",
        "last_updated": "Unknown"
    },
    "Dietary_Guidelines_for_Americans_2020-2025.pdf": {
        "document_id": "DOC0003",
        "title": "Dietary Guidelines for Americans 2020–2025",
        "organization": "USDA",
        "last_updated": "2020"
    },
    "exercise-and-older-adults-nia.pdf": {
        "document_id": "DOC0004",
        "title": "Exercise and Older Adults",
        "organization": "National Institute on Aging",
        "last_updated": "Unknown"
    },
    "tips-take-medicines-safely.pdf": {
        "document_id": "DOC0005",
        "title": "Tips to Take Medicines Safely",
        "organization": "National Institute on Aging",
        "last_updated": "Unknown"
    },
    "who_icope_handbook.pdf": {
        "document_id": "DOC0006",
        "title": "WHO ICOPE Handbook",
        "organization": "World Health Organization",
        "last_updated": "Unknown"
    }
}


def chunk_text(
    text,
    source_document,
    category,
    metadata,
    chunk_size=180,
    overlap=40
):
    """
    Split text into overlapping word chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to zero and less than chunk_size."
        )

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 1

    while start < len(words):

        end = start + chunk_size
        chunk_words = words[start:end]

        chunk = {
            "chunk_id": chunk_id,
            "document_id": metadata["document_id"],
            "title": metadata["title"],
            "category": category,
            "organization": metadata["organization"],
            "source_document": source_document,
            "document_type": "pdf",
            "language": "English",
            "last_updated": metadata["last_updated"],
            "text": " ".join(chunk_words)
        }

        chunks.append(chunk)

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def main(input_file=None):

    print("Searching for cleaned text files...\n")

    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(
            f"Cleaned text folder not found: {INPUT_FOLDER}"
        )

    if input_file:

        input_file = os.path.abspath(input_file)

        if (
            not os.path.isfile(input_file)
            or not input_file.endswith("_cleaned.txt")
        ):
            raise ValueError(
                f"Input must be an existing *_cleaned.txt file: {input_file}"
            )

        input_paths = [input_file]

    else:

        input_paths = []

        for root, _, files in os.walk(INPUT_FOLDER):
            for file in sorted(files):
                if file.endswith("_cleaned.txt"):
                    input_paths.append(
                        os.path.join(root, file)
                    )

    total_documents = 0
    total_chunks = 0
    failed_count = 0

    all_chunks = []

    for input_path in input_paths:

        file = os.path.basename(input_path)

        total_documents += 1

        print(f"Chunking: {file}")

        try:

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Recover the original PDF filename
            source_document = file.removesuffix("_cleaned.txt") + ".pdf"
            metadata = DOCUMENT_METADATA.get(source_document)
            if metadata is None:
                raise ValueError(
                    f"No metadata found for {source_document}"
                )
                
            
                

            category = os.path.basename(
                os.path.dirname(input_path)
            )

            chunks = chunk_text(
                text=text,
                source_document=source_document,
                category=category,
                metadata=metadata,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP
            )

            output_name = (
                file.removesuffix("_cleaned.txt")
                + "_chunks.json"
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    chunks,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            all_chunks.extend(chunks)

            total_chunks += len(chunks)

            print(f"   Category : {category}")
            print(f"   Chunks   : {len(chunks)}")
            print(f"   Saved    : {output_name}\n")

        except Exception as e:

            failed_count += 1

            print(f"   Failed: {file}")
            print(f"   Error : {e}\n")

    master_output = os.path.join(
        OUTPUT_FOLDER,
        "knowledge_base_chunks.json"
    )

    with open(master_output, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print(f"Documents processed : {total_documents}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Chunk files saved in: {OUTPUT_FOLDER}")
    print(f"Master knowledge base saved as: {master_output}")

    if failed_count:
        raise RuntimeError(
            f"Failed to chunk {failed_count} document(s)."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split cleaned text files into chunks."
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional cleaned text file to process."
    )

    args = parser.parse_args()

    main(args.input_file)