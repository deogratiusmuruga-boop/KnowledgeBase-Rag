import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "extracted_text"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_text"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    # Normalize line endings
    text = re.sub(r"\r\n?", "\n", text)

    # Remove multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Remove page numbers appearing alone on a line
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def main():

    print("Searching for extracted text files...\n")

    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"Extracted text folder not found: {INPUT_FOLDER}")

    file_count = 0
    failed_count = 0

    for file in sorted(os.listdir(INPUT_FOLDER)):

        if not file.endswith(".txt"):
            continue

        file_count += 1

        input_path = os.path.join(INPUT_FOLDER, file)

        print(f"Cleaning: {file}")

        try:

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned = clean_text(text)

            output_name = file.removesuffix(".txt") + "_cleaned.txt"

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(cleaned)

            print(f"   Original : {len(text)} characters")
            print(f"   Cleaned  : {len(cleaned)} characters")
            print(f"   Saved    : {output_name}\n")

        except Exception as e:

            failed_count += 1
            print(f"   Failed: {file}")
            print(f"   Error : {e}\n")

    print("=" * 50)
    print(f"Finished cleaning {file_count} text file(s).")
    print(f"Cleaned files saved in: {OUTPUT_FOLDER}")

    if failed_count:
        raise RuntimeError(f"Failed to clean {failed_count} text file(s).")


if __name__ == "__main__":
    main()
