import os
import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DOCUMENTS = os.path.join(
    BASE_DIR,
    "data",
    "raw_documents"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "extracted_text"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_pdf_text(pdf_path):
    """
    Extract all text from a PDF.
    """

    with fitz.open(pdf_path) as document:
        return "".join(page.get_text() for page in document)


def main():

    print("Searching for PDF files...\n")

    if not os.path.isdir(RAW_DOCUMENTS):
        raise FileNotFoundError(f"Raw document folder not found: {RAW_DOCUMENTS}")

    pdf_count = 0
    failed_count = 0
    output_sources = {}

    for root, directories, files in os.walk(RAW_DOCUMENTS):
        directories.sort()

        for file in sorted(files):

            if not file.lower().endswith(".pdf"):
                continue

            pdf_count += 1

            pdf_path = os.path.join(root, file)

            filename = os.path.splitext(file)[0] + ".txt"
            previous_source = output_sources.get(filename)
            if previous_source:
                raise ValueError(
                    f"PDF filename collision: {previous_source} and {pdf_path} both "
                    f"would write {filename}. Rename one of the source PDFs."
                )
            output_sources[filename] = pdf_path

            print(f"Processing: {file}")

            try:

                text = extract_pdf_text(pdf_path)

                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    filename
                )

                with open(
                    output_path,
                    "w",
                    encoding="utf-8"
                ) as f:
                    f.write(text)

                print(f"   Characters: {len(text)}")
                print(f"   Saved: {filename}\n")

            except Exception as e:

                failed_count += 1
                print(f"   Failed: {file}")
                print(f"   Error: {e}\n")

    print("=" * 50)
    print(f"Finished processing {pdf_count} PDF file(s).")
    print(f"Extracted text saved in: {OUTPUT_FOLDER}")

    if failed_count:
        raise RuntimeError(f"Failed to extract {failed_count} PDF file(s).")


if __name__ == "__main__":
    main()
