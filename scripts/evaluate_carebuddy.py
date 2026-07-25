import os
import json
import re

from sentence_transformers import SentenceTransformer

from rag_chat import generate_answer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "evaluation_queries.json"
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# Filename normalization
# ============================================================

def normalize_filename(filename):
    """
    Normalize filenames for comparison.

    Example:
    tips-take-medicines-safely.pdf
    tipstakemedicinessafely.pdf

    become identical.
    """

    filename = filename.lower()

    # remove extension
    filename = filename.replace(".pdf", "")

    # remove all non-alphanumeric characters
    filename = re.sub(
        r"[^a-z0-9]",
        "",
        filename
    )

    return filename



# ============================================================
# Source Matching
# ============================================================

def check_source_match(
    expected_source,
    retrieved_sources
):

    expected = normalize_filename(
        expected_source
    )


    for source in retrieved_sources:

        normalized_source = normalize_filename(
            source
        )

        if expected in normalized_source:
            return True


    return False



# ============================================================
# Extract Sources From Answer
# ============================================================

def extract_sources(answer):

    sources = []


    if "Sources:" not in answer:
        return sources


    source_section = answer.split(
        "Sources:"
    )[1]


    for line in source_section.splitlines():

        line = line.strip()

        if line.startswith("-"):

            sources.append(
                line.replace(
                    "-",
                    ""
                ).strip()
            )


    return sources



# ============================================================
# Main Evaluation
# ============================================================


def main():

    print("=" * 60)
    print("CareBuddy Evaluation")
    print("=" * 60)


    print("\nLoading embedding model...\n")

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )


    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        evaluation_queries = json.load(f)



    total_questions = len(
        evaluation_queries
    )

    source_matches = 0



    for item in evaluation_queries:


        print("\n")
        print("=" * 60)

        print("Question:")
        print(
            item["query"]
        )

        print("=" * 60)


        print("\nGenerating answer...\n")


        answer = generate_answer(
            item["query"],
            
        )


        print("Generated Answer")
        print("-" * 60)

        print(answer)


        retrieved_sources = extract_sources(
            answer
        )


        print("\nEvaluation")
        print("-" * 60)


        print(
            "Expected Source:",
            item["expected_source"]
        )


        print(
            "Retrieved Sources:"
        )

        for source in retrieved_sources:

            print(
                "-",
                source
            )


        matched = check_source_match(
            item["expected_source"],
            retrieved_sources
        )


        if matched:

            source_matches += 1

            print(
                "Source Match: PASS"
            )

        else:

            print(
                "Source Match: FAIL"
            )



    accuracy = (
        source_matches /
        total_questions
    ) * 100



    print("\n")
    print("=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)


    print(
        "Total Questions :",
        total_questions
    )

    print(
        "Source Matches  :",
        source_matches
    )

    print(
        f"Source Accuracy : {accuracy:.2f}%"
    )



if __name__ == "__main__":

    main()