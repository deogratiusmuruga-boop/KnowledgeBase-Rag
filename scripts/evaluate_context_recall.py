import os
import json


from hybrid_retriever import hybrid_search


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



# ============================================================
# Normalize filenames
# ============================================================

def normalize_filename(filename):

    return (
        filename
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace(".pdf", "")
    )



# ============================================================
# Retrieve Sources
# ============================================================

def retrieve_sources(query):

    chunks = hybrid_search(
        query
    )

    sources = set()


    for chunk in chunks:

        source = chunk.get(
            "source_document",
            ""
        )

        if source:

            sources.add(
                normalize_filename(
                    source
                )
            )


    return sources



# ============================================================
# Context Recall Calculation
# ============================================================

def calculate_context_recall(
    expected_source,
    retrieved_sources
):

    expected = normalize_filename(
        expected_source
    )


    if expected in retrieved_sources:

        return 1.0


    return 0.0



# ============================================================
# Main Evaluation
# ============================================================

def main():


    print("=" * 60)
    print("CareBuddy Context Recall Evaluation")
    print("=" * 60)



    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        queries = json.load(
            file
        )



    total_score = 0.0



    for item in queries:


        question = item["query"]

        expected_source = item["expected_source"]



        print("\n")
        print("=" * 60)

        print(
            "Question:"
        )

        print(
            question
        )


        print(
            "Expected Source:"
        )

        print(
            expected_source
        )



        print(
            "\nRetrieving evidence..."
        )



        retrieved_sources = retrieve_sources(
            question
        )



        print(
            "\nRetrieved Sources:"
        )


        for source in retrieved_sources:

            print(
                "-",
                source
            )



        score = calculate_context_recall(
            expected_source,
            retrieved_sources
        )


        print(
            "\nContext Recall Score:",
            score
        )


        total_score += score



    average = (
        total_score /
        len(queries)
    )



    print("\n")
    print("=" * 60)
    print("FINAL CONTEXT RECALL REPORT")
    print("=" * 60)



    print(
        "Total Questions:",
        len(queries)
    )


    print(
        f"Average Context Recall Score: {average:.2f}"
    )


    print(
        "Target Context Recall: >= 0.85"
    )



if __name__ == "__main__":

    main()