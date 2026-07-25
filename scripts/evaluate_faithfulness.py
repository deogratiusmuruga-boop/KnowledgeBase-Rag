import os
import json

import ollama

from hybrid_retriever import hybrid_search
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

JUDGE_MODEL = "llama3.2:latest"



# ============================================================
# Retrieve Evidence
# ============================================================

def retrieve_evidence(query):

    chunks = hybrid_search(query)

    evidence_text = ""

    for i, chunk in enumerate(chunks, start=1):

        evidence_text += (
            f"\n"
            f"==============================\n"
            f"Evidence {i}\n"
            f"==============================\n"
            f"Source: {chunk.get('source_document')}\n\n"
            f"{chunk.get('text')}\n"
        )

    return evidence_text



# ============================================================
# Faithfulness Judge
# ============================================================

def evaluate_faithfulness(
    question,
    answer,
    evidence
):

    prompt = f"""

You are a strict RAG faithfulness evaluator.

Your ONLY job:

Check whether the generated answer contains
information that is NOT supported by the retrieved evidence.


IMPORTANT:

Do NOT punish the answer because:

- it is short
- it does not include every evidence detail
- it summarizes evidence
- it omits examples
- it does not use all retrieved documents


ONLY punish:

- invented facts
- unsupported medical claims
- information not present in evidence
- contradictions with evidence


====================================================
QUESTION
====================================================

{question}


====================================================
RETRIEVED EVIDENCE
====================================================

{evidence}


====================================================
GENERATED ANSWER
====================================================

{answer}


====================================================
SCORING
====================================================


1.0

All statements are supported by evidence.
No hallucination.


0.75

Almost completely supported.
Only tiny unsupported wording.


0.50

Some important statements are unsupported.


0.25

Many unsupported statements.


0.0

Answer is unrelated or contradicts evidence.


Return ONLY JSON:

{{
    "score": 1.0,
    "reason": "short explanation"
}}

"""


    response = ollama.chat(
        model=JUDGE_MODEL,
        format="json",
        messages=[
            {
                "role": "system",
                "content":
                "You evaluate RAG faithfulness only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    return response["message"]["content"]



# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("CareBuddy Faithfulness Evaluation")
    print("=" * 60)


    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        queries = json.load(f)



    total_score = 0.0



    for item in queries:


        question = item["query"]


        print("\n")
        print("=" * 60)
        print("Question:")
        print(question)
        print("=" * 60)



        print("\nGenerating answer...\n")


        answer = generate_answer(
            question
        )


        print("\nRetrieving evidence...\n")


        evidence = retrieve_evidence(
            question
        )


        print("\nEvaluating faithfulness...\n")


        result = evaluate_faithfulness(
            question,
            answer,
            evidence
        )


        print("Faithfulness Result")

        print(result)

        print("_" * 60)



        try:

            parsed = json.loads(
                result
            )

            score = float(
                parsed["score"]
            )

            total_score += score


        except Exception as e:

            print(
                "JSON parsing failed:",
                e
            )



    average = (
        total_score /
        len(queries)
    )


    print("\n")
    print("=" * 60)
    print("FINAL FAITHFULNESS REPORT")
    print("=" * 60)

    print(
        "Total Questions:",
        len(queries)
    )

    print(
        f"Average Faithfulness Score: {average:.2f}"
    )

    print(
        "Target Faithfulness: >= 0.85"
    )



if __name__ == "__main__":

    main()