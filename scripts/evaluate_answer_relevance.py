import os
import json
import ollama

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
# Answer Relevance Judge
# ============================================================

def evaluate_answer_relevance(
    question,
    answer
):

    prompt = f"""

You are a strict RAG answer relevance evaluator.

Your ONLY task is to evaluate whether the generated answer
directly answers the user's question.

Do NOT evaluate:
- factual correctness
- faithfulness
- missing details
- writing quality
- answer length


====================================================
QUESTION
====================================================

{question}


====================================================
GENERATED ANSWER
====================================================

{answer}


====================================================
SCORING RULES
====================================================

1.0

The answer directly and completely addresses the question.


0.75

The answer answers the main question but misses
minor aspects.


0.50

The answer is partially related but does not fully
answer the question.


0.25

The answer is mostly unrelated.


0.0

The answer does not address the question at all.


IMPORTANT:

A short answer can receive 1.0 if it correctly answers
the question.

Do NOT reduce the score because:
- the answer is concise
- examples are missing
- additional evidence was not included


Return ONLY JSON.

Format:

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
                "You evaluate RAG answer relevance only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    return response["message"]["content"]



# ============================================================
# Main Evaluation
# ============================================================

def main():

    print("=" * 60)
    print("CareBuddy Answer Relevance Evaluation")
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


        print("Generated Answer")
        print("-" * 60)
        print(answer)



        print("\nEvaluating relevance...\n")


        result = evaluate_answer_relevance(
            question,
            answer
        )


        print("Relevance Result")
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


        except Exception as error:

            print(
                "JSON parsing failed:",
                error
            )



    average = (
        total_score /
        len(queries)
    )



    print("\n")
    print("=" * 60)
    print("FINAL ANSWER RELEVANCE REPORT")
    print("=" * 60)


    print(
        "Total Questions:",
        len(queries)
    )


    print(
        f"Average Answer Relevance Score: {average:.2f}"
    )


    print(
        "Target Answer Relevance: >= 0.85"
    )



if __name__ == "__main__":

    main()