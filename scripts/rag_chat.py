import os

import ollama
from sentence_transformers import SentenceTransformer

from hybrid_retriever import hybrid_search
from build_grounded_prompt import build_grounded_prompt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

LLM_MODEL = "llama3.2:latest"


# ============================================================
# Convert Hybrid Results
# into Evidence Format
# expected by Prompt Builder
# ============================================================

def prepare_evidence(chunks):

    evidence_items = []

    for chunk in chunks:

        evidence_items.append(
            {
                "source_document": chunk.get(
                    "source_document",
                    "Unknown"
                ),

                "document_category": chunk.get(
                    "category",
                    "Unknown"
                ),

                "authority_score": 1.0,

                "similarity_score": chunk.get(
                    "rerank_score",
                    chunk.get(
                        "dense_score",
                        0.0
                    )
                ),

                "text": chunk.get(
                    "text",
                    ""
                )
            }
        )

    return evidence_items



# ============================================================
# Retrieval + Generation
# ============================================================

def generate_answer(query):


    print("\nRetrieving evidence...\n")


    # -----------------------------------------
    # Step 7:
    # Hybrid Retrieval
    # -----------------------------------------

    retrieved_chunks = hybrid_search(
        query
    )


    if not retrieved_chunks:

        return (
            "I couldn't find that information "
            "in the knowledge base."
        )


    # Convert to prompt format

    evidence_items = prepare_evidence(
        retrieved_chunks
    )


    # Temporary reliability values
    # Reliability will be integrated later

    reliability = {
        "authority": 1.0,
        "relevance": 1.0,
        "support": 1.0,
        "coverage": 1.0,
        "consistency": 1.0,
        "overall_reliability": 1.0
    }


    decision = {
        "decision": "ACCEPT"
    }



    # -----------------------------------------
    # Prompt Builder
    # -----------------------------------------

    prompt = build_grounded_prompt(
        query=query,
        evidence_items=evidence_items,
        reliability=reliability,
        decision=decision
    )


    # -----------------------------------------
    # LLM Generation
    # -----------------------------------------

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[

            {
                "role": "system",

                "content":
                """
You are CareBuddy,
an elderly-care assistant.

Rules:

- Answer ONLY from the provided evidence.
- Do not use outside knowledge.
- Do not invent medical facts.
- If evidence is insufficient,
say:

I couldn't find that information in the knowledge base.

Always include Sources at the end.
"""
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
    print("CareBuddy Elderly Care Assistant")
    print("Type 'exit' to quit.")
    print("=" * 60)


    while True:


        query = input(
            "\nQuestion: "
        ).strip()


        if query.lower() == "exit":

            break


        if not query:

            continue


        print(
            "\nGenerating answer...\n"
        )


        answer = generate_answer(
            query
        )


        print("=" * 60)
        print("Answer")
        print("=" * 60)

        print(answer)



if __name__ == "__main__":

    main()