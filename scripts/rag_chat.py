"""
CareBuddy RAG Generation Module

Responsible only for:
- retrieving evidence
- building grounded prompt
- generating LLM response

Designed for:
- FastAPI backend
- carebuddy_service.py
- frontend applications
"""

import ollama

from scripts.hybrid_retriever import hybrid_search
from scripts.build_grounded_prompt import build_grounded_prompt



# ============================================================
# Configuration
# ============================================================

LLM_MODEL = "llama3.2:latest"



# ============================================================
# Evidence Preparation
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
# Generate Answer
# ============================================================

def generate_answer(
        query,
        return_evidence=False
        ):

    print("\nRetrieving evidence...\n")


    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    retrieved_chunks = hybrid_search(
        query
    )


    if not retrieved_chunks:

        answer = (
            "I couldn't find that information "
            "in the knowledge base."
        )


        if return_evidence:

            return (
                answer,
                []
            )


        return answer



    # --------------------------------------------------------
    # Convert chunks into evidence format
    # --------------------------------------------------------

    evidence_items = prepare_evidence(
        retrieved_chunks
    )



    # --------------------------------------------------------
    # Temporary reliability values
    # --------------------------------------------------------

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



    # --------------------------------------------------------
    # Build grounded prompt
    # --------------------------------------------------------

    prompt = build_grounded_prompt(
        query=query,
        evidence_items=evidence_items,
        reliability=reliability,
        decision=decision
    )



    # --------------------------------------------------------
    # LLM Generation
    # --------------------------------------------------------

    response = ollama.chat(

        model=LLM_MODEL,

        options={
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 10
        },

        messages=[

            {
                "role": "system",

                "content":
                """
You are CareBuddy,
an evidence-grounded elderly-care assistant.

Strict rules:

1. Use ONLY retrieved evidence.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not add medical facts.
5. If evidence is insufficient, say:

I couldn't find that information in the knowledge base.

Keep answers short and factual.

Always finish with:

Sources:
- document_name.pdf
"""
            },


            {
                "role": "user",

                "content": prompt
            }

        ]
    )


    # ========================================================
    # Extract Answer
    # ========================================================

    answer = response["message"]["content"]



    # ========================================================
    # Debug Output
    # ========================================================

    print("\nLLM RESPONSE OBJECT:")
    print(response)



    # ========================================================
    # Return
    # ========================================================

    if return_evidence:

        return (
            answer,
            evidence_items
        )


    return answer