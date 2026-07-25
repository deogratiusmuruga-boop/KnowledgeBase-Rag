"""
Adaptive Reliability Decision Controller

Converts reliability score into
response action decision.

Actions:

ACCEPT
REFINE
RE-RETRIEVE
REJECT

"""


from scripts.reliability_config import (
    load_reliability_config
)



# ============================================================
# Decision Function
# ============================================================

def make_reliability_decision(
    reliability
):


    """
    reliability can be:

    - dictionary from reliability_evaluation.py
    - direct float score

    """


    # --------------------------------------------------------
    # Handle dictionary input
    # --------------------------------------------------------

    if isinstance(
        reliability,
        dict
    ):

        reliability_score = float(

            reliability.get(
                "overall_reliability",
                0.0
            )

        )


    else:

        reliability_score = float(
            reliability
        )



    # --------------------------------------------------------
    # Load thresholds
    # --------------------------------------------------------

    config = load_reliability_config()


    thresholds = config.get(
        "decision_thresholds",
        {}
    )



    accept_threshold = thresholds.get(
        "accept",
        0.75
    )


    refine_threshold = thresholds.get(
        "refine",
        0.60
    )


    retrieve_threshold = thresholds.get(
        "re_retrieve",
        0.40
    )



    # --------------------------------------------------------
    # Decision Logic
    # --------------------------------------------------------

    if reliability_score >= accept_threshold:


        return {

            "decision": "ACCEPT",

            "score": reliability_score,

            "reason":
            "Retrieved evidence meets the high reliability threshold."

        }



    elif reliability_score >= refine_threshold:


        return {

            "decision": "REFINE",

            "score": reliability_score,

            "reason":
            "Answer requires refinement before delivery."

        }



    elif reliability_score >= retrieve_threshold:


        return {

            "decision": "RE-RETRIEVE",

            "score": reliability_score,

            "reason":
            "Additional evidence retrieval is required."

        }



    else:


        return {

            "decision": "REJECT",

            "score": reliability_score,

            "reason":
            "Evidence reliability is insufficient."

        }