import os
import json
import time
import statistics

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


# Requirement from proposal
TARGET_LATENCY = 3.0



# ============================================================
# Main Latency Evaluation
# ============================================================

def main():

    print("=" * 60)
    print("CareBuddy End-to-End Latency Evaluation")
    print("=" * 60)



    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        queries = json.load(
            file
        )



    response_times = []

    passed = 0



    for index, item in enumerate(
        queries,
        start=1
    ):


        question = item["query"]


        print("\n")
        print("=" * 60)

        print(
            f"Question {index}/{len(queries)}"
        )

        print(
            question
        )

        print("=" * 60)



        print(
            "\nGenerating answer..."
        )



        start_time = time.perf_counter()



        answer = generate_answer(
            question
        )



        end_time = time.perf_counter()



        latency = (
            end_time -
            start_time
        )


        response_times.append(
            latency
        )



        if latency <= TARGET_LATENCY:

            passed += 1



        print(
            f"Response Time: {latency:.2f} seconds"
        )



        print(
            "Status:",
            "PASS"
            if latency <= TARGET_LATENCY
            else "FAIL"
        )



    # ========================================================
    # Statistics
    # ========================================================


    average_latency = statistics.mean(
        response_times
    )


    minimum_latency = min(
        response_times
    )


    maximum_latency = max(
        response_times
    )


    pass_rate = (
        passed /
        len(queries)
        *
        100
    )



    print("\n")
    print("=" * 60)
    print("FINAL LATENCY REPORT")
    print("=" * 60)


    print(
        "Total Questions:",
        len(queries)
    )


    print(
        f"Average Response Time: {average_latency:.2f} seconds"
    )


    print(
        f"Minimum Response Time: {minimum_latency:.2f} seconds"
    )


    print(
        f"Maximum Response Time: {maximum_latency:.2f} seconds"
    )


    print(
        f"Under 3 Seconds: {passed}/{len(queries)}"
    )


    print(
        f"Latency Pass Rate: {pass_rate:.2f}%"
    )


    print(
        "Target Response Time: <= 3 seconds"
    )



if __name__ == "__main__":

    main()