import requests
import pandas as pd
import time
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GOLDEN_FILE = BASE_DIR / "golden_set.csv"
OUTPUT_FILE = BASE_DIR / "evaluation_results.csv"

API_URL = "http://127.0.0.1:8000/chat"


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("Loading golden set...")

df = pd.read_csv(GOLDEN_FILE)

print("Total examples:", len(df))


# ============================================================
# EVALUATE
# ============================================================

results = []

for index, row in df.iterrows():

    evaluation_id = row["evaluation_id"]
    conversation = str(row["conversation"])

    # --------------------------------------------------------
    # GET CUSTOMER QUERY
    # --------------------------------------------------------
    #
    # Use the last Customer message from the conversation.
    #

    customer_messages = []

    for line in conversation.split("\n"):

        if line.startswith("Customer:"):

            message = line.replace(
                "Customer:",
                "",
                1
            ).strip()

            if message:
                customer_messages.append(message)


    if not customer_messages:

        print(
            f"Skipping {evaluation_id}: "
            "No customer message found"
        )

        continue


    query = customer_messages[-1]


    print(
        f"Evaluating {index + 1}/{len(df)} "
        f"(ID: {evaluation_id})..."
    )


    # --------------------------------------------------------
    # CALL FASTAPI
    # --------------------------------------------------------

    try:

        response = requests.post(

            API_URL,

            json={
                "query": query
            },

            timeout=120
        )


        response.raise_for_status()

        data = response.json()


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predicted_cluster = data.get(
            "cluster"
        )

        predicted_intent = data.get(
            "intent"
        )

        predicted_handling = data.get(
            "handling"
        )

        generated_response = data.get(
            "response"
        )


        # ----------------------------------------------------
        # GOLD LABELS
        # ----------------------------------------------------

        gold_cluster = row.get(
            "gold_cluster"
        )

        gold_intent = row.get(
            "gold_intent"
        )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        intent_correct = (

            str(predicted_intent).strip()
            ==
            str(gold_intent).strip()
        )


        cluster_correct = (

            str(predicted_cluster).strip()
            ==
            str(gold_cluster).strip()
        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        results.append({

            "evaluation_id":
                evaluation_id,

            "conversation_id":
                row.get(
                    "conversation_id",
                    ""
                ),

            "query":
                query,

            "gold_intent":
                gold_intent,

            "predicted_intent":
                predicted_intent,

            "intent_correct":
                intent_correct,

            "gold_cluster":
                gold_cluster,

            "predicted_cluster":
                predicted_cluster,

            "cluster_correct":
                cluster_correct,

            "handling":
                predicted_handling,

            "response":
                generated_response
        })


    except Exception as e:

        print(
            f"ERROR for ID {evaluation_id}:",
            e
        )


    # Small delay between requests
    time.sleep(0.2)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CALCULATE METRICS
# ============================================================

if len(results_df) > 0:

    intent_accuracy = (
        results_df["intent_correct"]
        .mean()
        * 100
    )

    cluster_accuracy = (
        results_df["cluster_correct"]
        .mean()
        * 100
    )


    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        "\nExamples evaluated:",
        len(results_df)
    )

    print(
        f"\nIntent Accuracy: "
        f"{intent_accuracy:.2f}%"
    )

    print(
        f"Cluster Accuracy: "
        f"{cluster_accuracy:.2f}%"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )

else:

    print(
        "\nNo results were generated."
    )