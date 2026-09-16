import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

EVALUATION_FILE = BASE_DIR / "evaluation_results.csv"
DATA_FILE = DATA_DIR / "apple_support_intents.csv"

OUTPUT_FILE = BASE_DIR / "tfidf_results_no_leakage.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

evaluation_df = pd.read_csv(EVALUATION_FILE)
data_df = pd.read_csv(DATA_FILE)

print("Evaluation examples:", len(evaluation_df))
print("Reference conversations:", len(data_df))


# ============================================================
# PREPARE TEXT
# ============================================================

reference_texts = (
    data_df["conversation"]
    .fillna("")
    .astype(str)
    .tolist()
)

evaluation_queries = (
    evaluation_df["query"]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# CREATE TF-IDF VECTORS
# ============================================================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000
)

reference_vectors = vectorizer.fit_transform(
    reference_texts
)

query_vectors = vectorizer.transform(
    evaluation_queries
)


print(
    "Reference matrix shape:",
    reference_vectors.shape
)


# ============================================================
# EVALUATE
# ============================================================

print("\nRunning leakage-safe TF-IDF evaluation...")

results = []

for i in range(len(evaluation_df)):

    evaluation_row = evaluation_df.iloc[i]

    evaluation_id = evaluation_row["evaluation_id"]
    evaluation_conversation_id = str(
        evaluation_row["conversation_id"]
    )

    query_vector = query_vectors[i]

    # --------------------------------------------------------
    # Calculate similarities
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        reference_vectors
    ).flatten()

    # --------------------------------------------------------
    # REMOVE THE EVALUATION CONVERSATION
    # --------------------------------------------------------

    for j in range(len(data_df)):

        reference_conversation_id = str(
            data_df.iloc[j]["conversation_id"]
        )

        if (
            reference_conversation_id
            ==
            evaluation_conversation_id
        ):

            similarities[j] = -1


    # --------------------------------------------------------
    # FIND BEST OTHER CONVERSATION
    # --------------------------------------------------------

    best_index = np.argmax(
        similarities
    )

    best_similarity = similarities[
        best_index
    ]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_intent = str(
        data_df.iloc[
            best_index
        ]["intent"]
    )

    predicted_cluster = data_df.iloc[
        best_index
    ]["cluster"]


    # --------------------------------------------------------
    # GOLD LABEL
    # --------------------------------------------------------

    gold_intent = str(
        evaluation_row["gold_intent"]
    )

    gold_cluster = evaluation_row[
        "gold_cluster"
    ]


    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    intent_correct = (
        predicted_intent
        ==
        gold_intent
    )

    cluster_correct = (
        str(predicted_cluster)
        ==
        str(gold_cluster)
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "evaluation_id":
            evaluation_id,

        "conversation_id":
            evaluation_conversation_id,

        "query":
            evaluation_row["query"],

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

        "similarity":
            best_similarity
    })


    if (i + 1) % 10 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(evaluation_df)}"
        )


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


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("LEAKAGE-SAFE TF-IDF BASELINE")
print("=" * 60)

print(
    "\nExamples evaluated:",
    len(results_df)
)

print(
    "Correct intent predictions:",
    int(
        results_df["intent_correct"].sum()
    )
)

print(
    f"Intent Accuracy: "
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

print("\nDone!")