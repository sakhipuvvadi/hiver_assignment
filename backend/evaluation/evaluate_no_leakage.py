import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "backend" / "models"

EVALUATION_FILE = BASE_DIR / "golden_set.csv"
DATA_FILE = DATA_DIR / "apple_support_intents.csv"
EMBEDDINGS_FILE = MODEL_DIR / "conversation_embeddings.npy"

OUTPUT_FILE = BASE_DIR / "evaluation_no_leakage_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading evaluation set...")

golden_df = pd.read_csv(
    EVALUATION_FILE
)

print(
    "Golden set rows:",
    len(golden_df)
)


print("\nLoading reference data...")

data_df = pd.read_csv(
    DATA_FILE
)

print(
    "Reference conversations:",
    len(data_df)
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

if len(data_df) != len(embeddings):

    raise ValueError(
        f"Mismatch!\n"
        f"Data rows: {len(data_df)}\n"
        f"Embeddings: {len(embeddings)}"
    )


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading SentenceTransformer...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CREATE QUERY EMBEDDINGS
# ============================================================

print("\nEncoding evaluation queries...")

queries = (
    golden_df["conversation"]
    .fillna("")
    .astype(str)
    .tolist()
)

query_embeddings = model.encode(
    queries,
    normalize_embeddings=True,
    show_progress_bar=True
)

query_embeddings = np.asarray(
    query_embeddings,
    dtype=np.float64
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float64
)


# ============================================================
# NORMALIZE REFERENCE EMBEDDINGS
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1

embeddings = (
    embeddings / norms
)


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning leakage-safe evaluation...")

results = []

for i in range(
    len(golden_df)
):

    row = golden_df.iloc[i]

    evaluation_id = row[
        "evaluation_id"
    ]

    evaluation_conversation_id = str(
        row["conversation_id"]
    )


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = query_embeddings[i]


    # --------------------------------------------------------
    # Cosine similarity
    #
    # Because vectors are normalized,
    # dot product = cosine similarity
    # --------------------------------------------------------

    similarities = (
        embeddings
        @
        query_embedding
    )


    # --------------------------------------------------------
    # REMOVE SELF-MATCH
    # --------------------------------------------------------

    reference_ids = (
        data_df[
            "conversation_id"
        ]
        .astype(str)
        .values
    )


    self_indices = np.where(
        reference_ids
        ==
        evaluation_conversation_id
    )[0]


    if len(self_indices) > 0:

        similarities[
            self_indices
        ] = -1


    # --------------------------------------------------------
    # FIND BEST OTHER CONVERSATION
    # --------------------------------------------------------

    best_index = int(
        np.argmax(similarities)
    )


    similarity_score = float(
        similarities[best_index]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_intent = str(
        data_df.iloc[
            best_index
        ]["intent"]
    )


    predicted_cluster = int(
        data_df.iloc[
            best_index
        ]["cluster"]
    )


    # --------------------------------------------------------
    # GOLD
    # --------------------------------------------------------

    gold_intent = str(
        row["gold_intent"]
    )


    gold_cluster = int(
        row["gold_cluster"]
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    intent_correct = (
        predicted_intent
        ==
        gold_intent
    )


    cluster_correct = (
        predicted_cluster
        ==
        gold_cluster
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
            row["conversation"],

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
            similarity_score
    })


    if (
        (i + 1) % 10 == 0
    ):

        print(
            f"Processed "
            f"{i + 1}/{len(golden_df)}"
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
# METRICS
# ============================================================

intent_accuracy = (
    results_df[
        "intent_correct"
    ]
    .mean()
    * 100
)


cluster_accuracy = (
    results_df[
        "cluster_correct"
    ]
    .mean()
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print(
    "LEAKAGE-SAFE SENTENCETRANSFORMER EVALUATION"
)
print("=" * 60)

print(
    "\nExamples evaluated:",
    len(results_df)
)

print(
    "Correct intent predictions:",
    int(
        results_df[
            "intent_correct"
        ].sum()
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