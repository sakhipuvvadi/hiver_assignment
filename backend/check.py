import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    "../data/apple_support_intents.csv"
)

print("CSV loaded:", len(df))


# ==============================
# LOAD KMEANS
# ==============================

with open(
    "models/kmeans_model.pkl",
    "rb"
) as f:

    kmeans = pickle.load(f)

kmeans.cluster_centers_ = np.asarray(
    kmeans.cluster_centers_,
    dtype=np.float64
)

print("KMeans loaded!")


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==============================
# TEST QUERIES
# ==============================

queries = [
    "My photos disappeared from my iPhone",
    "My iPhone battery is draining very quickly"
]


for query in queries:

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    embedding = np.asarray(
        embedding,
        dtype=np.float64
    )

    cluster = int(
        kmeans.predict(
            embedding.reshape(1, -1)
        )[0]
    )

    print("\n==============================")
    print("QUERY:", query)
    print("PREDICTED CLUSTER:", cluster)

    rows = df[
        df["cluster"] == cluster
    ]

    print(
        "INTENT:",
        rows["intent"].iloc[0]
    )

    print(
        "REASON:",
        rows["reason"].iloc[0]
    )