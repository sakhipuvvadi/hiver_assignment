import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/apple_support_knowledge_base.csv"
OUTPUT_FILE = "data/apple_support_clusters.csv"

NUMBER_OF_CLUSTERS = 20


# ==========================================
# 1. Load knowledge base
# ==========================================

print("Loading knowledge base...")

df = pd.read_csv(INPUT_FILE)

print(f"Conversations: {len(df):,}")


# ==========================================
# 2. Load embedding model
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded.")


# ==========================================
# 3. Create embeddings
# ==========================================

print("\nCreating conversation embeddings...")

embeddings = model.encode(
    df["conversation"].astype(str).tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings created.")


# ==========================================
# 4. Cluster conversations
# ==========================================

print("\nClustering conversations...")

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    embeddings
)


# ==========================================
# 5. Show cluster sizes
# ==========================================

print("\n" + "=" * 60)
print("CLUSTER SIZES")
print("=" * 60)

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


# ==========================================
# 6. Show examples from each cluster
# ==========================================

print("\n" + "=" * 60)
print("CLUSTER EXAMPLES")
print("=" * 60)


for cluster in sorted(df["cluster"].unique()):

    cluster_df = df[
        df["cluster"] == cluster
    ]

    print("\n")
    print("-" * 60)
    print(f"CLUSTER {cluster}")
    print(
        f"Conversations: {len(cluster_df)}"
    )
    print("-" * 60)

    # Show up to 5 examples
    examples = cluster_df.head(5)

    for _, row in examples.iterrows():

        print(
            f"\nConversation ID: "
            f"{row['conversation_id']}"
        )

        text = str(row["conversation"])

        if len(text) > 700:
            text = text[:700] + "..."

        print(text)


# ==========================================
# 7. Save clustered dataset
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Saved to: {OUTPUT_FILE}"
)