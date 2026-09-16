import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "apple_support_intents.csv"
OUTPUT_FILE = BASE_DIR / "golden_set.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total conversations:", len(df))


# ============================================================
# RANDOMLY SELECT 200
# ============================================================

golden_df = df.sample(
    n=200,
    random_state=42
).copy()


# ============================================================
# CREATE EVALUATION COLUMNS
# ============================================================

golden_df.insert(
    0,
    "evaluation_id",
    range(1, len(golden_df) + 1)
)


# Use existing intent as the current gold label
golden_df["gold_intent"] = golden_df["intent"]


# Existing cluster
golden_df["gold_cluster"] = golden_df["cluster"]


# ============================================================
# SAVE
# ============================================================

golden_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("GOLDEN SET CREATED")
print("========================================")

print("Number of examples:", len(golden_df))

print("\nIntent distribution:")

print(
    golden_df["gold_intent"]
    .value_counts()
)

print(
    "\nSaved to:",
    OUTPUT_FILE
)

print("\nDone!")