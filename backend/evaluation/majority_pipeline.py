import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GOLDEN_FILE = BASE_DIR / "evaluation_results.csv"


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

df = pd.read_csv(GOLDEN_FILE)

print("Examples:", len(df))


# ============================================================
# FIND MOST COMMON INTENT
# ============================================================

majority_intent = (
    df["gold_intent"]
    .value_counts()
    .idxmax()
)

majority_count = (
    df["gold_intent"]
    .value_counts()
    .max()
)


print("\nMost common intent:")
print(majority_intent)

print(
    "Number of examples:",
    majority_count
)


# ============================================================
# PREDICT MAJORITY INTENT FOR EVERY EXAMPLE
# ============================================================

df["baseline_prediction"] = majority_intent


# ============================================================
# CALCULATE ACCURACY
# ============================================================

df["correct"] = (
    df["baseline_prediction"]
    ==
    df["gold_intent"]
)


accuracy = (
    df["correct"].mean()
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("MAJORITY-CLASS BASELINE")
print("========================================")

print(
    f"Examples evaluated: {len(df)}"
)

print(
    f"Correct predictions: {df['correct'].sum()}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)