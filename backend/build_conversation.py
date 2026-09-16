import pandas as pd
import re

# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/apple_support.csv"
OUTPUT_FILE = "data/apple_support_tagged.csv"


# ==========================================
# 1. Load dataset
# ==========================================

print("Loading AppleSupport dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total tweets: {len(df):,}")


# ==========================================
# 2. Clean tweet IDs
# ==========================================

def clean_id(value):
    """
    Extract tweet IDs from values such as:
    712
    712,713
    7,06,704
    """
    
    if pd.isna(value):
        return []

    value = str(value)

    # Find numbers
    ids = re.findall(r'\d+', value)

    return ids


# ==========================================
# 3. Create connection graph
# ==========================================

# All tweet IDs in our AppleSupport dataset

tweet_ids = set(
    df["tweet_id"]
    .astype(str)
)


# Dictionary:
# tweet_id -> connected tweet IDs

connections = {
    tweet_id: set()
    for tweet_id in tweet_ids
}


# ==========================================
# 4. Connect using in_response_to_tweet_id
# ==========================================

for _, row in df.iterrows():

    current_id = str(row["tweet_id"])

    parent_ids = clean_id(
        row["in_response_to_tweet_id"]
    )

    for parent_id in parent_ids:

        if parent_id in tweet_ids:

            connections[current_id].add(parent_id)
            connections[parent_id].add(current_id)


# ==========================================
# 5. Connect using response_tweet_id
# ==========================================

for _, row in df.iterrows():

    current_id = str(row["tweet_id"])

    response_ids = clean_id(
        row["response_tweet_id"]
    )

    for response_id in response_ids:

        if response_id in tweet_ids:

            connections[current_id].add(response_id)
            connections[response_id].add(current_id)


# ==========================================
# 6. Find complete conversations
# ==========================================

visited = set()

conversation_number = 0

conversation_map = {}


for tweet_id in tweet_ids:

    if tweet_id in visited:
        continue

    conversation_number += 1

    # Start a new conversation
    stack = [tweet_id]

    visited.add(tweet_id)

    while stack:

        current = stack.pop()

        conversation_map[current] = conversation_number

        for connected in connections[current]:

            if connected not in visited:

                visited.add(connected)

                stack.append(connected)


# ==========================================
# 7. Add conversation ID
# ==========================================

df["conversation_id"] = (
    df["tweet_id"]
    .astype(str)
    .map(conversation_map)
)


# ==========================================
# 8. Sort conversations
# ==========================================

df["created_at"] = pd.to_datetime(
    df["created_at"]
)

df = df.sort_values(
    ["conversation_id", "created_at"]
)


# ==========================================
# 9. Display conversation information
# ==========================================

print("\n" + "=" * 50)
print("CONVERSATION RESULTS")
print("=" * 50)

print(
    f"Total conversations: "
    f"{df['conversation_id'].nunique():,}"
)


# ==========================================
# 10. Show conversation sizes
# ==========================================

conversation_sizes = (
    df.groupby("conversation_id")
      .size()
      .sort_values(ascending=False)
)

print("\nLargest conversations:")

print(
    conversation_sizes.head(10)
)


# ==========================================
# 11. Save
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 50)
print("SUCCESS")
print("=" * 50)

print(
    f"Tagged dataset saved to:\n"
    f"{OUTPUT_FILE}"
)