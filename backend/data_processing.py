import pandas as pd

# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/twcs.csv"
OUTPUT_FILE = "data/apple_support.csv"

APPLE_ACCOUNT = "AppleSupport"


# ==========================================
# 1. Load the dataset
# ==========================================

print("Loading Twitter Customer Support dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")


# ==========================================
# 2. Check for Apple-related accounts
# ==========================================

print("\n" + "=" * 50)
print("Searching for Apple-related accounts...")
print("=" * 50)

apple_authors = (
    df[
        df["author_id"]
        .astype(str)
        .str.contains("apple", case=False, na=False)
    ]["author_id"]
    .value_counts()
)

if len(apple_authors) == 0:
    print("No Apple-related accounts found.")

else:
    print("\nApple-related accounts:")
    print(apple_authors)


# ==========================================
# 3. Find AppleSupport's own tweets
# ==========================================

print("\n" + "=" * 50)
print("Finding AppleSupport tweets...")
print("=" * 50)

apple_support_tweets = df[
    df["author_id"].astype(str) == APPLE_ACCOUNT
].copy()

print(
    f"\nAppleSupport tweets found: "
    f"{len(apple_support_tweets):,}"
)


# ==========================================
# 4. Check if AppleSupport exists
# ==========================================

if apple_support_tweets.empty:

    print("\nERROR: AppleSupport was not found.")

    print("\nAvailable author IDs containing 'apple':")
    print(apple_authors)

    print("\nPlease check the exact AppleSupport account name.")

    exit()


# ==========================================
# 5. Get AppleSupport tweet IDs
# ==========================================

apple_ids = set(
    apple_support_tweets["tweet_id"]
    .astype(str)
)


# ==========================================
# 6. Find customer tweets belonging to
#    AppleSupport conversations
# ==========================================

print("\n" + "=" * 50)
print("Finding customer messages...")
print("=" * 50)


# Condition 1:
# Customer replied directly to an AppleSupport tweet

replied_to_apple = (
    df["in_response_to_tweet_id"]
    .astype(str)
    .isin(apple_ids)
)


# Condition 2:
# Customer started/continued a conversation
# by mentioning @AppleSupport in the text

mentioned_apple = (
    df["text"]
    .astype(str)
    .str.contains(
        "@AppleSupport",
        case=False,
        na=False
    )
)


# Combine both conditions
customer_tweets = df[
    replied_to_apple | mentioned_apple
].copy()


print(
    f"Customer tweets found: "
    f"{len(customer_tweets):,}"
)


# ==========================================
# 7. Combine BOTH directions
# ==========================================

print("\n" + "=" * 50)
print("Combining both directions...")
print("=" * 50)


apple_df = pd.concat(
    [
        apple_support_tweets,
        customer_tweets
    ],
    ignore_index=True
)


# ==========================================
# 8. Remove duplicate tweets
# ==========================================

before_duplicates = len(apple_df)

apple_df = apple_df.drop_duplicates(
    subset=["tweet_id"]
)

after_duplicates = len(apple_df)

print(
    f"\nDuplicate tweet IDs removed: "
    f"{before_duplicates - after_duplicates:,}"
)


# ==========================================
# 9. Convert inbound column to TRUE/FALSE
# ==========================================

apple_df["inbound"] = (
    apple_df["inbound"]
    .astype(str)
    .str.upper()
    .eq("TRUE")
)


# ==========================================
# 10. Separate both directions
# ==========================================

customer_messages = apple_df[
    apple_df["inbound"] == True
]

apple_replies = apple_df[
    apple_df["inbound"] == False
]


# ==========================================
# 11. Sort by time
# ==========================================

apple_df["created_at"] = pd.to_datetime(
    apple_df["created_at"]
)

apple_df = apple_df.sort_values(
    "created_at"
)


# ==========================================
# 12. Display AppleSupport breakdown
# ==========================================

print("\n" + "=" * 50)
print("AppleSupport breakdown")
print("=" * 50)

print(
    f"Total AppleSupport conversation tweets : "
    f"{len(apple_df):,}"
)

print(
    f"Customer messages (TRUE)               : "
    f"{len(customer_messages):,}"
)

print(
    f"AppleSupport replies (FALSE)           : "
    f"{len(apple_replies):,}"
)


# ==========================================
# 13. Save AppleSupport dataset
# ==========================================

apple_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 14. Success message
# ==========================================

print("\n" + "=" * 50)
print("SUCCESS")
print("=" * 50)

print(
    "AppleSupport dataset saved to:"
)

print(OUTPUT_FILE)

print(
    f"\nFinal AppleSupport rows: "
    f"{len(apple_df):,}"
)