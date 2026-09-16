import pandas as pd

# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/apple_support_tagged.csv"
OUTPUT_FILE = "data/apple_support_knowledge_base.csv"


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv(INPUT_FILE)

print(f"Total tweets: {len(df):,}")
print(f"Total conversations: {df['conversation_id'].nunique():,}")


# ==========================================
# 2. Convert time
# ==========================================

df["created_at"] = pd.to_datetime(
    df["created_at"]
)


# ==========================================
# 3. Sort conversations chronologically
# ==========================================

df = df.sort_values(
    ["conversation_id", "created_at"]
)


# ==========================================
# 4. Build conversation text
# ==========================================

knowledge_base = []


for conversation_id, conversation in df.groupby(
    "conversation_id"
):

    messages = []

    for _, row in conversation.iterrows():

        if row["inbound"] == True:
            speaker = "Customer"
        else:
            speaker = "AppleSupport"

        text = str(row["text"]).strip()

        messages.append(
            f"{speaker}: {text}"
        )

    conversation_text = "\n".join(messages)

    knowledge_base.append({
        "conversation_id": conversation_id,
        "conversation": conversation_text
    })


# ==========================================
# 5. Create knowledge-base dataframe
# ==========================================

kb_df = pd.DataFrame(
    knowledge_base
)


# ==========================================
# 6. Save
# ==========================================

kb_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 7. Show result
# ==========================================

print("\n" + "=" * 50)
print("KNOWLEDGE BASE CREATED")
print("=" * 50)

print(
    f"Conversations: "
    f"{len(kb_df):,}"
)

print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)


# Show first conversation
print("\nExample conversation:")
print(kb_df.iloc[0]["conversation"])