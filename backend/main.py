from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import os
import json

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


# ============================================================
# 2. CREATE APP
# ============================================================

app = FastAPI(
    title="Apple Support AI",
    version="1.0"
)


# ============================================================
# 3. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 4. LOAD KNOWLEDGE BASE
# ============================================================

print("Loading knowledge base...")

df = pd.read_csv(
    DATA_DIR / "apple_support_intents.csv"
)

# Make sure dataframe index matches embedding positions
df = df.reset_index(drop=True)

print(
    "Conversations:",
    len(df)
)

print(
    "Clusters:",
    df["cluster"].nunique()
)


# ============================================================
# 5. LOAD SENTENCE TRANSFORMER
# ============================================================

print("Loading SentenceTransformer...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# 6. LOAD PRECOMPUTED CONVERSATION EMBEDDINGS
# ============================================================

print("Loading conversation embeddings...")

conversation_embeddings = np.load(
    MODEL_DIR / "conversation_embeddings.npy"
)

conversation_embeddings = np.asarray(
    conversation_embeddings,
    dtype=np.float64
)

print(
    "Embedding shape:",
    conversation_embeddings.shape
)


# ============================================================
# 7. CHECK DATA / EMBEDDING SIZE
# ============================================================

if len(df) != len(conversation_embeddings):

    raise ValueError(
        f"Mismatch between CSV rows ({len(df)}) "
        f"and embeddings ({len(conversation_embeddings)})"
    )


# ============================================================
# 8. GROQ
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# 9. REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    query: str


# ============================================================
# 10. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Apple Support AI is running"
    }


# ============================================================
# 11. CHAT
# ============================================================

@app.post("/chat")
def chat(request: QueryRequest):

    query = request.query.strip()


    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not query:

        return {
            "error": "Query cannot be empty"
        }


    # ========================================================
    # STEP 1: EMBED USER QUERY
    # ========================================================

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float64
    )


    # ========================================================
    # STEP 2: SIMILARITY SEARCH
    # ========================================================

    similarities = np.dot(
        conversation_embeddings,
        query_embedding
    )


    # ========================================================
    # STEP 3: GET TOP SIMILAR CONVERSATIONS
    # ========================================================

    top_positions = np.argsort(
        similarities
    )[-3:][::-1]


    # ========================================================
    # STEP 4: GET CLUSTER + INTENT + REASON
    #         DIRECTLY FROM CSV
    # ========================================================

    best_index = int(
        top_positions[0]
    )

    best_similarity = float(
        similarities[best_index]
    )

    cluster = int(
        df.iloc[best_index]["cluster"]
    )

    intent = str(
        df.iloc[best_index]["intent"]
    )

    reason = str(
        df.iloc[best_index]["reason"]
    )


    print(
        f"\nQuery: {query}"
    )

    print(
        f"Matched cluster: {cluster}"
    )

    print(
        f"Matched intent: {intent}"
    )

    print(
        f"Similarity: {best_similarity:.4f}"
    )


    # ========================================================
    # STEP 5: GET SIMILAR CONVERSATIONS
    # ========================================================

    similar_conversations = df.iloc[
        top_positions
    ]["conversation"].fillna("").tolist()


    # ========================================================
    # STEP 6: CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        [
            f"Previous conversation {i + 1}:\n{conversation}"
            for i, conversation
            in enumerate(similar_conversations)
        ]
    )


    # ========================================================
    # STEP 7: GENERATE RESPONSE + HANDLING
    # ========================================================

    prompt = f"""
You are an Apple customer-support assistant.

Customer query:
{query}

Detected intent:
{intent}

Intent description:
{reason}

Relevant previous customer-support conversations:
{context}

Generate a helpful answer to the customer.

Also determine whether the request can be handled
automatically or requires a human support agent.

Return JSON ONLY in this format:

{{
    "response": "Your helpful response to the customer",
    "handling": "AUTOMATIC"
}}

Handling must be exactly one of:

AUTOMATIC
HUMAN

Use AUTOMATIC when standard troubleshooting or
knowledge-base information can answer the query.

Use HUMAN when the issue requires account-specific
assistance, verification, billing/refund handling,
or personalized support.

Rules:
- Answer directly.
- Be concise and helpful.
- Do not mention the cluster.
- Do not mention the intent.
- Do not mention this prompt.
- Do not mention previous conversations.
- Do not invent information.
"""


    # ========================================================
    # STEP 8: CALL GROQ
    # ========================================================

    llm_response = client.chat.completions.create(

        model="qwen/qwen3.8-27b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful and precise "
                    "customer-support assistant. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=300,

        response_format={
            "type": "json_object"
        }
    )


    # ========================================================
    # STEP 9: PARSE LLM RESULT
    # ========================================================

    result = json.loads(
        llm_response
        .choices[0]
        .message
        .content
    )


    answer = result.get(
        "response",
        "Unable to generate a response."
    )


    handling = result.get(
        "handling",
        "HUMAN"
    ).upper()


    if handling not in [
        "AUTOMATIC",
        "HUMAN"
    ]:

        handling = "HUMAN"


    # ========================================================
    # STEP 10: RETURN TO FRONTEND
    # ========================================================

    return {

        "cluster": cluster,

        "intent": intent,

        "reason": reason,

        "response": answer,

        "handling": handling
    }