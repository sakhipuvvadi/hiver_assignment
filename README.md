# AI-Powered Customer Support System

An AI-powered customer support system built using the **Twitter Customer Support (TWCS) dataset**, focusing on AppleSupport conversations.

The system combines **conversation reconstruction, semantic embeddings, clustering, automatic intent discovery, semantic retrieval, LLM-based response generation, and automatic/human escalation** to provide support for new customer queries.

---

## Project Overview

Customer-support conversations often contain repeated questions and recurring support issues. This project uses historical AppleSupport conversations to build a knowledge base that can understand a new customer query and generate an appropriate support response.

### The system performs four major tasks:

1. **Intent Detection**
   - Identifies the type of problem in a new customer query.
   - Maps the query to one of the discovered support intents.

2. **Knowledge Retrieval**
   - Searches historical AppleSupport conversations using semantic similarity.
   - Retrieves the most relevant previous conversations.

3. **Response Generation**
   - Uses retrieved conversations and the detected intent as context.
   - Generates a customer-support response using an LLM.

4. **Handling Decision**
   - Determines whether the query can be handled automatically.
   - Returns either `AUTOMATIC` or `HUMAN`.

---
# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/sakhipuvvadi/hiver.git
cd hiver
```

---

#  2. Backend Setup

Go to the backend directory:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---


#  3. Configure the Groq API Key

Inside the `backend` folder, create:

```text
.env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace:

```text
your_groq_api_key_here
```

with your own Groq API key.

### Important

Never commit your API key to GitHub.

The `.env` file is excluded using `.gitignore`.

---

#  4. Start the Backend

Make sure the virtual environment is activated.

From:

```text
hiver/backend
```

run:

```bash
uvicorn main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

You can also check:

```text
http://127.0.0.1:8000/
```

---

#  5. Frontend Setup

Open a **new terminal**.

Go to the frontend:

```bash
cd hiver/frontend
```

Install Node dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The frontend will normally run at:

```text
http://127.0.0.1:5173
```

Open the URL shown in the terminal.

---

#  Running the Complete Application

You need two terminals.

### Terminal 1 — Backend

```bash
cd hiver/backend
venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

### Terminal 2 — Frontend

```bash
cd hiver/frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

---

# System Architecture

```text
                    ┌──────────────────────────┐
                    │   TWCS Dataset           │
                    │ Twitter Customer Support │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Processing           │
                    │                          │
                    │ • Filter AppleSupport    │
                    │ • Link conversations      │
                    │ • Preserve both speakers  │
                    │ • Sort chronologically    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Knowledge Base            │
                    │                          │
                    │ One row = one conversation│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ SentenceTransformer      │
                    │ all-MiniLM-L6-v2         │
                    │                          │
                    │ Conversation embeddings  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ K-Means Clustering       │
                    │                          │
                    │ 20 clusters              │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Automatic Intent Naming  │
                    │                          │
                    │ Qwen via Groq API        │
                    └──────────────────────────┘


                    LIVE QUERY
                        │
                        ▼
              ┌──────────────────────┐
              │ SentenceTransformer │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Semantic Retrieval   │
              │ Top-3 conversations  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Intent Mapping       │
              │                      │
              │ Cluster + Intent     │
              │ + Reason             │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Qwen LLM             │
              │ Response Generation  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ FastAPI Backend      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ React Frontend       │
              │                      │
              │ • Cluster            │
              │ • Intent             │
              │ • Support Response   │
              │ • Handling           │
              └──────────────────────┘
```

---

#  Dataset

The project uses the **Twitter Customer Support (TWCS)** dataset.

The AppleSupport subset is extracted from the original dataset.

Important fields include:

| Field | Description |
|---|---|
| `tweet_id` | Unique tweet identifier |
| `author_id` | Author of the tweet |
| `inbound` | Indicates customer/support direction |
| `created_at` | Timestamp |
| `text` | Tweet text |
| `response_tweet_id` | IDs of responses |
| `in_response_to_tweet_id` | Parent tweet ID |


# 🔗 Conversation Reconstruction

Tweets are connected using both:

- `in_response_to_tweet_id`
- `response_tweet_id`

This allows the system to reconstruct conversations even when the relationship is represented from either direction.

Each connected group is assigned a `conversation_id`.

Messages inside each conversation are then sorted chronologically.

Example:

```text
Customer: My iPhone battery is draining very quickly.

AppleSupport: We'd like to help with that. Please send us a DM.

Customer: Sure, I'll message you now.

AppleSupport: Thanks. We'll continue troubleshooting there.
```

This conversation-level representation is used to build the knowledge base.

---

# Knowledge Base

The reconstructed conversations are stored as:

```text
apple_support_knowledge_base.csv
```

Each row represents one complete conversation.

The conversation contains both sides of the interaction:

```text
Customer: ...
AppleSupport: ...
Customer: ...
AppleSupport: ...
```

The final processed dataset contains approximately **81,856 conversations**.

---

#  Semantic Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

from SentenceTransformers.

Each complete conversation is converted into a vector representation.

Embeddings are normalized so that dot-product similarity corresponds to cosine similarity.

The resulting embeddings are stored locally as:

```text
backend/models/conversation_embeddings.npy
```

---

#  Clustering

K-Means clustering is used to organize conversations into:

```text
20 clusters
```

The clustering is performed on the conversation embeddings.

Representative conversations close to each cluster centroid are selected to understand what each cluster represents.

---

# Automatic Intent Discovery

Instead of manually naming every cluster, representative conversations are sent to a Qwen model through the Groq API.

The model generates:

```json
{
  "intent": "SHORT_INTENT_NAME",
  "reason": "Explanation of the type of customer issue represented by the cluster."
}
```

Intent names are designed to be short and descriptive.

Examples include:

```text
IOS_UPDATE_PROBLEM
APPLE_ID_LOGIN
APPLE_ID_PASSWORD_RESET
APP_STORE_PAYMENT
ICLOUD_SYNC_PROBLEM
IPHONE_BATTERY_PROBLEM
APPLE_MUSIC_PROBLEM
```

The final intent mapping is stored in:

```text
apple_support_intents.csv
```

---

#  Live Query Processing

When a customer sends a new query:

```text
"My iPhone battery is draining very quickly"
```

the following pipeline is executed:

### Step 1 — Query embedding

The query is converted into an embedding using:

```text
all-MiniLM-L6-v2
```

### Step 2 — Semantic retrieval

The query embedding is compared against the stored conversation embeddings.

The most similar conversations are retrieved.

The system uses the **top 3 conversations** as context.

### Step 3 — Intent selection

The best matching conversation provides:

- Cluster
- Intent
- Intent reason

The system uses the cluster/intent mapping from the processed intent dataset.

### Step 4 — Response generation

The query, detected intent, intent reason, and retrieved conversations are passed to the Qwen model.

The model generates the final support response.

### Step 5 — Handling decision

The model also determines whether the request should be:

```text
AUTOMATIC
```

or

```text
HUMAN
```

### Step 6 — API response

The FastAPI backend returns:

```json
{
  "cluster": 16,
  "intent": "APPLE_WATCH_ISSUE",
  "reason": "...",
  "response": "...",
  "handling": "AUTOMATIC"
}
```

The frontend displays the relevant information to the user.

---

#  Technology Stack

## Frontend

- React
- Vite
- JavaScript
- CSS

## Backend

- Python
- FastAPI
- Uvicorn

## Machine Learning / NLP

- SentenceTransformers
- `all-MiniLM-L6-v2`
- Scikit-learn
- K-Means clustering

## LLM

- Qwen
- Groq API

## Data Processing

- Pandas
- NumPy


``

