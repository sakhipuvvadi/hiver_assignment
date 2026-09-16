# AI Customer Support Knowledge Base & Live Support System

An AI-powered customer-support system built using historical
AppleSupport conversations from the Twitter Customer Support dataset.

The system converts historical customer-support conversations into a
semantic knowledge base, automatically discovers support intents, and
uses those conversations to answer new customer queries through a live
React + FastAPI application.

---

# 1. Project Overview

Customer-support teams receive many repetitive queries involving
common problems such as:

- iOS updates
- Apple ID issues
- Battery problems
- Wi-Fi problems
- App Store issues
- iCloud problems
- Apple Music problems
- Device bugs
- Requests for human support

Historical customer-support conversations contain useful information
about how these problems were handled.

The goal of this project is to transform those historical conversations
into an AI-powered support system.

The system has two major stages:

### Offline Knowledge-Building Stage

Historical AppleSupport conversations are:

1. Extracted from the dataset.
2. Reconstructed into complete conversations.
3. Converted into semantic embeddings.
4. Grouped into semantic clusters.
5. Given meaningful intent names using an LLM.
6. Stored as a searchable knowledge base.

### Online Support Stage

When a new customer submits a query:

1. The query is converted into an embedding.
2. Similar historical conversations are retrieved.
3. The closest conversation determines the support intent.
4. The retrieved conversations are given to an LLM.
5. The LLM generates a support response.
6. The system decides whether the query should be handled automatically
   or sent to a human.
7. The result is displayed in the React frontend.

---

# 2. System Architecture

The overall architecture is divided into two pipelines.

```text
                    ┌──────────────────────────────┐
                    │ Twitter Customer Support     │
                    │ Dataset                      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ AppleSupport Conversation    │
                    │ Extraction                   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Conversation Reconstruction   │
                    │ using response relationships  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Chronological Conversations  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ SentenceTransformer          │
                    │ all-MiniLM-L6-v2             │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Conversation Embeddings      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ KMeans Clustering             │
                    │ 20 Semantic Clusters          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ LLM Intent Naming             │
                    │ Intent + Reason                │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ AppleSupport Knowledge Base   │
                    └──────────────────────────────┘