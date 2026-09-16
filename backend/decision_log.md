# Decision Log

## 1. Conversation reconstruction using both response fields

The dataset contains both `response_tweet_id` and
`in_response_to_tweet_id`. I used both fields when connecting tweets
so that conversations could be reconstructed even when one direction
of the relationship was missing.

## 2. Preserve both customer and AppleSupport messages

I retained both inbound customer messages and outbound AppleSupport
messages. This preserves the complete support interaction instead of
building a knowledge base only from customer questions.

## 3. Chronological ordering within conversations

Messages were sorted by `conversation_id` and `created_at`. This was
necessary because the order of messages is important for understanding
the context of a support interaction.

## 4. Conversation-level clustering

I clustered complete conversations rather than individual tweets.
This gives the embedding model more context about the customer's
problem and the corresponding support interaction.

## 5. SentenceTransformer embeddings

I used `all-MiniLM-L6-v2` to represent conversations semantically.
This was selected because semantic similarity is more suitable for
support queries than exact keyword matching.

## 6. Twenty clusters

I selected 20 clusters to create a manageable number of support
categories while still separating different types of Apple support
issues.

## 7. Representative examples for intent naming

Instead of sending all conversations in a cluster to the LLM, I used
representative conversations close to the cluster centroid. This
reduced token usage while retaining examples that represented the
cluster.

## 8. Short intent names

Intent names were constrained to short 2–5 word labels such as
`IOS_UPDATE_PROBLEM` and `APPLE_ID_LOGIN`. This makes the intents easier
to display, evaluate, and use in the live application.

## 9. Cluster labels taken from the intent mapping

The live system uses the cluster and intent mapping stored in
`apple_support_intents.csv`. This avoids relying on separately fitted
KMeans labels whose numeric cluster IDs can differ between model fits.

## 10. Semantic retrieval for live queries

For a new query, the system retrieves the most semantically similar
historical conversations using SentenceTransformer embeddings rather
than directly predicting a KMeans cluster.

## 11. Top-3 retrieved conversations

The live generator receives the top three retrieved conversations.
This provides multiple examples of historical support behavior rather
than relying on a single potentially noisy match.

## 12. LLM-based response generation

The retrieved conversations and predicted intent are provided to an
LLM to generate the final support response. This allows the system to
produce natural-language answers rather than returning a stored
response verbatim.

## 13. Automatic vs human handling

The live system also produces an `AUTOMATIC` or `HUMAN` handling
decision. This separates routine support responses from cases where
human assistance may be more appropriate.

## 14. Leakage-safe retrieval evaluation

During semantic evaluation, the exact evaluation conversation was
excluded from retrieval. This prevents the system from simply
retrieving the same conversation that generated the evaluation query.

## 15. Separate response-quality evaluation

Intent accuracy alone does not measure whether a generated answer is
actually useful. Therefore, generated responses were separately
evaluated for correctness, relevance, helpfulness, grounding, and
overall quality using an LLM judge.