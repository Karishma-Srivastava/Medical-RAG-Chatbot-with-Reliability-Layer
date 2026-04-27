from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pathlib import Path

# ----------------------------
# Load environment
# ----------------------------
from dotenv import load_dotenv
from pathlib import Path
import os

# ALWAYS load .env from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("PINECONE_API_KEY")

print("ENV FILE USED:", env_path)
print("FINAL API KEY:", repr(api_key))
api_key = api_key.strip() if api_key else None

# ----------------------------
# Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=api_key)
index = pc.Index("medibot")

# ----------------------------
# Load embedding model (ONLY ONCE)
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_docs(query, top_k=20):

    # Encode query
    query_emb = model.encode(query).tolist()

    # Query Pinecone
    results = index.query(
        vector=query_emb,
        top_k=top_k * 2,
        include_metadata=True
    )

    print("PINECONE RAW RESULTS:", results)

    # Handle empty results
    if not results["matches"]:
        return []

    # ----------------------------
    # Extract documents and scores
    # ----------------------------
    dense_docs = [m["metadata"]["text"] for m in results["matches"]]
    dense_scores = [m["score"] for m in results["matches"]]

    print("DENSE DOCS (TOP 5):", dense_docs[:5])

    # ----------------------------
    # OPTIONAL: filter based on query keywords (AFTER extraction)
    # ----------------------------
    filtered = []
    for doc in dense_docs:
        if any(word in doc.lower() for word in query.lower().split()):
            filtered.append(doc)

    if filtered:
        dense_docs = filtered
        dense_scores = dense_scores[:len(filtered)]  # keep alignment

    # ----------------------------
    # FIX 1: Keyword scoring
    # ----------------------------
    def keyword_score(query, doc):
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        return len(q_words & d_words) / (len(q_words) + 1)

    combined = []
    for doc, d_score in zip(dense_docs, dense_scores):
        k_score = keyword_score(query, doc)
        final_score = 0.7 * d_score + 0.3 * k_score
        combined.append(final_score)

    # ----------------------------
    # FIX 2: Boost for "manage"
    # ----------------------------
    if "manage" in query or "control" in query:
        for i, doc in enumerate(dense_docs):
            if i >= len(combined):
                break
            if any(word in doc.lower() for word in ["diet", "exercise", "lifestyle"]):
                combined[i] += 0.1

    # ----------------------------
    # FIX 3: Filter low-quality docs
    # ----------------------------
    filtered_docs = []
    filtered_scores = []

    for doc, score in zip(dense_docs, combined):
        if score > 0.05:
            filtered_docs.append(doc)
            filtered_scores.append(score)

    dense_docs = filtered_docs
    combined = filtered_scores

    # ----------------------------
    # FIX 4: Query-aware boost
    # ----------------------------
    query_terms = query.lower().split()

    for i, doc in enumerate(dense_docs):
        if i >= len(combined):
            break
        if any(term in doc.lower() for term in query_terms):
            combined[i] += 0.05

    # ----------------------------
    # Final safety check
    # ----------------------------
    if not dense_docs or not combined:
        return []

    # ----------------------------
    # Final ranking
    # ----------------------------
    ranked = sorted(
        zip(dense_docs, combined),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked][:top_k]