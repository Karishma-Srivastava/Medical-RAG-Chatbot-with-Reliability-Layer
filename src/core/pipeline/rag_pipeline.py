from src.core.retrieval.hybrid_retriever import retrieve_docs
from src.core.reranking.cross_encoder import rerank
from src.core.generator.llm_generator import generate_answer

from src.reliability.diversity import diversify
from src.reliability.confidence import compute_confidence
from src.reliability.retry import should_retry

from src.optimization.caching import get_from_cache, save_to_cache
from src.utils.logger import log
from textblob import TextBlob

from symspellpy import SymSpell, Verbosity

from sklearn.metrics.pairwise import cosine_similarity

from src.shared.models import embedding_model

def filter_relevant(query, docs, embedding_model, threshold=0.45):
    if not docs:
        return []

    try:
        q_emb = embedding_model.encode([query])
        d_emb = embedding_model.encode(docs)

        sims = cosine_similarity(q_emb, d_emb)[0]

        filtered = [doc for doc, sim in zip(docs, sims) if sim > threshold]

        return filtered if filtered else docs[:2]

    except Exception as e:
        print("ERROR in filter:", e)
        return docs[:2]

def correct_query(query):
    suggestions = sym_spell.lookup(query, Verbosity.CLOSEST)
    return suggestions[0].term if suggestions else query

def correct_query(query):
    return str(TextBlob(query).correct())


    
# ----------------------------
# Query Rewriting
# ----------------------------

def rewrite_query(query):
    if "manage" in query or "how to" in query:
        return query + " treatment lifestyle diet exercise control"
    elif "what is" in query:
        return query + " definition explanation"
    else:
        return query


# ----------------------------
# RAG Pipeline
# ----------------------------
def rag_pipeline(query):

    # Cache check
    cached = get_from_cache(query)
    if cached:
        return cached

    # Query rewrite
    q = rewrite_query(query)

    # Retrieval
    docs = retrieve_docs(q, top_k=20)

    if not docs:
        print("⚠️ NO DOCUMENTS RETRIEVED")
        return {
            "query": query,
            "response": "No relevant documents found. Please refine your query.",
            "confidence": 0.0
        }

    # Reranking + diversity
    docs = rerank(q, docs)

    # 🔥 SAFETY CHECK
    if not docs:
        return {
            "query": query,
            "response": "I don't know",
            "confidence": 0.0
        }

    # 🔥 SAFE FILTER
    try:
        docs = filter_relevant(q, docs, embedding_model)
    except Exception as e:
        print("FILTER ERROR:", e)

    docs = docs[:3]
    
    docs = [d for d in docs if len(d.split()) > 30]
    docs = docs[:3]
  
    
    docs = diversify(docs)

    print("TOP DOCS:", docs[:3])

    # Dynamic context size
    docs = docs[:3]  # after rerank + filter
    context = "\n\n".join(docs)

    # Generate answer
    response = generate_answer(q, context)

    # Confidence
    confidence = compute_confidence(q, response, docs)

    # Low confidence guard
    if confidence < 0.3:
        return {
            "query": query,
            "response": "Query unclear or misspelled. Please rephrase.",
            "confidence": confidence
        }

    # Retry mechanism
    if should_retry(confidence, query):
        docs = retrieve_docs(q + " detailed", top_k=40)
        docs = rerank(q, docs)[:5]
    

        context = "\n\n".join(docs[:5])
        response = generate_answer(q, context)
        confidence = compute_confidence(q, response, docs)

    # Final result
    result = {
        "query": query,
        "response": response,
        "confidence": confidence,
        "sources": docs[:2]
    }

    log(result)
    save_to_cache(query, result)

    return result