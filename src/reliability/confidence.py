from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_confidence(query, response, docs):
    print("CONFIDENCE DOCS:", docs)
    if not docs:
        return 0.0

    q = model.encode([query])
    r = model.encode([response])
    d = model.encode(docs)

    retrieval = np.mean(cosine_similarity(q, d))
    grounding = np.max(cosine_similarity(r, d))

    return float(0.3 * retrieval + 0.7 * grounding)