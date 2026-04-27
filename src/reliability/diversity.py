from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def diversify(docs, threshold=0.85):
    selected = []
    emb = model.encode(docs)

    for i, d in enumerate(docs):
        if not selected:
            selected.append((d, emb[i]))
            continue

        if all(cosine_similarity([emb[i]], [e])[0][0] < threshold for _, e in selected):
            selected.append((d, emb[i]))

    return [d for d, _ in selected]