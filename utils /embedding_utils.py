import numpy as np
def embed_chunks(model, chunks):
    emb = model.encode(chunks, convert_to_tensor=False)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

def get_top_k_chunks(query, model, chunks, chunk_embeddings, k=3):
    query_emb = model.encode([query])[0]
    query_emb = query_emb / np.linalg.norm(query_emb)
    # protect shapes
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return []
    scores = np.dot(chunk_embeddings, query_emb)

    boost = np.array([3 if "preamble" in c.lower() else 1 for c in chunks])
    scores = scores * boost

    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]
