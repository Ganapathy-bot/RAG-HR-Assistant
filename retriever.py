from embedder import embed_chunks
from pinecone_db import get_index

def retrieve(query, top_k=5):
    index = get_index()
    query_embedding = embed_chunks([query])[0]
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [m["metadata"]["text"] for m in results["matches"]]
