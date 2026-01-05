from embedder import embed_chunks
from pinecone_db import create_index, get_index
import uuid

def load_data():
    with open("data/hr_policy.txt", "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def run():
    create_index()
    index = get_index()
    text = load_data()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors)
    print("Data embedded & stored")

if __name__ == "__main__":
    run()
