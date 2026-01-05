import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

EMBED_MODEL = "text-embedding-3-large"

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk,
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "RAG HR Assistant",
            },
        )
        embeddings.append(response.data[0].embedding)
    return embeddings
