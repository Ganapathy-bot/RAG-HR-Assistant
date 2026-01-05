import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def ask_hr(question):
    contexts = retrieve(question)
    prompt = f"""You are an HR assistant.
Context:
{chr(10).join(contexts)}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        extra_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG HR Assistant",
        }
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        q = input("Ask HR: ")
        if q.lower() == "exit":
            break
        print(ask_hr(q))
