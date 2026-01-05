import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def delete_index():
    """Delete the index if it exists. Use with caution!"""
    if INDEX_NAME in [i["name"] for i in pc.list_indexes()]:
        pc.delete_index(INDEX_NAME)
        print(f"Index {INDEX_NAME} deleted successfully")

def create_index():
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )

def get_index():
    return pc.Index(INDEX_NAME)
