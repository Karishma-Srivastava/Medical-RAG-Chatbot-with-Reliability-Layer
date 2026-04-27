from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# Load env
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("PINECONE_API_KEY")

print("ENV FILE USED:", env_path)
print("API KEY USED:", repr(api_key))
# Init Pinecone
pc = Pinecone(api_key=api_key)

index_name = "medibot"

# 🔥 ADD THIS BLOCK (DELETE OLD INDEX)
existing_indexes = [i.name for i in pc.list_indexes()]

if index_name in existing_indexes:
    pc.delete_index(index_name)
    print("Old index deleted")

# CREATE FRESH INDEX
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index(index_name)

# 🔥 LOAD PDF DATA
pdf_path = r"C:\Medical RAG Chatbot with Evaluation & Reliability Layer\Medical-Chatbot-With-LLMs-Langchain-Pinecone-API\data\Medical.pdf"   # <-- CHANGE THIS PATH

reader = PdfReader(pdf_path)

texts = []

for page in reader.pages:
    content = page.extract_text()
    if content:
        chunks = chunk_text(content)
        texts.extend(chunks)

print("TOTAL CHUNKS CREATED:", len(texts))
print("TOTAL PAGES LOADED:", len(texts))

# 🔥 LOAD MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 CREATE EMBEDDINGS
embeddings = model.encode(texts).tolist()

# 🔥 UPSERT INTO PINECONE
vectors = [
    {
        "id": str(i),
        "values": embeddings[i],
        "metadata": {"text": texts[i]}
    }
    for i in range(len(texts))
]

batch_size = 50   # safe size

for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(batch)
    print(f"Uploaded batch {i // batch_size + 1}")

# VERIFY
print("INDEX STATS:", index.describe_index_stats())