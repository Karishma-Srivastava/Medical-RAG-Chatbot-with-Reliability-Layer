from fastapi import FastAPI
from pydantic import BaseModel

from src.core.pipeline.rag_pipeline import rag_pipeline

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(req: QueryRequest):
    result = rag_pipeline(req.query)
    return result