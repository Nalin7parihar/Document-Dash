from pydantic import BaseModel,Field
from typing import List, Optional

class QueryRequest(BaseModel):
  question : str
  top_k : int = 3

class Source(BaseModel):
    text: str
    metadata: dict


class Match(Source):
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    retrieved_documents: List[Source] = []
    top_k: int = 3
  
class AnswerResponse(BaseModel):
  answer: str = Field(description="The generated answer to the question")
  sources: list[str] = Field(
      description="List of sources used to generate the answer"
  )
    
  class Config:
      json_schema_extra = {
          "example": {
              "answer": "The capital of France is Paris.",
              "sources": ["Document 1: Geography textbook", "Document 2: Encyclopedia"]
          }
      }