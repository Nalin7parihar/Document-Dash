from fastapi import APIRouter, Query, status, HTTPException
from typing import Optional
from chromy import collection
from schema import QueryResponse, QueryRequest
from llmservice import retrieve_and_generate

router = APIRouter(tags=["qna"], prefix="/qna")


@router.post("/", status_code=status.HTTP_200_OK, response_model=QueryResponse)
async def get_detailed_answer(query: QueryRequest):
    """
    Detailed endpoint - returns both the LLM answer and retrieved documents.
    """
    if not query.question or not query.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    if query.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0"
        )
    
    try:
        # Use RAG chain with full details
        result = retrieve_and_generate(query.question, query.top_k)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Unknown error occurred")
            )
        
        # Add question and top_k to the response
        return {
            "question": query.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_documents": result["retrieved_documents"],
            "top_k": query.top_k
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )
  