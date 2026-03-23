# rag_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel, RunnableLambda
from schema import AnswerResponse
from dotenv import load_dotenv
from chromy import langchain_embeddings
import os
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
load_dotenv()



# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize parser
parser = PydanticOutputParser(pydantic_object=AnswerResponse)

# Create prompt template
prompt_template = """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, reply "I could not find the answer in the documents."

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

{format_instructions}

Provide your response as valid JSON matching the schema above. Extract source references from the metadata."""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

vectorstore=  Chroma(
  collection_name="documents",
  embedding_function=langchain_embeddings,
  persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(
  search_type="similarity",
  search_kwargs={"k":3}
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":6, "fetch_k":20, "lambda_mult":0.5}
)

memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=False
)

#optional high level function
""" 
document_chain = create_stuff_documents_chain(llm,prompt)

retrieval_chain = create_retrieval_chain(retriever,document_chain)
"""
def format_docs(docs):
    """Format retrieved documents into a single context string"""
    return "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}"
        for i, doc in enumerate(docs)
    ])


def hybrid_retrieve(question: str, top_k: int):
        """Combine similarity and MMR retrieval, then deduplicate and cap to top_k."""
        retriever.search_kwargs["k"] = max(top_k, 1)
        mmr_retriever.search_kwargs["k"] = max(top_k * 2, top_k)

        similarity_docs = retriever.invoke(question)
        mmr_docs = mmr_retriever.invoke(question)

        merged_docs = []
        seen = set()

        for doc in similarity_docs + mmr_docs:
                doc_key = (
                        doc.metadata.get("source"),
                        doc.metadata.get("page_number"),
                        doc.metadata.get("chunk_index"),
                        doc.page_content
                )
                if doc_key in seen:
                        continue
                seen.add(doc_key)
                merged_docs.append(doc)
                if len(merged_docs) >= top_k:
                        break

        return merged_docs


def load_chat_history(_: str) -> str:
        return memory.load_memory_variables({}).get("chat_history", "")


def build_rag_chain(top_k: int):
        return (
            RunnableParallel(
                {
                    "context": RunnableLambda(lambda question: format_docs(hybrid_retrieve(question, top_k))),
                    "question": RunnablePassthrough(),
                    "chat_history": RunnableLambda(load_chat_history)
                }
            )
            | prompt
            | llm
            | parser
        )
    

def retrieve_and_generate(question: str, top_k: int = 3) -> dict:
    """
    Use LangChain retrieval chain to get RAG response.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve (updates retriever)
        
    Returns:
        Dictionary containing success status, answer, sources, and retrieved documents
    """
    try:
        logger.info(f"Processing question: {question} with top_k={top_k}")

        retrieved_docs = hybrid_retrieve(question, top_k)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for question")
            return {
                "success": False,
                "error": "No relevant documents found in the database",
                "answer": "I could not find any relevant documents to answer your question.",
                "sources": [],
                "retrieved_documents": []
            }
        
        logger.info(f"Successfully retrieved {len(retrieved_docs)} documents")
        
        # Invoke RAG chain with hybrid retrieval and memory context
        response = build_rag_chain(top_k).invoke(question)
        
        if not isinstance(response, AnswerResponse):
            logger.error(f"Unexpected response type: {type(response)}")
            return {
                "success": False,
                "error": "LLM returned invalid format",
                "answer": str(response),
                "sources": [],
                "retrieved_documents": []
            }
        
        # Format metadata for response
        sources_metadata = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]
        
        logger.info(f"Generated answer with {len(response.answer)} characters")
        logger.info(f"Found {len(response.sources)} sources")

        memory.save_context(
            {"question": question},
            {"answer": response.answer}
        )
        
        return {
            "success": True,
            "answer": response.answer,
            "sources": response.sources,
            "retrieved_documents": sources_metadata
        }
      
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "answer": "An error occurred while processing your question.",
            "sources": [],
            "retrieved_documents": []
        }


def get_rag_response(question: str, top_k: int = 3) -> AnswerResponse:
    """
    Simplified wrapper that returns AnswerResponse directly.
    """
    result = retrieve_and_generate(question, top_k)
    
    if not result["success"]:
        return AnswerResponse(
            answer=result.get("answer", "I encountered an error processing your question."),
            sources=[]
        )
    
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )
