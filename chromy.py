import chromadb
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-V2")

""" 
class SentenceTransformerEmbeddings(Embeddings):
  def __init__(self,model_name:str="all-MiniLM-L6-V2"):
    self.model = SentenceTransformer(model_name)
  
  def embed_documents(self, texts : List[str]) -> List[List[float]]:
    embeddings = self.model.encode(texts,convert_to_numpy=True)
    return embeddings.tolist()
  
  def embed_query(self, text:str) ->List[float]:
    embedding = self.model.encode([text],convert_to_numpy=True)
    return embedding[0].tolist()

"""

langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={'device':'cpu'},encode_kwargs={'normalize_embeddings':True})
 
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="documents",embedding_function=sentence_transformer_ef)