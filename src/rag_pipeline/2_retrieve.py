import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ollama

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding_model, 
    collection_metadata={"hnsw:space": "cosine"} 
)

# query the document
query = "In what year did Tesla launch the Roadster?"

# creating the retriever component - getting top k (5) chunks
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)

# getting the top 5 relevant chunks
relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")

# More Questions

# 1. How much did Microsoft pay to acquire GitHub?
# 2. In what year did Tesla begin production of the Roadster?