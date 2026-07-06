import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ollama

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
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

# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}: \n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# create Ollama model
model = OllamaLLM(model="llama3.2:3b")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
print("Response from LLM:")
print(result)

# More Questions

# 1. How much did Microsoft pay to acquire GitHub?
# 2. In what year did Tesla launch the Roadster?