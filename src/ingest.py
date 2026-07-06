import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import ollama

# are classes - helps reading files from a directory - separate classes for pdfs etc
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# used for chunking
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):

    print(f"Loading documents from {docs_path}...")

    # check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your files.")

    # load all .txt files from the docs directory
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt", # only look for .txt files
        loader_cls = TextLoader,
        loader_kwargs = {
            "encoding": "utf-8",
            "autodetect_encoding": True
        }
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    # for i, doc in enumerate(documents):
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1536, chunk_overlap=0):
    
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)

    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):

    print("Creating embeddings and storing in ChromaDB...")

    # Initialize the mxbai-embed-large model
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    
    return vectorstore

def main():

    # 1. loading the files
    documents = load_documents()

    # 2. chunking the files
    chunks = split_documents(documents)

    # 3. create embedding
    vector_store = create_vector_store(chunks)

    # 4. store vector embeddings in vector db



if __name__ == "__main__":
    main()