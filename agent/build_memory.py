from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Define paths
PDF_PATH = "../data/company-reference.pdf"
DB_PATH = "faiss_index"

def build_vector_db():
    print(f"Loading document from {PDF_PATH}...")
    
    # 1. Load the PDF
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find {PDF_PATH}. Please ensure the file is in the data folder.")
        return
        
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    
    print("Chunking text...")
    # 2. Slice the document into smaller chunks the LLM can easily read
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(pages)
    
    print("Generating embeddings and building FAISS database (this may take a few seconds)...")
    # 3. Convert text to vectors using a fast, local open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Create and save the vector database
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(DB_PATH)
    
    print(f"Success! Embedded {len(docs)} chunks into the local vector database.")

if __name__ == "__main__":
    build_vector_db()
