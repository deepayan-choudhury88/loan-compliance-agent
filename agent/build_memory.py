import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# REMOVED: from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings # ADDED: Azure integration

# Load credentials from .env
load_dotenv()

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
    
    print("Generating embeddings via Azure and building FAISS database...")
    # 3. Convert text to vectors using Azure's text-embedding-3-small
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    )
    
    # 4. Create and save the vector database
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(DB_PATH)
    
    print(f"✅ Success! Embedded {len(docs)} chunks into the local vector database using Azure OpenAI.")

if __name__ == "__main__":
    build_vector_db()
