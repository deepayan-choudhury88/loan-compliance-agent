import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load credentials
load_dotenv()

# 1. Initialize the LLM (The Brain)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.2, 
)

# 2. Load the Vector Database (The Memory)
print("Loading company policy memory...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Create the Prompt Template
prompt_template = """
You are a strict financial compliance agent. A loan has failed the minimum value requirement (must be > 25,000 EUR).
Use the provided company policy context to suggest an alternative asset the client can pledge to meet the requirements.

Company Policy Context:
{context}

Failed Loan Details:
{question}

Output a short, professional remediation recommendation.
"""
prompt = PromptTemplate.from_template(prompt_template)

# 4. Wire the LCEL Chain (Modern LangChain syntax)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute a test remediation for Loan L000005
failed_loan_data = """
LoanID: L000005
Current Asset: Office Equipment
Asset Value: 6,165.57 EUR
"""

print(f"\nProcessing failed loan: L000005...")
print("-" * 50)
response = rag_chain.invoke(failed_loan_data)
print("\nRemediation Strategy:")
print(response)
