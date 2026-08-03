import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load credentials
load_dotenv()

# 1. Initialize the LLM (The Brain)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.1, # Keep it highly strict for compliance
)

# 2. Load the Vector Database (The Memory)
print("Loading company policy memory...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 3. Create the Smart Prompt Template
prompt_template = """
You are a strict financial compliance AI agent.
A loan has failed deterministic compliance checks. 

Rule Engine Violations:
{violations}

Failed Loan Details:
{loan_details}

Company Policy Context (for alternative assets):
{context}

Provide a remediation strategy based strictly on these guidelines:
- If Rule 1 Failed: Recommend removing the loan from the report. Do not suggest assets.
- If Rule 2 Failed: Recommend changing the reported currency to the expected HQ currency.
- If Rule 3 Failed: Use the Company Policy Context to suggest an alternative asset (owned by the company or related entities) that equals or exceeds 50% of the loan value. If no such asset exists, explicitly state that no adequate collateral is available.

Format as a professional, concise summary. If there are multiple violations, address each one clearly.
"""
prompt = PromptTemplate.from_template(prompt_template)

# 4. Wire the LCEL Chain with Dynamic Company Searching
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": itemgetter("company_name") | retriever | format_docs,
        "loan_details": itemgetter("loan_details"),
        "violations": itemgetter("violations")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute a test remediation for Loan L000005 (Rule 1 & 2 Failure)
print("\nProcessing Loan L000005...")
print("-" * 50)
test_loan_5 = {
    "company_name": "Copperfield Timber Ltd",
    "loan_details": "LoanID: L000005 | Value: 12,000 | Currency: USD | Expected: EUR",
    "violations": "Rule 1 Failed: Loan value is 25,000 EUR or less\nRule 2 Failed: Currency does not match HQ expected currency"
}
print(rag_chain.invoke(test_loan_5))

# 6. Execute a test remediation for Loan L000010 (Rule 3 Failure)
print("\nProcessing Loan L000010...")
print("-" * 50)
test_loan_10 = {
    "company_name": "Solaris Timber Ltd",
    "loan_details": "LoanID: L000010 | Loan Value: 1,000,000 EUR | Current Asset: 200,000 EUR",
    "violations": "Rule 3 Failed: Asset value is less than 50% of the loan value"
}
print(rag_chain.invoke(test_loan_10))
