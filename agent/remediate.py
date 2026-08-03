import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from operator import itemgetter
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load credentials
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

print("Loading company policy memory & AI components...")

# 1. This uses the gpt-5.4-mini model via the env variable
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.1,
)

# 2. UPDATED: Now using Azure's text-embedding-3-small instead of local HuggingFace
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
)

# 3. Load FAISS (WARNING: See Step 3 below)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

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

@app.route("/api/remediate", methods=["POST"])
def remediate():
    data = request.json
    if not data:
        return jsonify({"error": "No payload provided"}), 400
    
    try:
        print(f"Processing AI remediation for {data.get('company_name')}...")
        response = rag_chain.invoke(data)
        return jsonify({"remediation": response})
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting AI Remediation API on http://localhost:5000 ...")
    app.run(port=5000, debug=True)
