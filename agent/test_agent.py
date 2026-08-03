import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Load credentials from the .env file
load_dotenv()

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
)

# Test the connection
print("Pinging Azure OpenAI...")
message = HumanMessage(content="You are a strict financial compliance agent. Acknowledge your role in one sentence.")
response = llm.invoke([message])

print("\nResponse from Agent:")
print(response.content)
