from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Get Huggingface API key from .env
load_dotenv()
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Huggingface model
# It is an Image + Text model
# repo_id = "llava-hf/llava-1.5-7b-hf"
# Test below for text only model
repo_id = "meta-llama/Llama-3.3-70B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=hf_key,
    provider="groq",
)

# Use model
response = llm.invoke("What is the capital of France?")
print(response)
