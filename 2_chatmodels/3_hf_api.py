# can do both api and local inference

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Verify token loading
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") # This is what it should be looking for
if hf_token:
    print("Hugging Face API Token loaded successfully.")
else:
    print("Error: Hugging Face API Token not found in environment variables.")
    # Exit or raise an error if the token is critical
    exit()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", # Let's try with TinyLlama again now that token is fixed
    task="text-generation",
    huggingfacehub_api_token=hf_token # Explicitly pass the token (optional but good for clarity)
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of India")
    print(result.content)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your Hugging Face API token and model availability.")