from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# --- Hugging Face Model Setup ---
# Verify HUGGINGFACEHUB_API_TOKEN is loaded
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("Error: Hugging Face API Token not found. Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")
    exit() # Exit if token is not found

# Initialize Hugging Face Endpoint
# Using Zephyr as it worked for you.
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation", # 'text-generation' is the common task for chat models via HF endpoint
    huggingfacehub_api_token=hf_token,
    temperature=0.7,      # Adjust as needed for creativity vs. focus
    max_new_tokens=500    # Max tokens for the AI's response
)

# Wrap the HuggingFaceEndpoint with ChatHuggingFace
model = ChatHuggingFace(llm=llm_endpoint)

# --- Initial Messages ---
# LangChain's ChatHuggingFace handles SystemMessage, HumanMessage, AIMessage seamlessly.
messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

# --- Invoke the model with the initial messages ---
try:
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content)) # Append AI's response to messages
    print(f"AI: {result.content}") # Print AI's response
except Exception as e:
    print(f"An error occurred during AI response generation: {e}")
    print("Please check your model configuration, API token, or network connectivity.")

print("\n--- Full Message History ---")
# Print the entire conversation history, including the initial prompt and AI's response
for message in messages:
    if isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")