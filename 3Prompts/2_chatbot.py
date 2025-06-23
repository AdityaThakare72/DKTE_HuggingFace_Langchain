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
    task="text-generation", # 'text-generation' is appropriate for chat models on HF endpoint
    huggingfacehub_api_token=hf_token,
    temperature=0.7,      # Adjust as needed for creativity vs. focus
    max_new_tokens=500    # Max tokens for the AI's response
)

# Wrap the HuggingFaceEndpoint with ChatHuggingFace
model = ChatHuggingFace(llm=llm_endpoint)

# --- Chat Logic ---
chat_history = [
    SystemMessage(content='You are a helpful AI assistant.')
    # Zephyr's chat template often starts with a system message,
    # then alternates user/assistant.
]

print("Type 'exit' to end the chat.")

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit': # Use .lower() for case-insensitive exit
        break

    chat_history.append(HumanMessage(content=user_input))

    try:
        # Pass the entire chat_history list to the model
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI: ", result.content)
    except Exception as e:
        print(f"An error occurred during AI response generation: {e}")
        print("Please check your model configuration, API token, or network connectivity.")
        # Optionally, you might want to remove the last HumanMessage if the API call failed
        # to prevent it from being included in the next valid request.
        chat_history.pop()
        continue # Continue to next loop iteration for new user input

print("\n--- Chat Ended ---")
print("Final Chat History:")
for message in chat_history:
    if isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")