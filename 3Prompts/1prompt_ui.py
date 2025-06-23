from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt # Keep load_prompt if template.json is simple
import os

load_dotenv()

# --- Hugging Face Model Setup ---
# Verify HUGGINGFACEHUB_API_TOKEN is loaded
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("Hugging Face API Token not found. Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")
    st.stop() # Stop the Streamlit app if token is missing

# Initialize Hugging Face Endpoint
# Using Zephyr as it worked for you.
# Make sure this model is suitable for your use case and has a good chat interface.
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation", # Even for chat models, 'text-generation' is often the task type
    huggingfacehub_api_token=hf_token,
    temperature=0.7, # Adjust as needed
    max_new_tokens=500 # Adjust output length as needed
)

# Wrap the HuggingFaceEndpoint with ChatHuggingFace
model = ChatHuggingFace(llm=llm_endpoint)

# --- Streamlit UI ---
st.set_page_config(layout="wide") # Optional: makes the app wider
st.title('Research Paper Explanation Tool') # Changed header to title

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# --- Prompt Loading ---
# Assuming 'template.json' is a simple PromptTemplate JSON.
# If it's a ChatPromptTemplate or more complex, you might need to adjust.
try:
    template_prompt = load_prompt('/home/aditya/dktegenai/d4langchain/3Prompts/template.json')
except FileNotFoundError:
    st.error("template.json not found. Please make sure it's in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading prompt from template.json: {e}")
    st.stop()

# --- Chain and Execution ---
if st.button('Summarize'):
    with st.spinner('Generating explanation...'):
        try:
            # The template is directly piped to the ChatHuggingFace model
            chain = template_prompt | model
            result = chain.invoke({
                'paper_input': paper_input,
                'style_input': style_input,
                'length_input': length_input
            })
            st.markdown(result.content) # Use markdown to render content
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            st.info("Please try again or check your model configuration/API token.")