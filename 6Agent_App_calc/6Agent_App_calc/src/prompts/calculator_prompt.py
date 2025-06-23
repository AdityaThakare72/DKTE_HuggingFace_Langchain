from langchain_core.prompts import PromptTemplate

def get_calculator_prompt():
    return PromptTemplate(
        template="You are a calculator assistant. Please provide a mathematical expression to evaluate.",
        input_variables=["expression"]
    )