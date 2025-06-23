from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from src.agents.calculator_agent import CalculatorAgent
from src.chains.calculator_chain import CalculatorChain

load_dotenv()

def main():
    # Initialize the local Ollama Mistral model
    model = ChatOllama(model="mistral")

    # Set up the calculator agent
    calculator_agent = CalculatorAgent(model_name="mistral")

    # Set up the calculator chain
    calculator_chain = CalculatorChain(calculator_agent)

    print("Welcome to the Calculator App!")
    while True:
        user_input = input("Enter a calculation (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        result = calculator_chain.run_chain(user_input)
        print(f"Result: {result}")

if __name__ == "__main__":
    main()