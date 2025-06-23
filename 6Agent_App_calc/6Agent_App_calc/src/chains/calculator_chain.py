from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agents.calculator_agent import CalculatorAgent
from src.prompts.calculator_prompt import get_calculator_prompt

class CalculatorChain:
    def __init__(self, calculator_agent: CalculatorAgent):
        self.calculator_agent = calculator_agent

    def run_chain(self, user_input: str) -> str:
        return self.calculator_agent.run(user_input)