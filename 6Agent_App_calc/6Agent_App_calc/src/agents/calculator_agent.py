from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from src.tools.calculator_tool import CalculatorTool

CALCULATOR_PROMPT = """You are a calculator agent that helps with mathematical calculations.
You have access to these tools: {tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the mathematical expression to calculate
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

class CalculatorAgent:
    def __init__(self, model_name: str = "mistral"):
        self.llm = ChatOllama(model=model_name)
        self.tools = [CalculatorTool()]
        
        # Create prompt with all required variables
        prompt = PromptTemplate(
            template=CALCULATOR_PROMPT,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create the agent
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )

    def run(self, query: str) -> str:
        return self.agent_executor.invoke({"input": query})["output"]