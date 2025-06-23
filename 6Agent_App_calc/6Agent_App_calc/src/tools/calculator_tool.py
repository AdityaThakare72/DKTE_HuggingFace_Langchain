from typing import Optional
from langchain.tools import BaseTool
import numexpr

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Useful for performing mathematical calculations. Input should be a mathematical expression."
    return_direct: bool = True

    def _run(self, query: str) -> str:
        try:
            result = numexpr.evaluate(query).item()
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Calculator tool does not support async")