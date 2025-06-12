# agent_utils.py
import os
from pydantic_ai.agent import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define and export the agent
agent = Agent(
    "groq:llama-3.1-8b-instant",
    tools=[tavily_search_tool(TAVILY_API_KEY)],
    system_prompt="Search Tavily for the given query and return the results.",
)

def get_search_results(query: str) -> str:
    result = agent.run_sync(query)

    # Case 1: If result has an .output attribute
    if hasattr(result, "output"):
        return str(result.output).strip()

    # Case 2: If result is a dict-like response (tool traces or intermediate steps)
    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"]).strip()
        elif "steps" in result:
            outputs = []
            for step in result["steps"]:
                if "output" in step:
                    outputs.append(str(step["output"]))
            return "\n".join(outputs)

    return "❌ Unable to retrieve proper output. Try rephrasing your query."
# If no output is found, return a default message
    return "❌ No results found. Please try a different query." 