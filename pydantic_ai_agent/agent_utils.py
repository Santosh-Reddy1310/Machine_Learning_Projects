import streamlit as st
import os
from pydantic_ai.agent import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

# Fetch keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]

# Set for downstream usage
os.environ["GROQ_API_KEY"] = groq_api_key

# Define and export the agent
agent = Agent(
    "groq:llama-3.1-8b-instant",
    tools=[tavily_search_tool(tavily_api_key)],
    system_prompt="Search Tavily for the given query and return the results.",
)

def get_search_results(query: str) -> str:
    result = agent.run_sync(query)

    # Handle different result types
    if hasattr(result, "output"):
        return str(result.output).strip()

    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"]).strip()
        elif "steps" in result:
            return "\n".join(str(step.get("output", "")) for step in result["steps"])

    return "❌ Unable to retrieve proper output. Try rephrasing your query."
