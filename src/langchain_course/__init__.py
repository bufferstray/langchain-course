from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel, Field
from typing import List
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

class Source(BaseModel):
    """ Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources : List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


@tool
def search(query:str) -> str:
    """
    Tool that searches over the internet
    Args:
        query:
            The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm,tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings in london UK for an ai engineer using langchain on linkedin and list their details")})
    print(result)
