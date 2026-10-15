
"""
Web Research Agent using LangGraph + Tavily Search + Ollama.

Searches the web for a given topic, synthesizes findings using a
local Ollama model, and returns a structured research report.

Usage:
    python agent.py
    python agent.py --query "latest advances in quantum computing"
"""

import argparse
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables from .env
load_dotenv()


# Define the state shared between LangGraph nodes
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_results: list[dict]
    report: str


# Step 1: Search the web using Tavily
def search_web(state: ResearchState) -> dict:
    print("🔎 Searching the web with Tavily...")

    tool = TavilySearch(max_results=5)

    # Tavily expects the query as a dictionary
    raw_results = tool.invoke({"query": state["query"]})

    # Normalize the search results
    if isinstance(raw_results, dict):
        results = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        results = raw_results
    else:
        results = []

    print(f"✅ Retrieved {len(results)} search results.")

    return {"search_results": results}


# Step 2: Generate a research report using Ollama locally
def synthesize_report(state: ResearchState) -> dict:
    print("🤖 Generating the report with Ollama...")

    # This model runs locally through Ollama.
    # Make sure you have downloaded it with:
    # ollama pull llama3.2:3b
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    # Prepare the search results for the language model
    results_text = "\n\n".join(
        (
            f"Source: {result.get('url', 'N/A')}\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"Content: {str(result.get('content', ''))[:500]}"
        )
        for result in state["search_results"]
    )

    if not results_text:
        results_text = "No useful search results were returned."

    messages = [
        SystemMessage(
            content=(
                "You are a web research analyst. "
                "Create a clear, structured research report with these sections: "
                "1. Summary "
                "2. Key Findings (bullet points) "
                "3. Sources (include source URLs supplied in the search results). "
                "Use the provided search results as evidence. "
                "Do not invent facts or sources. "
                "If the evidence is insufficient, clearly say so."
            )
        ),
        HumanMessage(
            content=(
                f"Research query: {state['query']}\n\n"
                f"Search results:\n{results_text}"
            )
        ),
    ]

    response = llm.invoke(messages)

    # Ensure the report is returned as printable text
    report = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    print("✅ Report generated successfully.")

    return {
        "report": report,
        "messages": [response],
    }


# Step 3: Build the LangGraph workflow
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_web)
    graph.add_node("synthesize", synthesize_report)

    # Workflow starts with web search
    graph.set_entry_point("search")

    # After searching, generate the report
    graph.add_edge("search", "synthesize")

    # Finish after generating the report
    graph.add_edge("synthesize", END)

    return graph.compile()


# Step 4: Run the agent
def main():
    parser = argparse.ArgumentParser(
        description="Web Research Agent using Ollama"
    )

    parser.add_argument(
        "--query",
        default="WHaT is Gen aI",
        help="Topic to research",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔍 WEB RESEARCH AGENT")
    print("=" * 60)
    print(f"Research query: {args.query}\n")

    agent = build_graph()

    initial_state = {
        "query": args.query,
        "messages": [],
        "search_results": [],
        "report": "",
    }

    result = agent.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📄 RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])
    print("\n" + "=" * 60)
    print("✅ Research completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
