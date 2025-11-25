# -*- coding: utf-8 -*-
"""Example of using native LangGraph support with a real agent."""

import os
from contextlib import asynccontextmanager

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from typing_extensions import TypedDict

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)

# Import the native LangGraph decorator
# from agentscope_runtime.engine.frameworks.langgraph.decorators import native_langgraph_handler

# Set environment variables for LangChain
os.environ["LANGSMITH_OTLP_HTTP_ENDPOINT"] = "http://localhost:4318"
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_OTEL_ONLY"] = "true"
# other environment variables, such as: OTEL_EXPORTER_OTLP_PROTOCOL, OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_SERVICE_NAME


# Define a simple tool
@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny with a temperature of 25°C."


# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Create tools
tools = [get_weather]

# Create the prompt
prompt = """You are a helpful assistant that can provide weather information.
Use the get_weather tool when asked about weather."""


# Define State type
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    """Build LangGraph workflow"""
    # Create the agent
    agent = create_agent(llm, tools, prompt)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add a single node that runs the agent
    workflow.add_node(
        "agent", lambda state: {"messages": [agent.invoke(state)]}
    )

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # Compile graph
    graph = workflow.compile()
    return graph


# Create the AgentApp instance
agent_app = AgentApp(
    app_name="LangGraphWeatherAgent",
    app_description="A LangGraph-based weather assistant",
)


# Initialize services as instance variables
@agent_app.init
async def init_func(self):
    self.state_service = InMemoryStateService()
    self.session_service = InMemorySessionHistoryService()

    await self.state_service.start()
    await self.session_service.start()


@agent_app.shutdown
async def shutdown_func(self):
    await self.state_service.stop()
    await self.session_service.stop()


# For LangGraph framework, native support is automatically enabled
@agent_app.query(framework="langgraph")
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    **kwargs,
):
    """LangGraph query handler with automatic native support.

    This function demonstrates the simplified usage of LangGraph with native support.
    With the @agent_app.query(framework="langgraph") decorator, native support is
    automatically enabled, allowing you to directly yield LangGraph messages without
    manual conversion.
    """
    # Extract session information
    session_id = request.session_id
    user_id = request.user_id

    # Load state if available
    state = await self.state_service.export_state(
        session_id=session_id,
        user_id=user_id,
    )

    # Build the graph agent
    weather_agent = build_graph()

    # Process the messages through the agent with streaming
    async for chunk in weather_agent.astream({"messages": msgs}):
        # Directly yield LangGraph messages without manual conversion
        # The native adapter will handle the conversion automatically
        yield chunk

    # Save state if needed
    # For this example, we don't need to save complex state


if __name__ == "__main__":
    agent_app.run(host="127.0.0.1", port=8090)
