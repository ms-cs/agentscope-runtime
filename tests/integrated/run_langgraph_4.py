# -*- coding: utf-8 -*-
from langchain_core.messages import AIMessage, HumanMessage
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langchain.tools import tool


from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 定义 State 类型
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def add_time(state: AgentState):
    new_message = HumanMessage(content="今天是 2025年 8 月 21 日")
    return {"messages": [new_message]}


@tool
def get_weather(location: str, date: str) -> str:
    """Get the weather for a location and date."""
    print(f"Getting weather for {location} on {date}...")
    return f"The weather in {location} is sunny with a temperature of 25°C."


tools = [get_weather]
# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = """You are a proactive research assistant. """

agent = create_agent(llm, tools, system_prompt=prompt)

# Create the graph
workflow = StateGraph(AgentState)

# Add a single node that runs the agent
workflow.add_node("agent", agent)
workflow.add_node("add_time", add_time)

# Add edges
workflow.add_edge(START, "add_time")
workflow.add_edge("add_time", "agent")
workflow.add_edge("agent", END)

# Compile graph
graph = workflow.compile()


# Create the AgentApp instance
agent_app = AgentApp(
    app_name="LangGraphAgent",
    app_description="A LangGraph-based research assistant",
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


@agent_app.query(framework="langgraph")
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    **kwargs,
):
    # Extract session information
    session_id = request.session_id
    user_id = request.user_id

    input = {"messages": [HumanMessage(content="北京天气如何？")]}

    complete_chunk = {}
    for chunk in graph.stream(input, stream_mode="messages"):
        complete_chunk.update(chunk[0])
        yield chunk[0], False

    # Yield the final message with last flag set to True
    yield None, True


if __name__ == "__main__":
    agent_app.run(host="127.0.0.1", port=8090)
