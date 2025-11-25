# -*- coding: utf-8 -*-
from langgraph.graph import StateGraph, MessagesState, START, END
import os
from contextlib import asynccontextmanager

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from my_tools import iqs_generic_search
from my_tools import read_plan_file
from my_tools import update_plan_file
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

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}


graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)
graph_nodes = list(graph.get_graph().nodes.keys())
final_node = graph_nodes[-1]


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

    # Process the messages through the agent with streaming
    # Using stream instead of invoke for better streaming support
    complete_chunk = {}
    for chunk in graph.stream({"topic": "ice cream"}, stream_mode="updates"):
        complete_chunk.update(chunk)
        # # 检查当前chunk是否来自最终节点
        # current_node = list(chunk.keys())[0]
        # is_last = current_node == final_node
        yield chunk, False

    # Yield the final message with last flag set to True
    yield complete_chunk, True


if __name__ == "__main__":
    agent_app.run(host="127.0.0.1", port=8090)
