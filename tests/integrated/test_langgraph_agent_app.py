# -*- coding: utf-8 -*-
# pylint:disable=redefined-outer-name, unused-argument
"""Integration test for LangGraph AgentApp."""
import os
import multiprocessing
import time
import json
from typing import TypedDict, Annotated

import aiohttp
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from agentscope_runtime.adapters.langgraph.memory import (
    LangGraphSessionHistoryMemory,
)
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)
from langgraph.graph import StateGraph, MessagesState, START, END


PORT = 8091  # Use different port from other tests


# def mock_llm(state: MessagesState):
#     return {"messages": [{"role": "ai", "content": "hello world"}]}
#
#
# def build_graph():
#     """Build a simple LangGraph workflow."""
#     # Create the graph
#     graph = StateGraph(MessagesState)
#     graph.add_node(mock_llm)
#     graph.add_edge(START, "mock_llm")
#     graph.add_edge("mock_llm", END)
#     graph = graph.compile()
#     return graph


class State(TypedDict):
    topic: str
    joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}


def build_graph():
    graph = (
        StateGraph(State)
        .add_node(refine_topic)
        .add_node(generate_joke)
        .add_edge(START, "refine_topic")
        .add_edge("refine_topic", "generate_joke")
        .add_edge("generate_joke", END)
        .compile()
    )
    return graph


def run_langgraph_app():
    """Start LangGraph AgentApp with streaming output enabled."""
    agent_app = AgentApp(
        app_name="LangGraphAgent",
        app_description="A LangGraph-based assistant",
    )

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
        request: AgentRequest = None,
        **kwargs,
    ):
        session_id = request.session_id
        user_id = request.user_id

        # Process input messages from request
        graph = build_graph()

        # Run the chain
        # response = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
        for chunk in graph.stream(
            {"topic": "ice cream"}, stream_mode="updates"
        ):
            # print("chunk:",chunk)
            yield chunk, False

        # Yield the response with last flag
        a = chunk
        yield chunk, True

    agent_app.run(host="127.0.0.1", port=PORT)


@pytest.fixture(scope="module")
def start_langgraph_app():
    """Launch LangGraph AgentApp in a separate process before the async tests."""
    proc = multiprocessing.Process(target=run_langgraph_app)
    proc.start()
    import socket

    for _ in range(50):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("localhost", PORT))
            s.close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.terminate()
        pytest.fail("LangGraph server did not start within timeout")

    yield
    proc.terminate()
    proc.join()


@pytest.mark.asyncio
async def test_langgraph_process_endpoint_stream_async(start_langgraph_app):
    """
    Async test for streaming /process endpoint (SSE, multiple JSON events) with LangGraph.
    """
    url = f"http://localhost:{PORT}/process"
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"},
                ],
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_response = False

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue

                line = chunk.decode("utf-8").strip()
                print("line:", line)


@pytest.mark.asyncio
async def test_langgraph_multi_turn_stream_async(start_langgraph_app):
    """
    Async test for multi-turn conversation with streaming output using LangGraph.
    """
    session_id = "langgraph_test_session"

    url = f"http://localhost:{PORT}/process"

    # First turn
    async with aiohttp.ClientSession() as session:
        payload1 = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello LangGraph!"}],
                },
            ],
            "session_id": session_id,
        }
        async with session.post(url, json=payload1) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if (
                    line.startswith("data:")
                    and line[len("data:") :].strip() == "[DONE]"
                ):
                    break

    # Second turn
    payload2 = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "How are you?"}],
            },
        ],
        "session_id": session_id,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload2) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_response = False

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "output" in event:
                        try:
                            text_content = event["output"][0]["content"][0][
                                "text"
                            ]
                            if "LangGraph Echo:" in text_content:
                                found_response = True
                        except Exception:
                            pass

            assert (
                found_response
            ), "Did not find expected response in the second turn output"


if __name__ == "__main__":
    pytest.main([__file__])
