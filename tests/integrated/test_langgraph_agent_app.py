# -*- coding: utf-8 -*-
# pylint: disable=all

import json
import multiprocessing
import time

import aiohttp
import pytest
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


PORT = 8090  # Use different port from other tests

@tool
def get_weather(location: str, date: str) -> str:
    """Get the weather for a location and date."""
    print(f"Getting weather for {location} on {date}...")
    return f"The weather in {location} is sunny with a temperature of 25°C."

def run_langgraph_app():
    """Start LangGraph AgentApp with streaming output enabled."""
    agent_app = AgentApp(
        app_name="LangGraphAgent",
        app_description="A LangGraph-based assistant",
    )

    @agent_app.init
    async def init_func(self):
        pass

    @agent_app.shutdown
    async def shutdown_func(self):
        pass

    @agent_app.query(framework="langgraph")
    async def query_func(
        self,
        msgs,
        request: AgentRequest = None,
        **kwargs,
    ):
        session_id = request.session_id
        user_id = request.user_id
        print(f"Received query from user {user_id} with session {session_id}")
        tools = [get_weather]
        # Choose the LLM that will drive the agent
        llm = ChatOpenAI(
            model="Qwen3-30B-A3B-Instruct-2507",
            api_key="Y2VhOWM1ZmZmZjgzMTJmN2NjNzNiZWE5MzRmYzEzNDlhZjBmNmNlMA==",
            base_url="http://1095312831785714.cn-shanghai.pai-eas.aliyuncs.com/api/predict/wzy_debug2/v1",
            streaming=True,
        )

        prompt = """You are a proactive research assistant. """

        agent = create_agent(
            llm,
            tools,
            system_prompt=prompt,
            name="LangGraphAgent",
        )
        async for chunk, meta_data in agent.astream(
                input={"messages": msgs, "session_id": session_id, "user_id": user_id},
                stream_mode="messages",
                config={"configurable": {"thread_id": session_id}},
        ):
            print("herehere chunk: ", chunk)
            os.system("pip list | grep fast")
            is_last_chunk = (
                True if getattr(chunk, "chunk_position", "") == "last" else False
            )
            if getattr(chunk, "response_metadata", "") and "finish_reason" in chunk.response_metadata:
                is_last_chunk = True
                print("herehere is last_chunk: ", is_last_chunk)
            yield chunk, is_last_chunk

    agent_app.run(host="127.0.0.1", port=PORT)


@pytest.fixture(scope="module")
def start_langgraph_app():
    """Launch AgentApp in a separate process before the async tests."""
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
async def test_langgraph_stream_async(start_langgraph_app):
    print("=="*20)
    print("test_langgraph_stream_async")
    @dataclass
    class MyState:
        topic: str
        joke: str = ""

    model = ChatOpenAI(
        model="Qwen3-30B-A3B-Instruct-2507",
        api_key="Y2VhOWM1ZmZmZjgzMTJmN2NjNzNiZWE5MzRmYzEzNDlhZjBmNmNlMA==",
        base_url="http://1095312831785714.cn-shanghai.pai-eas.aliyuncs.com/api/predict/wzy_debug2/v1",
        streaming=True,
    )
    found_response = False

    def call_model(state: MyState):
        """Call the LLM to generate a joke about a topic"""
        # Note that message events are emitted even when the LLM is run using .invoke rather than .stream
        model_response = model.invoke(
            [
                {"role": "user", "content": f"Generate a joke about {state.topic}"}
            ]
        )
        return {"joke": model_response.content}

    graph = (
        StateGraph(MyState)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .compile()
    )

    for message_chunk, metadata in graph.stream(
            {"topic": "ice cream"},
            stream_mode="messages",
    ):
        print("message_chunk: ", message_chunk)
        if message_chunk.content:
            print(message_chunk.content)

    print("==" * 20)

@pytest.mark.asyncio
async def test_langgraph_process_endpoint_stream_async(start_langgraph_app):
    """
    Async test for streaming /process endpoint (SSE, multiple JSON events).
    """
    url = f"http://localhost:{PORT}/process"
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, What is the capital of France?"},
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
                # chunks.append(chunk.decode("utf-8").strip())
                line = chunk.decode("utf-8").strip()
                print("herehere line: ", line)
                # SSE lines start with "data:"
                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        # Ignore non‑JSON keepalive messages or partial lines
                        continue

                    # Check if this event has "output" from the assistant
                    if "output" in event:
                        try:
                            text_content = event["output"][0]["content"][0][
                                "text"
                            ].lower()
                            print("text_content: ", text_content)
                            if "paris" in text_content:
                                found_response = True
                        except Exception:
                            # Structure may differ; ignore
                            pass

            # # Final assertion — we must have seen "paris" in at least one event
            # assert (
            #     found_response
            # ), "Did not find 'paris' in any streamed output event"


@pytest.mark.asyncio
async def test_langgraph_multi_turn_stream_async(start_langgraph_app):
    """
    Async test for multi-turn conversation with streaming output.
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
            # Simply consume the stream without detailed checking
            chunks = []
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            found_response = False
            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

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

            chunks = []
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            found_response = False
            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

                # Check if this event has "output" from the assistant
                if "output" in event:
                    try:
                        text_content = event["output"][-1]["content"][0][
                            "text"
                        ].lower()
                        if text_content:
                            found_response = True
                    except Exception:
                        # Structure may differ; ignore
                        pass

            assert (
                found_response
            ), "Did not find expected response in the second turn output"


if __name__ == "__main__":
    pytest.main([__file__])
